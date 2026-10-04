// PulseX CCTV — live RTSP camera view with on-NPU AUTO-FRAMING and automatic recording of passers-by.
// Media Foundation decodes the stream on the GPU (DXVA) into D3D11 textures; the D3D11 video processor scales them
// for the screen, for the NPU (two square 640x640 tiles) and for the 1080p recording ring. YOLO runs on the Hexagon
// NPU through QNN every frame; person boxes drive a spring-damped digital pan/zoom (a UV sub-rect of the video,
// like Windows Studio Effects "Automatic Framing"). ffmpeg on a pipe is the fallback decoder.
#define STB_IMAGE_IMPLEMENTATION
#include "third_party/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STBIW_WINDOWS_UTF8                       // 09-30: file names are UTF-8 (photos in folders with non-ASCII names)
#include "third_party/stb_image_write.h"
#include "pcore_yolo_npu.hpp"                 // pcore_npu::NpuDetector (raw QNN YOLOv10 on the NPU)
#include <pcore/gui/app_shell.hpp>
#include <d3d11.h>
#include <windows.h>
#include <shellapi.h>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include <utility>
#include <filesystem>
#include <algorithm>
#include <ctime>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <memory>
#include <chrono>
#include <cstring>
#include <deque>
#include <cmath>
#include <psapi.h>
#include <dwmapi.h>
#pragma comment(lib, "dwmapi.lib")

using namespace pcore::gui;

static std::string exe_dir(){ char b[MAX_PATH]{}; GetModuleFileNameA(nullptr,b,MAX_PATH); std::string p=b;
    auto s=p.find_last_of("\\/"); return s==std::string::npos?std::string("."):p.substr(0,s); }
#define PULSEX_CCTV_VERSION "0.6"
// 09-30 v0.2: every path this app builds itself is UTF-8 (the folder picker returns UTF-8; the recorder opens UTF-8)
static std::wstring u8w(const std::string& s){ int n=MultiByteToWideChar(CP_UTF8,0,s.c_str(),(int)s.size(),nullptr,0);
    std::wstring w(n,L'\0'); if(n) MultiByteToWideChar(CP_UTF8,0,s.c_str(),(int)s.size(),&w[0],n); return w; }
static std::string w2u8(const wchar_t* w){ if(!w) return std::string(); int n=WideCharToMultiByte(CP_UTF8,0,w,-1,nullptr,0,nullptr,nullptr);
    std::string r(n>0? n-1 : 0,'\0'); if(n>1) WideCharToMultiByte(CP_UTF8,0,w,-1,&r[0],n,nullptr,nullptr); return r; }
static std::string env_u8(const wchar_t* name){ const wchar_t* v=_wgetenv(name); return (v && *v)? w2u8(v) : std::string(); }
// 10-01: a line from a file the user wrote by hand -> UTF-8. Drops a UTF-8 BOM; text that is not valid UTF-8 was saved
// as Windows-1252 (the old Notepad default) and is converted. ASCII passes unchanged.
static std::string text_to_utf8(std::string t){
    if(t.size()>=3 && (unsigned char)t[0]==0xEF && (unsigned char)t[1]==0xBB && (unsigned char)t[2]==0xBF) t.erase(0,3);
    if(t.empty() || MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, t.c_str(), (int)t.size(), nullptr, 0) > 0) return t;
    const int n=MultiByteToWideChar(1252, 0, t.c_str(), (int)t.size(), nullptr, 0);
    std::wstring w(n, L'\0'); if(n) MultiByteToWideChar(1252, 0, t.c_str(), (int)t.size(), &w[0], n);
    return w2u8(w.c_str()); }
static std::string known_folder_u8(REFKNOWNFOLDERID id){          // Videos / Pictures, wherever the user moved them
    PWSTR p=nullptr; std::string r; if(SUCCEEDED(SHGetKnownFolderPath(id,0,nullptr,&p))) r=w2u8(p); CoTaskMemFree(p); return r; }
// The tool bin dir (ffmpeg's working folder) is overridable; default = this exe's folder.
static std::string bin_dir(){ if(const char* e=std::getenv("PULSECORE_BIN")){ if(*e) return e; } return exe_dir(); }
static std::string ffmpeg_exe(){ if(const char* e=std::getenv("PULSECORE_FFMPEG")){ if(*e) return e; }
    { std::string p=exe_dir()+"\\ffmpeg.exe"; if(GetFileAttributesA(p.c_str())!=INVALID_FILE_ATTRIBUTES) return p; }   // 10-01: next to the exe
    { char b[MAX_PATH*2]={}; if(SearchPathA(nullptr,"ffmpeg.exe",nullptr,(DWORD)sizeof b,b,nullptr)) return b; }       // ... or on PATH
    return exe_dir()+"\\..\\tools\\ffmpeg\\ffmpeg.exe"; }
// Tapo RTSP URL (rtsp://user:pass@ip:554/stream1) — from env PULSECORE_TAPO_RTSP, else a one-line file
// tapo_rtsp.txt next to the exe. Kept OUT of the code so your Camera-Account password stays local.
static std::string tapo_rtsp(){
    { std::string e=env_u8(L"PULSECORE_TAPO_RTSP"); if(!e.empty()) return e; }
    std::ifstream f(std::filesystem::u8path(exe_dir()+"\\tapo_rtsp.txt")); std::string u; std::getline(f,u);
    while(!u.empty() && (u.back()=='\r'||u.back()=='\n'||u.back()==' '||u.back()=='\t')) u.pop_back();
    return text_to_utf8(u);                                           // 10-01: BOM / Windows-1252 file -> UTF-8
}
// The NPU YOLO context bin — overridable; defaults to the one pcore_yolo_npu.hpp ships with.
static const char* yolo_ctx(){ if(const char* e=std::getenv("PULSECORE_YOLO_CTX")){ if(*e) return e; }
    static std::string local = exe_dir()+"\\models\\yolov10_det_qairt_context.bin";   // 09-30: models next to the exe first
    if(GetFileAttributesA(local.c_str())!=INVALID_FILE_ATTRIBUTES) return local.c_str();
    return pcore_npu::CTX_BIN_DEFAULT; }
// Where captures go — a stable user location (Pictures), NOT next to the exe, so they never land in the repo/build.
// 09-30: one storage folder for clips and photos, chosen by the user (right click on ●), kept in %APPDATA%\PulseX\cctv.ini
static std::string g_save_dir; static bool g_auto_record=true;
static float g_ignore_top=0.0f;   // 09-30: ignore detections + motion ABOVE this normalized y (0 = off), cctv.ini ignore_top
static std::string g_camera_name;   // 10-01: optional cctv.ini camera_name, shown in the subtitle
static std::string settings_path(){ std::string a=env_u8(L"APPDATA"); return (a.empty()? std::string(".") : a)+"\\PulseX\\cctv.ini"; }
static void load_settings(){
    std::ifstream f(std::filesystem::u8path(settings_path())); std::string line;
    while(std::getline(f,line)){ line=text_to_utf8(line); auto eq=line.find('='); if(eq==std::string::npos) continue;   // 10-01: hand-edited ini
        std::string k=line.substr(0,eq), v=line.substr(eq+1); while(!v.empty() && (v.back()=='\r'||v.back()==' ')) v.pop_back();
        if(k=="save_dir") g_save_dir=v; else if(k=="auto_record") g_auto_record=(v!="0");
        else if(k=="camera_name") g_camera_name=v;
        else if(k=="ignore_top"){ float t=(float)std::atof(v.c_str()); g_ignore_top = t<0? 0 : t>0.95f? 0.95f : t; } } }
static void save_settings(){
    std::error_code ec; std::filesystem::create_directories(std::filesystem::u8path(settings_path()).parent_path(), ec);
    std::ofstream f(std::filesystem::u8path(settings_path())); f<<"save_dir="<<g_save_dir<<"\nauto_record="<<(g_auto_record?1:0)<<"\nignore_top="<<g_ignore_top<<"\n"; if(!g_camera_name.empty()) f<<"camera_name="<<g_camera_name<<"\n"; }
static std::string clips_dir(){
    { std::string e=env_u8(L"PULSECORE_CCTV_REC_DIR"); if(!e.empty()) return e; }
    if(!g_save_dir.empty()) return g_save_dir;
    { std::string v=known_folder_u8(FOLDERID_Videos); if(!v.empty()) return v+"\\PulseX CCTV"; }
    return exe_dir()+"\\clips"; }
static std::string snapshots_dir(){
    { std::string e=env_u8(L"PULSECORE_SNAPSHOTS"); if(!e.empty()) return e; }
    return clips_dir();               // 09-30: one folder for clips AND photos (the chosen one, else Videos\PulseX CCTV)
}

// Live pipe format: RAW RGBA (no MJPEG re-encode/decode) — resolution is nearly CPU-free, only memory bandwidth.
// Full 2K (2560x1440) @ 20 fps for maximum zoom sharpness — raw RGBA keeps CPU near-zero (memory bandwidth only).
static const int PIPE_W = 2560, PIPE_H = 1440, PIPE_FPS = 20;

// ── Exclusion masks: polygons (normalized coords) where detection + motion are ignored (road, trees, …) ──
typedef std::vector<std::pair<float,float>> Poly;
static bool point_in_poly(float x,float y,const Poly& p){        // standard ray-casting point-in-polygon
    bool in=false; size_t n=p.size(); if(n<3) return false;
    for(size_t i=0,j=n-1;i<n;j=i++){ float xi=p[i].first,yi=p[i].second,xj=p[j].first,yj=p[j].second;
        if(((yi>y)!=(yj>y)) && (x<(xj-xi)*(y-yi)/(yj-yi)+xi)) in=!in; }
    return in;
}
static bool in_any(float x,float y,const std::vector<Poly>& ms){ for(auto& p:ms) if(point_in_poly(x,y,p)) return true; return false; }
static std::string mask_path(){ if(const char* e=std::getenv("PULSECORE_TAPO_MASK")){ if(*e) return e; } return exe_dir()+"\\tapo_mask.txt"; }
static std::vector<Poly> load_masks(){
    std::vector<Poly> out; std::ifstream f(std::filesystem::u8path(mask_path())); std::string line;
    while(std::getline(f,line)){
        line=text_to_utf8(line);                                     // 10-01: a BOM would hide the first polygon
        if(line.empty()||line[0]=='#') continue;
        Poly poly; std::stringstream ss(line); std::string tok;
        while(ss>>tok){ auto c=tok.find(','); if(c==std::string::npos) continue;
            poly.push_back({(float)std::atof(tok.substr(0,c).c_str()),(float)std::atof(tok.substr(c+1).c_str())}); }
        if(poly.size()>=3) out.push_back(poly);
    }
    return out;
}
static void save_masks(const std::vector<Poly>& ms){
    std::ofstream f(mask_path());
    f<<"# PulseCore CCTV exclusion zones — one polygon per line, space-separated normalized x,y points.\n";
    f<<"# Detections + motion whose center is INSIDE any polygon are ignored (road, trees, …). Edit in-app.\n";
    for(auto& p:ms){ for(size_t i=0;i<p.size();i++){ if(i) f<<' '; f<<p[i].first<<','<<p[i].second; } f<<'\n'; }
}

// Frames are shared, not copied (09-30): a buffer is a shared_ptr; the pool reuses any buffer only the pool holds.
using FrameBuf = std::shared_ptr<std::vector<unsigned char>>;
struct FramePool {
    std::vector<FrameBuf> bufs;
    FrameBuf take(size_t bytes){
        for(auto& b : bufs) if(b.use_count()==1){ if(b->size()!=bytes) b->resize(bytes); return b; }
        bufs.push_back(std::make_shared<std::vector<unsigned char>>(bytes)); return bufs.back();
    }
};
// ── Memory purge like zimage_launch.c / PulseCore Memory (09-30) ─────────────────────────────────────────────────
static long long system_cache_mb(){
    PERFORMANCE_INFORMATION pi{}; pi.cb=sizeof(pi);
    if(!K32GetPerformanceInfo(&pi,sizeof(pi))) return -1;
    return (long long)pi.SystemCache*(long long)pi.PageSize/(1024*1024); }
static bool enable_privilege(const wchar_t* name){
    HANDLE tok=nullptr; if(!OpenProcessToken(GetCurrentProcess(),TOKEN_ADJUST_PRIVILEGES|TOKEN_QUERY,&tok)) return false;
    LUID luid; bool ok=false;
    if(LookupPrivilegeValueW(nullptr,name,&luid)){ TOKEN_PRIVILEGES tp{}; tp.PrivilegeCount=1; tp.Privileges[0].Luid=luid;
        tp.Privileges[0].Attributes=SE_PRIVILEGE_ENABLED; AdjustTokenPrivileges(tok,FALSE,&tp,sizeof(tp),nullptr,nullptr);
        ok=(GetLastError()==ERROR_SUCCESS); }                 // ERROR_NOT_ALL_ASSIGNED without admin
    CloseHandle(tok); return ok; }
// returns MB released from the system cache, -2 = needs administrator, -1 = failed/off
static long long purge_memory(bool full){
    if(const char* e=std::getenv("PULSECORE_CCTV_PURGE")){ if(*e && (!strcmp(e,"off")||!strcmp(e,"0"))) return -1; }
    typedef LONG (WINAPI* nt_set_t)(int,PVOID,ULONG);
    HMODULE nt=GetModuleHandleW(L"ntdll.dll"); nt_set_t set=nt? (nt_set_t)GetProcAddress(nt,"NtSetSystemInformation") : nullptr;
    if(!set) return -1;
    if(!enable_privilege(L"SeProfileSingleProcessPrivilege")) return -2;
    const long long before=system_cache_mb();
    enum { SystemMemoryListInformation=80, MemoryEmptyWorkingSets=2, MemoryFlushModifiedList=3, MemoryPurgeStandbyList=4 };
    int cmd;
    if(full){ cmd=MemoryEmptyWorkingSets; set(SystemMemoryListInformation,&cmd,sizeof(cmd)); }
    cmd=MemoryFlushModifiedList; set(SystemMemoryListInformation,&cmd,sizeof(cmd));
    cmd=MemoryPurgeStandbyList; if(set(SystemMemoryListInformation,&cmd,sizeof(cmd))!=0) return -1;
    const long long after=system_cache_mb();
    return (before>=0&&after>=0)? before-after : 0; }
static std::atomic<long long> g_purged_at_start{-1};   // 09-30: written by the purge thread

static const bool g_follow_old=[](){ const char* e=std::getenv("PULSECORE_CCTV_FOLLOW"); return e && !strcmp(e,"old"); }();   // A/B hook
static double now_ms(){ return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static double g_t_start=0;                                   // 09-30: startup timeline origin (main)


#include "sr_npu.hpp"                         // 09-30: QuickSRNet on the NPU (AI sharpening)
#include "mf_decoder.hpp"                     // 09-30: in-app decode, GPU -> NPU (Studio Effects design)
#include "rec_recorder.hpp"                   // 09-30: auto-recording of passers-by (GPU H.264)
#include <shobjidl.h>

struct Cctv {
    void* tex=nullptr; int tw=0,th=0;
    ID3D11Texture2D* dyn_tex=nullptr;                 // ONE dynamic texture, rewritten each new frame (09-30)
    // receipt (09-30): frames in / shown per second, last upload time
    std::atomic<int> n_in{0}; std::atomic<long long> n_total{0}; int n_shown=0; double stat_t0=0, fps_in=0, fps_shown=0, up_ms=0;
    // Tapo LIVE via pipe: ffmpeg decodes RTSP -> MJPEG onto a PIPE (no file -> no reader/writer lock). A reader
    // thread pulls the freshest JPEG and decodes it to RGBA for the render thread.
    void* ff_proc=nullptr; void* ff_rd=nullptr; std::string ff_log;
    void* ff_in=nullptr;                                 // ffmpeg's stdin: 'q' = graceful stop (TEARDOWN to the camera)
    double ff_started_ms=0; std::atomic<long long> last_frame_ms{0}; int reconnects=0;
    // 09-30 timing log: arrival (reader) and display (render) time of each frame, first 90 s after tapo_start
    std::FILE* tlog=nullptr; double tlog_t0=0;
    void tlog_put(const char* what, unsigned long long seq){
        if(!tlog) return; const double t=now_ms()-tlog_t0;
        if(t>90000.0){ std::fclose(tlog); tlog=nullptr; return; }
        std::fprintf(tlog, "%s %llu %.1f\n", what, seq, t); std::fflush(tlog); }
    std::thread pipe_thread; std::atomic<bool> pipe_run{false};
    std::mutex frame_mtx; FrameBuf frame_rgba; int fw=0,fh=0; unsigned long long frame_seq=0;
    std::condition_variable frame_cv;          // 09-30: signalled per published frame -> the detector runs in step
    FramePool pool;                                    // reader-thread only
    // playout buffer (09-30): queued frames (seq, buffer, arrival ms), guarded by frame_mtx
    struct QF { unsigned long long seq; FrameBuf buf; double t_in; int slot=-1; };   // slot: MF ring texture (09-30)
    std::deque<QF> playq; double period_ms=1000.0/30.0;                   // P: frame interval (least squares)
    std::deque<std::pair<unsigned long long,double>> arr;                // (seq, arrival ms) of the last 1800 frames
    void fit_period(){                                                   // slope of t vs seq, local origin
        const size_t n=arr.size(); if(n<30) return;
        const double s0=(double)arr.front().first, t0=arr.front().second; double Ss=0,St=0,Sss=0,Sst=0;
        for(auto& a: arr){ const double x=(double)a.first-s0, y=a.second-t0; Ss+=x; St+=y; Sss+=x*x; Sst+=x*y; }
        const double den=n*Sss-Ss*Ss; if(den>0){ const double p=(n*Sst-Ss*St)/den; if(p>5 && p<500) period_ms=p; } }
    double due_ms(unsigned long long seq) const {                       // earliest-arrival line + buffer
        double best=1e300; const size_t n=arr.size(), from=n>300? n-300 : 0;
        for(size_t i=from;i<n;i++){ const double d=arr[i].second+((double)seq-(double)arr[i].first)*period_ms; if(d<best) best=d; }
        return best+buffer_ms; }
    int buffer_ms=[](){ const char* e=std::getenv("PULSECORE_CCTV_BUFFER_MS"); return (e&&*e)? std::atoi(e) : 250; }();
    double lat_ms=0;                                                     // receipt: current display delay
    // last frame handed to the display — kept so "Take photo" can save exactly what's on screen (a pointer, no copy).
    FrameBuf last_rgba; int last_w=0,last_h=0; unsigned long long shown_seq=0;
    std::string saved_msg; double saved_at=-1e9;
    double photo_flash_at=-1e9; bool photo_ok=false; std::string photo_path;    // 10-04: the flash and the card
    bool show_info=[](){ const char* e=std::getenv("PULSECORE_CCTV_INFO"); return e && *e=='1'; }();   // 09-30: numbers on demand

    // ── in-app decode (09-30): Media Foundation + D3D11 video processor; ffmpeg pipe = fallback ──
    int dec_mode=[](){ const char* e=std::getenv("PULSECORE_CCTV_DECODER"); return (e && !strcmp(e,"ffmpeg"))? 1 : 0; }();  // 0 MF, 1 ffmpeg
    std::shared_ptr<MfDecoder> mf; std::atomic<bool> mf_on{false}; bool stream_started=false; int mf_fail=0; std::string dec_note;
    std::vector<int> slot_state; int shown_slot=-1;                  // guarded by frame_mtx: 0 free, 1 queued/reserved, 2 shown
    unsigned long long mf_seq=0;                                     // display frame counter in MF mode (frame_seq = NPU images)
    std::atomic<bool> det_inflight{false}, sr_inflight{false};               // skip-when-busy (Studio Effects)
    FrameBuf sr_src; float sr_src_rect[4]={0,0,1,1}; unsigned long long sr_src_seq=0;   // MF: GPU-made SR crop (sr_mtx)
    FramePool det_pool, sr_pool;                                     // decoder thread only
    std::FILE* mflog=nullptr; std::mutex mflog_mtx;
    void mlog(const std::string& m){ std::lock_guard<std::mutex> lk(mflog_mtx);
        if(!mflog){ char tmp[MAX_PATH]{}; GetTempPathA(MAX_PATH,tmp); mflog=std::fopen((std::string(tmp)+"pulsecore_cctv_mf.log").c_str(),"w"); }
        if(mflog){ std::time_t t=std::time(nullptr); std::tm lt{}; localtime_s(&lt,&t); char ts[16]; std::strftime(ts,sizeof ts,"%H:%M:%S",&lt);
            std::fprintf(mflog,"%s %s\n",ts,m.c_str()); std::fflush(mflog); } }
    int mf_acquire(){                                                // a slot the decoder may overwrite
        std::lock_guard<std::mutex> lk(frame_mtx);
        for(size_t i=0;i<slot_state.size();i++) if(slot_state[i]==0){ slot_state[i]=1; return (int)i; }
        while(!playq.empty()){ QF f=playq.front(); playq.pop_front(); if(f.slot>=0) return f.slot; }   // drop the oldest queued frame
        return -1; }
    void mf_publish(int slot, double tn){
        unsigned long long sq;
        { std::lock_guard<std::mutex> lk(frame_mtx); sq=++mf_seq;
          arr.push_back({sq,tn}); if(arr.size()>1800) arr.pop_front();
          if(sq%30==0) fit_period();
          playq.push_back({sq,nullptr,tn,slot});
          const size_t cap=(size_t)(std::max(0,buffer_ms)/period_ms)+6;
          while(playq.size()>cap){ QF f=playq.front(); playq.pop_front(); if(f.slot>=0) slot_state[f.slot]=0; } }
        tlog_put("in", sq); t_mark(4,"first frame in (decoded)");
        n_in.fetch_add(1, std::memory_order_relaxed); n_total.fetch_add(1, std::memory_order_relaxed);
        last_frame_ms.store((long long)now_ms()); }
    int det_tile_in=-1; float det_u0_in=0, det_u1_in=1;              // 09-30: which tile frame_rgba is (frame_mtx)
    void mf_publish_det(const uint8_t* px,int w,int h,int pitch,bool bgra,int tile,float u0,float u1){
        FrameBuf b=det_pool.take((size_t)w*h*4); MfDecoder::copy_rgba(b->data(),px,w,h,pitch,bgra);
        { std::lock_guard<std::mutex> lk(frame_mtx); frame_rgba=b; fw=w; fh=h; ++frame_seq; det_tile_in=tile; det_u0_in=u0; det_u1_in=u1; }
        frame_cv.notify_all(); }
    bool mf_want_sr(float r[4]){
        if(!sr_on.load() || sr_state.load()!=1 || sr_inflight.load()) return false;
        std::lock_guard<std::mutex> lk(sr_mtx); if(!sr_want) return false; std::memcpy(r,sr_req,4*sizeof(float)); return true; }
    void mf_publish_sr(const uint8_t* px,int w,int h,int pitch,bool bgra,const float r[4]){
        FrameBuf b=sr_pool.take((size_t)w*h*4); MfDecoder::copy_rgba(b->data(),px,w,h,pitch,bgra);
        std::lock_guard<std::mutex> lk(sr_mtx); sr_src=b; std::memcpy(sr_src_rect,r,4*sizeof(float)); sr_src_seq++; }
    void mf_start(){
        std::string url=tapo_rtsp(); if(url.empty() || !device()) return;
        mf_stop();
        const int cap=(int)(std::max(0,buffer_ms)/(1000.0/30.0))+6;
        { std::lock_guard<std::mutex> lk(frame_mtx); playq.clear(); arr.clear(); shown_slot=-1;
          slot_state.assign((size_t)std::min(40,cap+4),0); }
        auto d=std::make_shared<MfDecoder>(); d->nslots=(int)slot_state.size();
        d->det_w=pcore_npu::YS; d->det_h=pcore_npu::YS; d->sr_w=SR_W; d->sr_h=SR_H;
        d->acquire_slot=[this](){ return mf_acquire(); };
        d->publish=[this](int sl,double t){ mf_publish(sl,t); };
        d->want_det=[this](){ return det_state.load()==1 && !det_inflight.load(); };
        d->det_enabled=[this](){ return det_state.load()==1; };
        d->publish_det=[this](const uint8_t* p,int w,int h,int pi,bool bg,int tl,float a,float b){ mf_publish_det(p,w,h,pi,bg,tl,a,b); };
        { const char* e=std::getenv("PULSECORE_CCTV_TILES"); d->tiles=!(e && *e=='0'); }
        d->want_sr=[this](float r[4]){ return mf_want_sr(r); };
        d->publish_sr=[this](const uint8_t* p,int w,int h,int pi,bool bg,const float r[4]){ mf_publish_sr(p,w,h,pi,bg,r); };
        d->log=[this](const std::string& m){ mlog("[mf] "+m); };
        char tmp[MAX_PATH]{}; GetTempPathA(MAX_PATH,tmp);
        if(!tlog){ tlog=std::fopen((std::string(tmp)+"pulsecore_cctv_timing.log").c_str(),"w"); tlog_t0=now_ms(); }
        ff_started_ms=now_ms(); last_frame_ms.store(0);
        d->rec_want=rec.armed.load(); d->on_rec_frame=[this](int i,double t){ rec.frame(i,t); };
        mf=MfDecoder::start(d, device(), url); mf_on=true; rec.set_decoder(mf); }
    void mf_stop(){ rec.set_decoder(nullptr); if(mf){ mf->stop(); mf.reset(); } mf_on=false; }
    CctvRecorder rec;                                                // 09-30: auto-recording
    std::mutex note_mtx; std::string pending_note;                  // recorder thread -> status line
    std::deque<std::pair<double,bool>> rwin;                        // af_loop: recording trigger window
    std::mutex pick_mtx; std::string picked_dir; std::atomic<bool> picking{false};
    std::atomic<int> t_marks{0};                                      // bit per milestone already logged
    void t_mark(int bit, const std::string& what){ if(t_marks.fetch_or(1<<bit) & (1<<bit)) return;
        char b[160]; std::snprintf(b,sizeof b,"[t] +%.0f ms %s", now_ms()-g_t_start, what.c_str()); mlog(b); }
    void start_stream(){ if(dec_mode==0) mf_start(); else tapo_start(); }
    void stop_stream(){ mf_stop(); tapo_stop(); }

    // ── Auto-framing (on-NPU person tracking -> smoothed digital pan/zoom) ─────────────────────────────
    pcore_npu::NpuDetector det_;
    std::thread af_thread; std::atomic<bool> af_run{false};
    std::atomic<bool> auto_frame{false};      // the "Auto-frame" toggle
    std::atomic<int>  det_state{0};           // 0=loading, 1=ready, 2=unavailable (no context)
    // ── AI sharpening (09-30): QuickSRNet on the zoomed view ──
    std::mutex npu_mtx;                        // ONE graphExecute at a time on the shared QNN stack (YOLO + SR)
    // 09-30 steady NPU: detector receipt (written by af_loop, read by the status line once a frame)
    bool det_steady=true; std::atomic<int> det_n{0}; std::atomic<int> det_people{0};
    std::atomic<float> det_ms{0}, det_prep_ms{0}; double det_rate=0;
    std::atomic<long long> det_us_sum{0}, det_us_max{0};
    std::atomic<int> ps_n{0}, ps_30{0}, ps_50{0}, ps_max{0};  // 09-30: person-score evidence per second (af_loop -> log)     // exec time over the current second (af_loop adds)
    double det_mean_ms=0, det_max_ms=0, det_busy=0;          // last second's receipt (render thread)
    std::thread sr_thread; std::atomic<bool> sr_run{false}, sr_on{false};
    std::atomic<int> sr_state{0};              // 0=not loaded, 1=ready, 2=unavailable
    std::mutex sr_mtx; float sr_req[4]={0,0,1,1}; bool sr_want=false;     // requested region (render -> SR thread)
    FrameBuf sr_img; float sr_rect[4]={0,0,1,1}; unsigned long long sr_seq=0, sr_shown=0; double sr_ms=0;
    ID3D11Texture2D* sr_dyn=nullptr; void* sr_tex=nullptr;
    float sr_mix=[](){ const char* e=std::getenv("PULSECORE_CCTV_SR_MIX"); return (e&&*e)? (float)std::atof(e) : 0.5f; }();
    void sr_loop(){
        SrNpu sr; std::vector<uint8_t> in((size_t)3*SR_W*SR_H), out((size_t)3*SR_OW*SR_OH); FramePool opool;
        unsigned long long done_seq=0;
        while(sr_run.load()){
            if(!sr_on.load() || det_state.load()!=1){ Sleep(60); continue; }
            if(sr_state.load()==0){ std::lock_guard<std::mutex> lk(npu_mtx); sr_state.store(sr.load(det_,sr_ctx())? 1 : 2);
                mlog("[sr] "+std::string(sr_state.load()==1? "ready: " : "unavailable: ")+(sr.info.empty()? std::string("the context could not be opened") : sr.info)); }
            if(sr_state.load()!=1){ Sleep(200); continue; }
            float r[4]; bool want; { std::lock_guard<std::mutex> lk(sr_mtx); want=sr_want; std::memcpy(r,sr_req,sizeof r); }
            FrameBuf fb; int fw_=0,fh_=0; unsigned long long sq=0; float rc[4]={r[0],r[1],r[2],r[3]};
            if(mf_on.load()){ std::lock_guard<std::mutex> lk(sr_mtx);          // 09-30: GPU-made crop, already 512x288
                fb=sr_src; fw_=SR_W; fh_=SR_H; sq=sr_src_seq; std::memcpy(r,sr_src_rect,sizeof r); rc[0]=0; rc[1]=0; rc[2]=1; rc[3]=1; }
            else { std::lock_guard<std::mutex> lk(frame_mtx); fb=frame_rgba; fw_=fw; fh_=fh; sq=frame_seq; }
            if(!want || !fb || sq==done_seq){ Sleep(5); continue; }
            done_seq=sq; sr_inflight.store(true);
            // crop rc (normalized) -> 512x288 NCHW uint8, bilinear
            const double X0=rc[0]*fw_, Y0=rc[1]*fh_, SX=(rc[2]-rc[0])*fw_/SR_W, SY=(rc[3]-rc[1])*fh_/SR_H; const uint8_t* src=fb->data();
            for(int y=0;y<SR_H;y++){ double sy=Y0+(y+0.5)*SY-0.5; int y0=(int)std::floor(sy); double fy=sy-y0; int y1=std::min(y0+1,fh_-1); y0=std::max(0,std::min(y0,fh_-1));
                for(int x=0;x<SR_W;x++){ double sx=X0+(x+0.5)*SX-0.5; int x0=(int)std::floor(sx); double fx=sx-x0; int x1=std::min(x0+1,fw_-1); x0=std::max(0,std::min(x0,fw_-1));
                    const uint8_t *a=src+((size_t)y0*fw_+x0)*4, *b=src+((size_t)y0*fw_+x1)*4, *c=src+((size_t)y1*fw_+x0)*4, *d=src+((size_t)y1*fw_+x1)*4;
                    for(int k=0;k<3;k++){ double v=(a[k]*(1-fx)+b[k]*fx)*(1-fy)+(c[k]*(1-fx)+d[k]*fx)*fy; in[(size_t)k*SR_W*SR_H+(size_t)y*SR_W+x]=(uint8_t)(v+0.5); } } }
            bool ok; double ms; { std::lock_guard<std::mutex> lk(npu_mtx); ok=sr.run(in.data(),out.data()); ms=sr.last_ms; }
            if(!ok){ sr_state.store(2); sr_inflight.store(false); continue; }
            // NCHW -> RGBA, mixed with a bilinear enlargement of the 512x288 input (tames the edge halos)
            FrameBuf img=opool.take((size_t)SR_OW*SR_OH*4); uint8_t* o=img->data(); const float m=std::min(1.0f,std::max(0.0f,sr_mix));
            const size_t pl=(size_t)SR_OW*SR_OH;
            for(int y=0;y<SR_OH;y++){ const float sy=(y+0.5f)/4-0.5f; int y0=(int)std::floor(sy); float fy=sy-y0; int y1=std::min(y0+1,SR_H-1); y0=std::max(0,y0);
                for(int x=0;x<SR_OW;x++){ const float sx=(x+0.5f)/4-0.5f; int x0=(int)std::floor(sx); float fx=sx-x0; int x1=std::min(x0+1,SR_W-1); x0=std::max(0,x0);
                    uint8_t* px=o+((size_t)y*SR_OW+x)*4;
                    for(int k=0;k<3;k++){ const uint8_t* pk=in.data()+(size_t)k*SR_W*SR_H;
                        const float bl=(pk[y0*SR_W+x0]*(1-fx)+pk[y0*SR_W+x1]*fx)*(1-fy)+(pk[y1*SR_W+x0]*(1-fx)+pk[y1*SR_W+x1]*fx)*fy;
                        const float v=m*out[(size_t)k*pl+(size_t)y*SR_OW+x]+(1-m)*bl; px[k]=(uint8_t)(v+0.5f); }
                    px[3]=255; } }
            { std::lock_guard<std::mutex> lk(sr_mtx); sr_img=img; std::memcpy(sr_rect,r,sizeof r); sr_seq++; sr_ms=ms; }
            sr_inflight.store(false);
        }
    }
    std::mutex af_mtx;                         // guards the target crop below
    float t_cx=0.5f, t_cy=0.5f, t_h=0.5f;     // TARGET crop: center + half-size (normalized, frame-aspect square)
    struct Tgt { double t; float cx, cy, h; };   // 09-30: time-stamped targets (af_mtx) -> render applies the one of the shown frame
    std::deque<Tgt> tq;
    float v_cx=0, v_cy=0, v_h=0;               // spring velocities (render thread)
    void push_target(float cx,float cy,float h){ std::lock_guard<std::mutex> lk(af_mtx);
        t_cx=cx; t_cy=cy; t_h=h; tq.push_back({now_ms(),cx,cy,h}); while(tq.size()>96) tq.pop_front(); }
    float c_cx=0.5f, c_cy=0.5f, c_h=0.5f;     // CURRENT crop (render thread only) — eases toward the target
    float m_cx=0.5f, m_cy=0.5f, m_h=0.5f;     // MANUAL crop (render thread) — scroll to zoom, drag to pan
    unsigned long long last_person=0;          // GetTickCount64 of the last person seen
    // Exclusion masks — read by the detector thread, edited by the render thread (guarded by mask_mtx).
    std::vector<Poly> masks; std::mutex mask_mtx;
    std::atomic<bool> edit_mask{false};        // in-app "draw zones" mode (overlay + click-to-add points)
    Poly mask_wip;                             // polygon being drawn (render thread only)

    // Detection + control loop: runs YOLO on the NPU, keeps the union of person boxes, updates the target crop.
    // The heavy lifting stays off the render thread; the render thread just eases toward whatever target is set.
    void af_loop(){
        { const double t0=now_ms(); det_state.store(det_.load(yolo_ctx()) ? 1 : 2);
          char b[96]; std::snprintf(b,sizeof b,"detector %s (load %.0f ms)", det_state.load()==1? "ready" : "UNAVAILABLE", now_ms()-t0); t_mark(6,b); }
        if(det_state.load()!=1) return;
        std::vector<unsigned char> rgb;
        // Temporal filter ("must persist"): a person must be seen in ACQUIRE_HITS consecutive
        // detections before we commit to tracking — a shadow / 1-frame ghost never yanks the zoom. Once locked
        // we keep following through brief misses and only let go after RELEASE_MS with nobody in view.
        const int   DET_INTERVAL_MS = 55;  // ~15-18 fps detection (NPU YOLO is ~6 ms; matches the 15 fps stream)
        const int   ACQUIRE_HITS = 6;      // ~0.4 s at ~15 fps before the camera commits to a subject
        const float PERSON_MIN = 0.50f;
        const float KEEP_MIN = 0.30f;      // 09-30: once locked, a person counts from here (hysteresis)
        std::deque<std::pair<unsigned long long,bool>> win;   // 09-30: acquire window (time, person seen)
        std::vector<pcore_npu::Det> tile_dets[2]; double tile_t[2]={0,0};   // 09-30: latest detections per tile (full frame)    // 09-30: 0.60 -> 0.50 (30 det/s + the 400 ms streak filter ghosts)
        const unsigned long long ACQUIRE_MS = 400;   // steady mode: the same 0.4 s, by time (30 Hz gives more hits)
        unsigned long long hit_since=0, last_sq=0;
        const unsigned long long RELEASE_MS = 1500;  // nobody for this long -> glide back to full view
        int hit = 0; bool locked = false;
        float sx0=0,sy0=0,sx1=0,sy1=0; bool sm=false;          // 09-30: smoothed person box
        float tx=0.5f, ty=0.5f, th_=0.5f;                        // the target last published
        // Motion-gate: while IDLE (no locked subject), skip the NPU when the scene is still — saves night power.
        // Without exclusion zones, BOTH motion and person detection can be restricted to BELOW `ignore_top` (a normalized
        // y from cctv.ini, default 0 = the whole image counts); env PULSECORE_CCTV_IGNORE_TOP overrides.
        float ignore_top = g_ignore_top; if(const char* e=std::getenv("PULSECORE_CCTV_IGNORE_TOP")){ if(*e) ignore_top=(float)std::atof(e); }
        const int MSTRIDE=8;  /* 09-30: half-res frames -> same grid as 16 at full res */ const int MDIFF=18; const float MFRAC=0.004f;   // motion grid stride + diff/coverage thresholds
        std::vector<unsigned char> prevg;
        while(af_run.load()){
          std::vector<pcore_npu::Det> dets; std::vector<Poly> lm; bool use_mask=false;
          if(det_steady){
            // STEADY (09-30): one detection per camera frame, whether or not Auto-frame is on — like Studio Effects,
            // a small constant NPU load in step with the video instead of bursts.
            FrameBuf fb; int sw=0,sh=0; unsigned long long sq=0; int tl=-1; float tu0=0,tu1=1;
            { std::unique_lock<std::mutex> lk(frame_mtx);
              frame_cv.wait_for(lk, std::chrono::milliseconds(200), [&]{ return !af_run.load() || frame_seq!=last_sq; });
              fb=frame_rgba; sw=fw; sh=fh; sq=frame_seq; tl=det_tile_in; tu0=det_u0_in; tu1=det_u1_in; }
            if(!af_run.load()) break;
            if(!fb || sw<=0 || sq==last_sq) continue;
            last_sq=sq; det_inflight.store(true);                                  // MF: no new NPU image until done (09-30)
            const double pm=det_.prep_rgba(fb->data(), sw, sh); fb.reset();   // the frame buffer goes back to the pool
            std::string tag; bool al=false;
            { std::lock_guard<std::mutex> lk(npu_mtx); dets = det_.infer(nullptr, sw, sh, 0.25f, tag, al, false); }
            det_inflight.store(false);
            if(tl>=0 && tl<2){                                  // 09-30: tile -> full frame, then merge both tiles' latest
                const float S=(float)pcore_npu::YS;
                for(auto& dd: dets){ dd.x0=tu0*S+dd.x0*(tu1-tu0); dd.x1=tu0*S+dd.x1*(tu1-tu0); }   // y: the tile spans the full height
                tile_dets[tl]=dets; tile_t[tl]=now_ms(); dets.clear();
                for(int k=0;k<2;k++) if(now_ms()-tile_t[k]<250.0) dets.insert(dets.end(), tile_dets[k].begin(), tile_dets[k].end());
                // the tiles overlap by 320 px: one person there is found twice -> keep the best of same-class boxes with IoU > 0.5
                std::sort(dets.begin(), dets.end(), [](const pcore_npu::Det& a, const pcore_npu::Det& b){ return a.score>b.score; });
                std::vector<pcore_npu::Det> keep;
                for(const auto& c: dets){ bool dup=false;
                    for(const auto& k: keep){ if(k.cls!=c.cls) continue;
                        const float ix=std::max(0.f,std::min(k.x1,c.x1)-std::max(k.x0,c.x0)), iy=std::max(0.f,std::min(k.y1,c.y1)-std::max(k.y0,c.y0));
                        const float in=ix*iy, un=(k.x1-k.x0)*(k.y1-k.y0)+(c.x1-c.x0)*(c.y1-c.y0)-in;
                        if(un>0 && in/un>0.5f){ dup=true; break; } }
                    if(!dup) keep.push_back(c); }
                dets.swap(keep); }
            det_prep_ms.store((float)pm); det_ms.store((float)det_.last_ms()); det_n.fetch_add(1, std::memory_order_relaxed);
            { const long long us=(long long)(det_.last_ms()*1000.0); det_us_sum.fetch_add(us);
              long long m=det_us_max.load(); while(us>m && !det_us_max.compare_exchange_weak(m,us)){} }
            { std::lock_guard<std::mutex> lk(mask_mtx); lm=masks; } use_mask=!lm.empty();
            int np=0; float pbest=0; for(auto& d: dets){ if(d.cls!=0) continue;
                float pcy=(d.y0+d.y1)*0.5f/(float)pcore_npu::YS, pcx=(d.x0+d.x1)*0.5f/(float)pcore_npu::YS;
                if(use_mask ? in_any(pcx,pcy,lm) : (pcy < ignore_top)) continue;
                pbest=std::max(pbest,d.score); if(d.score>=PERSON_MIN) np++; }
            det_people.store(np);
            ps_n.fetch_add(1); if(pbest>=0.30f) ps_30.fetch_add(1); if(pbest>=0.50f) ps_50.fetch_add(1);   // 09-30 evidence
            { const int pm=(int)(pbest*1000); int o=ps_max.load(); while(pm>o && !ps_max.compare_exchange_weak(o,pm)){} }
            if(!auto_frame.load()){ hit=0; locked=false; continue; }
          } else {
            if(!auto_frame.load()){ hit=0; locked=false; Sleep(120); continue; }
            int w=0,h=0;
            FrameBuf fb; int sw=0,sh=0;
            { std::lock_guard<std::mutex> lk(frame_mtx); fb=frame_rgba; sw=fw; sh=fh; }   // pointer only (09-30)
            if(fb && sw>0){                  // RGBA -> RGB at HALF resolution, outside the lock (YOLO sees 640x640)
                w=sw/2; h=sh/2; rgb.resize((size_t)w*h*3);
                const unsigned char* src=fb->data();
                for(int y=0;y<h;y++){ const unsigned char* r=src+(size_t)(2*y)*sw*4; unsigned char* o=&rgb[(size_t)y*w*3];
                    for(int x=0;x<w;x++){ const unsigned char* p=r+(size_t)(2*x)*4; o[x*3]=p[0]; o[x*3+1]=p[1]; o[x*3+2]=p[2]; } }
            }
            if(w<=0){ Sleep(80); continue; }
            // snapshot the exclusion masks (edited on the render thread); if any exist they REPLACE the ignore_top line.
            { std::lock_guard<std::mutex> lk(mask_mtx); lm=masks; }
            use_mask = !lm.empty();
            // ── motion-gate: coarse grayscale grid of the scene, diffed vs last cycle; ignore samples inside a mask ──
            int ry0 = use_mask ? 0 : (int)(ignore_top*h); if(ry0<0)ry0=0; if(ry0>h)ry0=h;
            std::vector<unsigned char> curg;
            for(int y=ry0;y<h;y+=MSTRIDE){ const unsigned char* row=&rgb[(size_t)y*w*3];
                for(int x=0;x<w;x+=MSTRIDE){
                    if(use_mask && in_any((float)x/(float)w,(float)y/(float)h,lm)) continue;   // masked area -> no motion
                    const unsigned char* p=row+(size_t)x*3; curg.push_back((unsigned char)((p[0]+p[1]+p[2])/3)); } }
            bool motion=true;
            if(prevg.size()==curg.size() && !curg.empty()){ size_t ch=0;
                for(size_t i=0;i<curg.size();i++){ int d=(int)curg[i]-(int)prevg[i]; if(d<0)d=-d; if(d>MDIFF)ch++; }
                motion = ((float)ch/(float)curg.size()) > MFRAC; }
            prevg.swap(curg);
            if(!locked && !motion){ Sleep(120); continue; }   // idle + still scene -> let the NPU sleep (light motion poll)
            std::string tag; bool al=false;
            { std::lock_guard<std::mutex> lk(npu_mtx); dets = det_.detect(rgb.data(), w, h, 0.35f, tag, al, /*draw=*/false); }
          }
            { static const bool fake=[](){ const char* e=std::getenv("PULSECORE_CCTV_FAKE_PERSON"); return e && *e=='1'; }();   // test hook
              static const double f0=now_ms(); static unsigned rs=930;
              if(fake){ dets.clear(); const double tt=(now_ms()-f0)/1000.0; rs=rs*1103515245u+12345u;
                  if(tt>1.0 && tt<13.0 && ((rs>>16)%100)>=5){
                      auto nz=[&](){ rs=rs*1103515245u+12345u; return (((rs>>16)%2001)/1000.0f-1.0f)*0.006f; };
                      const float cx=0.08f+0.84f*(float)((tt-1.0)/12.0), cy=0.72f, w=0.045f, h=0.20f; const float S=(float)pcore_npu::YS;
                      dets.push_back({0,0.8f,(cx-w/2+nz())*S,(cy-h/2+nz())*S,(cx+w/2+nz())*S,(cy+h/2+nz())*S,true}); } } }
            // union of CONFIDENT person boxes (bar raised to 0.60), normalized (detect maps frame->640x640)
            float ux0=1,uy0=1,ux1=0,uy1=0; bool any=false;
            for(auto& d: dets){ if(d.cls!=0 || d.score<(locked? KEEP_MIN : PERSON_MIN)) continue;
                float x0=d.x0/(float)pcore_npu::YS, y0=d.y0/(float)pcore_npu::YS,
                      x1=d.x1/(float)pcore_npu::YS, y1=d.y1/(float)pcore_npu::YS;
                float pcx=(x0+x1)*0.5f, pcy=(y0+y1)*0.5f;         // ignore a person centered in a mask (or above ignore_top)
                if(use_mask ? in_any(pcx,pcy,lm) : (pcy < ignore_top)) continue;
                ux0=std::min(ux0,x0); uy0=std::min(uy0,y0); ux1=std::max(ux1,x1); uy1=std::max(uy1,y1); any=true; }
            { int rp=0; for(auto& d: dets){ if(d.cls!=0 || d.score<PERSON_MIN) continue;       // 09-30: recording trigger
                  float pcx=(d.x0+d.x1)*0.5f/(float)pcore_npu::YS, pcy=(d.y0+d.y1)*0.5f/(float)pcore_npu::YS;
                  if(use_mask ? in_any(pcx,pcy,lm) : (pcy < ignore_top)) continue; rp++; }
              const double tn=now_ms(); rwin.push_back({tn,rp>0}); while(!rwin.empty() && tn-rwin.front().first>700) rwin.pop_front();
              if(rp>0){ rec.person(tn); int h=0; for(auto& w: rwin) h+=w.second; if(h>=4) rec.trigger(tn); } }
            unsigned long long now=GetTickCount64();
            win.push_back({now,any}); while(!win.empty() && now-win.front().first>700) win.pop_front();
            if(any){
                if(!locked){ if(hit++==0) hit_since=now;                 // build up trust before committing
                    if(det_steady){ int h=0; for(auto& w: win) h+=w.second;   // 09-30: window, a miss no longer resets
                        if(h>=5 && h*2>=(int)win.size() && now-hit_since>=ACQUIRE_MS) locked=true; }
                    else if(hit>=ACQUIRE_HITS) locked=true; }
                if(locked){
                    if(g_follow_old){ sx0=ux0; sy0=uy0; sx1=ux1; sy1=uy1; sm=true; }   // A/B: no smoothing
                    else if(!sm){ sx0=ux0; sy0=uy0; sx1=ux1; sy1=uy1; sm=true; }
                    else { const float k=0.35f; sx0+=(ux0-sx0)*k; sy0+=(uy0-sy0)*k; sx1+=(ux1-sx1)*k; sy1+=(uy1-sy1)*k; }
                    ux0=sx0; uy0=sy0; ux1=sx1; uy1=sy1;
                    float cx=(ux0+ux1)*0.5f, cy=(uy0+uy1)*0.5f, bw=ux1-ux0, bh=uy1-uy0;
                    cy -= bh*0.08f;                                   // a little headroom above the subject
                    float hw=bw*0.85f*0.5f + bw*0.35f + 0.04f;        // padded half-extents
                    float hh=bh*0.85f*0.5f + bh*0.35f + 0.06f;
                    float need=std::max(hw,hh);                       // frame-aspect crop needs hw==hh (see render)
                    if(need<0.20f) need=0.20f;                        // cap max zoom (~2.5x)
                    if(need>0.5f)  need=0.5f;
                    if(cx<need) cx=need; if(cx>1-need) cx=1-need;      // keep the crop inside the frame
                    if(cy<need) cy=need; if(cy>1-need) cy=1-need;
                    // dead zone + zoom hysteresis (09-30): move only when the person leaves the middle of the crop,
                    // zoom out at +12 % need, zoom in only at -20 % — the view stops "breathing" with every box wobble
                    const bool first = th_>=0.499f && tx==0.5f && ty==0.5f;
                    const bool rezoom = first || need > th_*1.12f || need < th_*0.80f;
                    float nh = (rezoom || g_follow_old)? need : th_;
                    const float dz = 0.18f*nh;
                    const bool move = g_follow_old || first || rezoom || std::fabs(cx-tx)>dz || std::fabs(cy-ty)>dz;
                    if(move){ float ncx=cx, ncy=cy;
                        if(ncx<nh) ncx=nh; if(ncx>1-nh) ncx=1-nh; if(ncy<nh) ncy=nh; if(ncy>1-nh) ncy=1-nh;
                        tx=ncx; ty=ncy; th_=nh; push_target(tx,ty,th_); }
                    last_person=now;
                }
            } else {
                if(!det_steady) hit=0;                               // gated mode: streak broken; steady: the window decides
                else { int h=0; for(auto& w: win) h+=w.second; if(h==0) hit=0; }
                if(locked && now-last_person > RELEASE_MS){           // gone long enough -> release to full view
                    locked=false; sm=false; tx=0.5f; ty=0.5f; th_=0.5f;
                    push_target(0.5f,0.5f,0.5f);
                }
            }
            if(!det_steady) Sleep(DET_INTERVAL_MS);   // gated mode's cadence; steady mode is paced by the camera
        }
    }

    // Save the current live frame as a timestamped JPEG in the clips-and-photos folder.
    void save_photo(){
        std::string dir=snapshots_dir(); std::error_code ec; std::filesystem::create_directories(std::filesystem::u8path(dir),ec);
        std::time_t t=std::time(nullptr); std::tm lt{}; localtime_s(&lt,&t);
        char stamp[32]; std::strftime(stamp,sizeof(stamp),"%Y%m%d_%H%M%S",&lt);
        std::string path=dir+"\\cctv_"+stamp+".jpg";
        int ok=0, w=last_w, h=last_h; FrameBuf px=last_rgba;   // hold the shown buffer (no copy)
        if(mf_on.load() && mf){ auto d=mf; px=std::make_shared<std::vector<unsigned char>>();   // 09-30: from the GPU
            if(d->read_display(*px)){ w=d->W; h=d->H; } else px.reset(); }
        if(px && !px->empty() && w>0 && h>0) ok=stbi_write_jpg(path.c_str(), w, h, 4, px->data(), 92);  // stb takes RGB from RGBA
        saved_msg = ok ? ("Saved "+path) : (!px? std::string("No frame to save yet\xE2\x80\xA6") : ("Write failed: "+path));
        saved_at=ImGui::GetTime();
        photo_ok=ok!=0; photo_path=path; photo_flash_at=ImGui::GetTime();
    }
    // Open the captures folder in Explorer (creating it if empty) — the "view my captures" button.
    void open_snapshots(){
        std::string dir=snapshots_dir(); std::error_code ec; std::filesystem::create_directories(std::filesystem::u8path(dir),ec);
        ShellExecuteW(nullptr,L"open",u8w(dir).c_str(),nullptr,nullptr,SW_SHOWNORMAL);
    }

    // Reader thread: pull fixed-size RAW RGBA frames off the pipe (no decode needed) straight to the display buffer.
    void pipe_reader(){
        const size_t fsz=(size_t)PIPE_W*PIPE_H*4;
        FrameBuf frame=pool.take(fsz); size_t got=0; DWORD n=0;
        HANDLE rd=(HANDLE)ff_rd;
        while(pipe_run.load()){
            DWORD want=(DWORD)std::min(fsz-got, (size_t)(4u<<20));
            if(!ReadFile(rd, frame->data()+got, want, &n, nullptr) || n==0) break;
            got+=n;
            if(got>=fsz){                          // one full frame -> PUBLISH THE POINTER (no copy, 09-30)
                unsigned long long sq; const double tn=now_ms();
                { std::lock_guard<std::mutex> lk(frame_mtx); frame_rgba=frame; fw=PIPE_W; fh=PIPE_H; sq=++frame_seq;
                  arr.push_back({sq,tn}); if(arr.size()>1800) arr.pop_front();
                  if(sq%30==0) fit_period();
                  if(buffer_ms>0){ playq.push_back({sq,frame,tn});      // cap: buffer length + 6 frames (each 14.7 MB)
                      const size_t cap=(size_t)(buffer_ms/period_ms)+6; while(playq.size()>cap) playq.pop_front(); } }
                frame_cv.notify_all();
                tlog_put("in", sq);
                n_in.fetch_add(1, std::memory_order_relaxed); n_total.fetch_add(1, std::memory_order_relaxed);
                last_frame_ms.store((long long)now_ms());
                frame=pool.take(fsz);              // a buffer nobody else holds (display/detector keep theirs)
                got=0;
            }
        }
    }
    void tapo_start(){ std::string url=tapo_rtsp(); if(url.empty()) return; tapo_stop();
        SECURITY_ATTRIBUTES sa{}; sa.nLength=sizeof(sa); sa.bInheritHandle=TRUE;
        HANDLE rd=nullptr, wr=nullptr;
        if(!CreatePipe(&rd,&wr,&sa,4u<<20)) return;                       // 4 MB pipe — raw frames are big
        SetHandleInformation(rd, HANDLE_FLAG_INHERIT, 0);                 // parent's read end must NOT be inherited
        HANDLE nul=CreateFileA("NUL",GENERIC_READ|GENERIC_WRITE,FILE_SHARE_READ|FILE_SHARE_WRITE,&sa,OPEN_EXISTING,0,nullptr);
        HANDLE in_rd=nullptr, in_wr=nullptr;                              // stdin pipe: the child reads, we keep the write end
        if(CreatePipe(&in_rd,&in_wr,&sa,0)) SetHandleInformation(in_wr, HANDLE_FLAG_INHERIT, 0);
        // -hwaccel d3d11va: decode H.264 on the SoC's hardware video engine (not the CPU). Output RAW RGBA (no MJPEG
        // re-encode/decode) so higher resolution costs only memory bandwidth, not CPU. scale sets the display/track
        // resolution; the camera's 2K SD recordings are untouched.
        std::string cmd="\""+ffmpeg_exe()+"\" -hide_banner -loglevel level+warning -fflags nobuffer -flags low_delay -rtbufsize 32M"
            +std::string(url.find("://")==std::string::npos? " -re -stream_loop -1" : " -rtsp_transport tcp -timeout 5000000")   // 09-30: a test FILE plays in real time
            +" -hwaccel d3d11va -i \""+url+"\" -an -vf setpts=N,scale="
            +std::to_string(PIPE_W)+":"+std::to_string(PIPE_H)
            +" -fps_mode passthrough -f rawvideo -pix_fmt rgba pipe:1";   // 09-30: camera cadence, no fps=20 duplicates;
            // setpts=N: the Tapo sends repeated/backwards timestamps - with passthrough the rawvideo muxer DROPPED those
            // frames ("non monotonically increasing dts", 1647 in ~10 min = the stutter). Frame numbers are strictly increasing.
        // 09-30: ffmpeg's warnings/errors to a log instead of NUL (level+warning: no info line with the URL/password)
        char tmp[MAX_PATH]{}; GetTempPathA(MAX_PATH,tmp); ff_log=std::string(tmp)+"pulsecore_cctv_ffmpeg.log";
        if(!tlog){ tlog=std::fopen((std::string(tmp)+"pulsecore_cctv_timing.log").c_str(),"w"); tlog_t0=now_ms(); }
        HANDLE lg=CreateFileA(ff_log.c_str(),GENERIC_WRITE,FILE_SHARE_READ|FILE_SHARE_WRITE,&sa,CREATE_ALWAYS,FILE_ATTRIBUTE_NORMAL,nullptr);
        STARTUPINFOA si{}; si.cb=sizeof(si); si.dwFlags=STARTF_USESTDHANDLES; si.hStdInput=in_rd? in_rd : nul; si.hStdOutput=wr;
        si.hStdError=(lg!=INVALID_HANDLE_VALUE)? lg : nul;
        PROCESS_INFORMATION pi{};
        BOOL ok=CreateProcessA(nullptr, cmd.data(), nullptr,nullptr, TRUE, CREATE_NO_WINDOW, nullptr, bin_dir().c_str(), &si,&pi);
        CloseHandle(wr);                                                 // child owns the write end now
        if(nul!=INVALID_HANDLE_VALUE) CloseHandle(nul);
        if(lg!=INVALID_HANDLE_VALUE) CloseHandle(lg);
        if(in_rd) CloseHandle(in_rd);
        if(!ok){ CloseHandle(rd); if(in_wr) CloseHandle(in_wr); return; }
        CloseHandle(pi.hThread); ff_proc=pi.hProcess; ff_rd=rd; ff_in=in_wr; ff_started_ms=now_ms();
        pipe_run=true; pipe_thread=std::thread(&Cctv::pipe_reader,this);
    }
    void tapo_stop(){
        // graceful first: 'q' -> ffmpeg sends TEARDOWN and exits; the reader keeps draining meanwhile, so ffmpeg is
        // never stuck on a half-written frame. Only if it has not exited in 1.5 s is it killed.
        if(ff_proc){
            if(ff_in){ DWORD w=0; WriteFile((HANDLE)ff_in,"q",1,&w,nullptr); }
            if(WaitForSingleObject((HANDLE)ff_proc,1500)!=WAIT_OBJECT_0) TerminateProcess((HANDLE)ff_proc,0);
            CloseHandle((HANDLE)ff_proc); ff_proc=nullptr; }   // pipe write end closes -> ReadFile returns 0
        if(ff_in){ CloseHandle((HANDLE)ff_in); ff_in=nullptr; }
        pipe_run=false;
        if(pipe_thread.joinable()) pipe_thread.join();
        if(ff_rd){ CloseHandle((HANDLE)ff_rd); ff_rd=nullptr; }
    }
    ~Cctv(){ sr_run=false; if(sr_thread.joinable()) sr_thread.join(); af_run=false; frame_cv.notify_all(); if(af_thread.joinable()) af_thread.join(); stop_stream(); }
};
static Cctv g;

// ── Hand-drawn vector icons (no font dependency) — each paints inside a box of side `s` centered at `c` ──
static ImU32 g_icon_bg = IM_COL32(40,44,50,255);           // what is behind the icon (for knock-outs), set by dock_btn
static float ic_th(float s){ return std::max(1.6f, s*0.068f); }
static const float IC_PI = 3.14159265f;
static void ic_eye(ImDrawList* d, ImVec2 c, float s, ImU32 col){
    const float th=ic_th(s), w=s*0.42f, h=s*0.40f; const ImVec2 L(c.x-w,c.y), R(c.x+w,c.y);
    d->PathLineTo(L); d->PathBezierCubicCurveTo(ImVec2(c.x-w*0.45f,c.y-h), ImVec2(c.x+w*0.45f,c.y-h), R, 24);
    d->PathBezierCubicCurveTo(ImVec2(c.x+w*0.45f,c.y+h), ImVec2(c.x-w*0.45f,c.y+h), L, 24);
    d->_Path.Size--; d->PathStroke(col, ImDrawFlags_Closed, th);
    d->AddCircle(c, s*0.165f, col, 32, th);                                   // iris
    d->AddCircleFilled(c, s*0.088f, col, 24);                                  // pupil
    d->AddCircleFilled(ImVec2(c.x-s*0.032f,c.y-s*0.036f), s*0.030f, g_icon_bg, 12);   // highlight
}
static void ic_box(ImDrawList* d, ImVec2 c, float s, ImU32 col){              // exclusion zones
    const float th=ic_th(s), r=s*0.29f, rr=s*0.085f;
    const ImVec2 P[4]={ImVec2(c.x-r,c.y-r),ImVec2(c.x+r,c.y-r),ImVec2(c.x+r,c.y+r),ImVec2(c.x-r,c.y+r)};
    for(int i=0;i<4;i++){ const ImVec2 p=P[i], q=P[(i+1)%4]; const float dx=(q.x-p.x)/(2*r), dy=(q.y-p.y)/(2*r);
        d->AddLine(ImVec2(p.x+dx*rr,p.y+dy*rr), ImVec2(q.x-dx*rr,q.y-dy*rr), col, th); }
    for(const ImVec2& p: P) d->AddCircle(p, rr, col, 20, th);
}
static void ic_cam(ImDrawList* d, ImVec2 c, float s, ImU32 col){
    const float th=ic_th(s), x0=c.x-s*0.42f, x1=c.x+s*0.42f, top=c.y-s*0.17f, bot=c.y+s*0.32f, rr=s*0.10f;
    const float bx0=c.x-s*0.17f, bx1=c.x+s*0.17f, bh=s*0.11f;
    d->PathLineTo(ImVec2(bx0-s*0.03f, top));                                   // viewfinder bump
    d->PathLineTo(ImVec2(bx0+s*0.03f, top-bh)); d->PathLineTo(ImVec2(bx1-s*0.03f, top-bh)); d->PathLineTo(ImVec2(bx1+s*0.03f, top));
    d->PathArcTo(ImVec2(x1-rr, top+rr), rr, -IC_PI*0.5f, 0.0f, 8);             // body
    d->PathArcTo(ImVec2(x1-rr, bot-rr), rr, 0.0f, IC_PI*0.5f, 8);
    d->PathArcTo(ImVec2(x0+rr, bot-rr), rr, IC_PI*0.5f, IC_PI, 8);
    d->PathArcTo(ImVec2(x0+rr, top+rr), rr, IC_PI, IC_PI*1.5f, 8);
    d->PathStroke(col, ImDrawFlags_Closed, th);
    d->AddCircle(ImVec2(c.x, (top+bot)*0.5f+s*0.005f), s*0.145f, col, 32, th); // lens
    d->AddCircleFilled(ImVec2(x1-s*0.12f, top+s*0.10f), s*0.036f, col, 12);    // dot
}
static void ic_folder(ImDrawList* d, ImVec2 c, float s, ImU32 col){
    const float th=ic_th(s), x0=c.x-s*0.42f, x1=c.x+s*0.42f, y0=c.y-s*0.30f, y1=c.y+s*0.30f, rr=s*0.08f;
    const float tw=s*0.30f, tabh=s*0.11f;
    d->PathArcTo(ImVec2(x0+rr, y0+rr), rr, IC_PI, IC_PI*1.5f, 8);
    d->PathLineTo(ImVec2(x0+tw, y0)); d->PathLineTo(ImVec2(x0+tw+s*0.07f, y0+tabh));
    d->PathArcTo(ImVec2(x1-rr, y0+tabh+rr), rr, -IC_PI*0.5f, 0.0f, 8);
    d->PathArcTo(ImVec2(x1-rr, y1-rr), rr, 0.0f, IC_PI*0.5f, 8);
    d->PathArcTo(ImVec2(x0+rr, y1-rr), rr, IC_PI*0.5f, IC_PI, 8);
    d->PathStroke(col, ImDrawFlags_Closed, th);
}
static void ic_star4(ImDrawList* d, ImVec2 c, float R, ImU32 col, bool fill, float th){   // concave four-point star
    const float k=R*0.13f;
    d->PathLineTo(ImVec2(c.x,c.y-R));
    d->PathBezierQuadraticCurveTo(ImVec2(c.x+k,c.y-k), ImVec2(c.x+R,c.y), 12);
    d->PathBezierQuadraticCurveTo(ImVec2(c.x+k,c.y+k), ImVec2(c.x,c.y+R), 12);
    d->PathBezierQuadraticCurveTo(ImVec2(c.x-k,c.y+k), ImVec2(c.x-R,c.y), 12);
    d->PathBezierQuadraticCurveTo(ImVec2(c.x-k,c.y-k), ImVec2(c.x,c.y-R), 12);
    d->_Path.Size--;                                                           // last point == first
    if(fill) d->PathFillConcave(col); else d->PathStroke(col, ImDrawFlags_Closed, th);
}
static void ic_spark(ImDrawList* d, ImVec2 c, float s, ImU32 col){            // AI sharpening
    const float th=ic_th(s)*0.8f;
    ic_star4(d, ImVec2(c.x-s*0.07f, c.y+s*0.06f), s*0.34f, col, true, th);
    ic_star4(d, ImVec2(c.x+s*0.27f, c.y-s*0.25f), s*0.16f, col, false, th*0.7f);
    ic_star4(d, ImVec2(c.x+s*0.30f, c.y+s*0.27f), s*0.075f, col, true, th);
}
static bool g_rec_live=false;                                            // painter hint: a clip is being written
static void ic_rec(ImDrawList* d, ImVec2 c, float s, ImU32 col){              // auto-record: ring + dot (red while recording)
    const float th=ic_th(s);
    if(g_rec_live){                                            // recording: the whole button red, a pulsing white dot
        const float a=0.70f+0.30f*(float)std::sin(ImGui::GetTime()*5.0);
        d->AddCircleFilled(c, s*0.5f/0.86f, IM_COL32(214,48,49,255), 40);
        d->AddCircle(c, s*0.34f, IM_COL32(255,255,255,235), 40, th);
        d->AddCircleFilled(c, s*0.19f, IM_COL32(255,255,255,(int)(255*a)), 32); return; }
    d->AddCircle(c, s*0.34f, col, 40, th);
    d->AddCircleFilled(c, s*0.17f, col, 32);
}
static void ic_info(ImDrawList* d, ImVec2 c, float s, ImU32 col){             // live numbers on/off
    const float th=ic_th(s);
    d->AddCircle(c, s*0.36f, col, 40, th);
    d->AddCircleFilled(ImVec2(c.x, c.y-s*0.17f), th*0.95f, col, 12);          // dot
    d->AddLine(ImVec2(c.x, c.y-s*0.05f), ImVec2(c.x, c.y+s*0.20f), col, th*1.1f);   // stem
}
static void ic_zoom(ImDrawList* d, ImVec2 c, float s, ImU32 col){             // reset zoom: magnifier
    const float th=ic_th(s), r=s*0.22f; const ImVec2 o(c.x-s*0.07f,c.y-s*0.07f);
    d->AddCircle(o, r, col, 32, th);
    const ImVec2 h0(o.x+r*0.72f,o.y+r*0.72f), h1(c.x+s*0.34f,c.y+s*0.34f);
    d->AddLine(h0, h1, col, th*1.35f); d->AddCircleFilled(h1, th*0.67f, col, 12);
    d->AddLine(ImVec2(o.x-r*0.48f,o.y), ImVec2(o.x+r*0.48f,o.y), col, th);   // minus: zoom out to full view
}
static bool icon_btn(const char* id, float sz, bool active, const char* tip, void(*paint)(ImDrawList*,ImVec2,float,ImU32)){
    ImVec2 p=ImGui::GetCursorScreenPos();
    ImGui::InvisibleButton(id, ImVec2(sz,sz));
    bool hov=ImGui::IsItemHovered(), clicked=ImGui::IsItemClicked();
    ImDrawList* dl=ImGui::GetWindowDrawList();
    if(active)   dl->AddRectFilled(p, ImVec2(p.x+sz,p.y+sz), IM_COL32(38,120,110,255), 5.0f);
    else if(hov) dl->AddRectFilled(p, ImVec2(p.x+sz,p.y+sz), IM_COL32(255,255,255,26), 5.0f);
    ImU32 col = active? IM_COL32(232,255,250,255) : IM_COL32(208,208,208,255);
    paint(dl, ImVec2(p.x+sz*0.5f, p.y+sz*0.5f), sz, col);
    if(hov && tip) ImGui::SetTooltip("%s", tip);
    return clicked;
}

// ── the dock (09-30): centred, rounded, under the video; mode text + receipt under it ───────────────────────────────
// 09-30: the Windows folder picker in its own STA thread (the render loop keeps drawing); the result is taken over
// by the render thread (g.picked_dir). Used by the folder button's menu and the right-click on the record button.
static void pick_folder(){
    if(g.picking.exchange(true)) return;
    std::string start=clips_dir(); HWND owner=GetActiveWindow();
    std::thread([start,owner](){
        std::string res; if(SUCCEEDED(CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED|COINIT_DISABLE_OLE1DDE))){
            IFileOpenDialog* dlg=nullptr;
            if(SUCCEEDED(CoCreateInstance(CLSID_FileOpenDialog,nullptr,CLSCTX_INPROC_SERVER,IID_PPV_ARGS(&dlg)))){
                DWORD o=0; dlg->GetOptions(&o); dlg->SetOptions(o|FOS_PICKFOLDERS|FOS_FORCEFILESYSTEM|FOS_PATHMUSTEXIST);
                dlg->SetTitle(L"Where should PulseX CCTV save clips and photos?");
                IShellItem* si=nullptr; std::wstring ws=u8w(start);          // v0.2: UTF-8, and the nearest folder that exists
                while(!ws.empty() && GetFileAttributesW(ws.c_str())==INVALID_FILE_ATTRIBUTES){ auto k=ws.find_last_of(L"\\/"); if(k==std::wstring::npos || k<3) break; ws.resize(k); }
                if(SUCCEEDED(SHCreateItemFromParsingName(ws.c_str(),nullptr,IID_PPV_ARGS(&si)))){ dlg->SetFolder(si); si->Release(); }
                if(SUCCEEDED(dlg->Show(owner))){ IShellItem* r=nullptr;
                    if(SUCCEEDED(dlg->GetResult(&r))){ PWSTR p=nullptr;
                        if(SUCCEEDED(r->GetDisplayName(SIGDN_FILESYSPATH,&p))){ int n=WideCharToMultiByte(CP_UTF8,0,p,-1,nullptr,0,nullptr,nullptr);
                            res.resize(n>0? n-1 : 0); WideCharToMultiByte(CP_UTF8,0,p,-1,&res[0],n,nullptr,nullptr); CoTaskMemFree(p); }
                        r->Release(); } }
                dlg->Release(); }
            CoUninitialize(); }
        { std::lock_guard<std::mutex> lk(g.pick_mtx); g.picked_dir=res; }
        g.picking=false; }).detach();
}
static bool dock_btn(const char* id, float sz, bool active, const char* tip, void(*paint)(ImDrawList*,ImVec2,float,ImU32)){
    ImVec2 p=ImGui::GetCursorScreenPos();
    ImGui::InvisibleButton(id, ImVec2(sz,sz));
    const bool hov=ImGui::IsItemHovered(), clicked=ImGui::IsItemClicked();
    ImDrawList* dl=ImGui::GetWindowDrawList(); const ImVec2 c(p.x+sz*0.5f, p.y+sz*0.5f);
    g_icon_bg = active? (hov? IM_COL32(56,196,172,255) : IM_COL32(40,164,146,255)) : (hov? IM_COL32(69,73,79,255) : IM_COL32(40,44,50,255));
    if(active)   dl->AddCircleFilled(c, sz*0.5f, g_icon_bg, 32);
    else if(hov) dl->AddCircleFilled(c, sz*0.5f, IM_COL32(255,255,255,30), 32);
    paint(dl, c, sz*0.86f, active? IM_COL32(250,255,254,255) : IM_COL32(218,222,228,255));
    if(hov && tip) ImGui::SetTooltip("%s", tip);
    return clicked; }
static void centred_text(const std::string& t){        // one line centred, or wrapped when wider than the window
    if(t.empty()) return;
    const float aw=ImGui::GetContentRegionAvail().x, tw=ImGui::CalcTextSize(t.c_str()).x;
    ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled));
    if(tw<aw){ ImGui::SetCursorPosX(ImGui::GetCursorPosX()+(aw-tw)*0.5f); ImGui::TextUnformatted(t.c_str()); }
    else { ImGui::PushTextWrapPos(0.0f); ImGui::TextUnformatted(t.c_str()); ImGui::PopTextWrapPos(); }
    ImGui::PopStyleColor(); }
static void draw_dock(bool af, bool em, int ds){
    const float B=44.0f, GAP=10.0f, SEP=22.0f, PAD=10.0f;
    const bool rz = !af && !em && g.m_h<0.499f;
    const int n = rz? 8 : 7;
    const float dock_w = n*B + (n-1)*GAP + 2*(SEP-GAP) + 2*PAD, dock_h = B + 2*PAD;
    ImGui::Dummy(ImVec2(0,10));
    const float aw=ImGui::GetContentRegionAvail().x;
    ImVec2 o=ImGui::GetCursorScreenPos(); o.x += std::max(0.0f,(aw-dock_w)*0.5f);
    ImDrawList* dl=ImGui::GetWindowDrawList();
    dl->AddRectFilled(o, ImVec2(o.x+dock_w,o.y+dock_h), IM_COL32(40,44,50,240), dock_h*0.5f);
    dl->AddRect(o, ImVec2(o.x+dock_w,o.y+dock_h), IM_COL32(255,255,255,20), dock_h*0.5f, 0, 1.0f);
    float x=o.x+PAD; const float y=o.y+PAD;
    auto at=[&](float adv){ ImGui::SetCursorScreenPos(ImVec2(x,y)); x+=B+adv; };
    auto sep=[&](){ const float sx=x-GAP+(SEP)*0.5f-1.0f; dl->AddLine(ImVec2(sx,y+B*0.22f),ImVec2(sx,y+B*0.78f),IM_COL32(255,255,255,28),1.0f); x+=SEP-GAP; };
    at(GAP); if(dock_btn("##af", B, af, af?"Auto-frame: following people (click to stop)":"Auto-frame: zoom in on people (NPU)", ic_eye)){ g.auto_frame.store(!af); }
    at(GAP); if(dock_btn("##ez", B, em, em?"Editing exclusion zones (click to finish)":"Edit exclusion zones (road, trees)", ic_box)){ g.edit_mask.store(!em); g.mask_wip.clear(); }
    sep();
    { const bool armed=g.rec.armed.load(); g_rec_live=g.rec.recording.load();
      std::string tip = g.dec_mode!=0? std::string("Recording needs GPU decode (Media Foundation) - it is using ffmpeg now")
          : std::string(armed? "Auto-record ON: a clip is saved when someone passes" : "Auto-record people who pass by")
            + "\nSaved to " + clips_dir();
      at(GAP); if(dock_btn("##rec", B, armed, tip.c_str(), ic_rec)){ g.rec.armed=!armed; g_auto_record=!armed; save_settings(); }
      if(ImGui::IsItemClicked(ImGuiMouseButton_Right)) pick_folder();                 // shortcut; the menu under the folder button is the way
    }
    at(GAP); if(dock_btn("##cam", B, false, "Take photo", ic_cam)) g.save_photo();
    { const bool open=ImGui::IsPopupOpen("##foldmenu");                     // 09-30: folder menu - where, open, choose
      at(GAP); if(dock_btn("##fold", B, open, open? nullptr : "Clips and photos folder", ic_folder)) ImGui::OpenPopup("##foldmenu");
      const ImVec2 mn=ImGui::GetItemRectMin(), mx=ImGui::GetItemRectMax();
      ImGui::SetNextWindowPos(ImVec2((mn.x+mx.x)*0.5f, mn.y-10.0f), ImGuiCond_Appearing, ImVec2(0.5f,1.0f));
      ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(14,12)); ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(8,9));
      ImGui::PushStyleVar(ImGuiStyleVar_PopupRounding, 10.0f);
      if(ImGui::BeginPopup("##foldmenu")){
          ImGui::TextDisabled("Clips and photos are saved in");
          ImGui::TextUnformatted(clips_dir().c_str());
          ImGui::Separator();
          if(ImGui::MenuItem("Open folder")) g.open_snapshots();
          if(ImGui::MenuItem("Choose folder\xE2\x80\xA6", nullptr, false, !g.picking.load())) pick_folder();
          ImGui::EndPopup(); }
      ImGui::PopStyleVar(3); }
    sep();
    { bool so=g.sr_on.load();
      static const bool sr_model=[](){ const bool f=GetFileAttributesW(u8w(sr_ctx()).c_str())!=INVALID_FILE_ATTRIBUTES;   // 10-01
          g.mlog(std::string("[cctv] sharpening model: ")+(f? "found" : "missing - the button says so")); return f; }();
      at(GAP); if(dock_btn("##sr", B, so, !sr_model? "AI sharpening needs models\\quicksrnetlarge_288x512_ctx_qnn.bin next to the app (see README)"
                                            : so? "AI sharpening on (QuickSRNet on the NPU, when zoomed in) - click to turn off"
                                            : "AI sharpening on the NPU when zoomed in (QuickSRNet)", ic_spark)) g.sr_on.store(!so); }
    if(rz){ at(GAP); if(dock_btn("##rz", B, false, "Reset zoom", ic_zoom)){ g.m_cx=0.5f; g.m_cy=0.5f; g.m_h=0.5f; } }
    at(GAP); if(dock_btn("##info", B, g.show_info, g.show_info? "Hide the live numbers" : "Show live numbers (decode, fps, NPU) \xE2\x80\x94 PulseX CCTV " PULSEX_CCTV_VERSION, ic_info)) g.show_info=!g.show_info;
    ImGui::SetCursorScreenPos(ImVec2(ImGui::GetCursorStartPos().x+ImGui::GetWindowPos().x, o.y+dock_h));
    ImGui::Dummy(ImVec2(0,8));
    // mode text, then the receipt
    std::string mode;
    if(g.rec.recording.load()){ const int sec=(int)((now_ms()-g.rec.rec_t0.load())/1000.0);
        char b[96]; std::snprintf(b,sizeof b,"\xE2\x97\x8F Recording \xE2\x80\x94 someone is passing (%d:%02d)", sec/60, sec%60); mode=b; }
    else if(em)           mode="Editing zones \xE2\x80\x94 click the image to add points";
    else if(af && ds==0)  mode="loading the NPU detector\xE2\x80\xA6";
    else if(af && ds==2)  mode="detector unavailable \xE2\x80\x94 YOLO context not found";
    else if(af)           mode="Auto-frame: the NPU follows people and zooms in";
    else if(g.m_h<0.499f) mode="manual zoom \xE2\x80\x94 scroll / drag";
    else                  mode="scroll to zoom \xC2\xB7 drag to pan";
    centred_text(mode);
    { // status row 2 (09-30): the receipt on its own line, wrapped to the window, decode path first
      std::string rx; char t[200]; auto add=[&](const char* x){ if(!rx.empty()) rx+=" \xC2\xB7 "; rx+=x; };
      if(g.show_info && (g.fps_in>0 || g.mf_on.load())){ add(g.mf_on.load()? "GPU decode" : "ffmpeg");
          if(g.fps_in>0){ std::snprintf(t,sizeof t,"%.0f fps (shown %.0f)",g.fps_in,g.fps_shown); add(t);
              std::snprintf(t,sizeof t,"%s %.1f ms",g.mf_on.load()? "convert" : "upload",g.up_ms); add(t);
              if(g.buffer_ms>0){ std::snprintf(t,sizeof t,"buffer %.0f ms",g.lat_ms); add(t); } } }
      if(g.show_info && ds==1 && g.det_steady && g.det_rate>0){ const int np=g.det_people.load();
          std::snprintf(t,sizeof t,"NPU %.0f/s %.1f ms (max %.0f)",g.det_rate,g.det_mean_ms,g.det_max_ms); add(t);
          std::snprintf(t,sizeof t,"busy %.0f%%",g.det_busy); add(t);
          std::snprintf(t,sizeof t,"%d %s",np,np==1? "person" : "people"); add(t); }
      if(g.sr_on.load()){ const int st=g.sr_state.load();
          if(st==2) add("AI sharpening unavailable (context not found)");
          else if(!g.sr_want) add("AI sharpening: zoom in to use it");
          else if(g.show_info){ std::snprintf(t,sizeof t,"AI sharp %.1f ms",g.sr_ms); add(t); } }
      if(!g.dec_note.empty()) add(g.dec_note.c_str());
      if(g_purged_at_start==-2) add("run as administrator to empty the memory cache");
      if(!g.saved_msg.empty() && ImGui::GetTime()-g.saved_at < 5.0) add(g.saved_msg.c_str());
      centred_text(rx); }
    // zone editor actions (only while editing), centred
    if(em){
        ImGui::Dummy(ImVec2(0,4));
        const char* lb[4]={"Finish zone","Undo point","Clear all","Save"}; float tw=0;
        for(auto* l: lb) tw+=ImGui::CalcTextSize(l).x+ImGui::GetStyle().FramePadding.x*2; tw+=3*8.0f;
        ImGui::SetCursorPosX(ImGui::GetCursorPosX()+std::max(0.0f,(ImGui::GetContentRegionAvail().x-tw)*0.5f));
        if(ImGui::SmallButton("Finish zone")){ if(g.mask_wip.size()>=3){ std::lock_guard<std::mutex> lk(g.mask_mtx); g.masks.push_back(g.mask_wip); } g.mask_wip.clear(); }
        ImGui::SameLine(0,8); if(ImGui::SmallButton("Undo point")){ if(!g.mask_wip.empty()) g.mask_wip.pop_back(); }
        ImGui::SameLine(0,8); if(ImGui::SmallButton("Clear all")){ std::lock_guard<std::mutex> lk(g.mask_mtx); g.masks.clear(); g.mask_wip.clear(); }
        ImGui::SameLine(0,8); if(ImGui::SmallButton("Save")){ std::vector<Poly> cp; { std::lock_guard<std::mutex> lk(g.mask_mtx); cp=g.masks; } save_masks(cp); g.saved_msg="Saved zones"; g.saved_at=ImGui::GetTime(); }
    }
}

static BOOL CALLBACK find_own_window(HWND h, LPARAM lp){                // 09-30: this process's visible top window
    DWORD pid=0; GetWindowThreadProcessId(h,&pid);
    if(pid==GetCurrentProcessId() && IsWindowVisible(h) && !GetWindow(h,GW_OWNER)){ *(HWND*)lp=h; return FALSE; }
    return TRUE; }
static void draw_cctv(){
    { static int tries=0; static bool done=false;                         // 09-30: rounded window corners + real title
      if(!done && tries++<120){ HWND h=nullptr; EnumWindows(find_own_window,(LPARAM)&h);
          if(h){ const DWORD DWMWA_WINDOW_CORNER_PREFERENCE=33; const int DWMWCP_ROUND=2;
              DwmSetWindowAttribute(h, DWMWA_WINDOW_CORNER_PREFERENCE, &DWMWCP_ROUND, sizeof(DWMWCP_ROUND));
              SetWindowTextW(h, L"PulseX CCTV"); done=true; } } }
    { static bool themed=false; if(!themed){ themed=true;                     // 09-30: a softer slate than near-black
        int r=60,gg=65,b=73; if(const char* e=std::getenv("PULSECORE_CCTV_BG")) std::sscanf(e,"%d,%d,%d",&r,&gg,&b);
        ImGui::GetStyle().Colors[ImGuiCol_WindowBg]=ImVec4(r/255.0f,gg/255.0f,b/255.0f,1.0f); } }
    { static const double t_start=now_ms(); static bool snapped=false;   // 09-30 test hooks (env only)
      static const double snap_s=[](){ const char* e=std::getenv("PULSECORE_CCTV_AUTOSNAP_S"); return (e&&*e)? std::atof(e) : -1.0; }();
      static const double exit_s=[](){ const char* e=std::getenv("PULSECORE_CCTV_EXIT_S"); return (e&&*e)? std::atof(e) : -1.0; }();
      const double el=(now_ms()-t_start)/1000.0;
      if(snap_s>0 && !snapped && el>=snap_s){ snapped=true; g.save_photo(); g.mlog("[cctv] autosnap: "+g.saved_msg); }
      if(exit_s>0 && el>=exit_s){ g.mlog("[cctv] exit hook"); PostQuitMessage(0); }
      static bool sr_hook=false;                                     // PULSECORE_CCTV_TEST_SR=cx,cy,h: zoom there + AI sharpening
      if(!sr_hook){ sr_hook=true; if(const char* e=std::getenv("PULSECORE_CCTV_TEST_SR")){ float a=0.5f,b=0.5f,c=0.18f;
          if(std::sscanf(e,"%f,%f,%f",&a,&b,&c)>=1){ g.m_cx=a; g.m_cy=b; g.m_h=c; g.sr_on.store(true); } } } }
    ImGui::Dummy(ImVec2(0,4));
    { static const std::string src=tapo_rtsp();                    // 10-01: what is shown, never the address
      std::string sub = g_camera_name.empty()? std::string() : g_camera_name+" \xC2\xB7 ";
      sub += src.empty()? "no camera set (tapo_rtsp.txt)" : (src.find("://")==std::string::npos? "video file" : "RTSP live");
      const int sw = g.mf? g.mf->W : g.last_w, sh = g.mf? g.mf->H : g.last_h;
      if(sw>0 && sh>0){ char b[48]; std::snprintf(b,sizeof b," \xC2\xB7 %d \xC3\x97 %d",sw,sh); sub+=b; }
      ImGui::TextDisabled("%s", sub.c_str());
      { static std::string logged; if(sub!=logged){ logged=sub; g.mlog("[cctv] subtitle: "+sub); } } }   // witness (10-01)
    ImGui::Dummy(ImVec2(0,6));
    // LIVE: take the freshest decoded frame from the pipe reader — no file, no reader/writer lock.
    g.t_mark(2,"window: first draw");
    if(g.mf) g.mf->rec_want = g.rec.armed.load();                  // 09-30: the ring runs only while armed
    { std::lock_guard<std::mutex> lk(g.note_mtx); if(!g.pending_note.empty()){ g.saved_msg=g.pending_note; g.saved_at=ImGui::GetTime(); g.pending_note.clear(); } }
    { std::string pd; { std::lock_guard<std::mutex> lk(g.pick_mtx); pd.swap(g.picked_dir); }
      if(!pd.empty()){ g_save_dir=pd; save_settings(); g.rec.dir=clips_dir(); g.saved_msg="Clips and photos now go to "+pd; g.saved_at=ImGui::GetTime(); } }
    if(!g.stream_started && device()){ g.stream_started=true; g.t_mark(3, g.dec_mode==0? "stream start (Media Foundation)" : "stream start (ffmpeg)"); g.start_stream(); }   // 09-30: needs the D3D11 device -> first frame
    if(g.dec_mode==0){ // MF watchdog: failed/ended -> reopen after 3 s; no frame for 15 s -> reopen; 2 opens without a frame -> ffmpeg
      const double t=now_ms(); auto d=g.mf; const int st=d? d->state.load() : 0;
      const double since=t-(double)std::max<long long>(g.last_frame_ms.load(), (long long)g.ff_started_ms);
      if(d && ((st==3||st==4) ? t-g.ff_started_ms>3000 : since>15000)){
          g.mlog("[cctv] decoder state "+std::to_string(st)+" ("+d->err+"), "+std::to_string(d->frames.load())+" frames");
          if(d->frames.load()==0) g.mf_fail++; else g.mf_fail=0;
          if(g.mf_fail>=2){ g.dec_note="Media Foundation could not open the stream ("+d->err+") \xE2\x80\x94 using ffmpeg";
              g.mlog("[cctv] falling back to ffmpeg"); g.stop_stream(); g.dec_mode=1; g.start_stream(); }
          else { g.reconnects++; g.start_stream(); } }
    } else { // watchdog (09-30): no frame for 10 s, or ffmpeg exited -> restart it (the camera may still hold an old session)
      const double t=now_ms(); DWORD code=STILL_ACTIVE; if(g.ff_proc) GetExitCodeProcess((HANDLE)g.ff_proc,&code);
      const double since=t-(double)std::max<long long>(g.last_frame_ms.load(), (long long)g.ff_started_ms);
      if(g.ff_proc && (code!=STILL_ACTIVE ? t-g.ff_started_ms>3000 : since>10000)){ g.reconnects++; g.tapo_start(); } }
    FrameBuf cur; int lw=0,lh=0; int cur_slot=-1;
    { std::lock_guard<std::mutex> lk(g.frame_mtx);
      if(g.buffer_ms<=0 && g.dec_mode!=0){                  // no buffer: newest frame, as before (ffmpeg pipe)
          if(g.frame_rgba && g.frame_seq!=g.shown_seq){ cur=g.frame_rgba; lw=g.fw; lh=g.fh; g.shown_seq=g.frame_seq; }
      } else if(!g.playq.empty()){                          // PLAYOUT v2: show the newest frame whose due time passed
          const double t=now_ms(); int pick=-1;
          for(int i=0;i<(int)g.playq.size();i++){ if(t>=g.due_ms(g.playq[i].seq)) pick=i; else break; }
          if(pick>=0){
              Cctv::QF f=g.playq[pick];
              for(int i=0;i<pick;i++) if(g.playq[i].slot>=0) g.slot_state[g.playq[i].slot]=0;   // skipped MF frames free their slot
              g.playq.erase(g.playq.begin(), g.playq.begin()+pick+1);
              if(f.slot>=0){                                    // MF: the picked slot is shown, the previous one is free
                  if(g.shown_slot>=0 && g.shown_slot!=f.slot) g.slot_state[g.shown_slot]=0;
                  g.shown_slot=f.slot; g.slot_state[f.slot]=2; cur_slot=f.slot; g.shown_seq=f.seq; g.lat_ms=t-f.t_in; }
              else if(f.seq!=g.shown_seq){ cur=f.buf; lw=g.fw; lh=g.fh; g.shown_seq=f.seq; g.lat_ms=t-f.t_in; }
          }
      } }
    if(cur || cur_slot>=0){ g.tlog_put("show", g.shown_seq); g.t_mark(5,"first frame shown"); }
    if(cur_slot>=0 && g.mf){                                  // MF: NV12 slot -> display texture on the GPU (no upload)
        const double t0=now_ms(); g.mf->render_slot(cur_slot); g.up_ms=now_ms()-t0; g.n_shown++; }
    // what the view shows this frame: the MF display texture, or the ffmpeg path's dynamic texture
    void* vtex=g.tex; int vw=g.tw, vh=g.th;
    if(g.mf_on.load() && g.mf && g.mf->state.load()==2 && g.mf->display_srv() && g.mf->frames.load()>0){
        vtex=g.mf->display_srv(); vw=g.mf->W; vh=g.mf->H; }
    if(cur){
        const double t0=now_ms();
        if(!g.dyn_tex || g.tw!=lw || g.th!=lh){          // (re)create the ONE dynamic texture only on a size change
            if(g.tex){ ((ID3D11ShaderResourceView*)g.tex)->Release(); g.tex=nullptr; }
            if(g.dyn_tex){ g.dyn_tex->Release(); g.dyn_tex=nullptr; }
            D3D11_TEXTURE2D_DESC td{}; td.Width=lw; td.Height=lh; td.MipLevels=1; td.ArraySize=1;
            td.Format=DXGI_FORMAT_R8G8B8A8_UNORM; td.SampleDesc.Count=1; td.Usage=D3D11_USAGE_DYNAMIC;
            td.BindFlags=D3D11_BIND_SHADER_RESOURCE; td.CPUAccessFlags=D3D11_CPU_ACCESS_WRITE;
            if(device() && device()->CreateTexture2D(&td,nullptr,&g.dyn_tex)==S_OK){
                ID3D11ShaderResourceView* srv=nullptr; device()->CreateShaderResourceView(g.dyn_tex,nullptr,&srv); g.tex=srv; }
            g.tw=lw; g.th=lh;
        }
        D3D11_MAPPED_SUBRESOURCE ms{};
        if(g.dyn_tex && detail::ctx->Map(g.dyn_tex,0,D3D11_MAP_WRITE_DISCARD,0,&ms)==S_OK){
            const unsigned char* src=cur->data(); const size_t row=(size_t)lw*4;
            if(ms.RowPitch==row) memcpy(ms.pData, src, row*lh);
            else for(int y=0;y<lh;y++) memcpy((unsigned char*)ms.pData+(size_t)y*ms.RowPitch, src+(size_t)y*row, row);
            detail::ctx->Unmap(g.dyn_tex,0);
        }
        g.last_rgba=cur; g.last_w=lw; g.last_h=lh;      // "Take photo" saves exactly what's on screen (pointer)
        g.up_ms=now_ms()-t0; g.n_shown++;
    }
    { const double t=now_ms(); if(g.stat_t0==0) g.stat_t0=t;   // receipt, once a second
      if(t-g.stat_t0>=1000.0){ g.fps_in=g.n_in.exchange(0)*1000.0/(t-g.stat_t0); g.fps_shown=g.n_shown*1000.0/(t-g.stat_t0);
          { const int n=g.det_n.exchange(0); const double sum=g.det_us_sum.exchange(0)/1000.0;
            g.det_rate=n*1000.0/(t-g.stat_t0); g.det_mean_ms=n? sum/n : 0; g.det_max_ms=g.det_us_max.exchange(0)/1000.0;
            g.det_busy=sum/(t-g.stat_t0)*100.0; }
          if(g.mf_on.load() && g.mf){ const int pn=g.ps_n.exchange(0), p30=g.ps_30.exchange(0), p50=g.ps_50.exchange(0);
              const double pmx=g.ps_max.exchange(0)/1000.0; char b[400]; std::snprintf(b,sizeof b,"[rx] in %.1f shown %.1f convert %.2f ms buffer %.0f ms | npu %.0f/s %.1f ms (max %.0f) busy %.0f%% gpu-img %.2f ms people %d pmax %.2f p30 %d/%d p50 %d/%d | sr %s %.1f ms #%llu",
              g.fps_in, g.fps_shown, g.up_ms, g.lat_ms, g.det_rate, g.det_mean_ms, g.det_max_ms, g.det_busy, (double)g.mf->det_gpu_ms.load(), g.det_people.load(), pmx, p30, pn, p50, pn,
              !g.sr_on.load()? "off" : g.sr_state.load()==1? (g.sr_want? "on" : "idle") : g.sr_state.load()==2? "unavailable" : "loading",
              g.sr_ms, (unsigned long long)g.sr_seq); g.mlog(b); }
          g.n_shown=0; g.stat_t0=t; } }

    // controls live in the dock under the video (09-30, draw_dock)
    bool af=g.auto_frame.load();
    bool em=g.edit_mask.load();
    int ds=g.det_state.load();

    // UV sub-rect. In EDIT mode we force FULL view so clicks map straight to normalized coords and you see it all.
    // Otherwise a frame-aspect crop needs half-width == half-height in normalized coords, so one scalar `c_h` drives both.
    float u0=0,v0=0,u1=1,v1=1; bool manual=false;
    if(!em && af && ds==1){                                   // AUTO-FRAME: eased crop toward the tracked subject
        // 09-30: the target that belongs to the frame on screen (detections run ~display-delay ahead), then a
        // critically damped spring on real time (smooth start/stop, frame-rate independent)
        static float acx=0.5f, acy=0.5f, ah=0.5f;             // target currently applied
        { std::lock_guard<std::mutex> lk(g.af_mtx);
          const double due = now_ms() - (g.buffer_ms>0? g.lat_ms : 0.0) + 40.0;   // +40: detector image is ~1 frame + exec old
          while(!g.tq.empty() && g.tq.front().t <= due){ acx=g.tq.front().cx; acy=g.tq.front().cy; ah=g.tq.front().h; g.tq.pop_front(); } }
        static double lt=0; const double tn=now_ms(); float dt = lt>0? (float)((tn-lt)/1000.0) : 0.016f; lt=tn;
        if(dt>0.1f) dt=0.1f; if(dt<0.0f) dt=0.0f;
        auto spring=[&](float& x, float& v, float target, float w){ const float acc=w*w*(target-x)-2.0f*w*v; v+=acc*dt; x+=v*dt; };
        if(g_follow_old){ float tcx,tcy,th; { std::lock_guard<std::mutex> lk(g.af_mtx); tcx=g.t_cx; tcy=g.t_cy; th=g.t_h; }   // A/B: old
            const float a=0.15f; g.c_cx+=(tcx-g.c_cx)*a; g.c_cy+=(tcy-g.c_cy)*a; g.c_h+=(th-g.c_h)*a; acx=tcx; acy=tcy; ah=th; }
        else { spring(g.c_cx, g.v_cx, acx, 3.2f); spring(g.c_cy, g.v_cy, acy, 3.2f); spring(g.c_h, g.v_h, ah, 2.6f); }
        if(g.c_h>0.5f) g.c_h=0.5f; if(g.c_h<0.08f) g.c_h=0.08f;
        if(g.c_cx<g.c_h) g.c_cx=g.c_h; if(g.c_cx>1-g.c_h) g.c_cx=1-g.c_h; if(g.c_cy<g.c_h) g.c_cy=g.c_h; if(g.c_cy>1-g.c_h) g.c_cy=1-g.c_h;
        { static std::FILE* tf=[](){ const char* e=std::getenv("PULSECORE_CCTV_TRACE_FRAMING"); return (e&&*e)? std::fopen(e,"w") : (std::FILE*)nullptr; }();
          if(tf){ std::fprintf(tf,"%.1f %.5f %.5f %.5f %.5f %.5f %.5f\n", tn, g.c_cx, g.c_cy, g.c_h, acx, acy, ah); std::fflush(tf); } }   // test hook
        u0=g.c_cx-g.c_h; v0=g.c_cy-g.c_h; u1=g.c_cx+g.c_h; v1=g.c_cy+g.c_h;
    } else if(!em){                                           // MANUAL zoom/pan (auto-frame off) — scroll + drag
        manual=true; g.c_cx=0.5f; g.c_cy=0.5f; g.c_h=0.5f; g.v_cx=g.v_cy=g.v_h=0;
        if(g.m_h<0.10f) g.m_h=0.10f; if(g.m_h>0.5f) g.m_h=0.5f;              // clamp zoom (max ~5x)
        if(g.m_cx<g.m_h) g.m_cx=g.m_h; if(g.m_cx>1-g.m_h) g.m_cx=1-g.m_h;    // keep crop inside the frame
        if(g.m_cy<g.m_h) g.m_cy=g.m_h; if(g.m_cy>1-g.m_h) g.m_cy=1-g.m_h;
        u0=g.m_cx-g.m_h; v0=g.m_cy-g.m_h; u1=g.m_cx+g.m_h; v1=g.m_cy+g.m_h;
    } else { g.c_cx=0.5f; g.c_cy=0.5f; g.c_h=0.5f; }          // EDIT mode -> full view

    // AI sharpening (09-30): ask for the current view (+10 % margin) when zoomed in enough, show it when it covers the view
    bool sr_draw=false; float su0=0,sv0=0,su1=1,sv1=1;
    { const float hh=(u1-u0)*0.5f; const bool zoomed=!em && vw>0 && (u1-u0)*vw<=1024.0f;
      { std::lock_guard<std::mutex> lk(g.sr_mtx);
        g.sr_want = g.sr_on.load() && zoomed;
        if(g.sr_want){ float mh=std::min(0.5f,hh*1.1f), cx=(u0+u1)*0.5f, cy=(v0+v1)*0.5f;
            cx=std::min(std::max(cx,mh),1-mh); cy=std::min(std::max(cy,mh),1-mh);
            g.sr_req[0]=cx-mh; g.sr_req[1]=cy-mh; g.sr_req[2]=cx+mh; g.sr_req[3]=cy+mh; }
        if(g.sr_want && g.sr_img){
            const float* r=g.sr_rect; const float rw=r[2]-r[0], rh=r[3]-r[1];
            if(u0>=r[0]-1e-4f && v0>=r[1]-1e-4f && u1<=r[2]+1e-4f && v1<=r[3]+1e-4f && rw>0 && rh>0){
                su0=(u0-r[0])/rw; sv0=(v0-r[1])/rh; su1=(u1-r[0])/rw; sv1=(v1-r[1])/rh; sr_draw=true;
                if(g.sr_seq!=g.sr_shown){                            // upload the new sharpened frame (one dynamic texture)
                    if(!g.sr_dyn){ D3D11_TEXTURE2D_DESC td{}; td.Width=SR_OW; td.Height=SR_OH; td.MipLevels=1; td.ArraySize=1;
                        td.Format=DXGI_FORMAT_R8G8B8A8_UNORM; td.SampleDesc.Count=1; td.Usage=D3D11_USAGE_DYNAMIC;
                        td.BindFlags=D3D11_BIND_SHADER_RESOURCE; td.CPUAccessFlags=D3D11_CPU_ACCESS_WRITE;
                        if(device() && device()->CreateTexture2D(&td,nullptr,&g.sr_dyn)==S_OK){
                            ID3D11ShaderResourceView* v=nullptr; device()->CreateShaderResourceView(g.sr_dyn,nullptr,&v); g.sr_tex=v; } }
                    D3D11_MAPPED_SUBRESOURCE ms{};
                    if(g.sr_dyn && detail::ctx->Map(g.sr_dyn,0,D3D11_MAP_WRITE_DISCARD,0,&ms)==S_OK){
                        const uint8_t* sp=g.sr_img->data();
                        for(int y=0;y<SR_OH;y++) memcpy((uint8_t*)ms.pData+(size_t)y*ms.RowPitch, sp+(size_t)y*SR_OW*4, (size_t)SR_OW*4);
                        detail::ctx->Unmap(g.sr_dyn,0); }
                    g.sr_shown=g.sr_seq; }
                if(!g.sr_tex) sr_draw=false;
            } } } }
    if(vtex){
        ImVec2 avail = ImGui::GetContentRegionAvail();
        const float room = std::max(120.0f, avail.y - 150.0f);   // 09-30: keep room for the dock + two text lines
        float w = avail.x, h = w*vh/(float)vw;                 // crop preserves frame aspect -> same fit math
        if(h > room){ h = room; w = h*vw/(float)vh; }
        if(w < avail.x) ImGui::SetCursorPosX(ImGui::GetCursorPosX()+(avail.x-w)*0.5f);   // centred
        ImGui::Dummy(ImVec2(w,h));                               // 09-30: rounded video (the item keeps the layout)
        ImVec2 p0=ImGui::GetItemRectMin(), p1=ImGui::GetItemRectMax(); float rw=p1.x-p0.x, rh=p1.y-p0.y;
        { const float R=12.0f; ImDrawList* dl=ImGui::GetWindowDrawList();
          if(sr_draw) dl->AddImageRounded((ImTextureID)g.sr_tex, p0, p1, ImVec2(su0,sv0), ImVec2(su1,sv1), IM_COL32_WHITE, R);
          else        dl->AddImageRounded((ImTextureID)vtex,     p0, p1, ImVec2(u0,v0),   ImVec2(u1,v1),   IM_COL32_WHITE, R); }
        if(em){
            // click adds a point (full view -> screen maps straight to normalized frame coords)
            if(ImGui::IsMouseHoveringRect(p0,p1) && ImGui::IsMouseClicked(ImGuiMouseButton_Left)){ ImVec2 m=ImGui::GetMousePos();
                float nx=(m.x-p0.x)/rw, ny=(m.y-p0.y)/rh; if(nx>=0&&nx<=1&&ny>=0&&ny<=1) g.mask_wip.push_back({nx,ny}); }
            // overlay: committed zones red, work-in-progress yellow with point handles
            ImDrawList* dl=ImGui::GetWindowDrawList(); dl->PushClipRect(p0,p1,true);
            auto S=[&](float nx,float ny){ return ImVec2(p0.x+nx*rw, p0.y+ny*rh); };
            std::vector<Poly> cp; { std::lock_guard<std::mutex> lk(g.mask_mtx); cp=g.masks; }
            for(auto& poly:cp){ std::vector<ImVec2> pts; for(auto& pt:poly) pts.push_back(S(pt.first,pt.second));
                dl->AddConvexPolyFilled(pts.data(),(int)pts.size(), IM_COL32(255,40,40,55));
                dl->AddPolyline(pts.data(),(int)pts.size(), IM_COL32(255,60,60,200), ImDrawFlags_Closed, 2.0f); }
            if(!g.mask_wip.empty()){ std::vector<ImVec2> pts; for(auto& pt:g.mask_wip) pts.push_back(S(pt.first,pt.second));
                dl->AddPolyline(pts.data(),(int)pts.size(), IM_COL32(255,220,40,230), 0, 2.0f);
                for(auto& q:pts) dl->AddCircleFilled(q,4.0f, IM_COL32(255,220,40,255)); }
            dl->PopClipRect();
        }
        else if(manual){
            // Overlay an invisible button on the image — a real interactive item reliably captures the mouse
            // wheel (zoom) and drag (pan); IsItemHovered() on a plain Image is flaky for wheel input.
            ImGui::SetCursorScreenPos(p0);
            ImGui::InvisibleButton("##view", ImVec2(rw,rh));
            if(ImGui::IsItemHovered()){
                float wh=ImGui::GetIO().MouseWheel;
                if(wh!=0.0f){
                    ImVec2 mp=ImGui::GetMousePos(); float fx=(mp.x-p0.x)/rw, fy=(mp.y-p0.y)/rh;
                    float curx=u0+fx*(u1-u0), cury=v0+fy*(v1-v0);            // frame point under the cursor
                    float nh=g.m_h*(wh>0? 0.85f : 1.0f/0.85f); if(nh<0.10f)nh=0.10f; if(nh>0.5f)nh=0.5f;
                    g.m_cx=curx-(fx*2.0f-1.0f)*nh; g.m_cy=cury-(fy*2.0f-1.0f)*nh; g.m_h=nh;   // keep that point fixed
                }
            }
            if(ImGui::IsItemActive()){                                       // holding + moving -> pan the view
                ImVec2 dd=ImGui::GetIO().MouseDelta;
                g.m_cx-=dd.x/rw*(u1-u0); g.m_cy-=dd.y/rh*(v1-v0);
            }
        }
        // 10-04: a photo taken - the flash, then a card that says where it went
        { const double el=ImGui::GetTime()-g.photo_flash_at;
          if(el>=0 && el<4.0){
            ImDrawList* dl=ImGui::GetWindowDrawList();      // (the window's: the buttons below draw on top)
            if(g.photo_ok && el<0.35){ const int a=(int)(230*(1.0-el/0.35)); dl->AddRectFilled(p0,p1,IM_COL32(255,255,255,a),12.0f); }
            const float fade=(float)(el<3.6? 1.0 : (4.0-el)/0.4);
            const float cw=360.0f, ch=g.photo_ok? 112.0f : 76.0f, m=14.0f;
            const ImVec2 c0(p1.x-cw-m, p1.y-ch-m), c1(p1.x-m, p1.y-m);
            const ImU32 bg = g.photo_ok? IM_COL32(32,36,42,(int)(235*fade)) : IM_COL32(120,36,36,(int)(235*fade));
            dl->AddRectFilled(c0,c1,bg,10.0f);
            dl->AddRect(c0,c1,IM_COL32(110,210,180,(int)(200*fade)),10.0f,0,1.5f);
            const std::string full=g.photo_path; const size_t sl=full.find_last_of("\\/");
            const std::string name= sl==std::string::npos? full : full.substr(sl+1), dir= sl==std::string::npos? std::string() : full.substr(0,sl);
            float tx=c0.x+14.0f;
            if(g.photo_ok){                                 // a small preview of the frame (what was saved: what is on screen)
                const float th=ch-28.0f, tw=th*vw/(float)vh;
                dl->AddImageRounded((ImTextureID)vtex, ImVec2(c0.x+14.0f,c0.y+14.0f), ImVec2(c0.x+14.0f+tw,c0.y+14.0f+th), ImVec2(u0,v0), ImVec2(u1,v1), IM_COL32(255,255,255,(int)(255*fade)), 6.0f);
                tx=c0.x+28.0f+tw; }
            const ImU32 tc=IM_COL32(240,244,248,(int)(255*fade)), dc=IM_COL32(170,180,192,(int)(255*fade));
            dl->AddText(ImVec2(tx,c0.y+12.0f), tc, g.photo_ok? "Photo saved" : "Photo not saved");
            dl->AddText(ImVec2(tx,c0.y+34.0f), dc, g.photo_ok? name.c_str() : g.saved_msg.c_str());
            if(g.photo_ok){
                std::string d2=dir; if(d2.size()>34) d2="\xE2\x80\xA6"+d2.substr(d2.size()-33);
                dl->AddText(ImVec2(tx,c0.y+52.0f), dc, d2.c_str());
                const ImVec2 keep=ImGui::GetCursorScreenPos();
                ImGui::SetCursorScreenPos(ImVec2(tx,c1.y-34.0f));
                ImGui::PushStyleVar(ImGuiStyleVar_Alpha, fade);
                if(ImGui::SmallButton("Open##photo")) ShellExecuteW(nullptr,L"open",u8w(g.photo_path).c_str(),nullptr,nullptr,SW_SHOWNORMAL);
                ImGui::SameLine();
                if(ImGui::SmallButton("Folder##photo")){ const std::wstring arg=L"/select,\""+u8w(g.photo_path)+L"\"";
                    ShellExecuteW(nullptr,L"open",L"explorer.exe",arg.c_str(),nullptr,SW_SHOWNORMAL); }
                ImGui::PopStyleVar();
                ImGui::SetCursorScreenPos(keep); }
          } }
    } else {
        ImGui::Dummy(ImVec2(0,20));
        ImGui::TextDisabled("Connecting to the camera\xE2\x80\xA6");
        // 09-30 diagnostics: is ffmpeg alive, did any frame arrive?
        DWORD code=STILL_ACTIVE; bool have=g.ff_proc!=nullptr;
        if(have) GetExitCodeProcess((HANDLE)g.ff_proc,&code);
        if(g.reconnects>0) ImGui::TextDisabled("reconnecting to the camera (attempt %d) \xE2\x80\x94 it may still hold an earlier session", g.reconnects);
        if(g.dec_mode==0){ auto d=g.mf; const int st=d? d->state.load() : -1;
            static const char* sn[]={"idle","opening the stream","running","failed","ended"};
            ImGui::TextDisabled("GPU decode (Media Foundation): %s%s%s \xC2\xB7 frames %lld \xC2\xB7 log: %%TEMP%%\\pulsecore_cctv_mf.log",
                st<0? "not started (tapo_rtsp.txt?)" : sn[st], (d && !d->err.empty())? " \xE2\x80\x94 " : "", d? d->err.c_str() : "", (long long)g.n_total.load()); }
        else
        ImGui::TextDisabled("frames received: %lld \xC2\xB7 ffmpeg: %s (exit %lu) \xC2\xB7 log: %s", (long long)g.n_total.load(),
            !have? "not started (tapo_rtsp.txt?)" : (code==STILL_ACTIVE? "running" : "EXITED"), (unsigned long)code, g.ff_log.c_str());
    }
    draw_dock(af, em, ds);                                    // 09-30: controls under the video
}

int main(int, char**){
    if(const char* e=std::getenv("PULSECORE_CCTV_NPU")){ if(*e && !strcmp(e,"gate")) g.det_steady=false; }
    if(g.dec_mode==0) g.det_steady=true;               // MF mode feeds the detector per frame (no gated path)
    pcore::gui::detail::device_flags = D3D11_CREATE_DEVICE_VIDEO_SUPPORT | D3D11_CREATE_DEVICE_BGRA_SUPPORT;   // 09-30: DXVA + video processor
    MFStartup(MF_VERSION, MFSTARTUP_FULL);
    g_t_start = now_ms(); g.t_mark(0,"main");
    std::thread([](){                                  // 09-30: purge thread — the window and the stream do not wait for it
      const double t0=now_ms();
      const long long r = purge_memory(true);            // like zimage purge_on_start=full (needs admin)
      g_purged_at_start = r;
      char b[120]; std::snprintf(b,sizeof b,"purge done (%.0f ms, %s)", now_ms()-t0,
          r==-2? "not admin - skipped" : r==-1? "off/failed" : (std::to_string(r)+" MB freed").c_str());
      g.t_mark(1,b); }).detach();
    { const char* e=std::getenv("PULSECORE_CCTV_AUTOFRAME"); g.auto_frame.store(!(e && *e=='0')); }   // 09-30: people-zoom on at start
    g.masks = load_masks();                            // exclusion zones from tapo_mask.txt (empty -> ignore_top fallback)
    // 09-30: the stream starts on the first drawn frame (Media Foundation needs the D3D11 device run_app creates)
    load_settings();                                   // 09-30: storage folder + auto-record (%APPDATA%\PulseX\cctv.ini)
    g.rec.dir=clips_dir(); g.rec.armed=g_auto_record;
    g.mlog(std::string("[cctv] PulseX CCTV ")+PULSEX_CCTV_VERSION);
    { char b[160]; std::snprintf(b,sizeof b,"[cctv] settings: auto_record %d ignore_top %.2f save_dir %s", g_auto_record?1:0, g_ignore_top,
          g_save_dir.empty()? "(default)" : "(chosen)"); g.mlog(b); }                 // 09-30: witness what cctv.ini gave
    g.rec.log=[](const std::string& m){ g.mlog(m); };
    g.rec.notice=[](const std::string& m){ std::lock_guard<std::mutex> lk(g.note_mtx); g.pending_note=m; };
    g.rec.start();
    g.af_run=true; g.af_thread=std::thread(&Cctv::af_loop,&g);   // on-NPU auto-framing worker (idle until toggled on)
    g.sr_run=true; g.sr_thread=std::thread(&Cctv::sr_loop,&g);   // 09-30: AI sharpening worker (idle until toggled on)
    int r = run_app("PulseX CCTV", draw_cctv, 620, 820);
    g.rec.stop();                                      // 09-30: finalize a clip in progress first
    g.stop_stream();                                   // decoder / ffmpeg gone before the purge
    g.sr_run=false; if(g.sr_thread.joinable()) g.sr_thread.join();          // 09-30: NPU workers end, then the
    g.af_run=false; g.frame_cv.notify_all(); if(g.af_thread.joinable()) g.af_thread.join();
    { std::lock_guard<std::mutex> lk(g.npu_mtx); g.det_.close(); }         // contexts + device are freed HERE
    MFShutdown();
    purge_memory(false);                               // 09-30: like zimage purge_on_exit=standby
    return r;
}
