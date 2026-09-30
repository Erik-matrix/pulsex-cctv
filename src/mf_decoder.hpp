// mf_decoder.hpp — 2026-09-30: in-app video decode the way Windows Studio Effects does it (memory:
// studio-effects-runtime-chain-and-design-0930). Media Foundation reads the RTSP stream and decodes it on the SoC's
// video engine (DXVA) straight into D3D11 NV12 textures on OUR device; the D3D11 video processor (MF's "XVP") converts
// and scales on the GPU: full-res RGB for the display, a small image for the NPU, a 512x288 crop for QuickSRNet.
// Only small images ever come back to the CPU — the 14.7 MB RGBA frame of the ffmpeg pipe is gone.
//
//   ring of NV12 slot textures (GPU)  <- CopySubresourceRegion from the decoder surface (decoder pool never starved)
//   render thread : slot -> display texture (video processor #1)
//   decoder thread: slot -> 640x640 -> staging -> CPU (detector)      only when the detector is idle
//                   slot crop -> 512x288 -> staging -> CPU (SR)       only when SR wants it and is idle
//
// The owner (pulsecore_cctv.cpp) supplies the hooks: which ring slot to fill, where a finished frame goes.
// The D3D11 device must be created with D3D11_CREATE_DEVICE_VIDEO_SUPPORT (app_shell detail::device_flags).
#pragma once
#include <d3d11.h>
#include <d3d11_4.h>
#include <mfapi.h>
#include <mfidl.h>
#include <mfreadwrite.h>
#include <mferror.h>
#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#pragma comment(lib, "mfplat.lib")
#pragma comment(lib, "mfreadwrite.lib")
#pragma comment(lib, "mfuuid.lib")
#pragma comment(lib, "mf.lib")                 // MFCreateCredentialCache
#pragma comment(lib, "ole32.lib")

template<class T> static void mf_rel(T*& p){ if(p){ p->Release(); p=nullptr; } }

// RTSP credentials: Media Foundation's network source asks an IMFNetCredentialManager instead of reading user:pass
// from the URL. The owner's URL rtsp://user:pass@host/... is split; the password never goes into a log.
class MfCredMgr : public IMFNetCredentialManager {
    long ref_=1; std::wstring user_, pass_; IMFNetCredential* cred_=nullptr;
public:
    MfCredMgr(std::wstring u, std::wstring p): user_(std::move(u)), pass_(std::move(p)) {}
    virtual ~MfCredMgr(){ mf_rel(cred_); }
    STDMETHODIMP QueryInterface(REFIID r, void** o) override {
        if(r==__uuidof(IUnknown) || r==__uuidof(IMFNetCredentialManager)){ *o=static_cast<IMFNetCredentialManager*>(this); AddRef(); return S_OK; }
        *o=nullptr; return E_NOINTERFACE; }
    STDMETHODIMP_(ULONG) AddRef() override { return (ULONG)InterlockedIncrement(&ref_); }
    STDMETHODIMP_(ULONG) Release() override { long r=InterlockedDecrement(&ref_); if(!r) delete this; return (ULONG)r; }
    STDMETHODIMP BeginGetCredentials(MFNetCredentialManagerGetParam* p, IMFAsyncCallback* cb, IUnknown* st) override {
        IMFNetCredentialCache* cache=nullptr; HRESULT hr=MFCreateCredentialCache(&cache);
        if(SUCCEEDED(hr)){
            mf_rel(cred_); DWORD req=0;
            hr=cache->GetCredential(p->pszUrl, p->pszRealm, p->fClearTextPackage? MFNET_AUTHENTICATION_CLEAR_TEXT : 0, &cred_, &req);
            if(SUCCEEDED(hr) && cred_){
                cred_->SetUser((BYTE*)user_.c_str(), (DWORD)((user_.size()+1)*sizeof(wchar_t)), FALSE);
                cred_->SetPassword((BYTE*)pass_.c_str(), (DWORD)((pass_.size()+1)*sizeof(wchar_t)), FALSE); }
            cache->Release(); }
        IMFAsyncResult* res=nullptr;
        if(SUCCEEDED(MFCreateAsyncResult(nullptr, cb, st, &res))){ res->SetStatus(hr); MFInvokeCallback(res); res->Release(); }
        return S_OK; }
    STDMETHODIMP EndGetCredentials(IMFAsyncResult*, IMFNetCredential** c) override {
        *c=cred_; if(cred_){ cred_->AddRef(); return S_OK; } return E_FAIL; }
    STDMETHODIMP SetGood(IMFNetCredential*, BOOL) override { return S_OK; }
};

struct MfDecoder {
    // ── hooks (set by the owner before start) ──
    std::function<int()> acquire_slot;                                   // a ring slot the decoder may overwrite, -1 = none
    std::function<void(int slot, double t_in)> publish;                  // that slot now holds the newest frame
    std::function<bool()> want_det;                                      // detector idle -> hand it the ready image now
    std::function<bool()> det_enabled;                                   // detector loaded -> keep one image in flight
    std::function<void(const uint8_t* px, int w, int h, int pitch, bool bgra, int tile, float u0, float u1)> publish_det;
    bool tiles=true;                       // 09-30: two square H x H tiles, alternating (aspect-correct NPU input)
    std::function<bool(float rect[4])> want_sr;                          // SR wants this normalized crop now
    std::function<void(const uint8_t* px, int w, int h, int pitch, bool bgra, const float rect[4])> publish_sr;
    std::function<void(const std::string&)> log;

    int det_w=640, det_h=640, sr_w=512, sr_h=288, nslots=16;
    std::atomic<int> state{0};             // 0 idle, 1 opening, 2 running, 3 failed, 4 ended
    std::atomic<long long> frames{0};
    std::atomic<float> det_gpu_ms{0};      // video-processor blt + readback for the NPU image (decoder thread)
    std::string err;                       // last failure (no URL/password in it)
    int W=0, H=0; bool bgra=false; bool bt709=true, full_range=false; std::string bind_note;

    ~MfDecoder(){ free_gpu(); }

    // ── recording ring (09-30): rec_w x rec_h NV12 copies of the recent frames for the recorder ──
    std::function<void(int idx, double t)> on_rec_frame;           // decoder thread, after a slot was filled
    std::atomic<bool> rec_want{false}; int rec_w=1920, rec_h=1080, rec_n=105;
    ID3D11Device* device() const { return dev_; }
    ID3D11Texture2D* rec_tex(int i){ std::lock_guard<std::mutex> lk(rec_mtx_); return (i>=0 && i<(int)rec_.size())? rec_[i].tex : nullptr; }
    void rec_pin(int i){ std::lock_guard<std::mutex> lk(rec_mtx_); if(i>=0 && i<(int)rec_.size()) rec_[i].pins++; }
    void rec_unpin(int i){ std::lock_guard<std::mutex> lk(rec_mtx_); if(i>=0 && i<(int)rec_.size() && rec_[i].pins>0) rec_[i].pins--; }
    // frames with arrival time >= t0, oldest first, each PINNED for the caller
    std::vector<std::pair<int,double>> rec_since(double t0){
        std::lock_guard<std::mutex> lk(rec_mtx_); std::vector<std::pair<int,double>> v;
        for(int i=0;i<(int)rec_.size();i++) if(rec_[i].t>0 && rec_[i].t>=t0){ rec_[i].pins++; v.push_back({i,rec_[i].t}); }
        std::sort(v.begin(), v.end(), [](const std::pair<int,double>& a, const std::pair<int,double>& b){ return a.second<b.second; });
        return v; }
    std::atomic<int> rec_dropped{0};                               // frames not captured because every slot was pinned

    // start the decoder thread; the thread keeps this object alive (shared_ptr) until it has exited
    static std::shared_ptr<MfDecoder> start(std::shared_ptr<MfDecoder> self, ID3D11Device* dev, const std::string& url){
        self->dev_=dev; dev->AddRef(); dev->GetImmediateContext(&self->ctx_);
        self->run_=true; self->exited_=CreateEventA(nullptr,TRUE,FALSE,nullptr);
        std::thread([self, url](){ self->body(url); SetEvent(self->exited_); }).detach();
        return self; }
    // ask the thread to stop; wait up to `ms` (an RTSP open can block inside Media Foundation)
    bool stop(DWORD ms=3000){ run_=false; if(!exited_) return true; return WaitForSingleObject(exited_, ms)==WAIT_OBJECT_0; }

    // ── render thread ──
    ID3D11ShaderResourceView* display_srv() const { return disp_srv_; }
    bool render_slot(int slot){                                          // slot -> display texture
        if(slot<0 || slot>=(int)ring_.size() || !disp_ov_ || !vp_render_) return false;
        RECT src{0,0,W,H}; blt(vp_render_, ring_[slot].iv, src, disp_ov_, W, H); return true; }
    // the displayed frame as RGBA (for "Take photo"); render thread
    bool read_display(std::vector<uint8_t>& rgba){
        if(!disp_tex_) return false;
        if(!photo_stage_){ D3D11_TEXTURE2D_DESC d{}; disp_tex_->GetDesc(&d); d.Usage=D3D11_USAGE_STAGING; d.BindFlags=0;
            d.CPUAccessFlags=D3D11_CPU_ACCESS_READ; d.MiscFlags=0; if(FAILED(dev_->CreateTexture2D(&d,nullptr,&photo_stage_))) return false; }
        ctx_->CopyResource(photo_stage_, disp_tex_);
        D3D11_MAPPED_SUBRESOURCE m{}; if(FAILED(ctx_->Map(photo_stage_,0,D3D11_MAP_READ,0,&m))) return false;
        rgba.resize((size_t)W*H*4); copy_rgba(rgba.data(), (const uint8_t*)m.pData, W, H, m.RowPitch, bgra);
        ctx_->Unmap(photo_stage_,0); return true; }
    // pitched BGRA/RGBA -> tight RGBA
    static void copy_rgba(uint8_t* dst, const uint8_t* src, int w, int h, int pitch, bool swap){
        for(int y=0;y<h;y++){ const uint8_t* s=src+(size_t)y*pitch; uint8_t* d=dst+(size_t)y*w*4;
            if(!swap){ std::memcpy(d,s,(size_t)w*4); continue; }
            for(int x=0;x<w;x++){ d[4*x]=s[4*x+2]; d[4*x+1]=s[4*x+1]; d[4*x+2]=s[4*x]; d[4*x+3]=255; } } }

private:
    ID3D11Device* dev_=nullptr; ID3D11DeviceContext* ctx_=nullptr;
    ID3D11VideoDevice* vdev_=nullptr; ID3D11VideoContext* vctx_=nullptr;
    ID3D11VideoProcessorEnumerator* vpe_=nullptr; ID3D11VideoProcessor *vp_render_=nullptr, *vp_dec_=nullptr;
    struct Slot { ID3D11Texture2D* tex=nullptr; ID3D11VideoProcessorInputView* iv=nullptr; };
    std::vector<Slot> ring_;
    ID3D11Texture2D *disp_tex_=nullptr, *det_rt_=nullptr, *sr_rt_=nullptr, *sr_stage_=nullptr, *photo_stage_=nullptr;
    ID3D11Texture2D* det_stage_[2]={nullptr,nullptr}; int det_next_=0, det_pending_=-1;   // two staging buffers, one in flight
    int det_ptile_[2]={-1,-1}; float det_pu_[2][2]={{0,1},{0,1}}; int tile_next_=0;       // which tile each staging buffer holds
    ID3D11ShaderResourceView* disp_srv_=nullptr;
    ID3D11VideoProcessorOutputView *disp_ov_=nullptr, *det_ov_=nullptr, *sr_ov_=nullptr;
    IMFDXGIDeviceManager* mgr_=nullptr; UINT token_=0;
    IMFMediaSource* source_=nullptr; IMFSourceReader* reader_=nullptr;
    struct RecSlot { ID3D11Texture2D* tex=nullptr; ID3D11VideoProcessorOutputView* ov=nullptr; double t=0; int pins=0; };
    std::vector<RecSlot> rec_; std::mutex rec_mtx_; int rec_next_=0; bool rec_fail_=false;
    bool rec_alloc(){
        UINT f=0; vpe_->CheckVideoProcessorFormat(DXGI_FORMAT_NV12,&f);
        if(!(f & D3D11_VIDEO_PROCESSOR_FORMAT_SUPPORT_OUTPUT)){ L("recording: the video processor cannot write NV12 - off"); return false; }
        std::vector<RecSlot> v(rec_n);
        for(auto& r: v){
            D3D11_TEXTURE2D_DESC d{}; d.Width=rec_w; d.Height=rec_h; d.MipLevels=1; d.ArraySize=1; d.Format=DXGI_FORMAT_NV12;
            d.SampleDesc.Count=1; d.Usage=D3D11_USAGE_DEFAULT; d.BindFlags=D3D11_BIND_RENDER_TARGET;
            D3D11_VIDEO_PROCESSOR_OUTPUT_VIEW_DESC od{}; od.ViewDimension=D3D11_VPOV_DIMENSION_TEXTURE2D;
            if(FAILED(dev_->CreateTexture2D(&d,nullptr,&r.tex)) || FAILED(vdev_->CreateVideoProcessorOutputView(r.tex,vpe_,&od,&r.ov))){
                for(auto& q: v){ mf_rel(q.ov); mf_rel(q.tex); } L("recording: ring allocation failed - off"); return false; } }
        { std::lock_guard<std::mutex> lk(rec_mtx_); rec_.swap(v); rec_next_=0; }
        L("recording ring: "+std::to_string(rec_n)+" x "+std::to_string(rec_w)+"x"+std::to_string(rec_h)+" NV12 ("
          +std::to_string((long long)rec_n*rec_w*rec_h*3/2/(1024*1024))+" MB)");
        return true; }
    std::atomic<bool> run_{false}; HANDLE exited_=nullptr;

    void L(const std::string& s){ if(log) log(s); }
    void fail(const char* what, HRESULT hr){ char b[160]; std::snprintf(b,sizeof b,"%s failed, hr=0x%08lx",what,(unsigned long)hr);
        err=b; L(err); state=3; }
    static double now_ms(){ return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
    static std::wstring widen(const std::string& s){ if(s.empty()) return L""; int n=MultiByteToWideChar(CP_UTF8,0,s.c_str(),(int)s.size(),nullptr,0);
        std::wstring w(n,L'\0'); MultiByteToWideChar(CP_UTF8,0,s.c_str(),(int)s.size(),&w[0],n); return w; }
    static std::string pct_decode(const std::string& s){ std::string o; for(size_t i=0;i<s.size();i++){
        if(s[i]=='%' && i+2<s.size()){ o+=(char)std::strtol(s.substr(i+1,2).c_str(),nullptr,16); i+=2; } else o+=s[i]; } return o; }

    void body(std::string url){
        CoInitializeEx(nullptr, COINIT_MULTITHREADED);
        state=1;
        const bool is_file = url.find("://")==std::string::npos;
        std::string user, pass, clean=url;
        if(!is_file){                                                    // rtsp://user:pass@host -> rtspt://host (TCP) + creds
            size_t s=clean.find("://")+3, at=clean.find('@',s), sl=clean.find('/',s);
            if(at!=std::string::npos && (sl==std::string::npos || at<sl)){
                std::string ui=clean.substr(s,at-s); size_t c=ui.find(':');
                user=pct_decode(ui.substr(0,c)); pass= c==std::string::npos? "" : pct_decode(ui.substr(c+1));
                clean=clean.substr(0,s)+clean.substr(at+1); }
            const char* udp=std::getenv("PULSECORE_CCTV_MF_UDP");
            if(clean.compare(0,7,"rtsp://")==0 && !(udp && *udp=='1')) clean="rtspt://"+clean.substr(7);   // like ffmpeg -rtsp_transport tcp
            if(const char* k=std::getenv("PULSECORE_CCTV_MF_URLCREDS")){ if(*k=='1') clean=url; }            // A/B: creds left in the URL
        }
        L(std::string("opening ")+(is_file? "file" : "network stream")+(user.empty()? "" : " (credentials via IMFNetCredentialManager)"));
        HRESULT hr=open(clean, user, pass);
        if(FAILED(hr)){ if(state!=3) fail("open", hr); teardown_mf(); CoUninitialize(); return; }
        if(!run_){ teardown_mf(); CoUninitialize(); return; }
        state=2; L("running: "+std::to_string(W)+"x"+std::to_string(H)+(bgra? " BGRA" : " RGBA")+" out, "+(bt709? "BT.709" : "BT.601")
                   +(full_range? " full" : " limited")+" range, slot bind="+bind_note+", "+std::to_string(ring_.size())+" slots");
        double t0=0; LONGLONG ts0=-1;
        while(run_){
            DWORD si=0, fl=0; LONGLONG ts=0; IMFSample* s=nullptr;
            hr=reader_->ReadSample(MF_SOURCE_READER_FIRST_VIDEO_STREAM,0,&si,&fl,&ts,&s);
            if(FAILED(hr)){ fail("ReadSample", hr); break; }
            if(fl & MF_SOURCE_READERF_ENDOFSTREAM){
                if(is_file){ PROPVARIANT pv; PropVariantInit(&pv); pv.vt=VT_I8; pv.hVal.QuadPart=0;       // test file: loop
                    reader_->SetCurrentPosition(GUID_NULL,pv); ts0=-1; mf_rel(s); continue; }
                err="end of stream"; L(err); state=4; mf_rel(s); break; }
            if(fl & MF_SOURCE_READERF_CURRENTMEDIATYPECHANGED){
                UINT32 w=0,h=0; IMFMediaType* mt=nullptr;
                if(SUCCEEDED(reader_->GetCurrentMediaType(MF_SOURCE_READER_FIRST_VIDEO_STREAM,&mt))){ MFGetAttributeSize(mt,MF_MT_FRAME_SIZE,&w,&h); mt->Release(); }
                if((int)w!=W || (int)h!=H){ err="frame size changed "+std::to_string(W)+"x"+std::to_string(H)+" -> "+std::to_string(w)+"x"+std::to_string(h);
                    L(err); state=3; mf_rel(s); break; } }
            if(!s) continue;
            if(is_file){                                                  // a file plays as fast as it can: pace it like a camera
                if(ts0<0){ ts0=ts; t0=now_ms(); }
                const double due=t0+(double)(ts-ts0)/1e4, w=due-now_ms(); if(w>1) Sleep((DWORD)w); }
            on_sample(s); s->Release();
        }
        teardown_mf(); CoUninitialize();
    }

    HRESULT open(const std::string& url, const std::string& user, const std::string& pass){
        HRESULT hr;
        { ID3D11Multithread* mt=nullptr;                                 // MF + our render thread share the device
          if(SUCCEEDED(ctx_->QueryInterface(__uuidof(ID3D11Multithread),(void**)&mt))){ mt->SetMultithreadProtected(TRUE); mt->Release(); } }
        if(FAILED(hr=MFCreateDXGIDeviceManager(&token_,&mgr_))) { fail("MFCreateDXGIDeviceManager",hr); return hr; }
        if(FAILED(hr=mgr_->ResetDevice(dev_,token_))) { fail("DXGIDeviceManager::ResetDevice (device without VIDEO_SUPPORT?)",hr); return hr; }
        // source (network properties: credentials, a short buffer)
        IMFSourceResolver* res=nullptr; IPropertyStore* props=nullptr;
        if(FAILED(hr=MFCreateSourceResolver(&res))) { fail("MFCreateSourceResolver",hr); return hr; }
        if(SUCCEEDED(CreatePropertyStore(&props))){
            if(!user.empty()){ MfCredMgr* cm=new MfCredMgr(widen(user),widen(pass));
                PROPERTYKEY k{MFNETSOURCE_CREDENTIAL_MANAGER,0}; PROPVARIANT v; PropVariantInit(&v); v.vt=VT_UNKNOWN; v.punkVal=cm;
                props->SetValue(k,v); cm->Release(); }
            PROPERTYKEY kb{MFNETSOURCE_MAXBUFFERTIMEMS,0}; PROPVARIANT vb; PropVariantInit(&vb); vb.vt=VT_I4; vb.lVal=500; props->SetValue(kb,vb); }
        MF_OBJECT_TYPE ot=MF_OBJECT_INVALID; IUnknown* obj=nullptr;
        hr=res->CreateObjectFromURL(widen(url).c_str(), MF_RESOLUTION_MEDIASOURCE|MF_RESOLUTION_CONTENT_DOES_NOT_HAVE_TO_MATCH_EXTENSION_OR_MIME_TYPE,
                                    props, &ot, &obj);
        mf_rel(props); mf_rel(res);
        if(FAILED(hr)){ fail("CreateObjectFromURL",hr); return hr; }
        hr=obj->QueryInterface(IID_PPV_ARGS(&source_)); obj->Release(); if(FAILED(hr)){ fail("IMFMediaSource",hr); return hr; }
        // reader: hardware decode onto our device, low latency, NV12 out
        IMFAttributes* a=nullptr; MFCreateAttributes(&a,4);
        a->SetUnknown(MF_SOURCE_READER_D3D_MANAGER, mgr_);
        a->SetUINT32(MF_READWRITE_ENABLE_HARDWARE_TRANSFORMS, TRUE);
        a->SetUINT32(MF_LOW_LATENCY, TRUE);
        a->SetUINT32(MF_SOURCE_READER_DISABLE_DXVA, FALSE);
        hr=MFCreateSourceReaderFromMediaSource(source_, a, &reader_); a->Release();
        if(FAILED(hr)){ fail("MFCreateSourceReaderFromMediaSource",hr); return hr; }
        reader_->SetStreamSelection((DWORD)MF_SOURCE_READER_ALL_STREAMS, FALSE);
        reader_->SetStreamSelection((DWORD)MF_SOURCE_READER_FIRST_VIDEO_STREAM, TRUE);
        IMFMediaType* mt=nullptr; MFCreateMediaType(&mt);
        mt->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video); mt->SetGUID(MF_MT_SUBTYPE, MFVideoFormat_NV12);
        hr=reader_->SetCurrentMediaType((DWORD)MF_SOURCE_READER_FIRST_VIDEO_STREAM, nullptr, mt); mt->Release();
        if(FAILED(hr)){ fail("SetCurrentMediaType(NV12) - codec not decodable here?",hr); return hr; }
        IMFMediaType* cur=nullptr;
        if(FAILED(hr=reader_->GetCurrentMediaType((DWORD)MF_SOURCE_READER_FIRST_VIDEO_STREAM,&cur))){ fail("GetCurrentMediaType",hr); return hr; }
        UINT32 w=0,h=0; MFGetAttributeSize(cur,MF_MT_FRAME_SIZE,&w,&h); W=(int)w; H=(int)h;
        UINT32 m=0; if(SUCCEEDED(cur->GetUINT32(MF_MT_YUV_MATRIX,&m))) bt709=(m!=MFVideoTransferMatrix_BT601); else bt709=(H>=720);
        UINT32 nr=0; if(SUCCEEDED(cur->GetUINT32(MF_MT_VIDEO_NOMINAL_RANGE,&nr))) full_range=(nr==MFNominalRange_0_255);
        cur->Release();
        if(W<=0||H<=0){ err="no frame size"; state=3; L(err); return E_FAIL; }
        return setup_gpu();
    }

    HRESULT setup_gpu(){
        HRESULT hr;
        if(FAILED(hr=dev_->QueryInterface(IID_PPV_ARGS(&vdev_)))){ fail("ID3D11VideoDevice",hr); return hr; }
        if(FAILED(hr=ctx_->QueryInterface(IID_PPV_ARGS(&vctx_)))){ fail("ID3D11VideoContext",hr); return hr; }
        D3D11_VIDEO_PROCESSOR_CONTENT_DESC cd{}; cd.InputFrameFormat=D3D11_VIDEO_FRAME_FORMAT_PROGRESSIVE;
        cd.InputFrameRate={30,1}; cd.InputWidth=W; cd.InputHeight=H; cd.OutputFrameRate={30,1}; cd.OutputWidth=W; cd.OutputHeight=H;
        cd.Usage=D3D11_VIDEO_USAGE_OPTIMAL_SPEED;
        if(FAILED(hr=vdev_->CreateVideoProcessorEnumerator(&cd,&vpe_))){ fail("CreateVideoProcessorEnumerator",hr); return hr; }
        UINT f=0; vpe_->CheckVideoProcessorFormat(DXGI_FORMAT_R8G8B8A8_UNORM,&f);
        bgra = !(f & D3D11_VIDEO_PROCESSOR_FORMAT_SUPPORT_OUTPUT);
        const DXGI_FORMAT ofmt = bgra? DXGI_FORMAT_B8G8R8A8_UNORM : DXGI_FORMAT_R8G8B8A8_UNORM;
        if(FAILED(hr=vdev_->CreateVideoProcessor(vpe_,0,&vp_render_))){ fail("CreateVideoProcessor",hr); return hr; }
        if(FAILED(hr=vdev_->CreateVideoProcessor(vpe_,0,&vp_dec_))){ fail("CreateVideoProcessor",hr); return hr; }
        // NV12 ring: find bind flags this driver accepts for a video-processor input view
        const UINT tries[3]={0, D3D11_BIND_SHADER_RESOURCE, D3D11_BIND_RENDER_TARGET}; const char* names[3]={"none","shader-resource","render-target"};
        ring_.resize(nslots);
        for(int k=0;k<nslots;k++){
            bool ok=false;
            for(int t=0;t<3 && !ok;t++){
                if(k>0 && bind_note!=names[t]) continue;                 // all slots like the first
                D3D11_TEXTURE2D_DESC d{}; d.Width=W; d.Height=H; d.MipLevels=1; d.ArraySize=1; d.Format=DXGI_FORMAT_NV12;
                d.SampleDesc.Count=1; d.Usage=D3D11_USAGE_DEFAULT; d.BindFlags=tries[t];
                if(FAILED(dev_->CreateTexture2D(&d,nullptr,&ring_[k].tex))) continue;
                D3D11_VIDEO_PROCESSOR_INPUT_VIEW_DESC iv{}; iv.FourCC=0; iv.ViewDimension=D3D11_VPIV_DIMENSION_TEXTURE2D;
                if(FAILED(vdev_->CreateVideoProcessorInputView(ring_[k].tex,vpe_,&iv,&ring_[k].iv))){ mf_rel(ring_[k].tex); continue; }
                bind_note=names[t]; ok=true; }
            if(!ok){ fail("NV12 slot texture / input view",E_FAIL); return E_FAIL; } }
        auto mk_out=[&](int w,int h,bool srv,ID3D11Texture2D** tex,ID3D11VideoProcessorOutputView** ov,ID3D11ShaderResourceView** sv)->HRESULT{
            D3D11_TEXTURE2D_DESC d{}; d.Width=w; d.Height=h; d.MipLevels=1; d.ArraySize=1; d.Format=ofmt; d.SampleDesc.Count=1;
            d.Usage=D3D11_USAGE_DEFAULT; d.BindFlags=D3D11_BIND_RENDER_TARGET|(srv? D3D11_BIND_SHADER_RESOURCE : 0);
            HRESULT r=dev_->CreateTexture2D(&d,nullptr,tex); if(FAILED(r)) return r;
            D3D11_VIDEO_PROCESSOR_OUTPUT_VIEW_DESC od{}; od.ViewDimension=D3D11_VPOV_DIMENSION_TEXTURE2D;
            r=vdev_->CreateVideoProcessorOutputView(*tex,vpe_,&od,ov); if(FAILED(r)) return r;
            if(srv) r=dev_->CreateShaderResourceView(*tex,nullptr,sv);
            return r; };
        auto mk_stage=[&](int w,int h,ID3D11Texture2D** st)->HRESULT{
            D3D11_TEXTURE2D_DESC d{}; d.Width=w; d.Height=h; d.MipLevels=1; d.ArraySize=1; d.Format=ofmt; d.SampleDesc.Count=1;
            d.Usage=D3D11_USAGE_STAGING; d.CPUAccessFlags=D3D11_CPU_ACCESS_READ; return dev_->CreateTexture2D(&d,nullptr,st); };
        if(FAILED(hr=mk_out(W,H,true,&disp_tex_,&disp_ov_,&disp_srv_))){ fail("display texture",hr); return hr; }
        if(FAILED(hr=mk_out(det_w,det_h,false,&det_rt_,&det_ov_,nullptr)) || FAILED(hr=mk_stage(det_w,det_h,&det_stage_[0]))
           || FAILED(hr=mk_stage(det_w,det_h,&det_stage_[1]))){ fail("NPU image texture",hr); return hr; }
        if(FAILED(hr=mk_out(sr_w,sr_h,false,&sr_rt_,&sr_ov_,nullptr)) || FAILED(hr=mk_stage(sr_w,sr_h,&sr_stage_))){ fail("SR crop texture",hr); return hr; }
        return S_OK;
    }

    void blt(ID3D11VideoProcessor* vp, ID3D11VideoProcessorInputView* in, RECT src, ID3D11VideoProcessorOutputView* out, int ow, int oh, bool yuv_out=false){
        RECT dst{0,0,ow,oh};
        vctx_->VideoProcessorSetStreamFrameFormat(vp,0,D3D11_VIDEO_FRAME_FORMAT_PROGRESSIVE);
        vctx_->VideoProcessorSetStreamSourceRect(vp,0,TRUE,&src);
        vctx_->VideoProcessorSetStreamDestRect(vp,0,TRUE,&dst);
        vctx_->VideoProcessorSetOutputTargetRect(vp,TRUE,&dst);
        D3D11_VIDEO_PROCESSOR_COLOR_SPACE ci{}; ci.YCbCr_Matrix=bt709? 1 : 0;
        ci.Nominal_Range= full_range? D3D11_VIDEO_PROCESSOR_NOMINAL_RANGE_0_255 : D3D11_VIDEO_PROCESSOR_NOMINAL_RANGE_16_235;
        D3D11_VIDEO_PROCESSOR_COLOR_SPACE co{}; co.RGB_Range=0;         // full-range RGB out
        if(yuv_out){ co.YCbCr_Matrix=1; co.Nominal_Range=D3D11_VIDEO_PROCESSOR_NOMINAL_RANGE_16_235; }   // NV12 recording: BT.709 limited
        vctx_->VideoProcessorSetStreamColorSpace(vp,0,&ci); vctx_->VideoProcessorSetOutputColorSpace(vp,&co);
        vctx_->VideoProcessorSetStreamAutoProcessingMode(vp,0,FALSE);
        D3D11_VIDEO_PROCESSOR_STREAM st{}; st.Enable=TRUE; st.pInputSurface=in;
        vctx_->VideoProcessorBlt(vp,out,0,1,&st);
    }

    void on_sample(IMFSample* s){
        IMFMediaBuffer* mb=nullptr; if(FAILED(s->GetBufferByIndex(0,&mb))) return;
        IMFDXGIBuffer* db=nullptr; HRESULT hr=mb->QueryInterface(IID_PPV_ARGS(&db)); mb->Release();
        if(FAILED(hr)){ if(frames==0){ err="decoder output is not a D3D11 surface (no DXVA)"; L(err); state=3; run_=false; } return; }
        ID3D11Texture2D* tx=nullptr; UINT sub=0;
        if(FAILED(db->GetResource(IID_PPV_ARGS(&tx))) || FAILED(db->GetSubresourceIndex(&sub))){ db->Release(); mf_rel(tx); return; }
        db->Release();
        const int slot=acquire_slot? acquire_slot() : -1;
        if(slot<0 || slot>=(int)ring_.size()){ tx->Release(); return; }
        D3D11_BOX box{0,0,0,(UINT)W,(UINT)H,1};
        ctx_->CopySubresourceRegion(ring_[slot].tex,0,0,0,0,tx,sub,&box);
        tx->Release();
        const double tn=now_ms();
        if(publish) publish(slot, tn);
        frames++;
        if(rec_want.load() && !rec_fail_){                              // 09-30: recording ring (pre-roll + clip)
            if(rec_.empty() && !rec_alloc()) rec_fail_=true;
            if(!rec_fail_){
                int idx=-1; { std::lock_guard<std::mutex> lk(rec_mtx_); const int n=(int)rec_.size();
                    for(int k=0;k<n;k++){ const int j=(rec_next_+k)%n; if(rec_[j].pins==0){ idx=j; break; } }
                    if(idx>=0){ rec_next_=(idx+1)%n; rec_[idx].t=0; } }
                if(idx<0) rec_dropped++;
                else { RECT full{0,0,W,H}; blt(vp_dec_, ring_[slot].iv, full, rec_[idx].ov, rec_w, rec_h, true);
                    { std::lock_guard<std::mutex> lk(rec_mtx_); rec_[idx].t=tn; }
                    if(on_rec_frame) on_rec_frame(idx, tn); } } }
        // NPU image — skip when the detector is busy (Studio Effects: "Skipping frame, PerceptionCore is busy").
        // Pipelined readback (09-30 v2): the image copied at frame N is read at frame N+1 with DO_NOT_WAIT, so this
        // thread never waits for the GPU (v1 blocking Map: 7-34 ms per image, capped the NPU at ~25/s).
        // v3: an image is copied EVERY frame (one in flight) and handed over only if the detector is idle — v2 asked
        // only when idle, so every other frame was lost (18/s on a 30 fps source).
        if(want_det && publish_det && det_enabled && det_enabled()){
            if(det_pending_>=0){
                const double t0=now_ms(); D3D11_MAPPED_SUBRESOURCE m{};
                HRESULT mh=ctx_->Map(det_stage_[det_pending_],0,D3D11_MAP_READ,D3D11_MAP_FLAG_DO_NOT_WAIT,&m);
                if(SUCCEEDED(mh)){ if(want_det()) publish_det((const uint8_t*)m.pData,det_w,det_h,(int)m.RowPitch,bgra,
                                                               det_ptile_[det_pending_],det_pu_[det_pending_][0],det_pu_[det_pending_][1]);
                    ctx_->Unmap(det_stage_[det_pending_],0); det_pending_=-1; det_gpu_ms=(float)(now_ms()-t0); } }
            if(det_pending_<0){
                RECT src{0,0,W,H}; int tile=-1; float u0=0.0f, u1=1.0f;
                if(tiles && W>H){ tile=tile_next_; tile_next_^=1; const int x0= tile==0? 0 : W-H;   // square tile, left/right
                    src=RECT{x0,0,x0+H,H}; u0=(float)x0/W; u1=(float)(x0+H)/W; }
                blt(vp_dec_, ring_[slot].iv, src, det_ov_, det_w, det_h);
                ctx_->CopyResource(det_stage_[det_next_], det_rt_);
                det_ptile_[det_next_]=tile; det_pu_[det_next_][0]=u0; det_pu_[det_next_][1]=u1;
                det_pending_=det_next_; det_next_^=1; } }
        float r[4];
        if(want_sr && publish_sr && want_sr(r)){
            RECT src{ (LONG)(r[0]*W), (LONG)(r[1]*H), (LONG)(r[2]*W), (LONG)(r[3]*H) };
            if(src.right>src.left+1 && src.bottom>src.top+1){
                blt(vp_dec_, ring_[slot].iv, src, sr_ov_, sr_w, sr_h);
                ctx_->CopyResource(sr_stage_, sr_rt_);
                D3D11_MAPPED_SUBRESOURCE m{};
                if(SUCCEEDED(ctx_->Map(sr_stage_,0,D3D11_MAP_READ,0,&m))){ publish_sr((const uint8_t*)m.pData,sr_w,sr_h,(int)m.RowPitch,bgra,r); ctx_->Unmap(sr_stage_,0); } } }
    }

    void teardown_mf(){
        mf_rel(reader_);
        if(source_){ source_->Shutdown(); mf_rel(source_); }
        mf_rel(mgr_);
    }
    void free_gpu(){
        for(auto& r: rec_){ mf_rel(r.ov); mf_rel(r.tex); } rec_.clear();
        for(auto& s: ring_){ mf_rel(s.iv); mf_rel(s.tex); } ring_.clear();
        mf_rel(disp_ov_); mf_rel(disp_srv_); mf_rel(disp_tex_); mf_rel(det_ov_); mf_rel(det_rt_); mf_rel(det_stage_[0]); mf_rel(det_stage_[1]);
        mf_rel(sr_ov_); mf_rel(sr_rt_); mf_rel(sr_stage_); mf_rel(photo_stage_);
        mf_rel(vp_render_); mf_rel(vp_dec_); mf_rel(vpe_); mf_rel(vctx_); mf_rel(vdev_); mf_rel(ctx_); mf_rel(dev_);
        if(exited_){ CloseHandle(exited_); exited_=nullptr; }
    }
};
