// rec_recorder.hpp — 2026-09-30. Records a clip when someone passes, all on the GPU like the rest of the CCTV:
//   the decoder keeps the last seconds as 1080p NV12 textures (MfDecoder recording ring = pre-roll);
//   on a trigger this thread opens an MP4 Sink Writer on our D3D11 device (hardware H.264 on the video engine),
//   writes the pre-roll, then every new frame, until nobody has been seen for post_ms.
//   Each written frame is an IMFTrackedSample: when Media Foundation lets go of it the ring slot is unpinned, so the
//   decoder never overwrites a texture the encoder still reads.
#pragma once
#include "mf_decoder.hpp"
#include <deque>
#include <condition_variable>
#include <filesystem>
#include <ctime>

// 09-30 v3: encoder parameters (values from codecapi.h, local copies so no initguid/strmiids is needed)
static const GUID REC_AVEncCommonRateControlMode = {0x1c0608e9,0x370c,0x4710,{0x8a,0x58,0xcb,0x61,0x81,0xc4,0x24,0x23}};
static const GUID REC_AVEncCommonMeanBitRate     = {0xf7222374,0x2144,0x4815,{0xb5,0x50,0xa3,0x7f,0x8e,0x12,0xee,0x52}};
static const GUID REC_AVEncMPVGOPSize            = {0x95f31b26,0x95a4,0x41aa,{0x93,0x03,0x24,0x6a,0x7f,0xc6,0xee,0xf1}};

class RecReleaseCb : public IMFAsyncCallback {                  // tracked sample released -> unpin its ring slot
    long ref_=1; std::shared_ptr<MfDecoder> d_; int idx_;
public:
    RecReleaseCb(std::shared_ptr<MfDecoder> d, int idx): d_(std::move(d)), idx_(idx) {}
    virtual ~RecReleaseCb() = default;
    STDMETHODIMP QueryInterface(REFIID r, void** o) override {
        if(r==__uuidof(IUnknown) || r==__uuidof(IMFAsyncCallback)){ *o=static_cast<IMFAsyncCallback*>(this); AddRef(); return S_OK; }
        *o=nullptr; return E_NOINTERFACE; }
    STDMETHODIMP_(ULONG) AddRef() override { return (ULONG)InterlockedIncrement(&ref_); }
    STDMETHODIMP_(ULONG) Release() override { long r=InterlockedDecrement(&ref_); if(!r) delete this; return (ULONG)r; }
    STDMETHODIMP GetParameters(DWORD*, DWORD*) override { return E_NOTIMPL; }
    STDMETHODIMP Invoke(IMFAsyncResult*) override { if(d_) d_->rec_unpin(idx_); return S_OK; }
};

class CctvRecorder {
public:
    std::function<void(const std::string&)> log;                  // diagnostics
    std::function<void(const std::string&)> notice;               // one line for the user ("Saved clip …")
    std::atomic<bool> armed{true};                                 // the ● button
    std::atomic<bool> recording{false};
    std::atomic<double> rec_t0{0};                                 // when the current clip started (display)
    double pre_ms=2500, post_ms=5000; UINT32 bitrate=8000000; int fps=30;
    std::string dir;                                               // where clips go (set by the owner)
    std::string last_file;

    void set_decoder(std::shared_ptr<MfDecoder> d){ std::lock_guard<std::mutex> lk(m_); dec_=std::move(d); cv_.notify_all(); }
    void person(double t){ last_person_=t; }                       // any confident person in the watched area
    void trigger(double t){ trig_=t; cv_.notify_all(); }           // debounced "someone is here" -> start a clip
    // decoder thread: a ring slot was filled; queue it while a clip is being written
    void frame(int idx, double t){
        if(!recording.load()) return;
        std::shared_ptr<MfDecoder> d; { std::lock_guard<std::mutex> lk(m_); d=wdec_; }
        if(!d) return;
        d->rec_pin(idx);
        { std::lock_guard<std::mutex> lk(m_); q_.push_back({idx,t}); }
        cv_.notify_all(); }
    void start(){ run_=true; th_=std::thread([this]{ loop(); }); }
    void stop(){ run_=false; cv_.notify_all(); if(th_.joinable()) th_.join(); }
    ~CctvRecorder(){ stop(); }

private:
    std::mutex m_; std::condition_variable cv_; std::thread th_; std::atomic<bool> run_{false};
    std::shared_ptr<MfDecoder> dec_, wdec_;                         // current decoder / the one the clip is written from
    std::deque<std::pair<int,double>> q_;
    std::atomic<double> trig_{0}, last_person_{0};
    IMFSinkWriter* w_=nullptr; IMFDXGIDeviceManager* mgr_=nullptr; DWORD stream_=0; double t_first_=0, t_last_=-1; long long nwritten_=0;
    long long k_last_=-1;                                          // 09-30 v2: frame index on the 1/fps grid
    std::string path_;

    static double now_ms(){ return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
    void L(const std::string& s){ if(log) log("[rec] "+s); }
    static std::wstring widen(const std::string& s){ int n=MultiByteToWideChar(CP_UTF8,0,s.c_str(),(int)s.size(),nullptr,0);
        std::wstring w(n,L'\0'); MultiByteToWideChar(CP_UTF8,0,s.c_str(),(int)s.size(),&w[0],n); return w; }

    void loop(){
        CoInitializeEx(nullptr, COINIT_MULTITHREADED);
        while(run_){
            std::shared_ptr<MfDecoder> d;
            { std::unique_lock<std::mutex> lk(m_); cv_.wait_for(lk, std::chrono::milliseconds(100)); d=dec_; }
            const double t=now_ms();
            if(!recording.load()){
                const double tr=trig_.exchange(0);
                if(armed.load() && d && tr>0 && t-tr<2000 && d->rec_want.load()) begin(d, tr-pre_ms);
                continue; }
            write_queued();
            std::shared_ptr<MfDecoder> cur; { std::lock_guard<std::mutex> lk(m_); cur=dec_; }
            if(t-last_person_.load()>post_ms || !armed.load() || cur!=wdec_ || !run_) finish();
        }
        if(recording.load()) finish();
        CoUninitialize();
    }

    void begin(std::shared_ptr<MfDecoder> d, double t0){
        std::error_code ec; std::filesystem::create_directories(std::filesystem::u8path(dir), ec);
        std::time_t tt=std::time(nullptr); std::tm lt{}; localtime_s(&lt,&tt); char nm[64];
        std::strftime(nm,sizeof nm,"PulseX_CCTV_%Y-%m-%d_%H-%M-%S.mp4",&lt);
        path_=(std::filesystem::u8path(dir)/nm).u8string();
        HRESULT hr; UINT tok=0;
        if(FAILED(hr=MFCreateDXGIDeviceManager(&tok,&mgr_)) || FAILED(hr=mgr_->ResetDevice(d->device(),tok))){ fail("device manager",hr); return; }
        for(int attempt=0; attempt<2; attempt++){                  // 09-30 v2: High profile first, encoder default second
        IMFAttributes* a=nullptr; MFCreateAttributes(&a,4);
        a->SetUnknown(MF_SINK_WRITER_D3D_MANAGER, mgr_);
        a->SetUINT32(MF_READWRITE_ENABLE_HARDWARE_TRANSFORMS, TRUE);
        a->SetGUID(MF_TRANSCODE_CONTAINERTYPE, MFTranscodeContainerType_MPEG4);
        hr=MFCreateSinkWriterFromURL(widen(path_).c_str(), nullptr, a, &w_); a->Release();
        if(FAILED(hr)){ fail("MFCreateSinkWriterFromURL",hr); return; }
        IMFMediaType *out=nullptr, *in=nullptr; MFCreateMediaType(&out); MFCreateMediaType(&in);
        out->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video); out->SetGUID(MF_MT_SUBTYPE, MFVideoFormat_H264);
        if(attempt==0) out->SetUINT32(MF_MT_MPEG2_PROFILE, 100 /*eAVEncH264VProfile_High*/);
        out->SetUINT32(MF_MT_AVG_BITRATE, bitrate); out->SetUINT32(MF_MT_INTERLACE_MODE, MFVideoInterlace_Progressive);
        MFSetAttributeSize(out, MF_MT_FRAME_SIZE, d->rec_w, d->rec_h); MFSetAttributeRatio(out, MF_MT_FRAME_RATE, fps, 1);
        MFSetAttributeRatio(out, MF_MT_PIXEL_ASPECT_RATIO, 1, 1);
        in->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video); in->SetGUID(MF_MT_SUBTYPE, MFVideoFormat_NV12);
        in->SetUINT32(MF_MT_INTERLACE_MODE, MFVideoInterlace_Progressive);
        MFSetAttributeSize(in, MF_MT_FRAME_SIZE, d->rec_w, d->rec_h); MFSetAttributeRatio(in, MF_MT_FRAME_RATE, fps, 1);
        MFSetAttributeRatio(in, MF_MT_PIXEL_ASPECT_RATIO, 1, 1);
        hr=w_->AddStream(out,&stream_);
        // v3: the default (CBR) dropped 3 frames at every keyframe (~1.1 s: 47 of 544) -> unconstrained VBR, GOP 2 s
        IMFAttributes* enc=nullptr; MFCreateAttributes(&enc,3);
        enc->SetUINT32(REC_AVEncCommonRateControlMode, 2 /*eAVEncCommonRateControlMode_UnconstrainedVBR*/);
        enc->SetUINT32(REC_AVEncCommonMeanBitRate, bitrate); enc->SetUINT32(REC_AVEncMPVGOPSize, (UINT32)(2*fps));
        if(SUCCEEDED(hr)) hr=w_->SetInputMediaType(stream_, in, enc);
        enc->Release();
        if(SUCCEEDED(hr)) hr=w_->BeginWriting();
        out->Release(); in->Release();
        if(SUCCEEDED(hr)){ L(attempt==0? "encoder: H.264 High profile" : "encoder: H.264 (encoder default profile)"); break; }
        mf_rel(w_);
        if(attempt==1){ fail("sink writer setup (H.264 encoder)",hr); return; }
        }
        { std::lock_guard<std::mutex> lk(m_); wdec_=d; q_.clear(); }
        t_first_=0; t_last_=-1; nwritten_=0; k_last_=-1; rec_t0=now_ms();
        recording=true;                                            // from here frame() queues new frames
        auto pre=d->rec_since(t0);                                 // the pre-roll, already pinned
        { std::lock_guard<std::mutex> lk(m_); for(auto it=pre.rbegin(); it!=pre.rend(); ++it) q_.push_front(*it); }
        L("start "+path_+" ("+std::to_string(pre.size())+" pre-roll frames)");
        write_queued();
    }

    void write_queued(){
        std::deque<std::pair<int,double>> q; std::shared_ptr<MfDecoder> d;
        { std::lock_guard<std::mutex> lk(m_); q.swap(q_); d=wdec_; }
        for(auto& f: q){
            if(!w_ || !d || f.second<=t_last_){ if(d) d->rec_unpin(f.first); continue; }   // duplicate or no writer
            if(t_first_==0) t_first_=f.second;
            ID3D11Texture2D* tx=d->rec_tex(f.first);
            IMFMediaBuffer* buf=nullptr; IMF2DBuffer* b2=nullptr; IMFTrackedSample* ts=nullptr; IMFSample* s=nullptr;
            HRESULT hr=MFCreateDXGISurfaceBuffer(__uuidof(ID3D11Texture2D), tx, 0, FALSE, &buf);
            if(SUCCEEDED(hr) && SUCCEEDED(buf->QueryInterface(IID_PPV_ARGS(&b2)))){ DWORD len=0; b2->GetContiguousLength(&len); buf->SetCurrentLength(len); b2->Release(); }
            if(SUCCEEDED(hr)) hr=MFCreateTrackedSample(&ts);
            if(SUCCEEDED(hr)) hr=ts->QueryInterface(IID_PPV_ARGS(&s));
            if(SUCCEEDED(hr)){
                s->AddBuffer(buf);
                long long k=llround((f.second-t_first_)*fps/1000.0); if(k<=k_last_) k=k_last_+1; k_last_=k;   // 1/fps grid
                s->SetSampleTime(k*10000000LL/fps); s->SetSampleDuration(10000000LL/fps);
                RecReleaseCb* cb=new RecReleaseCb(d, f.first); ts->SetAllocator(cb, nullptr); cb->Release();   // unpins on release
                hr=w_->WriteSample(stream_, s);
                if(SUCCEEDED(hr)){ nwritten_++; t_last_=f.second; }
            } else d->rec_unpin(f.first);
            if(s) s->Release(); if(ts) ts->Release(); if(buf) buf->Release();
            if(FAILED(hr)){ fail("WriteSample",hr); break; }
        }
    }

    void finish(){
        write_queued();
        const double secs = t_last_>t_first_? (t_last_-t_first_)/1000.0 : 0.0;
        HRESULT hr = w_? w_->Finalize() : E_FAIL;
        mf_rel(w_); mf_rel(mgr_);
        { std::lock_guard<std::mutex> lk(m_); for(auto& f: q_) if(wdec_) wdec_->rec_unpin(f.first); q_.clear(); wdec_.reset(); }
        recording=false;
        if(SUCCEEDED(hr) && nwritten_>0){
            last_file=path_; char b[64]; std::snprintf(b,sizeof b," (%.0f s, %lld frames)", secs, nwritten_);
            L("saved "+path_+b); if(notice) notice("Saved clip "+path_+b); }
        else { char b[64]; std::snprintf(b,sizeof b,"clip not finalized, hr=0x%08lx", (unsigned long)hr); L(b); }
    }

    void fail(const char* what, HRESULT hr){
        char b[160]; std::snprintf(b,sizeof b,"%s failed, hr=0x%08lx - recording off", what, (unsigned long)hr); L(b);
        if(notice) notice(std::string("Recording failed: ")+what);
        mf_rel(w_); mf_rel(mgr_); armed=false;
        { std::lock_guard<std::mutex> lk(m_); for(auto& f: q_) if(wdec_) wdec_->rec_unpin(f.first); q_.clear(); wdec_.reset(); }
        recording=false; }
};
