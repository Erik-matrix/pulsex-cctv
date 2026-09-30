// pcore_yolo_npu.hpp — reusable YOLOv10 detector on the HEXAGON NPU (raw QNN, session in pcore_qnn_min.hpp).
// Loads the precompiled QNN context (yolov10_det_qairt_context.bin, 2.46) once; detect() runs graphExecute per frame.
// Graph "yolov10_det": in id=1 "image" f32 [1,640,640,3] NHWC; out 784 boxes f32[1,8400,4], 785 scores f32[1,8400],
// 788 classes u8[1,8400]. Shared by pulsecore_detect_npu (single-shot + --watch) and pulsecore_video (mp4 timeline).
#pragma once
#include "pcore_qnn_min.hpp"                  // 09-30: QNN session without pulsecore.dll
#include <QnnInterface.h>
#include <QnnContext.h>
#include <QnnGraph.h>
#include <QnnTypes.h>
#include <QnnMem.h>
#include <System/QnnSystemInterface.h>
#include <System/QnnSystemContext.h>
#include <windows.h>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>

namespace pcore_npu {

static const int YS = 640;
static const char* CTX_BIN_DEFAULT =
    "models\\yolov10_det_qairt_context.bin";
static const char* COCO80[80] = {
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light",
    "fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow",
    "elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
    "skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle",
    "wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange",
    "broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed",
    "dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven",
    "toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"};
inline const char* sv_name(int c){ return COCO80[c]; }   // English COCO label (kept name for call sites)
inline bool is_person(int c){return c==0;} inline bool is_animal(int c){return c>=14&&c<=23;}
inline bool is_vehicle(int c){return c==1||c==2||c==3||c==5||c==7;} inline bool is_bag(int c){return c==24||c==26||c==28;}
#if __has_include("pcore_yolo_zones_local.hpp")
#include "pcore_yolo_zones_local.hpp"          // a local zone grid (machine-specific, not published)
#else
inline bool read_zones(unsigned char[1024]){ return false; }   // no zone grid: everything below EDGE_Y counts
#endif
inline void drawbox(unsigned char* im,int W,int H,int x0,int y0,int x1,int y1,unsigned char r,unsigned char g,unsigned char b){
    auto px=[&](int x,int y){ if(x<0||y<0||x>=W||y>=H)return; size_t o=((size_t)y*W+x)*3; im[o]=r;im[o+1]=g;im[o+2]=b; };
    if(x0>x1)std::swap(x0,x1); if(y0>y1)std::swap(y0,y1);
    for(int t=0;t<3;t++){for(int x=x0;x<=x1;x++){px(x,y0+t);px(x,y1-t);}for(int y=y0;y<=y1;y++){px(x0+t,y);px(x1-t,y);}} }

struct Det { int cls; float score; float x0,y0,x1,y1; bool in_zone; };

inline Qnn_Tensor_t Tt(uint32_t id,const void*d,size_t bytes,Qnn_TensorType_t ty,Qnn_DataType_t dt){
    Qnn_Tensor_t x={}; x.version=QNN_TENSOR_VERSION_2; x.v2.id=id; x.v2.type=ty; x.v2.dataType=dt;
    x.v2.clientBuf.data=(void*)d; x.v2.clientBuf.dataSize=(uint32_t)bytes; return x; }
// one graph I/O tensor as the context binary describes it (09-30)
struct IoT { uint32_t id=0; Qnn_DataType_t dt=QNN_DATATYPE_FLOAT_32; float scale=0; int32_t offset=0; bool ok=false;
    std::vector<uint32_t> dims;                         // 09-30: for QnnMem_register (zero-copy)
    size_t esz() const { return dt==QNN_DATATYPE_FLOAT_32? 4 : (dt==QNN_DATATYPE_UFIXED_POINT_16||dt==QNN_DATATYPE_FLOAT_16)? 2 : 1; }
    bool quant() const { return dt==QNN_DATATYPE_UFIXED_POINT_8 || dt==QNN_DATATYPE_UFIXED_POINT_16; } };
inline IoT iot_of(const Qnn_Tensor_t& t){ IoT r; r.ok=true; r.id=t.v2.id; r.dt=t.v2.dataType;
    if(t.v2.dimensions) r.dims.assign(t.v2.dimensions, t.v2.dimensions+t.v2.rank);
    const Qnn_QuantizeParams_t& q=t.v2.quantizeParams;
    if(q.encodingDefinition==QNN_DEFINITION_DEFINED && q.quantizationEncoding==QNN_QUANTIZATION_ENCODING_SCALE_OFFSET){
        r.scale=q.scaleOffsetEncoding.scale; r.offset=q.scaleOffsetEncoding.offset; }
    return r; }
inline Qnn_ContextHandle_t g_npu_ctx=nullptr;
inline void npu_ctxcb(Qnn_ContextHandle_t c,Qnn_GraphHandle_t,const char*,QnnContext_createFromBinaryAsyncNotifyType_t ty,void*,Qnn_ErrorHandle_t){
    if(ty==QNN_CONTEXT_NOTIFY_TYPE_CONTEXT_INIT) g_npu_ctx=c; }

class NpuDetector {
public:
    bool load(const char* ctx_bin = CTX_BIN_DEFAULT, int log=0) {
        (void)log;                                          // QNN log: env PULSECORE_QNN_LOG (pcore_qnn_min.hpp)
        if (!qs_.open(L"QnnHtp.dll")) return false;
        iface_ = qs_.iface();
        std::ifstream f(ctx_bin, std::ios::binary); blob_.assign((std::istreambuf_iterator<char>(f)),{});
        if (blob_.empty()) return false;
        QnnContext_Params_t P{}; P.version=QNN_CONTEXT_PARAMS_VERSION_1; P.v1.binaryBuffer=blob_.data();
        *(Qnn_ContextBinarySize_t*)&P.v1.binaryBufferSize=blob_.size(); P.v1.notifyFunc=npu_ctxcb; P.v1.notifyParam=nullptr;
        const QnnContext_Params_t* arr[2]={&P,nullptr};
        #define QFN_ (iface_->QNN_INTERFACE_VER_NAME)
        if (QFN_.contextCreateFromBinaryListAsync(qs_.backend(),qs_.device(),arr,nullptr,nullptr)!=QNN_SUCCESS || !g_npu_ctx) return false;
        ctx_=g_npu_ctx;
        read_io();                                          // graph name + I/O from the binary (fallback: the float layout)
        if (QFN_.graphRetrieve(ctx_,gname_.c_str(),&graph_)!=QNN_SUCCESS || !graph_) return false;
        in_.assign((size_t)YS*YS*3,0.f); boxes_.assign((size_t)8400*4,0.f); scores_.assign(8400,0.f); classes_.assign(8400,0);
        inq_.assign((size_t)YS*YS*3*ti_.esz(),0); rb_.assign((size_t)8400*4*tb_.esz(),0); rs_.assign((size_t)8400*ts_.esz(),0);
        rc_.assign((size_t)8400*tc_.esz(),0);
        setup_zc();                                         // 09-30: shared, registered I/O buffers (Studio Effects style)
        for (int v=0;v<256;v++){ double q = ti_.scale>0 ? std::floor(v/255.0/ti_.scale - ti_.offset + 0.5) : v;
            q8_[v]=(uint8_t)(q<0?0:q>255?255:q); }
        return true;
    }
    // "float" / "u8 in" etc. — what the loaded context wants, for receipts
    std::string io_desc() const { auto d=[](const IoT& t){ return t.dt==QNN_DATATYPE_FLOAT_32? "f32" : t.dt==QNN_DATATYPE_UFIXED_POINT_8? "u8"
        : t.dt==QNN_DATATYPE_UINT_8? "uint8" : t.dt==QNN_DATATYPE_UFIXED_POINT_16? "u16" : t.dt==QNN_DATATYPE_FLOAT_16? "f16" : "?"; };
        return gname_ + (nchw_? " NCHW" : " NHWC") + (zc_on_? " zero-copy" : " copy") + " in " + d(ti_) + " out " + d(tb_) + "/" + d(ts_) + "/" + d(tc_); }
    // 09-30: free what load() made (zero-copy buffers, context) and close the QNN session; also on destruction
    void close(){ if (iface_){ free_zc(); if (ctx_) QFN_.contextFree(ctx_,nullptr); } ctx_=nullptr; graph_=nullptr; iface_=nullptr; qs_.close(); }
    ~NpuDetector(){ close(); }
    double last_ms() const { return last_ms_; }
    unsigned long long last_err() const { return last_err_; }
    const std::vector<uint8_t>& input_q() const { return inq_; }        // the uint8 input as last prepared (tests)            // graphExecute's result (0 = OK)
    float max_score() const { float m=0; for(float v: scores_) m=std::max(m,v); return m; }
    // 09-30: the loaded backend/device, so another graph (the CCTV's QuickSRNet) can share THIS QNN stack instead of
    // opening a second one in the same process. Valid after a successful load().
    const QnnInterface_t* iface() const { return iface_; }
    Qnn_BackendHandle_t backend() const { return qs_.backend(); }
    Qnn_DeviceHandle_t  device()  const { return qs_.device(); }
    const std::string& session_error() const { return qs_.error(); }
    // detect on interleaved RGB (W*H*3); if draw, paints boxes; fills tag/alert; returns NMS'd zone-tagged dets.
    std::vector<Det> detect(unsigned char* rgb, int W, int H, float thr, std::string& tag, bool& alert, bool draw) {
        // preprocess → NHWC [1,640,640,3] [0,1] (the graph's real layout)
        for (int y=0;y<YS;y++){ double sy=(y+0.5)*H/YS-0.5; int y0=(int)std::floor(sy); double fy=sy-y0; int y1=y0+1; if(y0<0)y0=0; if(y1>H-1)y1=H-1; if(y0>H-1)y0=H-1;
            for (int x=0;x<YS;x++){ double sx=(x+0.5)*W/YS-0.5; int x0=(int)std::floor(sx); double fx=sx-x0; int x1=x0+1; if(x0<0)x0=0; if(x1>W-1)x1=W-1; if(x0>W-1)x0=W-1;
                for (int c=0;c<3;c++){ double p00=rgb[((size_t)y0*W+x0)*3+c],p01=rgb[((size_t)y0*W+x1)*3+c],p10=rgb[((size_t)y1*W+x0)*3+c],p11=rgb[((size_t)y1*W+x1)*3+c];
                    double top=p00+(p01-p00)*fx, bot=p10+(p11-p10)*fx; in_[((size_t)y*YS+x)*3+c]=(float)(((top+(bot-top)*fy))/255.0); } } }
        if (ti_.quant()) { const double qmax = ti_.esz()==2? 65535.0 : 255.0;   // u16in (09-30): 8- or 16-bit input
            for (size_t i=0;i<in_.size();i++){ double q=std::floor(in_[i]/ti_.scale - ti_.offset + 0.5); q=q<0?0:q>qmax?qmax:q;
                if (ti_.esz()==2) ((uint16_t*)inq_.data())[i]=(uint16_t)q; else inq_[i]=(uint8_t)q; } }
        return infer(rgb, W, H, thr, tag, alert, draw);
    }
    // 09-30 (CCTV, every frame): bilinear from interleaved RGBA (W*H*4) straight into the NHWC float input. The x/y
    // taps and weights are tables rebuilt only when the frame size changes; 1/255 is folded into a 256-entry LUT.
    // Same sampling positions as detect()'s resample. Returns the time it took (ms).
    double prep_rgba(const unsigned char* rgba, int W, int H) {
        LARGE_INTEGER fq,a,b; QueryPerformanceFrequency(&fq); QueryPerformanceCounter(&a);
        if (W!=tw_ || H!=th_) { tw_=W; th_=H; tx0_.resize(YS); tx1_.resize(YS); tfx_.resize(YS); ty0_.resize(YS); ty1_.resize(YS); tfy_.resize(YS);
            auto tap=[](int i,int N,int& i0,int& i1,float& f){ double s=(i+0.5)*N/YS-0.5; int k=(int)std::floor(s); f=(float)(s-k);
                i0=k<0?0:(k>N-1?N-1:k); i1=k+1>N-1?N-1:(k+1<0?0:k+1); };
            for (int i=0;i<YS;i++){ int x0,x1,y0,y1; float fx,fy; tap(i,W,x0,x1,fx); tap(i,H,y0,y1,fy);
                tx0_[i]=x0*4; tx1_[i]=x1*4; tfx_[i]=fx; ty0_[i]=y0; ty1_[i]=y1; tfy_[i]=fy; }
            for (int v=0;v<256;v++) lut_[v]=v/255.0f; }
        if (ti_.quant() && ti_.esz()==2) {                   // u16in: uint16 input, q straight from the float bilinear
            const float k=1.f/(255.f*ti_.scale), off=(float)ti_.offset; uint16_t* q16=(uint16_t*)inq_.data();
            for (int y=0;y<YS;y++){ const unsigned char* r0=rgba+(size_t)ty0_[y]*W*4; const unsigned char* r1=rgba+(size_t)ty1_[y]*W*4;
                const float fy=tfy_[y]; uint16_t* o=q16+(size_t)y*YS*3;
                for (int x=0;x<YS;x++){ const unsigned char *a0=r0+tx0_[x], *a1=r0+tx1_[x], *b0=r1+tx0_[x], *b1=r1+tx1_[x]; const float fx=tfx_[x];
                    for (int c=0;c<3;c++){ const float t=a0[c]+(a1[c]-a0[c])*fx, u=b0[c]+(b1[c]-b0[c])*fx;
                        float q=(t+(u-t)*fy)*k - off + 0.5f; o[x*3+c]=(uint16_t)(q<0.f?0.f:q>65535.f?65535.f:q); } } }
            QueryPerformanceCounter(&b); return (double)(b.QuadPart-a.QuadPart)*1000.0/fq.QuadPart;
        }
        if (ti_.quant()) {                                   // uint8 input: bilinear in pixel units, then the 256-LUT
            for (int y=0;y<YS;y++){ const unsigned char* r0=rgba+(size_t)ty0_[y]*W*4; const unsigned char* r1=rgba+(size_t)ty1_[y]*W*4;
                const float fy=tfy_[y]; uint8_t* o=&inq_[(size_t)y*YS*3];
                for (int x=0;x<YS;x++){ const unsigned char *a0=r0+tx0_[x], *a1=r0+tx1_[x], *b0=r1+tx0_[x], *b1=r1+tx1_[x]; const float fx=tfx_[x];
                    for (int c=0;c<3;c++){ const float t=a0[c]+(a1[c]-a0[c])*fx, u=b0[c]+(b1[c]-b0[c])*fx;
                        o[x*3+c]=q8_[(int)(t+(u-t)*fy+0.5f)]; } } }
            QueryPerformanceCounter(&b); return (double)(b.QuadPart-a.QuadPart)*1000.0/fq.QuadPart;
        }
        for (int y=0;y<YS;y++){ const unsigned char* r0=rgba+(size_t)ty0_[y]*W*4; const unsigned char* r1=rgba+(size_t)ty1_[y]*W*4;
            const float fy=tfy_[y]; float* o=&in_[(size_t)y*YS*3];
            for (int x=0;x<YS;x++){ const unsigned char *a0=r0+tx0_[x], *a1=r0+tx1_[x], *b0=r1+tx0_[x], *b1=r1+tx1_[x]; const float fx=tfx_[x];
                for (int c=0;c<3;c++){ const float t=lut_[a0[c]]+(lut_[a1[c]]-lut_[a0[c]])*fx, u=lut_[b0[c]]+(lut_[b1[c]]-lut_[b0[c]])*fx;
                    o[x*3+c]=t+(u-t)*fy; } } }
        QueryPerformanceCounter(&b); return (double)(b.QuadPart-a.QuadPart)*1000.0/fq.QuadPart;
    }
    // run the graph on whatever the input already holds (after prep_rgba); rgb/draw as in detect() (nullptr = no draw)
    std::vector<Det> infer(unsigned char* rgb, int W, int H, float thr, std::string& tag, bool& alert, bool draw) {
        const bool fin = !ti_.quant();
        void* src = fin? (void*)in_.data() : (void*)inq_.data(); const size_t nb = fin? in_.size()*4 : inq_.size();
        if (nchw_) {                                         // HWC -> CHW for a channels-first input (09-30)
            trn_.resize(nb); const size_t P=(size_t)YS*YS;
            if (fin){ const float* s=in_.data(); float* d=(float*)trn_.data(); for(size_t p=0;p<P;p++){ d[p]=s[p*3]; d[P+p]=s[p*3+1]; d[2*P+p]=s[p*3+2]; } }
            else if (ti_.esz()==2) { const uint16_t* s=(const uint16_t*)inq_.data(); uint16_t* d=(uint16_t*)trn_.data(); for(size_t p=0;p<P;p++){ d[p]=s[p*3]; d[P+p]=s[p*3+1]; d[2*P+p]=s[p*3+2]; } }
            else    { const uint8_t* s=inq_.data(); uint8_t* d=trn_.data();    for(size_t p=0;p<P;p++){ d[p]=s[p*3]; d[P+p]=s[p*3+1]; d[2*P+p]=s[p*3+2]; } }
            src = trn_.data(); }
        Qnn_Tensor_t tin = TtQ(ti_, src, nb, QNN_TENSOR_TYPE_APP_WRITE);
        Qnn_Tensor_t tout[3] = { TtQ(tb_, rb_.data(), rb_.size(), QNN_TENSOR_TYPE_APP_READ),
                                 TtQ(ts_, rs_.data(), rs_.size(), QNN_TENSOR_TYPE_APP_READ),
                                 TtQ(tc_, rc_.data(), rc_.size(), QNN_TENSOR_TYPE_APP_READ) };
        const uint8_t *ob=rb_.data(), *os=rs_.data(), *oc=rc_.data();
        if (zc_on_) {                                        // zero-copy: point every tensor at its registered buffer
            std::memcpy(zb_[0].p, src, nb);
            Qnn_Tensor_t* all[4]={&tin,&tout[0],&tout[1],&tout[2]}; const IoT* io[4]={&ti_,&tb_,&ts_,&tc_};
            for (int k=0;k<4;k++){ Qnn_Tensor_t& t=*all[k]; t.v2.memType=QNN_TENSORMEMTYPE_MEMHANDLE; t.v2.memHandle=zb_[k].h;
                t.v2.rank=(uint32_t)io[k]->dims.size(); t.v2.dimensions=const_cast<uint32_t*>(io[k]->dims.data()); }
            ob=(const uint8_t*)zb_[1].p; os=(const uint8_t*)zb_[2].p; oc=(const uint8_t*)zb_[3].p; }
        LARGE_INTEGER fq,a,b; QueryPerformanceFrequency(&fq); QueryPerformanceCounter(&a);
        auto er = QFN_.graphExecute(graph_,&tin,1,tout,3,nullptr,nullptr);
        QueryPerformanceCounter(&b); last_ms_=(double)(b.QuadPart-a.QuadPart)*1000.0/fq.QuadPart;
        std::vector<Det> out; tag.clear(); alert=false;
        last_err_=(unsigned long long)er;
        if (er!=QNN_SUCCESS) return out;
        deq(tb_, ob, boxes_.data(), boxes_.size()); deq(ts_, os, scores_.data(), scores_.size());
        if (tc_.dt==QNN_DATATYPE_UINT_8 || tc_.dt==QNN_DATATYPE_INT_8) std::memcpy(classes_.data(), oc, 8400);   // raw class ids
        else if (tc_.dt==QNN_DATATYPE_UFIXED_POINT_8 && !(tc_.scale>0 && (tc_.scale!=1.f || tc_.offset!=0))) std::memcpy(classes_.data(), oc, 8400);
        else { std::vector<float> cf(8400); deq(tc_, oc, cf.data(), 8400);
            for (int i=0;i<8400;i++){ float v=std::floor(cf[i]+0.5f); classes_[i]=(uint8_t)(v<0?255:v>255?255:v); } }
        std::vector<Det> d;
        for (int i=0;i<8400;i++){ if(scores_[i]<thr)continue; int c=classes_[i]; if(c<0||c>=80)continue; d.push_back({c,scores_[i],boxes_[i*4+0],boxes_[i*4+1],boxes_[i*4+2],boxes_[i*4+3],false}); }
        std::sort(d.begin(),d.end(),[](const Det&a,const Det&b){return a.score>b.score;});
        auto iou=[](const Det&a,const Det&b){ float xx0=std::max(a.x0,b.x0),yy0=std::max(a.y0,b.y0),xx1=std::min(a.x1,b.x1),yy1=std::min(a.y1,b.y1);
            float w=std::max(0.f,xx1-xx0),h=std::max(0.f,yy1-yy0),inter=w*h; float ua=(a.x1-a.x0)*(a.y1-a.y0)+(b.x1-b.x0)*(b.y1-b.y0)-inter; return ua>0?inter/ua:0.f; };
        unsigned char zones[1024]; bool hz=read_zones(zones); const float EDGE_Y=0.32f*YS;
        std::vector<char> dead(d.size(),0); std::string t;
        for (size_t i=0;i<d.size();++i){ if(dead[i])continue;
            for (size_t j=i+1;j<d.size();++j) if(!dead[j]&&d[j].cls==d[i].cls&&iou(d[i],d[j])>0.5f)dead[j]=1;
            Det b=d[i]; float cx=(b.x0+b.x1)*0.5f, cy=(b.y0+b.y1)*0.5f;
            int gx=(int)(cx/YS*32),gy=(int)(cy/YS*32); gx=gx<0?0:gx>31?31:gx; gy=gy<0?0:gy>31?31:gy;
            b.in_zone = (hz?zones[gy*32+gx]:(cy<EDGE_Y?0:2))>0;
            if(is_person(b.cls)&&b.in_zone) alert=true;
            if(!t.empty())t+=", "; t+=std::string(sv_name(b.cls))+(b.in_zone?" (zone)":" (outside)");
            if (draw && rgb){ int sx0=(int)(b.x0*W/YS),sy0=(int)(b.y0*H/YS),sx1=(int)(b.x1*W/YS),sy1=(int)(b.y1*H/YS);
                unsigned char cr=60,cg=220,cb=255; if(is_person(b.cls)){cr=255;cg=60;cb=60;}else if(is_animal(b.cls)||is_bag(b.cls)){cr=255;cg=160;cb=30;}else if(is_vehicle(b.cls)){cr=255;cg=220;cb=40;}
                if(!b.in_zone){cr=cg=cb=140;} drawbox(rgb,W,H,sx0,sy0,sx1,sy1,cr,cg,cb); }
            out.push_back(b);
        }
        tag=t; return out;
    }
private:
    static Qnn_Tensor_t TtQ(const IoT& t, void* d, size_t bytes, Qnn_TensorType_t ty){
        Qnn_Tensor_t x=Tt(t.id,d,bytes,ty,t.dt);
        if (t.scale>0){ x.v2.quantizeParams.encodingDefinition=QNN_DEFINITION_DEFINED;
            x.v2.quantizeParams.quantizationEncoding=QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
            x.v2.quantizeParams.scaleOffsetEncoding.scale=t.scale; x.v2.quantizeParams.scaleOffsetEncoding.offset=t.offset; }
        return x; }
    static void deq(const IoT& t, const uint8_t* src, float* dst, size_t n){
        if (t.dt==QNN_DATATYPE_FLOAT_32){ std::memcpy(dst,src,n*4); return; }
        if (t.dt==QNN_DATATYPE_UFIXED_POINT_16){ const uint16_t* s=(const uint16_t*)src; for(size_t i=0;i<n;i++) dst[i]=t.scale*((float)s[i]+t.offset); return; }
        const float sc = t.scale>0? t.scale : 1.f; for(size_t i=0;i<n;i++) dst[i]=sc*((float)src[i]+t.offset); }
    // graph 0's name and I/O from the context binary; the float release's layout if anything is missing
    void read_io(){
        gname_="yolov10_det"; ti_={1,QNN_DATATYPE_FLOAT_32,0,0,true,{1,640,640,3}}; tb_={784,QNN_DATATYPE_FLOAT_32,0,0,true,{1,8400,4}};
        ts_={785,QNN_DATATYPE_FLOAT_32,0,0,true,{1,8400}}; tc_={788,QNN_DATATYPE_UFIXED_POINT_8,0,0,true,{1,8400}};
        HMODULE hs=LoadLibraryA("QnnSystem.dll"); if(!hs) return;
        typedef Qnn_ErrorHandle_t(*SGP)(const QnnSystemInterface_t***,uint32_t*);
        SGP sgp=(SGP)GetProcAddress(hs,"QnnSystemInterface_getProviders"); const QnnSystemInterface_t** sp=nullptr; uint32_t nsp=0;
        if(!sgp||sgp(&sp,&nsp)!=QNN_SUCCESS||nsp==0) return;
        QNN_SYSTEM_INTERFACE_VER_TYPE SI=sp[0]->QNN_SYSTEM_INTERFACE_VER_NAME; QnnSystemContext_Handle_t sc=nullptr;
        if(SI.systemContextCreate(&sc)!=QNN_SUCCESS) return;
        const QnnSystemContext_BinaryInfo_t* bi=nullptr; Qnn_ContextBinarySize_t bsz=0;
        if(SI.systemContextGetBinaryInfo(sc,blob_.data(),(uint64_t)blob_.size(),&bi,&bsz)==QNN_SUCCESS && bi){
            uint32_t ng=0; QnnSystemContext_GraphInfo_t* gs=nullptr;
            if(bi->version==QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1){ ng=bi->contextBinaryInfoV1.numGraphs; gs=bi->contextBinaryInfoV1.graphs; }
            else if(bi->version==QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2){ ng=bi->contextBinaryInfoV2.numGraphs; gs=bi->contextBinaryInfoV2.graphs; }
            else { ng=bi->contextBinaryInfoV3.numGraphs; gs=bi->contextBinaryInfoV3.graphs; }
            if(ng>0 && gs){ const char* gn=nullptr; uint32_t ni=0,no=0; Qnn_Tensor_t *ins=nullptr,*outs=nullptr; auto& g=gs[0];
                if(g.version==QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1){ gn=g.graphInfoV1.graphName; ni=g.graphInfoV1.numGraphInputs; ins=g.graphInfoV1.graphInputs; no=g.graphInfoV1.numGraphOutputs; outs=g.graphInfoV1.graphOutputs; }
                else if(g.version==QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_2){ gn=g.graphInfoV2.graphName; ni=g.graphInfoV2.numGraphInputs; ins=g.graphInfoV2.graphInputs; no=g.graphInfoV2.numGraphOutputs; outs=g.graphInfoV2.graphOutputs; }
                else { gn=g.graphInfoV3.graphName; ni=g.graphInfoV3.numGraphInputs; ins=g.graphInfoV3.graphInputs; no=g.graphInfoV3.numGraphOutputs; outs=g.graphInfoV3.graphOutputs; }
                // by NAME (boxes/scores/classes) or, for AI Hub's output_0/1/2, by SHAPE+TYPE: [1,8400,4] = boxes,
                // raw integer [1,8400] = classes, the other [1,8400] = scores
                IoT b,s,c; for(uint32_t i=0;i<no;i++){ const Qnn_Tensor_t& t=outs[i]; std::string sn(t.v2.name? t.v2.name : "");
                    const uint32_t r=t.v2.rank; const bool raw=t.v2.dataType==QNN_DATATYPE_UINT_8||t.v2.dataType==QNN_DATATYPE_INT_8;
                    if(sn.find("box")!=std::string::npos || (r==3 && t.v2.dimensions && t.v2.dimensions[2]==4)) b=iot_of(t);
                    else if(sn.find("class")!=std::string::npos || (r==2 && raw)) c=iot_of(t);
                    else if(sn.find("score")!=std::string::npos || r==2) s=iot_of(t); }
                if(gn && ni==1 && b.ok && s.ok && c.ok){ gname_=gn; ti_=iot_of(ins[0]); tb_=b; ts_=s; tc_=c;
                    nchw_ = ins[0].v2.rank==4 && ins[0].v2.dimensions && ins[0].v2.dimensions[1]==3; } } }
        SI.systemContextFree(sc);
    }
    unsigned long long last_err_=0;
    bool nchw_=false; std::vector<uint8_t> trn_;
    // ── zero-copy (09-30): rpcmem buffers registered with QnnMem_register; [0]=input [1]=boxes [2]=scores [3]=classes
    struct ZBuf { void* p=nullptr; size_t n=0; int fd=-1; Qnn_MemHandle_t h=nullptr; };
    ZBuf zb_[4]; bool zc_on_=false;
    typedef void* (*rpcmem_alloc_t)(int,uint32_t,int); typedef int (*rpcmem_to_fd_t)(void*); typedef void (*rpcmem_free_t)(void*);
    rpcmem_free_t rfree_=nullptr;
    void setup_zc(){
        if (const char* e=std::getenv("PULSECORE_YOLO_ZEROCOPY")) { if (*e=='0') return; }
        HMODULE m=GetModuleHandleA("libcdsprpc.dll"); if(!m) m=LoadLibraryA("libcdsprpc.dll"); if(!m) return;
        auto ra=(rpcmem_alloc_t)GetProcAddress(m,"rpcmem_alloc"); auto rf=(rpcmem_to_fd_t)GetProcAddress(m,"rpcmem_to_fd");
        rfree_=(rpcmem_free_t)GetProcAddress(m,"rpcmem_free"); if(!ra||!rf||!rfree_) return;
        const IoT* io[4]={&ti_,&tb_,&ts_,&tc_};
        const size_t nel[4]={(size_t)YS*YS*3,(size_t)8400*4,8400,8400};
        for (int k=0;k<4;k++){
            ZBuf& z=zb_[k]; z.n=nel[k]*io[k]->esz(); z.p=ra(25 /*RPCMEM_HEAP_ID_SYSTEM*/, 1 /*RPCMEM_DEFAULT_FLAGS*/, (int)z.n);
            if(!z.p){ free_zc(); return; }
            z.fd=rf(z.p); if(z.fd<0 || io[k]->dims.empty()){ free_zc(); return; }
            Qnn_MemDescriptor_t md=QNN_MEM_DESCRIPTOR_INIT;
            md.memShape.numDim=(uint32_t)io[k]->dims.size(); md.memShape.dimSize=const_cast<uint32_t*>(io[k]->dims.data());
            md.dataType=io[k]->dt; md.memType=QNN_MEM_TYPE_ION; md.ionInfo.fd=z.fd;
            if (QFN_.memRegister(ctx_,&md,1,&z.h)!=QNN_SUCCESS || !z.h){ z.h=nullptr; free_zc(); return; }
        }
        zc_on_=true;
    }
    void free_zc(){ for (auto& z: zb_){ if(z.h) QFN_.memDeRegister(&z.h,1); if(z.p && rfree_) rfree_(z.p); z=ZBuf{}; } zc_on_=false; }
    std::string gname_; IoT ti_, tb_, ts_, tc_; std::vector<uint8_t> inq_, rb_, rs_, rc_; uint8_t q8_[256];
    QnnHtpSession qs_; const QnnInterface_t* iface_=nullptr;
    Qnn_ContextHandle_t ctx_=nullptr; Qnn_GraphHandle_t graph_=nullptr;
    std::vector<uint8_t> blob_; std::vector<float> in_, boxes_, scores_; std::vector<uint8_t> classes_; double last_ms_=0;
    int tw_=0, th_=0; std::vector<int> tx0_, tx1_, ty0_, ty1_; std::vector<float> tfx_, tfy_; float lut_[256];   // prep_rgba tables
};

} // namespace pcore_npu
