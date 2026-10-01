// sr_npu.hpp — QuickSRNet-Large w8a8 on the NPU for pulsecore_cctv (2026-09-30). Shares the YOLO detector's QNN
// backend/device (pcore_yolo_npu.hpp). Moved out of pulsecore_cctv.cpp so tools/test can exercise the SAME code.
#pragma once
#include "pcore_yolo_npu.hpp"
#include <windows.h>
#include <string>
#include <vector>
#include <fstream>
#include <cstdlib>

static double sr_now_ms(){ LARGE_INTEGER f,c; QueryPerformanceFrequency(&f); QueryPerformanceCounter(&c); return (double)c.QuadPart*1000.0/(double)f.QuadPart; }
// ── QuickSRNet-Large w8a8 on the NPU (09-30) - shares the YOLO detector's QNN backend/device ─────────────────────────
static const int SR_W=512, SR_H=288, SR_OW=2048, SR_OH=1152;
static std::string sr_ctx(){ if(const char* e=std::getenv("PULSECORE_SR_CTX")){ if(*e) return e; }
    { char b[MAX_PATH]{}; GetModuleFileNameA(nullptr,b,MAX_PATH); std::string p=b;
      p=p.substr(0,p.find_last_of("\\/")+1)+"models\\quicksrnetlarge_288x512_ctx_qnn.bin";
      if(GetFileAttributesA(p.c_str())!=INVALID_FILE_ATTRIBUTES) return p; }          // 09-30: models next to the exe first
    return "models\\quicksrnetlarge_288x512_ctx_qnn.bin"; }
// The context's graph name and tensor ids are read from the binary (10-01); these are the fallback - the values of the
// context built here by ORT-QNN 2.46 from the ONNX (read with qnn-context-binary-utility).
static const char* SR_GRAPH="QNNExecutionProvider_QNN_16202934166401028404_1_0";
struct SrNpu {
    const QnnInterface_t* iface=nullptr; Qnn_ContextHandle_t ctx=nullptr; Qnn_GraphHandle_t graph=nullptr;
    std::vector<uint8_t> blob; double last_ms=0;
    std::string gname=SR_GRAPH, info;                 // info: what was found (or why the context was refused), for the log
    uint32_t in_id=1, out_id=94;
    ~SrNpu(){ if(ctx && iface) iface->QNN_INTERFACE_VER_NAME.contextFree(ctx,nullptr); }   // 09-30: before the device goes
    bool load(const pcore_npu::NpuDetector& d, const std::string& path){
        iface=d.iface(); if(!iface) return false;
        std::ifstream f(path, std::ios::binary); blob.assign((std::istreambuf_iterator<char>(f)),{}); if(blob.empty()) return false;
        QnnContext_Params_t P{}; P.version=QNN_CONTEXT_PARAMS_VERSION_1; P.v1.binaryBuffer=blob.data();
        *(Qnn_ContextBinarySize_t*)&P.v1.binaryBufferSize=blob.size(); P.v1.notifyFunc=pcore_npu::npu_ctxcb; P.v1.notifyParam=nullptr;
        const QnnContext_Params_t* arr[2]={&P,nullptr};
        pcore_npu::g_npu_ctx=nullptr;
        if(iface->QNN_INTERFACE_VER_NAME.contextCreateFromBinaryListAsync(d.backend(),d.device(),arr,nullptr,nullptr)!=QNN_SUCCESS
           || !pcore_npu::g_npu_ctx) return false;
        ctx=pcore_npu::g_npu_ctx;
        if(!read_io()) return false;
        return iface->QNN_INTERFACE_VER_NAME.graphRetrieve(ctx,gname.c_str(),&graph)==QNN_SUCCESS && graph;
    }
    // graph 0's name and I/O from the binary. false = the context does not fit (info says why); a binary QnnSystem cannot
    // read keeps the fallback values above.
    bool read_io(){
        info = "graph and tensors: built-in values (QnnSystem could not read the binary)";
        HMODULE hs=LoadLibraryA("QnnSystem.dll"); if(!hs) return true;
        typedef Qnn_ErrorHandle_t(*SGP)(const QnnSystemInterface_t***,uint32_t*);
        SGP sgp=(SGP)GetProcAddress(hs,"QnnSystemInterface_getProviders"); const QnnSystemInterface_t** sp=nullptr; uint32_t nsp=0;
        if(!sgp||sgp(&sp,&nsp)!=QNN_SUCCESS||nsp==0) return true;
        QNN_SYSTEM_INTERFACE_VER_TYPE SI=sp[0]->QNN_SYSTEM_INTERFACE_VER_NAME; QnnSystemContext_Handle_t sc=nullptr;
        if(SI.systemContextCreate(&sc)!=QNN_SUCCESS) return true;
        bool fits=true;
        const QnnSystemContext_BinaryInfo_t* bi=nullptr; Qnn_ContextBinarySize_t bsz=0;
        if(SI.systemContextGetBinaryInfo(sc,blob.data(),(uint64_t)blob.size(),&bi,&bsz)==QNN_SUCCESS && bi){
            uint32_t ng=0; QnnSystemContext_GraphInfo_t* gs=nullptr;
            if(bi->version==QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1){ ng=bi->contextBinaryInfoV1.numGraphs; gs=bi->contextBinaryInfoV1.graphs; }
            else if(bi->version==QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2){ ng=bi->contextBinaryInfoV2.numGraphs; gs=bi->contextBinaryInfoV2.graphs; }
            else { ng=bi->contextBinaryInfoV3.numGraphs; gs=bi->contextBinaryInfoV3.graphs; }
            if(ng>0 && gs){ const char* gn=nullptr; uint32_t ni=0,no=0; Qnn_Tensor_t *ins=nullptr,*outs=nullptr; auto& g=gs[0];
                if(g.version==QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1){ gn=g.graphInfoV1.graphName; ni=g.graphInfoV1.numGraphInputs; ins=g.graphInfoV1.graphInputs; no=g.graphInfoV1.numGraphOutputs; outs=g.graphInfoV1.graphOutputs; }
                else if(g.version==QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_2){ gn=g.graphInfoV2.graphName; ni=g.graphInfoV2.numGraphInputs; ins=g.graphInfoV2.graphInputs; no=g.graphInfoV2.numGraphOutputs; outs=g.graphInfoV2.graphOutputs; }
                else { gn=g.graphInfoV3.graphName; ni=g.graphInfoV3.numGraphInputs; ins=g.graphInfoV3.graphInputs; no=g.graphInfoV3.numGraphOutputs; outs=g.graphInfoV3.graphOutputs; }
                // one 8-bit image in and out, 0..255 = 0..1, NCHW (1 x 3 x H x W)
                auto shape=[](const Qnn_Tensor_t& t, int H, int W)->bool{
                    const uint32_t* d=t.v2.dimensions; if(t.v2.rank!=4 || !d || d[0]!=1) return false;
                    return d[1]==3 && (int)d[2]==H && (int)d[3]==W; };   // NCHW only - the layout this app feeds and reads
                auto u8=[](const Qnn_Tensor_t& t)->bool{ const auto dt=t.v2.dataType;
                    if(dt==QNN_DATATYPE_UINT_8) return true;
                    if(dt!=QNN_DATATYPE_UFIXED_POINT_8) return false;
                    const auto& q=t.v2.quantizeParams;
                    return q.quantizationEncoding==QNN_QUANTIZATION_ENCODING_SCALE_OFFSET && q.scaleOffsetEncoding.offset==0
                           && std::fabs(q.scaleOffsetEncoding.scale*255.0f-1.0f)<0.01f; };
                auto dims=[](const Qnn_Tensor_t& t){ std::string r="["; for(uint32_t i=0;i<t.v2.rank && t.v2.dimensions;i++) r+=(i?",":"")+std::to_string(t.v2.dimensions[i]); return r+"]"; };
                bool ni_ok=false, no_ok=false;
                if(gn && ni==1 && no==1){ ni_ok=shape(ins[0],SR_H,SR_W) && u8(ins[0]); no_ok=shape(outs[0],SR_OH,SR_OW) && u8(outs[0]); }
                if(ni_ok && no_ok){ gname=gn; in_id=ins[0].v2.id; out_id=outs[0].v2.id;
                    info="graph "+gname+", in id "+std::to_string(in_id)+" "+dims(ins[0])
                         +", out id "+std::to_string(out_id)+" "+dims(outs[0])+", uint8 1/255"; }
                else { fits=false;
                    info="this context does not fit: it needs one 8-bit NCHW image 512x288 in and 2048x1152 out (scale 1/255); it has "
                         +std::to_string(ni)+" input(s) "+(ni? dims(ins[0]) : std::string("-"))+" and "+std::to_string(no)+" output(s) "+(no? dims(outs[0]) : std::string("-")); } } }
        SI.systemContextFree(sc);
        return fits;
    }
    // in: uint8 NCHW [3][288][512], out: uint8 NCHW [3][1152][2048] (scale 1/255, offset 0 on both)
    bool run(const uint8_t* in, uint8_t* out){
        auto T=[](uint32_t id,const void* p,size_t n,Qnn_TensorType_t ty){ Qnn_Tensor_t x=pcore_npu::Tt(id,p,n,ty,QNN_DATATYPE_UFIXED_POINT_8);
            x.v2.quantizeParams.encodingDefinition=QNN_DEFINITION_DEFINED; x.v2.quantizeParams.quantizationEncoding=QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
            x.v2.quantizeParams.scaleOffsetEncoding.scale=1.0f/255.0f; x.v2.quantizeParams.scaleOffsetEncoding.offset=0; return x; };
        Qnn_Tensor_t ti=T(in_id,in,(size_t)3*SR_W*SR_H,QNN_TENSOR_TYPE_APP_WRITE), to=T(out_id,out,(size_t)3*SR_OW*SR_OH,QNN_TENSOR_TYPE_APP_READ);
        const double t0=sr_now_ms();
        const bool ok=iface->QNN_INTERFACE_VER_NAME.graphExecute(graph,&ti,1,&to,1,nullptr,nullptr)==QNN_SUCCESS;
        last_ms=sr_now_ms()-t0; return ok;
    }
};

