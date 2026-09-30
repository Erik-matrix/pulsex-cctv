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
// graph + tensor ids of that context (read with qnn-context-binary-utility; built by ORT-QNN 2.46 from the ONNX)
static const char* SR_GRAPH="QNNExecutionProvider_QNN_16202934166401028404_1_0";
struct SrNpu {
    const QnnInterface_t* iface=nullptr; Qnn_ContextHandle_t ctx=nullptr; Qnn_GraphHandle_t graph=nullptr;
    std::vector<uint8_t> blob; double last_ms=0;
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
        return iface->QNN_INTERFACE_VER_NAME.graphRetrieve(ctx,SR_GRAPH,&graph)==QNN_SUCCESS && graph;
    }
    // in: uint8 NCHW [3][288][512], out: uint8 NCHW [3][1152][2048] (scale 1/255, offset 0 on both)
    bool run(const uint8_t* in, uint8_t* out){
        auto T=[](uint32_t id,const void* p,size_t n,Qnn_TensorType_t ty){ Qnn_Tensor_t x=pcore_npu::Tt(id,p,n,ty,QNN_DATATYPE_UFIXED_POINT_8);
            x.v2.quantizeParams.encodingDefinition=QNN_DEFINITION_DEFINED; x.v2.quantizeParams.quantizationEncoding=QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
            x.v2.quantizeParams.scaleOffsetEncoding.scale=1.0f/255.0f; x.v2.quantizeParams.scaleOffsetEncoding.offset=0; return x; };
        Qnn_Tensor_t ti=T(1,in,(size_t)3*SR_W*SR_H,QNN_TENSOR_TYPE_APP_WRITE), to=T(94,out,(size_t)3*SR_OW*SR_OH,QNN_TENSOR_TYPE_APP_READ);
        const double t0=sr_now_ms();
        const bool ok=iface->QNN_INTERFACE_VER_NAME.graphExecute(graph,&ti,1,&to,1,nullptr,nullptr)==QNN_SUCCESS;
        last_ms=sr_now_ms()-t0; return ok;
    }
};

