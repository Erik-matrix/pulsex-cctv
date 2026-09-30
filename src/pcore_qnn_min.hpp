// pcore_qnn_min.hpp — 2026-09-30: the smallest QNN HTP session the detector (and CCTV's QuickSRNet) need:
// load QnnHtp.dll, backend, device (unsigned PD, V73), HTP power profile. Header-only, no pulsecore.dll.
//
// Replaces pcore::runtime::QnnLoader + DeviceManager for these tools. The old pair also opened an empty default
// context, a DEBUG-level QNN log with a crash-recovery callback (for graph COMPILATION in other tools) and a
// log file under the repo root — none of which a precompiled context binary uses.
//
// QNN log: off unless PULSECORE_QNN_LOG=<file> (WARN level; PULSECORE_QNN_LOG_LEVEL=debug|verbose|info|error).
// Power: PERFORMANCE, DCVS off, corner MAX — PULSECORE_HTP_CORNER=<NOM|TURBO|...> / PULSECORE_HTP_DCVS=1 as before.
#pragma once
#include <windows.h>
#include <QnnInterface.h>
#include <QnnDevice.h>
#include <QnnLog.h>
#include <HTP/QnnHtpDevice.h>
#include <HTP/QnnHtpPerfInfrastructure.h>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <mutex>

namespace pcore_npu {

inline FILE*& qnn_log_file(){ static FILE* f=nullptr; return f; }
inline std::mutex& qnn_log_mtx(){ static std::mutex m; return m; }
inline void qnn_log_cb(const char* fmt, QnnLog_Level_t lvl, uint64_t, va_list args){
    FILE* f=qnn_log_file(); if(!f) return;
    const char* p = lvl==QNN_LOG_LEVEL_ERROR? "[ERR ] " : lvl==QNN_LOG_LEVEL_WARN? "[WARN] " : lvl==QNN_LOG_LEVEL_INFO? "[INFO] " : "[DBG ] ";
    std::lock_guard<std::mutex> lk(qnn_log_mtx()); std::fputs(p,f); std::vfprintf(f,fmt,args); std::fputc('\n',f); std::fflush(f); }

class QnnHtpSession {
public:
    QnnHtpSession() = default;
    ~QnnHtpSession(){ close(); }
    QnnHtpSession(const QnnHtpSession&) = delete; QnnHtpSession& operator=(const QnnHtpSession&) = delete;

    bool open(const wchar_t* dll = L"QnnHtp.dll"){
        if(dev_) return true;
        lib_ = LoadLibraryW(dll); if(!lib_){ err_="cannot load QnnHtp.dll (install the Qualcomm AI Runtime / QAIRT and put its HTP libraries next to the exe or on PATH)"; return false; }
        using GetProviders = Qnn_ErrorHandle_t (*)(const QnnInterface_t***, uint32_t*);
        auto gp = reinterpret_cast<GetProviders>(GetProcAddress(lib_, "QnnInterface_getProviders"));
        const QnnInterface_t** pv=nullptr; uint32_t n=0;
        if(!gp || gp(&pv,&n)!=QNN_SUCCESS || !n || !pv || !pv[0]){ err_="QnnInterface_getProviders failed"; close(); return false; }
        iface_ = pv[0];
        auto& F = iface_->QNN_INTERFACE_VER_NAME;
        if(const char* lf=std::getenv("PULSECORE_QNN_LOG")){ if(*lf && F.logCreate && !qnn_log_file()){
            qnn_log_file() = std::fopen(lf,"w");
            QnnLog_Level_t lv=QNN_LOG_LEVEL_WARN; if(const char* l=std::getenv("PULSECORE_QNN_LOG_LEVEL")){ std::string s=l;
                if(s=="debug") lv=QNN_LOG_LEVEL_DEBUG; else if(s=="verbose") lv=QNN_LOG_LEVEL_VERBOSE; else if(s=="info") lv=QNN_LOG_LEVEL_INFO; else if(s=="error") lv=QNN_LOG_LEVEL_ERROR; }
            if(qnn_log_file() && F.logCreate(qnn_log_cb, lv, &log_)!=QNN_SUCCESS) log_=nullptr; } }
        if(!F.backendCreate || F.backendCreate(log_, nullptr, &be_)!=QNN_SUCCESS || !be_){ err_="QNN backendCreate failed"; close(); return false; }
        // unsigned process domain + arch V73 (Snapdragon X); retry without a config like the old DeviceManager
        QnnHtpDevice_CustomConfig_t pd{}; pd.option=QNN_HTP_DEVICE_CONFIG_OPTION_SIGNEDPD;
        pd.useSignedProcessDomain.deviceId=0; pd.useSignedProcessDomain.useSignedProcessDomain=false;
        QnnHtpDevice_CustomConfig_t ar{}; ar.option=QNN_HTP_DEVICE_CONFIG_OPTION_ARCH; ar.arch.deviceId=0; ar.arch.arch=QNN_HTP_DEVICE_ARCH_V73;
        QnnDevice_Config_t c1{}; c1.option=QNN_DEVICE_CONFIG_OPTION_CUSTOM; c1.customConfig=&pd;
        QnnDevice_Config_t c2{}; c2.option=QNN_DEVICE_CONFIG_OPTION_CUSTOM; c2.customConfig=&ar;
        const QnnDevice_Config_t* cfg[]={&c1,&c2,nullptr};
        if(!F.deviceCreate || (F.deviceCreate(log_, cfg, &dev_)!=QNN_SUCCESS && F.deviceCreate(nullptr, nullptr, &dev_)!=QNN_SUCCESS) || !dev_){
            dev_=nullptr; err_="QNN deviceCreate failed"; close(); return false; }
        perf_ok_ = set_performance();
        return true;
    }
    void close(){
        if(iface_){ auto& F = iface_->QNN_INTERFACE_VER_NAME;
            if(dev_ && F.deviceFree) F.deviceFree(dev_);
            if(be_ && F.backendFree) F.backendFree(be_);
            if(log_ && F.logFree) F.logFree(log_); }
        dev_=nullptr; be_=nullptr; log_=nullptr; iface_=nullptr;
        if(lib_){ FreeLibrary(lib_); lib_=nullptr; }
    }
    const QnnInterface_t* iface()   const { return iface_; }
    Qnn_BackendHandle_t   backend() const { return be_; }
    Qnn_DeviceHandle_t    device()  const { return dev_; }
    bool perf_ok() const { return perf_ok_; }
    const std::string& error() const { return err_; }

private:
    // DCVS v3: PERFORMANCE mode, fixed voltage corner (MAX unless PULSECORE_HTP_CORNER), DSP kept awake between executes
    bool set_performance(){
        auto& F = iface_->QNN_INTERFACE_VER_NAME;
        QnnDevice_Infrastructure_t inf=nullptr;
        if(!F.deviceGetInfrastructure || F.deviceGetInfrastructure(&inf)!=QNN_SUCCESS || !inf) return false;
        auto& perf = reinterpret_cast<QnnHtpDevice_Infrastructure_t*>(inf)->perfInfra;
        if(!perf.createPowerConfigId || !perf.setPowerConfig) return false;
        uint32_t id=0; if(perf.createPowerConfigId(0,0,&id)!=QNN_SUCCESS) return false;
        auto corner = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
        if(const char* ec=std::getenv("PULSECORE_HTP_CORNER")){ std::string c=ec;
            if(c=="TURBO_L3") corner=DCVS_VOLTAGE_VCORNER_TURBO_L3; else if(c=="TURBO_L2") corner=DCVS_VOLTAGE_VCORNER_TURBO_L2;
            else if(c=="TURBO_PLUS") corner=DCVS_VOLTAGE_VCORNER_TURBO_PLUS; else if(c=="TURBO") corner=DCVS_VOLTAGE_VCORNER_TURBO;
            else if(c=="NOM_PLUS") corner=DCVS_VOLTAGE_VCORNER_NOM_PLUS; else if(c=="NOM") corner=DCVS_VOLTAGE_VCORNER_NOM;
            else if(c=="SVS_PLUS") corner=DCVS_VOLTAGE_VCORNER_SVS_PLUS; else if(c=="SVS") corner=DCVS_VOLTAGE_VCORNER_SVS; }
        const bool dcvs = std::getenv("PULSECORE_HTP_DCVS")!=nullptr;
        QnnHtpPerfInfrastructure_PowerConfig_t pc = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIG_INIT;
        pc.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3;
        auto& d = pc.dcvsV3Config;
        d.contextId=id; d.setDcvsEnable=1; d.dcvsEnable=dcvs?1:0;
        d.powerMode=QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE;
        d.setSleepLatency=1; d.sleepLatency=40; d.setSleepDisable=1; d.sleepDisable=1;
        d.setBusParams=1;  d.busVoltageCornerMin =dcvs? DCVS_VOLTAGE_VCORNER_NOM : corner; d.busVoltageCornerTarget =corner; d.busVoltageCornerMax =DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
        d.setCoreParams=1; d.coreVoltageCornerMin=dcvs? DCVS_VOLTAGE_VCORNER_NOM : corner; d.coreVoltageCornerTarget=corner; d.coreVoltageCornerMax=DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
        const QnnHtpPerfInfrastructure_PowerConfig_t* cs[]={&pc,nullptr};
        return perf.setPowerConfig(id, cs)==QNN_SUCCESS;
    }
    HMODULE lib_=nullptr; const QnnInterface_t* iface_=nullptr;
    Qnn_LogHandle_t log_=nullptr; Qnn_BackendHandle_t be_=nullptr; Qnn_DeviceHandle_t dev_=nullptr;
    bool perf_ok_=false; std::string err_;
};

} // namespace pcore_npu
