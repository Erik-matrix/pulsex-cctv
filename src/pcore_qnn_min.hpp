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
#include <vector>
#include <filesystem>

namespace pcore_npu {

inline FILE*& qnn_log_file(){ static FILE* f=nullptr; return f; }
inline std::mutex& qnn_log_mtx(){ static std::mutex m; return m; }
inline void qnn_log_cb(const char* fmt, QnnLog_Level_t lvl, uint64_t, va_list args){
    FILE* f=qnn_log_file(); if(!f) return;
    const char* p = lvl==QNN_LOG_LEVEL_ERROR? "[ERR ] " : lvl==QNN_LOG_LEVEL_WARN? "[WARN] " : lvl==QNN_LOG_LEVEL_INFO? "[INFO] " : "[DBG ] ";
    std::lock_guard<std::mutex> lk(qnn_log_mtx()); std::fputs(p,f); std::vfprintf(f,fmt,args); std::fputc('\n',f); std::fflush(f); }

// 10-01: the NPU driver loads its runtime only from a folder whose path is plain ASCII. -> the folder to load QNN from:
// the exe's own when it is ASCII (or holds no QNN files), else a copy under %ProgramData%\PulseX\npu\. Empty = failed.
inline bool qnn_is_ascii(const std::wstring& t){ for(wchar_t c: t) if(c > 127) return false; return true; }
inline std::wstring qnn_exe_dir(){
    std::wstring b(32768, L'\0'); DWORD n = GetModuleFileNameW(nullptr, &b[0], (DWORD)b.size()); b.resize(n);
    size_t k = b.find_last_of(L"\\/"); return k == std::wstring::npos ? std::wstring(L".") : b.substr(0, k); }
inline std::wstring ascii_runtime_dir(const std::wstring& src, std::string& err){
    if(qnn_is_ascii(src)) return src;
    std::vector<std::wstring> files; unsigned long long h = 1469598103934665603ull;   // FNV-1a over name, size, time
    for(const wchar_t* pat : {L"\\Qnn*.dll", L"\\libQnn*.so", L"\\libqnn*.cat"}){
        WIN32_FIND_DATAW fd; HANDLE f = FindFirstFileW((src + pat).c_str(), &fd);
        if(f == INVALID_HANDLE_VALUE) continue;
        do { if(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) continue;
             files.push_back(fd.cFileName);
             auto mix = [&](const void* p, size_t n){ for(size_t i=0;i<n;i++){ h ^= ((const unsigned char*)p)[i]; h *= 1099511628211ull; } };
             mix(fd.cFileName, wcslen(fd.cFileName) * 2); mix(&fd.nFileSizeLow, 4); mix(&fd.nFileSizeHigh, 4); mix(&fd.ftLastWriteTime, 8);
        } while(FindNextFileW(f, &fd));
        FindClose(f);
    }
    if(files.empty()) return src;                                   // QNN comes from elsewhere (PATH): nothing to copy
    wchar_t pd[MAX_PATH] = {}; if(!GetEnvironmentVariableW(L"ProgramData", pd, MAX_PATH)) wcscpy_s(pd, L"C:\\ProgramData");
    wchar_t hx[24]; swprintf_s(hx, L"%016llx", h);
    const std::wstring dst = std::wstring(pd) + L"\\PulseX\\npu\\qnn-" + hx;
    if(!qnn_is_ascii(dst)){ err = "the NPU cannot load its files from a folder with letters outside A-Z, and ProgramData is not plain ASCII either"; return L""; }
    std::error_code ec; std::filesystem::create_directories(dst, ec);
    for(const auto& f : files){
        const std::wstring a = src + L"\\" + f, b = dst + L"\\" + f;
        WIN32_FILE_ATTRIBUTE_DATA sa{}, sb{};
        const bool same = GetFileAttributesExW(a.c_str(), GetFileExInfoStandard, &sa) && GetFileAttributesExW(b.c_str(), GetFileExInfoStandard, &sb)
                          && sa.nFileSizeLow == sb.nFileSizeLow && sa.nFileSizeHigh == sb.nFileSizeHigh;
        if(!same && !CopyFileW(a.c_str(), b.c_str(), FALSE)){
            err = "the NPU cannot load its files from a folder with letters outside A-Z, and copying them to ProgramData failed - "
                  "move the app to a folder such as C:\\PulseX";
            return L""; }
    }
    return dst;
}

class QnnHtpSession {
public:
    QnnHtpSession() = default;
    ~QnnHtpSession(){ close(); }
    QnnHtpSession(const QnnHtpSession&) = delete; QnnHtpSession& operator=(const QnnHtpSession&) = delete;

    bool open(const wchar_t* dll = L"QnnHtp.dll"){
        if(dev_) return true;
        { const std::wstring here = qnn_exe_dir(); rt_dir_ = ascii_runtime_dir(here, err_);   // 10-01: see ascii_runtime_dir
          if(rt_dir_.empty()) return false;
          if(rt_dir_ != here){                                        // QnnHtp.dll + its stub + the skel from the ASCII copy
              std::wstring old(32768, L'\0'); DWORD n = GetEnvironmentVariableW(L"ADSP_LIBRARY_PATH", &old[0], (DWORD)old.size());
              old.resize(n < old.size() ? n : 0);
              const std::wstring adsp = rt_dir_ + (old.empty() ? std::wstring() : L";" + old);
              SetEnvironmentVariableW(L"ADSP_LIBRARY_PATH", adsp.c_str());   // the process block ...
              _wputenv_s(L"ADSP_LIBRARY_PATH", adsp.c_str());                // ... and the CRT's copy
              lib_ = LoadLibraryExW((rt_dir_ + L"\\" + dll).c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH); } }
        if(!lib_) lib_ = LoadLibraryW(dll);
        if(!lib_){ err_="cannot load QnnHtp.dll (install the Qualcomm AI Runtime / QAIRT and put its HTP libraries next to the exe or on PATH)"; return false; }
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
    const std::wstring& runtime_dir() const { return rt_dir_; }     // where QnnHtp.dll was loaded from (10-01)

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
    bool perf_ok_=false; std::string err_; std::wstring rt_dir_;
};

} // namespace pcore_npu
