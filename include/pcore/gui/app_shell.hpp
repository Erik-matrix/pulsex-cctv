// app_shell.hpp — reusable borderless "microapp" shell for PulseCore (the pattern proven by PulseCore CV).
// A single function spins up a dark, frameless ImGui + DX11 window with a custom title bar (teal dot + title +
// min/max/close), draggable header, resize borders, work-area-aware maximize, a minimise/unfocus frame throttle,
// Segoe UI (with Nordic letters), and app-relative + Documents/Downloads path helpers. Each function (CV, CCTV, Image, …)
// becomes its own lean standalone exe:  int main(){ return pcore::gui::run_app("PulseCore CCTV", draw_cctv); }
//
// Header-with-impl: include it in exactly ONE .cpp per app (the app's main), alongside the Dear ImGui sources.
#pragma once
#include <atomic>
#include "imgui.h"
#include "imgui_impl_win32.h"
#include "imgui_impl_dx11.h"
#include <d3d11.h>
#ifndef WIN32_LEAN_AND_MEAN
#  define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <windowsx.h>
#include <shlobj.h>
#include <imm.h>
#include <string>
#include <functional>
#include <cstdlib>
#include <filesystem>
#pragma comment(lib,"d3d11.lib")
#pragma comment(lib,"shell32.lib")
#pragma comment(lib,"ole32.lib")
#pragma comment(lib,"imm32.lib")

// Provided by imgui_impl_win32.cpp (declared here at global scope so the borderless WndProc can forward to it).
extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

namespace pcore::gui {

// ── path helpers (portable: relative to the exe; user content under Documents; results in Downloads) ──
inline std::string app_dir(){
    char b[MAX_PATH]={0}; GetModuleFileNameA(nullptr,b,MAX_PATH);
    std::string p(b); auto s=p.find_last_of("\\/"); return s==std::string::npos?std::string("."):p.substr(0,s);
}
inline std::string _known(REFKNOWNFOLDERID id, const char* fallback){
    PWSTR wp=nullptr; std::string out;
    if(SUCCEEDED(SHGetKnownFolderPath(id,0,nullptr,&wp)) && wp){
        int n=WideCharToMultiByte(CP_ACP,0,wp,-1,nullptr,0,nullptr,nullptr);
        if(n>1){ out.resize(n-1); WideCharToMultiByte(CP_ACP,0,wp,-1,&out[0],n,nullptr,nullptr); } }
    if(wp) CoTaskMemFree(wp);
    if(out.empty()){ const char* up=std::getenv("USERPROFILE"); out = up? std::string(up)+"\\"+fallback : app_dir(); }
    return out;
}
inline std::string downloads(){ return _known(FOLDERID_Downloads,"Downloads"); }
// Shared model pool — every microapp resolves models from ONE location (never a per-app copy).
// Override with the PULSECORE_MODELS env var; otherwise fall back to the models\ folder next to the app.
inline std::string models_dir(){
    if(const char* e=std::getenv("PULSECORE_MODELS")){ if(*e) return std::string(e); }
    std::string a=app_dir(); std::error_code ec;
    for(std::string c : { a+"\\models", a+"\\..\\models", a+"\\..\\..\\models" })
        if(std::filesystem::exists(c,ec)) return std::filesystem::weakly_canonical(c,ec).string();
    return a+"\\models";
}
// A named model in the shared pool, e.g. model_path("qwen3-4b-4k\\genie_bundle").
inline std::string model_path(const std::string& rel){ return models_dir()+"\\"+rel; }
// <Documents>\<sub> (created). Pass the app's own name, e.g. documents("pulse_cv").
inline std::string documents(const std::string& sub){
    std::string d=_known(FOLDERID_Documents,"Documents")+"\\"+sub;
    std::error_code ec; std::filesystem::create_directories(d,ec); return d;
}

// ── DX11 + window internals (static: one set per app exe) ──
namespace detail {
    inline ID3D11Device*           dev  = nullptr;
    inline ID3D11DeviceContext*    ctx  = nullptr;
    inline IDXGISwapChain*         swap = nullptr;
    inline ID3D11RenderTargetView* rtv  = nullptr;
    // extra D3D11 device-creation flags an app may set BEFORE run_app (09-30: pulsecore_cctv decodes video in-app and
    // needs D3D11_CREATE_DEVICE_VIDEO_SUPPORT for Media Foundation DXVA + the video processor). 0 = as before.
    inline UINT device_flags = 0;
    inline void mkRTV(){ ID3D11Texture2D* bb=nullptr; swap->GetBuffer(0,IID_PPV_ARGS(&bb));
        if(bb){ dev->CreateRenderTargetView(bb,nullptr,&rtv); bb->Release(); } }
    inline void freeRTV(){ if(rtv){ rtv->Release(); rtv=nullptr; } }
    inline bool mkDevice(HWND h){
        DXGI_SWAP_CHAIN_DESC sd{}; sd.BufferCount=2; sd.BufferDesc.Format=DXGI_FORMAT_R8G8B8A8_UNORM;
        sd.BufferDesc.RefreshRate.Numerator=60; sd.BufferDesc.RefreshRate.Denominator=1;
        sd.BufferUsage=DXGI_USAGE_RENDER_TARGET_OUTPUT; sd.OutputWindow=h; sd.SampleDesc.Count=1; sd.Windowed=TRUE;
        sd.SwapEffect=DXGI_SWAP_EFFECT_DISCARD;
        D3D_FEATURE_LEVEL fl; const D3D_FEATURE_LEVEL lv[]={D3D_FEATURE_LEVEL_11_0,D3D_FEATURE_LEVEL_10_0};
        if(D3D11CreateDeviceAndSwapChain(nullptr,D3D_DRIVER_TYPE_HARDWARE,nullptr,device_flags,lv,2,D3D11_SDK_VERSION,
            &sd,&swap,&dev,&fl,&ctx)!=S_OK) return false; mkRTV(); return true; }
    inline void freeDevice(){ freeRTV(); if(swap){swap->Release();swap=nullptr;} if(ctx){ctx->Release();ctx=nullptr;} if(dev){dev->Release();dev=nullptr;} }
}
// DX11 device — for apps that upload textures (e.g. a camera frame). Valid only inside run_app's draw callback.
inline ID3D11Device* device(){ return detail::dev; }
// Upload an RGBA image and get an ImGui-usable texture handle (caller releases the previous one).
inline void* make_texture(const unsigned char* rgba, int w, int h){
    if(!detail::dev||!rgba||w<=0||h<=0) return nullptr;
    D3D11_TEXTURE2D_DESC td{}; td.Width=w; td.Height=h; td.MipLevels=1; td.ArraySize=1;
    td.Format=DXGI_FORMAT_R8G8B8A8_UNORM; td.SampleDesc.Count=1; td.Usage=D3D11_USAGE_DEFAULT; td.BindFlags=D3D11_BIND_SHADER_RESOURCE;
    D3D11_SUBRESOURCE_DATA sr{}; sr.pSysMem=rgba; sr.SysMemPitch=(UINT)w*4;
    ID3D11Texture2D* tex=nullptr; if(detail::dev->CreateTexture2D(&td,&sr,&tex)!=S_OK) return nullptr;
    ID3D11ShaderResourceView* srv=nullptr; detail::dev->CreateShaderResourceView(tex,nullptr,&srv); tex->Release(); return srv;
}
inline LRESULT WINAPI _wndproc(HWND h,UINT m,WPARAM w,LPARAM l){
    if(::ImGui_ImplWin32_WndProcHandler(h,m,w,l)) return true;
    using namespace detail;
    switch(m){
        case WM_SIZE: if(dev && w!=SIZE_MINIMIZED){ freeRTV(); swap->ResizeBuffers(0,(UINT)LOWORD(l),(UINT)HIWORD(l),DXGI_FORMAT_UNKNOWN,0); mkRTV(); } return 0;
        case WM_NCCALCSIZE: if(w) return 0; break;
        case WM_NCHITTEST: { const LONG b=8; POINT p={GET_X_LPARAM(l),GET_Y_LPARAM(l)}; RECT r; GetWindowRect(h,&r);
            const bool L=p.x<r.left+b,R=p.x>=r.right-b,T=p.y<r.top+b,B=p.y>=r.bottom-b;
            if(T&&L)return HTTOPLEFT; if(T&&R)return HTTOPRIGHT; if(B&&L)return HTBOTTOMLEFT; if(B&&R)return HTBOTTOMRIGHT;
            if(L)return HTLEFT; if(R)return HTRIGHT; if(T)return HTTOP; if(B)return HTBOTTOM;
            // Title-bar strip = native OS drag (reliable), excluding the right button cluster (min/max/close).
            // (Replaces the flaky ImGui IsMouseClicked+SendMessage drag, which raced the frame and sometimes missed.)
            LONG cy=p.y-r.top, cx=p.x-r.left, wcw=r.right-r.left;
            if(cy<52 && cx < wcw-104) return HTCAPTION;
            return HTCLIENT; }
        case WM_GETMINMAXINFO: { MINMAXINFO* mmi=(MINMAXINFO*)l; HMONITOR mo=MonitorFromWindow(h,MONITOR_DEFAULTTONEAREST);
            MONITORINFO mi{sizeof(mi)}; if(GetMonitorInfo(mo,&mi)){ mmi->ptMaxPosition.x=mi.rcWork.left-mi.rcMonitor.left; mmi->ptMaxPosition.y=mi.rcWork.top-mi.rcMonitor.top;
                mmi->ptMaxSize.x=mi.rcWork.right-mi.rcWork.left; mmi->ptMaxSize.y=mi.rcWork.bottom-mi.rcWork.top; } return 0; }
        case WM_SYSCOMMAND: if((w&0xfff0)==SC_KEYMENU) return 0; break;
        case WM_DESTROY: PostQuitMessage(0); return 0;
    }
    return DefWindowProcW(h,m,w,l);
}

// ** z282cr RAMKVITTO (se Present nedan). `frame_no` rakar upp for varje presenterad ram och
//    `frame_qpc` bar dess QPC-tidpunkt. En app som vill mata "rad mottagen -> syns pa skarmen"
//    laser frame_no fore, vantar tills den andras, och laser frame_qpc.
inline std::atomic<unsigned long long> frame_no{0};
inline std::atomic<long long>          frame_qpc{0};

// Run a borderless dark microapp. `draw` renders the content each frame (below the built-in title bar).
// `accent` recolours the title-bar dot + buttons/headers per app (default teal). Pass a bright base colour.
inline int run_app(const char* title_utf8, const std::function<void()>& draw, int prefW=980, int prefH=1200,
                   ImVec4 accent=ImVec4(0,0,0,0)){
    const bool hasAccent = accent.w>0.0f;
    WNDCLASSEXW wc{ sizeof(wc), CS_CLASSDC, _wndproc, 0,0, GetModuleHandle(nullptr), nullptr,nullptr,nullptr,nullptr, L"PulseCoreApp", nullptr };
    RegisterClassExW(&wc);
    RECT wa{0,0,1280,1024}; SystemParametersInfoW(SPI_GETWORKAREA,0,&wa,0);
    int waW=wa.right-wa.left, waH=wa.bottom-wa.top, W=prefW, H=(waH<prefH)?waH:prefH;
    int X=wa.left+(waW-W)/2; if(X<wa.left)X=wa.left; int Y=wa.top+(waH-H)/6; if(Y<wa.top)Y=wa.top;
    HWND h = CreateWindowExW(WS_EX_APPWINDOW, wc.lpszClassName, L"PulseCore",
                             WS_POPUP|WS_THICKFRAME|WS_MINIMIZEBOX|WS_MAXIMIZEBOX, X,Y,W,H, nullptr,nullptr,wc.hInstance,nullptr);
    if(!detail::mkDevice(h)){ detail::freeDevice(); UnregisterClassW(wc.lpszClassName,wc.hInstance); return 1; }
    ImmAssociateContext(h, NULL);   // detach IME — these apps are physical-keyboard only (Nordic letters are direct keys), no IME needed
    // ** FONSTRET SKA UPP, INTE BARA FINNAS. SW_SHOWDEFAULT arver visningslaget fran
    //    den som STARTADE processen (STARTUPINFO.wShowWindow) - startas appen fran ett
    //    skript, en genvag eller en annan process kan det vara minimerat eller
    //    icke-aktiverande, och da hamnar fonstret bara i aktivitetsfaltet.
    // ⚠ Och SetForegroundWindow ensamt racker inte: Windows VAGRAR ge fokus till en
    //   process som inte redan ager forgrunden, och gor det TYST - anropet returnerar
    //   bara FALSE. Standardvagen ar att tillfalligt koppla var indatako till den
    //   nuvarande forgrundstradens, gora bytet, och koppla loss igen.
    IMGUI_CHECKVERSION(); ImGui::CreateContext();
    ImGuiIO& io=ImGui::GetIO(); io.IniFilename=nullptr; ImGui::StyleColorsDark();
    { ImGuiStyle& s=ImGui::GetStyle();
      s.WindowRounding=0; s.ChildRounding=10; s.FrameRounding=7; s.PopupRounding=7; s.ScrollbarRounding=9; s.GrabRounding=7; s.TabRounding=7;
      s.WindowPadding=ImVec2(16,14); s.FramePadding=ImVec2(11,7); s.ItemSpacing=ImVec2(10,9); s.WindowBorderSize=0;
      ImVec4* c=s.Colors; c[ImGuiCol_WindowBg]=ImVec4(0.13f,0.14f,0.16f,1);
      ImVec4 a(0.13f,0.60f,0.55f,1),ah(0.22f,0.82f,0.74f,1),ad(0.10f,0.44f,0.41f,1);
      if(hasAccent){ ah=accent; a=ImVec4(accent.x*0.74f,accent.y*0.74f,accent.z*0.74f,1); ad=ImVec4(accent.x*0.55f,accent.y*0.55f,accent.z*0.55f,1); }
      c[ImGuiCol_Button]=a; c[ImGuiCol_ButtonHovered]=ah; c[ImGuiCol_ButtonActive]=ad;
      c[ImGuiCol_Header]=ad; c[ImGuiCol_HeaderHovered]=a; c[ImGuiCol_HeaderActive]=ah;
      c[ImGuiCol_FrameBg]=ImVec4(0.20f,0.21f,0.25f,1); c[ImGuiCol_CheckMark]=ah; }
    { static const ImWchar rg[]={0x0020,0x00FF,0x2010,0x2027,0x25A0,0x25FF,0};
      if(!io.Fonts->AddFontFromFileTTF("C:\\Windows\\Fonts\\segoeui.ttf",22.0f,nullptr,rg)) io.Fonts->AddFontDefault(); }
    ImGui_ImplWin32_Init(h); ImGui_ImplDX11_Init(detail::dev,detail::ctx);
    // ★ z282q AKTIVERINGEN FLYTTAD HIT, efter ImGui_ImplWin32_Init.
    // Den lag FORE ImGui::CreateContext(), och imgui_impl_win32.cpp:629 har en
    // kontextvakt: `if (ImGui::GetCurrentContext() == nullptr) return 0;`. Alla
    // WM_SETFOCUS/WM_ACTIVATE som ShowWindow/SetForegroundWindow/SetFocus genererade
    // kastades alltsa TYST - ImGui fick aldrig sin forsta fokushandelse.
    // ⚠ Att detta ar Ctrl+V-buggen ar EN LASNING, inte en matning. Men ordningen ar
    //   fel oavsett, och ratt ordning kostar ingenting.
    ShowWindow(h, SW_SHOW);
    if(IsIconic(h)) ShowWindow(h, SW_RESTORE);
    UpdateWindow(h);
    {
        HWND fg = GetForegroundWindow();
        DWORD vi = GetCurrentThreadId();
        DWORD fi = fg ? GetWindowThreadProcessId(fg, nullptr) : vi;
        // ⚠ z282q HYPOTES UNDER PROV: Ctrl+V/C fungerar i pulsecore_image.exe (byggd
        // 2026-08-08, alltsa med skalet FORE den har aktiveringsblocken) men INTE i
        // zimage.exe. Allt annat ar uteslutet: identiska byggflaggor, samma ImGui-kalla,
        // samma WndProc, samma meddelandeloop, och zimage.cpp ror inte tangentbords-IO.
        // AttachThreadInput kopplar ihop tradars INDATAKOER och ar en rimlig misstankt.
        // PULSE_GUI_NOATTACH=1 hoppar over kopplingen sa hypotesen kan provas utan att
        // forlora aktiveringen (fonstret tas anda fram, bara utan tradkoppling).
        const bool _noattach = [](){ const char* e=getenv("PULSE_GUI_NOATTACH"); return e && *e && atoi(e)!=0; }();
        if(!_noattach && fi != vi) AttachThreadInput(fi, vi, TRUE);
        BringWindowToTop(h);
        SetForegroundWindow(h);
        SetActiveWindow(h);
        SetFocus(h);
        if(!_noattach && fi != vi) AttachThreadInput(fi, vi, FALSE);
    }
    ImGui::GetPlatformIO().Platform_SetImeDataFn = nullptr;   // no per-frame IME candidate-window calls (keyboard-only apps)
    std::wstring wtitle; { int n=MultiByteToWideChar(CP_UTF8,0,title_utf8,-1,nullptr,0); wtitle.resize(n>0?n-1:0); if(n>0) MultiByteToWideChar(CP_UTF8,0,title_utf8,-1,&wtitle[0],n); }
    SetWindowTextW(h, wtitle.c_str());

    bool done=false;
    while(!done){
        MSG msg;
        while(PeekMessageW(&msg,nullptr,0,0,PM_REMOVE)){ TranslateMessage(&msg); DispatchMessageW(&msg); if(msg.message==WM_QUIT) done=true; }
        if(done) break;
        if(IsIconic(h)){ Sleep(120); continue; }
        if(GetForegroundWindow()!=h) Sleep(30);
        ImGui_ImplDX11_NewFrame(); ImGui_ImplWin32_NewFrame(); ImGui::NewFrame();
        const ImGuiViewport* vp=ImGui::GetMainViewport();
        ImGui::SetNextWindowPos(vp->Pos); ImGui::SetNextWindowSize(vp->Size);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize,0.0f); ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding,0.0f);
        ImGui::Begin("##app", nullptr, ImGuiWindowFlags_NoTitleBar|ImGuiWindowFlags_NoResize|ImGuiWindowFlags_NoMove|ImGuiWindowFlags_NoCollapse|ImGuiWindowFlags_NoBringToFrontOnFocus);
        ImGui::PopStyleVar(2);
        // custom title bar: teal dot + title + min/max/close, draggable
        const ImVec2 hmin=ImGui::GetCursorScreenPos();
        { float th=ImGui::GetTextLineHeight(); ImVec2 p=ImGui::GetCursorScreenPos();
          ImU32 dotc = hasAccent ? IM_COL32((int)(accent.x*255),(int)(accent.y*255),(int)(accent.z*255),255) : IM_COL32(90,210,190,255);
          ImGui::GetWindowDrawList()->AddCircleFilled(ImVec2(p.x+th*0.45f,p.y+th*0.55f), th*0.30f, dotc);
          ImGui::Dummy(ImVec2(th*0.9f,th)); }
        ImGui::SameLine(0,4); ImGui::TextUnformatted(title_utf8);
        ImGui::SameLine(ImGui::GetContentRegionMax().x-84.0f);
        ImGui::PushStyleColor(ImGuiCol_Button,ImVec4(0,0,0,0)); ImGui::PushStyleColor(ImGuiCol_ButtonHovered,ImVec4(1,1,1,0.10f));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive,ImVec4(1,1,1,0.18f)); ImGui::PushStyleColor(ImGuiCol_Border,ImVec4(0,0,0,0));
        { ImVec2 b=ImGui::GetCursorScreenPos(); if(ImGui::Button("##min",ImVec2(26,0))) ShowWindow(h,SW_MINIMIZE);
          ImDrawList* d=ImGui::GetWindowDrawList(); float bh=ImGui::GetItemRectSize().y,cx=b.x+13,cy=b.y+bh*0.5f; ImU32 ic=ImGui::GetColorU32(ImGuiCol_Text);
          d->AddLine(ImVec2(cx-6,cy),ImVec2(cx+6,cy),ic,1.3f); }
        ImGui::SameLine();
        { bool mx=IsZoomed(h); ImVec2 b=ImGui::GetCursorScreenPos(); if(ImGui::Button("##max",ImVec2(26,0))) ShowWindow(h,mx?SW_RESTORE:SW_MAXIMIZE);
          ImDrawList* d=ImGui::GetWindowDrawList(); float bh=ImGui::GetItemRectSize().y,cx=b.x+13,cy=b.y+bh*0.5f; ImU32 ic=ImGui::GetColorU32(ImGuiCol_Text);
          if(mx){ d->AddRect(ImVec2(cx-3,cy-5),ImVec2(cx+5,cy+3),ic,0,0,1.2f); d->AddRectFilled(ImVec2(cx-5,cy-3),ImVec2(cx+3,cy+5),ImGui::GetColorU32(ImGuiCol_WindowBg)); d->AddRect(ImVec2(cx-5,cy-3),ImVec2(cx+3,cy+5),ic,0,0,1.2f); }
          else d->AddRect(ImVec2(cx-5,cy-5),ImVec2(cx+5,cy+5),ic,0,0,1.2f); }
        ImGui::SameLine(); ImGui::PushStyleColor(ImGuiCol_ButtonHovered,ImVec4(0.82f,0.22f,0.22f,0.85f));
        { ImVec2 b=ImGui::GetCursorScreenPos(); bool cl=ImGui::Button("##close",ImVec2(26,0));
          ImDrawList* d=ImGui::GetWindowDrawList(); float bh=ImGui::GetItemRectSize().y,cx=b.x+13,cy=b.y+bh*0.5f; ImU32 ic=ImGui::GetColorU32(ImGuiCol_Text);
          d->AddLine(ImVec2(cx-5,cy-5),ImVec2(cx+5,cy+5),ic,1.3f); d->AddLine(ImVec2(cx-5,cy+5),ImVec2(cx+5,cy-5),ic,1.3f); if(cl) done=true; }
        ImGui::PopStyleColor(5);
        // whole header strip is draggable (dot + title + empty space); the rect already stops 84px before the
        // right edge so the min/max/close buttons are excluded — no need to block on IsAnyItemHovered (which
        // wrongly disabled dragging over the title text and dot).
        const ImVec2 hmax(hmin.x+ImGui::GetContentRegionMax().x-84.0f, hmin.y+ImGui::GetFrameHeight());
        if(ImGui::IsMouseHoveringRect(hmin,hmax) && ImGui::IsMouseClicked(ImGuiMouseButton_Left)){ ReleaseCapture(); SendMessageW(h,WM_NCLBUTTONDOWN,HTCAPTION,0); }
        ImGui::Separator(); ImGui::Dummy(ImVec2(0,2));
        draw();
        ImGui::End();
        ImGui::Render();
        const float clr[4]={0.09f,0.10f,0.12f,1.0f};
        detail::ctx->OMSetRenderTargets(1,&detail::rtv,nullptr);
        detail::ctx->ClearRenderTargetView(detail::rtv,clr);
        ImGui_ImplDX11_RenderDrawData(ImGui::GetDrawData());
        detail::swap->Present(1,0);
        // ** z282cr RAMKVITTO. Present kallas med SyncInterval=1, alltsa VSYNC: en fardig
        //    rad fran arbetaren kan ligga och vanta upp till en hel bildruta innan den syns.
        //    Det hoppet ar osynligt for varje matare vi har -- DSP-tiden ar mikrosekunder,
        //    en bildruta ar ~16 ms. Har stamplas VARJE presenterad ram sa en app kan mata
        //    "nar blev det jag just tog emot faktiskt synligt". Rent additivt: tva atomics
        //    och ett QPC-anrop (~20 ns), inget beteende andras for nagon app.
        { LARGE_INTEGER _q; QueryPerformanceCounter(&_q);
          frame_qpc.store(_q.QuadPart, std::memory_order_release);
          frame_no.fetch_add(1, std::memory_order_release); }
    }
    ImGui_ImplDX11_Shutdown(); ImGui_ImplWin32_Shutdown(); ImGui::DestroyContext();
    detail::freeDevice(); DestroyWindow(h); UnregisterClassW(wc.lpszClassName,wc.hInstance);
    return 0;
}

} // namespace pcore::gui
