cmake_minimum_required(VERSION 3.21)
project(pulsex_cctv LANGUAGES CXX RC)

# PulseX CCTV — Windows on ARM (Snapdragon X), live RTSP camera view with person tracking on the Hexagon NPU.
#
#   cmake -S . -B build -G "Visual Studio 17 2022" -A ARM64 -DQNN_SDK_ROOT=C:/Qualcomm/AIStack/QAIRT/2.47.0.260601
#   cmake --build build --config Release
#
# QNN_SDK_ROOT is the Qualcomm AI Runtime (QAIRT) SDK folder, the one that contains include/QNN and lib/.
# Its headers are used to build; its HTP runtime libraries are copied next to the exe after the build.

set(QNN_SDK_ROOT "$ENV{QNN_SDK_ROOT}" CACHE PATH "Qualcomm AI Runtime (QAIRT) SDK root (contains include/QNN)")
if(NOT EXISTS "${QNN_SDK_ROOT}/include/QNN/QnnInterface.h")
    message(FATAL_ERROR "Set QNN_SDK_ROOT to the QAIRT SDK folder (the one with include/QNN/QnnInterface.h).")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(IMGUI ${CMAKE_SOURCE_DIR}/third_party/imgui)

add_executable(pulsex_cctv WIN32
    src/pulsecore_cctv.cpp src/app.rc src/utf8.manifest   # utf8.manifest: ANSI code page = UTF-8 (paths, tapo_rtsp.txt)
    ${IMGUI}/imgui.cpp ${IMGUI}/imgui_draw.cpp ${IMGUI}/imgui_tables.cpp ${IMGUI}/imgui_widgets.cpp
    ${IMGUI}/backends/imgui_impl_win32.cpp ${IMGUI}/backends/imgui_impl_dx11.cpp)
target_include_directories(pulsex_cctv PRIVATE
    src include ${IMGUI} ${IMGUI}/backends ${QNN_SDK_ROOT}/include/QNN)
target_compile_definitions(pulsex_cctv PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX _CRT_SECURE_NO_WARNINGS)
target_link_libraries(pulsex_cctv PRIVATE d3d11 dxgi d3dcompiler)
if(MSVC)
    target_compile_options(pulsex_cctv PRIVATE /utf-8)
    target_link_options(pulsex_cctv PRIVATE /ENTRY:mainCRTStartup)
endif()

# The QNN HTP runtime from YOUR SDK install, next to the exe (not part of this repository).
set(QNN_LIB ${QNN_SDK_ROOT}/lib/aarch64-windows-msvc)
set(QNN_SKEL ${QNN_SDK_ROOT}/lib/hexagon-v73/unsigned)
# the skel runs ON the NPU; Windows only loads it together with its signature catalog (.cat) - without it: qnn_open 0x80000406
foreach(f ${QNN_LIB}/QnnHtp.dll ${QNN_LIB}/QnnHtpV73Stub.dll ${QNN_LIB}/QnnSystem.dll ${QNN_SKEL}/libQnnHtpV73Skel.so ${QNN_SKEL}/libqnnhtpv73.cat)
    if(EXISTS ${f})
        add_custom_command(TARGET pulsex_cctv POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different ${f} $<TARGET_FILE_DIR:pulsex_cctv>)
    else()
        message(WARNING "not found in the SDK: ${f}")
    endif()
endforeach()
