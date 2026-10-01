# PulseX CCTV — a live camera view that follows people, on the Snapdragon NPU

PulseX CCTV shows an RTSP security camera on a Windows on ARM laptop (Snapdragon X) and
does the heavy lifting where Windows' own camera effects do it: the **video is decoded on the GPU**,
**people are detected on the Hexagon NPU**, and the CPU mostly waits. When someone walks into the
picture the view zooms in and follows them smoothly, and a clip of the passer-by can be recorded
automatically.

Tested with a TP-Link Tapo C220 (2560 × 1440 H.264, 30 fps) on a Snapdragon X Plus.

| Stage | How | Runs on |
|---|---|---|
| RTSP + H.264 decode | Media Foundation (DXVA) → NV12 texture, TCP transport | GPU video engine |
| Scaling / colour | D3D11 video processor (display, 640 × 640 NPU tiles, 1080p recording) | GPU |
| Person detection | YOLO detection context through QNN, two square tiles per frame, zero-copy buffers | NPU (HTP) |
| Following | time-aligned targets + critically damped spring | CPU (tiny) |
| Recording | Media Foundation Sink Writer, hardware H.264, 1080p, 8 Mbit/s | GPU |
| AI sharpening (optional) | QuickSRNet on the zoomed-in crop | NPU |

Measured on a Snapdragon X Plus with the camera live: **CPU ~1.5 %**, NPU ~12–15 % at 25–30
detections per second (4.5 ms each), first picture about 0.5 s after start.

## What it does

* **Auto-frame** (eye button): a person has to be seen in several detections in a row before the
  view locks on (no zooming at shadows), then pan and zoom follow them with a spring instead of
  jumping from box to box. Wider when they leave.
* **Small, distant people**: the 16:9 frame is detected as two overlapping *square* tiles at the
  model's native size, so a person far away is not squeezed to half their width first.
* **Exclusion zones** (square button): draw polygons over a street, trees or anything that should
  never trigger. Motion and people whose centre is inside a zone are ignored.
* **Recording** (● button): when auto-record is on, a clip is written as soon as someone passes —
  with 2.5 s from *before* they appeared (a GPU ring buffer) and until 5 s after they left.
* **Photo** (camera button).
* **Folder** (folder button): a menu that shows where clips and photos are saved, with
  **Open folder** and **Choose folder…** (for example another drive). One folder for everything;
  default `Videos\PulseX CCTV`.
* **AI sharpening** (✦, optional): QuickSRNet on the NPU for the zoomed-in view.
* **Live numbers** (ⓘ): decode, frame rate, NPU timings, detections — hidden unless you want them.

## Privacy

Everything stays on the laptop. The camera login is read from a local file and never logged; with
Media Foundation the password is handed to Windows through a credential manager instead of being
part of the URL that is opened.
Clips, photos and zones are local files; nothing is uploaded anywhere.

## Requirements

* Windows 11 on a Snapdragon X laptop (Hexagon V73 NPU).
* Visual Studio 2022 with the ARM64 C++ tools, CMake 3.21+.
* The **Qualcomm AI Runtime (QAIRT) SDK** — its headers to build, its HTP runtime to run
  (`QnnHtp.dll`, `QnnHtpV73Stub.dll`, `QnnSystem.dll`, `libQnnHtpV73Skel.so` and its signature
  catalog `libqnnhtpv73.cat`; the build copies them from your SDK next to the exe — without the
  `.cat` the NPU refuses the library and the app runs without detection). They are not part of this repository.
* A **YOLO detection model as a QNN context binary** for Snapdragon X (see below).
* An RTSP camera.

## Build

```bat
cmake -S . -B build -G "Visual Studio 17 2022" -A ARM64 -DQNN_SDK_ROOT=C:\path\to\qairt\2.47.0.260601
cmake --build build --config Release
```

## Models

Models are **not** included — they come with their own licences (the YOLO models are AGPL-3.0,
from Ultralytics). Download a precompiled QNN context for Snapdragon X from
[Qualcomm AI Hub](https://aihub.qualcomm.com/) — for example *YOLOv10-Detection*, runtime
*Qualcomm AI Engine Direct (QNN) context binary*, device *Snapdragon X Elite* — and put it in a
`models` folder next to the exe:

```
build\Release\pulsex_cctv.exe
build\Release\models\yolov10_det_qairt_context.bin
```

The detector reads the graph's inputs and outputs from the binary itself, so float and quantized
(u8/u16) contexts both work as long as the outputs are boxes `[1,8400,4]`, scores `[1,8400]` and
classes `[1,8400]`. The QAIRT runtime should be the version the context was compiled with, or newer.

AI sharpening needs a QuickSRNet-Large QNN context in `models\quicksrnetlarge_288x512_ctx_qnn.bin`: one 8-bit
image in, NCHW `1 × 3 × 288 × 512`, and one out, `1 × 3 × 1152 × 2048`, both with scale 1/255. The graph and tensor
names are read from the file. One way to make it: take *QuickSRNetLarge* (ONNX, w8a8) from Qualcomm AI Hub, set its
input to `1 × 3 × 288 × 512` (and re-infer the shapes), and open it once with ONNX Runtime's QNN execution provider
with `ep.context_enable=1` — the `_ctx_qnn.bin` it writes is the file. A context that does not fit is refused with the
reason in `%TEMP%\pulsecore_cctv_mf.log`, and without one the ✦ tooltip names the missing file.

## Camera

Put your camera's RTSP address in a one-line file `tapo_rtsp.txt` next to the exe (see
`tapo_rtsp.example.txt`), or set `PULSECORE_TAPO_RTSP`:

```
rtsp://USER:PASSWORD@CAMERA-IP:554/stream1
```

On a Tapo camera the user and password are the *camera account* made in the Tapo app
(Advanced settings → Camera account), not your TP-Link login.

If Media Foundation cannot open the stream, the app falls back to an `ffmpeg.exe` pipe, which costs a lot more
CPU. It uses `PULSECORE_FFMPEG=<path to ffmpeg.exe>` if set, else an `ffmpeg.exe` next to the app, else one on `PATH`.

## Folder names and text files

Paths may contain letters outside A–Z (Nordic, Polish, …): the app's folder, the save folder, the video path. `tapo_rtsp.txt`
and `tapo_mask.txt` can be saved as UTF-8 (with or without a BOM) or in the old Windows encoding - both are read
right, also a camera password with Nordic letters.

One exception comes from the NPU driver: it only loads its runtime (`QnnHtp.dll`, the V73 stub and skel) from a folder
whose path is plain A–Z. When the app sits in a folder with other letters (for example under
`C:\Users\<a name with Nordic letters>\Downloads`), it copies those files once to `C:\ProgramData\PulseX\npu\` and loads them
from there. To avoid the copy, keep the app in a folder such as `C:\PulseX CCTV`.

## Settings

`%APPDATA%\PulseX\cctv.ini`, written by the app:

| Key | Meaning |
|---|---|
| `save_dir` | folder for clips and photos (folder button → Choose folder…) |
| `auto_record` | 1 = record passers-by |
| `ignore_top` | without exclusion zones: ignore everything above this height (0–0.95 of the picture, 0 = off) |
| `camera_name` | optional: a name shown under the title, before "RTSP live" and the picture size (the camera's address is never shown) |

Exclusion zones are stored in `tapo_mask.txt` next to the exe.

When started as administrator, the app empties the standby memory list at start and exit
(like a "free memory" button); `PULSECORE_CCTV_PURGE=off` turns that off.

## For developers

Environment switches (all optional):

| Variable | Effect |
|---|---|
| `PULSECORE_CCTV_DECODER=ffmpeg` | use the ffmpeg pipe instead of Media Foundation |
| `PULSECORE_TAPO_RTSP=<file.mp4>` | play a video file instead of the camera (loops, paced by timestamps) |
| `PULSECORE_YOLO_CTX`, `PULSECORE_SR_CTX` | model paths |
| `PULSECORE_CCTV_TILES=0` | one squeezed 640 × 640 frame instead of two square tiles |
| `PULSECORE_CCTV_AUTOFRAME=0` | start with Auto-frame off |
| `PULSECORE_CCTV_INFO=1` | start with the live numbers shown |
| `PULSECORE_CCTV_BG=r,g,b` | background colour |
| `PULSECORE_CCTV_REC_DIR` | recording folder (overrides the setting) |
| `PULSECORE_CCTV_FAKE_PERSON=1` | a synthetic walker, for testing the following |
| `PULSECORE_CCTV_TRACE_FRAMING=<file>` | write the framing (pan/zoom per frame) to a CSV |
| `PULSECORE_CCTV_EXIT_S=<s>` | quit after s seconds (tests) |
| `PULSECORE_QNN_LOG=<file>` | QNN's own log (`PULSECORE_QNN_LOG_LEVEL=debug`…) |
| `PULSECORE_HTP_CORNER`, `PULSECORE_HTP_DCVS=1` | NPU power corner / dynamic clocking |
| `PULSECORE_YOLO_ZEROCOPY=0` | copy the NPU inputs/outputs instead of sharing buffers |

A timeline and a once-a-second receipt (decode, NPU, people) are written to
`%TEMP%\pulsecore_cctv_mf.log`.

## Licence

MIT for the code in `src/` and `include/pcore/`. Dear ImGui (MIT) and stb (public domain / MIT)
keep their own licences. Models and the Qualcomm runtime are not included and have their own terms.
