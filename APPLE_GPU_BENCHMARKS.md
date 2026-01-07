# Apple MPS Benchmarks

System: Apple MacBook Pro (M4 Pro 12c CPU / 16c GPU, 24GB unified memory)

Speedup is relative to CPU (OpenBLAS) for the same config/video. CPU baselines are from the last OpenBLAS run; yolov4 CPU numbers were not run.

Output files follow the pattern: `<video>_<config>_<backend>.m4v` (for example `video1_LegoGears_mps_postproc.m4v`, `video1_LegoGears_mps.m4v`, `video1_LegoGears_openblas.m4v`, `video1_LegoGears_cpu_only.m4v`).

## LegoGears videos

| Config | Video | GPU (MPS postproc) FPS (x) | GPU (MPS no postproc) FPS (x) | CPU (OpenBLAS) FPS (x) |
| --- | --- | --- | --- | --- |
| LegoGears.cfg | video1.MOV | 336.207 (5.23x) | 663.265 (10.32x) | 64.303 (1.00x) |
| LegoGears.cfg | video2.MOV | 337.171 (5.19x) | 708.117 (10.90x) | 64.993 (1.00x) |
| yolov4-tiny.cfg | video1.MOV | 295.679 (11.90x) | 334.190 (13.44x) | 24.857 (1.00x) |
| yolov4-tiny.cfg | video2.MOV | 293.346 (11.32x) | 343.384 (13.25x) | 25.920 (1.00x) |
| yolov4.cfg | video1.MOV | 24.135 (n/a) | 23.880 (n/a) | n/a |
| yolov4.cfg | video2.MOV | 24.432 (n/a) | 23.909 (n/a) | n/a |

## Big Buck Bunny

| Config | Video | GPU (MPS postproc) FPS (x) | GPU (MPS no postproc) FPS (x) | CPU (OpenBLAS) FPS (x) |
| --- | --- | --- | --- | --- |
| yolov4-tiny.cfg | big_buck_bunny_720p.mov | 49.137 (2.02x) | 317.765 (13.05x) | 24.350 (1.00x) |
