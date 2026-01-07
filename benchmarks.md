# Benchmarks (Apple MPS vs CPU OpenBLAS)

This document captures how to reproduce the video benchmarks and naming/overlay conventions.

## Assumptions
- You have the `.cfg` and matching `.weights` files available (same directory is easiest).
- Input videos (for example `video1.MOV`, `video2.MOV`) are available locally.
- You will run the multithreaded video example: `darknet_05_process_videos_multithreaded`.

## Builds

### GPU (Apple MPS)
```
cmake -S . -B build_mps -DDARKNET_TRY_MPS=ON
cmake --build build_mps
```

### CPU (OpenBLAS)
```
cmake -S . -B build_cpu_openblas -DDARKNET_TRY_MPS=OFF -DDARKNET_TRY_OPENBLAS=ON
cmake --build build_cpu_openblas
```

## Output Naming and Overlay
The video example supports optional environment variables:
- `DARKNET_BENCH_LABEL`: overlay title string (for example `GPU - Apple MPS`)
- `DARKNET_BENCH_SUFFIX`: filename suffix (for example `mps`, `openblas`)

Output filename pattern:
`<video>_<config>_<suffix>.m4v`

Example: `video1_LegoGears_mps.m4v`

## Example Commands
Run the same config/video pair against both builds.

GPU (MPS):
```
DARKNET_BENCH_LABEL="GPU - Apple MPS" DARKNET_BENCH_SUFFIX=mps \
  build_mps/src-examples/darknet_05_process_videos_multithreaded LegoGears.cfg video1.MOV
```

CPU (OpenBLAS):
```
DARKNET_BENCH_LABEL="CPU - OpenBLAS" DARKNET_BENCH_SUFFIX=openblas \
  build_cpu_openblas/src-examples/darknet_05_process_videos_multithreaded LegoGears.cfg video1.MOV
```

Repeat the same pattern for `video2.MOV` and for each config you want to benchmark.

## Capturing FPS
The tool prints a summary line:
`-> processed frame rate ..... <FPS> FPS`

Use that value when updating `APPLE_GPU_BENCHMARKS.md`.
