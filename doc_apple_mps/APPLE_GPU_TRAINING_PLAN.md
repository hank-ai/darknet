# Apple GPU Training Plan (Metal/MPS)

## Scope and Goals
- Target: enable end-to-end training on Apple Silicon with Metal/MPS.
- Focus on tiny models (YOLOv4-tiny and similar). Full YOLOv4 is out of scope.
- Focus on FP32 first; mixed precision is optional later.
- Training correctness: loss curve and mAP within acceptable tolerance vs CPU baseline.

## Stage 0: Define Training Scope
- Pick target configs for validation (e.g., `yolov4-tiny.cfg`).
- Define baseline CPU training run (epochs, batch size, seed).
- Decide acceptable parity thresholds (loss curve delta, mAP delta).

## Stage 1: Core GPU Runtime (Metal Backend)
- Build a minimal backend interface for:
  - Device selection + info.
  - Command queue and command buffers.
  - Buffer allocation/free and CPU↔GPU copies.
  - Kernel dispatch and synchronization.
- Implement Metal compute kernels for basic tensor ops:
  - `fill`, `copy`, `axpy`, `scal`, `add`, `mul`, `div`.
  - `exp`, `log`, `pow` (needed for activations + loss).
- Add RNG support for dropout and augmentation.

## Stage 2: Backward Pass Coverage (Critical Path)
- Convolution backward:
  - Grad w.r.t. input (col2im) and weights (im2col + GEMM).
  - Bias gradients.
- Batchnorm forward+backward (including running mean/variance).
- Activation backward:
  - ReLU, leaky, mish, swish, hard-mish.
- Pooling backward:
  - Maxpool and avgpool.
- Data movement layers:
  - Route, shortcut, upsample, reorg, and scale-channels gradients.

## Stage 3: Losses + Detection Layers
- YOLO loss forward/backward for all detection heads.
- Softmax / logistic loss on GPU.
- Bounding box loss gradients on GPU.

## Stage 4: Optimizers + Training Loop
- Optimizers:
  - SGD + momentum (first).
  - Adam (second).
- Implement weight decay and learning rate schedules.
- Decide whether updates happen on GPU buffers or via CPU fallback initially.

## Stage 5: Data Pipeline + Augmentations
- Keep data pipeline on CPU for the first pass.
- Copy batches to GPU memory.
- Optional later:
  - GPU resize/letterbox.
  - GPU color augmentations.

## Stage 6: Memory + Performance
- Activation caching strategy (keep only what backprop needs).
- Gradient checkpointing for large models.
- Memory reuse for intermediate tensors.
- Mixed-precision training (FP16 + loss scaling) once stable.

## Stage 7: Validation + Tooling
- Deterministic smoke test with fixed seed.
- Add training parity tests in `src-test/`.
- Benchmark throughput vs CPU.
- Capture peak memory + swap usage per run.

## Suggested First Milestone
- Train YOLOv4-tiny for a few epochs on GPU without crashes.
- Loss trend should match CPU within tolerance.
- No runaway memory growth during a full epoch.

## Files to Anchor Work (Likely)
- Backend runtime: new `src-lib/gpu_backend_*` files (Metal).
- GPU kernels: new Metal `.metal` sources or an existing kernel host.
- Training layers: `src-lib/*_layer.cpp` backward paths.
- Optimizers: `src-lib/update.c` / `src-lib/optimizer.cpp` (or equivalent).
- Tests: `src-test/` training parity tests.
