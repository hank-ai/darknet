# Apple GPU Parity Matrix (Inference vs CUDA/ROCm)

This table summarizes feature parity between the CUDA/ROCm GPU backends and the current Apple MPS/Metal inference path.

Legend:
- **Yes**: implemented and used in inference
- **Partial**: supported for a subset of cases
- **No**: CPU fallback

| Layer/Op | CUDA/ROCm | MPS/Metal (Inference) | Notes |
| --- | --- | --- | --- |
| GEMM | Yes | Yes | MPSMatrixMultiplication in `src-lib/gemm.cpp` |
| Convolution (forward) | Yes | Yes | MPSCNNConvolution (inference-only) |
| Convolution (backward) | Yes | No | Training not supported on MPS |
| Batchnorm (conv) | Yes | Yes | MPSCNNBatchNormalization |
| Max/Avg Pooling | Yes | Yes (partial) | MPSCNNPooling* for standard cases + Metal fallback for larger padding |
| Activations (ReLU/leaky/linear) | Yes | Yes | MPSCNNNeuron for conv/shortcut |
| Activations (swish/mish/hard-mish) | Yes | Yes | Metal kernels |
| Activations (logistic/tanh/elu/etc.) | Yes | Yes (partial) | Metal kernels for elementwise activations; normalize_channels still CPU |
| Route (concat) | Yes | Yes | Metal kernel supports grouped + non-4-aligned channels |
| Shortcut add (simple) | Yes | Yes | MPSCNNAdd |
| Shortcut add (weighted) | Yes | Yes (partial) | PER_FEATURE/PER_CHANNEL, NO/RELU/SOFTMAX normalization |
| Upsample (nearest) | Yes | Yes | Metal kernel |
| Reorg | Yes | Yes | Metal kernel (forward + reverse) |
| Softmax | Yes | Yes (partial) | Metal kernel for non-tree, non-spatial softmax |
| YOLO layer / post-processing | Yes | Partial | GPU activation + box decode; class probs + NMS still CPU |
| General training + optimizers | Yes | No | MPS inference-only |

## Runtime Coverage Tool
Enable coverage summary per layer type:
```
DARKNET_MPS_COVERAGE=1
```
This prints a one-time summary at exit showing how many layers used MPS vs CPU fallback for MPS-enabled layer types.

## Coverage Snapshot (LegoGears.cfg, video1.MOV)
- conv: 21 MPS, 0 CPU (100%)
- max: 3 MPS, 0 CPU (100%)
- route: 11 MPS, 0 CPU (100%)
- upsample: 1 MPS, 0 CPU (100%)

## Coverage Snapshot (yolov4.cfg, video1.MOV)
- conv: 110 MPS, 0 CPU (100%)
- max: 3 MPS, 0 CPU (100%)
- route: 21 MPS, 0 CPU (100%)
- shortcut: 23 MPS, 0 CPU (100%)
- upsample: 2 MPS, 0 CPU (100%)

## Plan A: Full Inference Parity (Prioritized)
Goal: eliminate CPU fallbacks for inference across common Darknet configs.

Priority order:
1. ~~**Softmax (CPU fallback)**~~  
   - ~~Add Metal kernel for softmax (channel-wise).~~  
   - ~~Wire into `softmax_layer.cpp` forward path when MPS is enabled.~~
2. ~~**Reorg (CPU fallback)**~~  
   - ~~Implement Metal kernel for reorg and use in `reorg_layer.cpp`.~~  
   - ~~Validate against CPU output for typical YOLO configs that still use it.~~
3. ~~**Route edge cases**~~  
   - ~~Grouped route support + 4-channel aligned concat~~  
   - ~~Extend route to handle non-4-channel-aligned cases.~~  
   - ~~Validate grouped route + concat for YOLO variants.~~
4. ~~**Pooling edge cases**~~  
   - ~~Max/avg pooling forward path on common padding/stride cases~~  
   - ~~Metal fallback for larger padding/stride combos.~~
5. ~~**Activation parity**~~  
   - ~~ReLU/leaky/linear via MPSCNNNeuron; swish/mish/hard-mish via Metal kernels~~  
   - ~~Added Metal kernels for remaining elementwise activations.~~

Exit criteria:
- Coverage summary shows 100% MPS for conv/max/route/shortcut/upsample/softmax/reorg on representative models.
- CPU fallbacks reduced to YOLO post-processing only.

## Plan B: GPU Post-Processing (Prioritized)
Goal: keep detection post-processing on GPU after inference to avoid CPU stalls.

Priority order:
1. **Decode outputs on GPU (partial)**  
   - ~~Add a Metal kernel to apply YOLO activation + scale (in-place on output buffer).~~  
   - ~~Add a Metal kernel to transform raw outputs into boxes (x/y/w/h).~~  
   - Keep CPU path as fallback for validation.  
   - Note: class probabilities + thresholding still run on CPU.
2. **Score thresholding + top-K (partial)**  
   - ~~GPU candidate filtering (objectness > thresh) to avoid CPU scanning.~~  
   - Optional cap via `DARKNET_MPS_POSTPROC_TOPK` (defaults to full parity).  
   - Validate output parity with CPU.
3. **NMS on GPU (partial)**  
   - Per-class greedy NMS via Metal (default NMS only).  
   - Enable with `DARKNET_MPS_POSTPROC=1`, CPU fallback remains.
4. **End-to-end GPU pipeline (partial)**  
   - GPU decode + candidate filtering + NMS wired into the YOLO postproc path.  
   - Debug compare mode: `DARKNET_MPS_POSTPROC=2` or `DARKNET_MPS_POSTPROC=compare`.
   - Single flag controls postproc stages; `DARKNET_MPS_POSTPROC=1` enables all GPU postproc steps.

Exit criteria:
- CPU is only used for final result formatting/logging.  
- Measured reduction in CPU time per frame with the same detection outputs.
