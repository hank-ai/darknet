# Metal Kernel Registry

This file documents the Metal compute kernels embedded in `src-lib/metal_backend.mm`.

All kernels are compiled at runtime from the embedded source and dispatched via `metal_dispatch_*` helpers.

## Kernels

### swish_kernel
- Textures:
  - `texture(0)`: input/output `texture2d_array<float>` (in place)
- Buffers: none
- Bytes: none

### mish_kernel
- Textures:
  - `texture(0)`: input/output `texture2d_array<float>` (in place)
- Buffers: none
- Bytes: none

### hard_mish_kernel
- Textures:
  - `texture(0)`: input/output `texture2d_array<float>` (in place)
- Buffers: none
- Bytes: none

### upsample_kernel
- Textures:
  - `texture(0)`: input `texture2d_array<float>`
  - `texture(1)`: output `texture2d_array<float>`
- Buffers: none
- Bytes:
  - `buffer(0)`: `uint stride`
  - `buffer(1)`: `float scale`

### shortcut_weighted_kernel
- Textures:
  - `texture(0)`: input `texture2d_array<float>`
  - `texture(1)`: addend `texture2d_array<float>`
  - `texture(2)`: output `texture2d_array<float>`
- Buffers: none
- Bytes:
  - `buffer(0)`: `float w_in`
  - `buffer(1)`: `float w_add`

### shortcut_weighted_relu_kernel
- Textures:
  - `texture(0)`: input `texture2d_array<float>`
  - `texture(1)`: addend `texture2d_array<float>`
  - `texture(2)`: output `texture2d_array<float>`
- Buffers: none
- Bytes:
  - `buffer(0)`: `float w_in`
  - `buffer(1)`: `float w_add`

### shortcut_weighted_softmax_kernel
- Textures:
  - `texture(0)`: input `texture2d_array<float>`
  - `texture(1)`: addend `texture2d_array<float>`
  - `texture(2)`: output `texture2d_array<float>`
- Buffers: none
- Bytes:
  - `buffer(0)`: `float w_in`
  - `buffer(1)`: `float w_add`

### shortcut_per_channel_kernel
- Textures:
  - `texture(0)`: input `texture2d_array<float>`
  - `texture(1)`: addend `texture2d_array<float>`
  - `texture(2)`: output `texture2d_array<float>`
- Buffers:
  - `buffer(0)`: `float* w_in` (length = channels)
  - `buffer(1)`: `float* w_add` (length = channels)
- Bytes:
  - `buffer(2)`: `uint channels`

### shortcut_per_channel_relu_kernel
- Textures:
  - `texture(0)`: input `texture2d_array<float>`
  - `texture(1)`: addend `texture2d_array<float>`
  - `texture(2)`: output `texture2d_array<float>`
- Buffers:
  - `buffer(0)`: `float* w_in` (length = channels)
  - `buffer(1)`: `float* w_add` (length = channels)
- Bytes:
  - `buffer(2)`: `uint channels`

### shortcut_per_channel_softmax_kernel
- Textures:
  - `texture(0)`: input `texture2d_array<float>`
  - `texture(1)`: addend `texture2d_array<float>`
  - `texture(2)`: output `texture2d_array<float>`
- Buffers:
  - `buffer(0)`: `float* w_in` (length = channels)
  - `buffer(1)`: `float* w_add` (length = channels)
- Bytes:
  - `buffer(2)`: `uint channels`

### activation kernels (elementwise)
Kernels:
`logistic_kernel`, `loggy_kernel`, `tanh_kernel`, `relu6_kernel`, `elu_kernel`, `selu_kernel`,
`gelu_kernel`, `relie_kernel`, `ramp_kernel`, `hardtan_kernel`, `lhtan_kernel`, `plse_kernel`,
`stair_kernel`, `revleaky_kernel`
- Textures:
  - `texture(0)`: input/output `texture2d_array<float>` (in place)
- Buffers: none
- Bytes: none

### route_clear_kernel
- Textures:
  - `texture(0)`: output `texture2d_array<float>` (in place)
- Buffers: none
- Bytes: none

### route_copy_kernel
- Textures:
  - `texture(0)`: input `texture2d_array<float>`
  - `texture(1)`: output `texture2d_array<float>` (read/write)
- Buffers: none
- Bytes:
  - `buffer(0)`: `uint in_channels_total`
  - `buffer(1)`: `uint out_channels_total`
  - `buffer(2)`: `uint in_channel_offset`
  - `buffer(3)`: `uint out_channel_offset`
  - `buffer(4)`: `uint copy_channels`

### reorg_forward_kernel
- Textures:
  - `texture(0)`: input `texture2d_array<float>`
  - `texture(1)`: output `texture2d_array<float>`
- Buffers: none
- Bytes:
  - `buffer(0)`: `uint in_channels`
  - `buffer(1)`: `uint out_channels`
  - `buffer(2)`: `uint stride`

### reorg_reverse_kernel
- Textures:
  - `texture(0)`: input `texture2d_array<float>`
  - `texture(1)`: output `texture2d_array<float>`
- Buffers: none
- Bytes:
  - `buffer(0)`: `uint in_channels`
  - `buffer(1)`: `uint out_channels`
  - `buffer(2)`: `uint stride`

### maxpool_kernel
- Textures:
  - `texture(0)`: input `texture2d_array<float>`
  - `texture(1)`: output `texture2d_array<float>`
- Buffers: none
- Bytes:
  - `buffer(0)`: `uint channels`
  - `buffer(1)`: `uint size`
  - `buffer(2)`: `uint stride_x`
  - `buffer(3)`: `uint stride_y`
  - `buffer(4)`: `int pad`

### avgpool_kernel
- Textures:
  - `texture(0)`: input `texture2d_array<float>`
  - `texture(1)`: output `texture2d_array<float>`
- Buffers: none
- Bytes:
  - `buffer(0)`: `uint channels`
  - `buffer(1)`: `uint size`
  - `buffer(2)`: `uint stride_x`
  - `buffer(3)`: `uint stride_y`
  - `buffer(4)`: `int pad`

### softmax_kernel
- Buffers:
  - `buffer(0)`: `float* input`
  - `buffer(1)`: `float* output`
- Bytes:
  - `buffer(2)`: `uint n`
  - `buffer(3)`: `uint batch`
  - `buffer(4)`: `uint batch_offset`
  - `buffer(5)`: `uint groups`
  - `buffer(6)`: `uint group_offset`
  - `buffer(7)`: `uint stride`
  - `buffer(8)`: `float temp`

### yolo_activate_kernel
- Buffers:
  - `buffer(0)`: `float* data` (in place)
- Bytes:
  - `buffer(1)`: `uint entries`
  - `buffer(2)`: `uint wh`
  - `buffer(3)`: `float scale_x_y`
  - `buffer(4)`: `float bias`
  - `buffer(5)`: `uint new_coords`
  - `buffer(6)`: `uint total`

### yolo_decode_boxes_kernel
- Buffers:
  - `buffer(0)`: `float* data`
  - `buffer(1)`: `float4* boxes`
  - `buffer(2)`: `float* biases`
  - `buffer(3)`: `int* mask`
- Bytes:
  - `buffer(4)`: `uint w`
  - `buffer(5)`: `uint h`
  - `buffer(6)`: `uint entries`
  - `buffer(7)`: `uint n`
  - `buffer(8)`: `uint batch`
  - `buffer(9)`: `uint outputs`
  - `buffer(10)`: `uint netw`
  - `buffer(11)`: `uint neth`
  - `buffer(12)`: `uint new_coords`

### yolo_candidates_kernel
- Buffers:
  - `buffer(0)`: `float* data`
  - `buffer(1)`: `uint* indices`
  - `buffer(2)`: `atomic_uint* count`
- Bytes:
  - `buffer(3)`: `uint w`
  - `buffer(4)`: `uint h`
  - `buffer(5)`: `uint entries`
  - `buffer(6)`: `uint n`
  - `buffer(7)`: `uint batch`
  - `buffer(8)`: `uint outputs`
  - `buffer(9)`: `float thresh`
  - `buffer(10)`: `uint max_candidates`

### nms_suppress_kernel
- Buffers:
  - `buffer(0)`: `float4* boxes`
  - `buffer(1)`: `float* scores`
  - `buffer(2)`: `uint* order`
- Bytes:
  - `buffer(3)`: `uint count`
  - `buffer(4)`: `uint base`
  - `buffer(5)`: `float thresh`

### buffer_scale
- Buffers:
  - `buffer(0)`: `float* data`
- Bytes:
  - `buffer(1)`: `float scale`
