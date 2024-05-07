# Optimized ops in Kelvin TFLM

The following table is a list of currently optimized ops in Kelvin TFLM. The
relevant source code can be found located [here](https://opensecura.googlesource.com/sw/kelvin/+/refs/heads/master/tflm/opt).

## Non-Convolutional Ops

| Op              | Supported Data Type | Comments                                  |
| :-------------- | :-----------------: | :---------------------------------------- |
| Elementwise Add | s8, s16, s32        | Rescaling with offset and shift, clamping |
| Leaky ReLU      | s8, s16             |                                           |
| Max Pooling     | s8                  |                                           |

## Convolutional Ops

| Op               | Weights | Activation | Bias | Comments                                |
| :--------------- | :-----: | :--------: | :--: | :-------------------------------------- |
| Depthwise Conv2d | s8      |     s16    | s64  | filter size 3x1                         |
| Depthwise Conv2d | s8      |     s8     | s64  | output depth % 32 == 0                  |
| Conv2d           | s8      |     s16    | s32  |                                         |
| Conv2d           | s8      |     s16    | s64  | filter size 1x1, filter depth % 32 == 0 |
| Conv2d           | s8      |     s16    | s64  | filter size 1xn, grouped or ungroups    |
| Conv2d           | s8      |     s8     | s32  | filter size 1x1, output depth % 8 == 0  |
| Conv2d           | s8      |     s8     | s32  | filter depth % 32 == 0                  |
| Conv2d           | s8      |     s8     | s32  | filter shape  == (48x3x1x48)            |
