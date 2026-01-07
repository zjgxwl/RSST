#include <torch/extension.h>
#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA 内核函数
__global__ void conv_skip_zero_channels_kernel(
    float* input, float* output, float* weights,
    float* mask,  // 掩码参数
    int N, int C, int H, int W, int K, int stride, int padding, int output_H, int output_W
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * C * output_H * output_W) return;  // 防止越界

    int n = idx / (C * output_H * output_W);  // 批量索引
    int c = (idx % (C * output_H * output_W)) / (output_H * output_W);  // 通道索引
    int h = (idx % (output_H * output_W)) / output_W; // 高度索引
    int w = (idx % output_W);  // 宽度索引

    // 通过掩码直接跳过全零通道
    if (mask[c] == 0) {
        return;  // 跳过全零通道
    }

    // 计算卷积
    int output_index = (n * C + c) * output_H * output_W + h * output_W + w;
    output[output_index] = 0;

    for (int k = 0; k < K; ++k) {
        for (int j = 0; j < K; ++j) {
            int input_x = w * stride - padding + j;  // 修改输入索引
            int input_y = h * stride - padding + k;  // 修改输入索引
            if (input_x >= 0 && input_x < W && input_y >= 0 && input_y < H) {
                int input_idx = (n * C + c) * H * W + input_y * W + input_x;
                int weight_idx = (c * K + k) * K + j;  // 修改权重索引
                output[output_index] += input[input_idx] * weights[weight_idx];
            }
        }
    }
}

// C++ 接口
void conv_skip_zero_channels(
    at::Tensor input, at::Tensor weights, at::Tensor output,
    at::Tensor mask,  // 掩码参数
    int stride, int padding
) {
    int N = input.size(0);  // 批量大小
    int C = input.size(1);  // 输入通道数
    int H = input.size(2);  // 输入高度
    int W = input.size(3);  // 输入宽度
    int K = weights.size(2);  // 卷积核大小
    int output_H = (H - K + 2 * padding) / stride + 1;  // 输出高度
    int output_W = (W - K + 2 * padding) / stride + 1;  // 输出宽度

    int total_threads = N * C * output_H * output_W;
    int block_size = 256;  // 每个块的线程数量
    int num_blocks = (total_threads + block_size - 1) / block_size;

    // 启动内核
    conv_skip_zero_channels_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), weights.data_ptr<float>(),
        mask.data_ptr<float>(),  // 传递掩码
        N, C, H, W, K, stride, padding, output_H, output_W
    );
}

// PyTorch 扩展初始化
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv_skip_zero_channels", &conv_skip_zero_channels, "Skip zero channels during convolution");
}
