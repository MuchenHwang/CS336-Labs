// file: vector_add_cuda.cu

#include <torch/extension.h>
#include <cuda_runtime.h>

// =======================================================
// 1. CUDA 内核实现
//    这是将在GPU上并行执行的底层计算逻辑。
// =======================================================
__global__ void vector_add_kernel(const float* a, const float* b, float* c, int n) {
    // 使用内置变量计算当前线程的全局唯一ID
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 边界检查，防止线程访问越界内存
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// =======================================================
// 2. C++ 包装函数 (主机端代码)
//    这个函数负责连接PyTorch和CUDA内核。
// =======================================================
torch::Tensor vector_add(torch::Tensor a, torch::Tensor b) {
    // 检查输入张量是否在CUDA设备上
    TORCH_CHECK(a.is_cuda(), "Input tensor 'a' must be on a CUDA device");
    TORCH_CHECK(b.is_cuda(), "Input tensor 'b' must be on a CUDA device");

    // 检查输入张量的形状和数据类型
    TORCH_CHECK(a.sizes() == b.sizes(), "Input tensors must have the same shape");
    TORCH_CHECK(a.scalar_type() == torch::kFloat32, "Input tensors must be of type float32");

    // 创建一个与输入形状和类型相同的输出张量
    auto c = torch::empty_like(a);
    int n = a.numel(); // 获取元素总数

    // 计算CUDA内核的启动配置
    const int threads_per_block = 256;
    const int blocks_per_grid = (n + threads_per_block - 1) / threads_per_block;

    // ★ 关键：调用CUDA内核。
    // 因为这段代码在.cu文件中，NVCC编译器能正确解析<<<...>>>语法。
    vector_add_kernel<<<blocks_per_grid, threads_per_block>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        c.data_ptr<float>(),
        n
    );

    return c;
}

// =======================================================
// 3. 模块定义 (绑定到Python)
//    使用PYBIND11将C++函数暴露给Python。
// =======================================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
      "forward", // 在Python中调用的函数名
      &vector_add, // 指向要调用的C++函数
      "Vector Add (CUDA implementation)" // 函数的文档字符串
    );
}
