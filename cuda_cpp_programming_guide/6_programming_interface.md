# 6.1 Compailation with NVCC
可以用cuda或者ptx来编写kernel，然后用nvcc编译。

## 6.1.1 编译流程
### 6.1.1.1 Offline Compailation

nvcc会先将device code和host code分开，然后：

- 将device code编译为汇编格式，即PTX code，或者二进制格式（cubin）。
- 修改host code，将<<<...>>>部分的代码替换为kernel，并添加必要的cuda runtime函数调用来加载并启动每个kernel。这里的kernel是指上面生成的ptx或者cubin。

### 6.1.1.2 Just-in-Time 编译
在运行时实时地将ptx代码编译为二进制格式（cubin），并加载到GPU上运行，这就是jit编译。
NVRTC 可用于将cuda 代码编译为ptx代码。
## 6.1.2 Binary Compatibility
cuda编译生成的二进制文件不是通用的。
In other words, a cubin object generated for compute capability X.y will only execute on devices of compute capability X.z where z≥y.

## 6.1.3 PTX Compatibility
ptx的部分指令是只针对部分架构的。比如warp shuffle functions，只支持sm 50以上。
# 6.2 CUDA Runtime
这里内容太多了，只适合按需进行阅读。