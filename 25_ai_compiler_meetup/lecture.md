# TVM FFI

framework: torch

specialized library: xxx

compiler dsl : tilelang

多个系统需要交互：

tensor

交互诉求：

- ABI：Application Binary Interface
app如何和memory做交互

- FFI: Foreign Function Interface
cross-language communicaton, python call c++

use C ABI
c++ 不是一个稳定的ABI

dlpack zero-copy

tvm-ffi 支持不同的python版本

支持inline kernel

tilelang + tvm-ffi: cpu overhead 优化, shape 检查、dim检查等

TVM组件化

graph -> TBD

tensor -> TIR(X)

Runtime FFI -> TVM FFI

TVM vs MLIR


## QA
1. triton支持？
没有打入triton社区，所以不支持

2. MLIR兼容？
tvm-ffi兼容，但是graph、tir不兼容

3. distributed? 
暂不支持
torch 稳定的API






# TileRT

rollout

聚焦 First token time cost
理论速度 一个数量级提升
聚焦：编译、算子优化、调度、分布式、MTP

- kernel launch和计算达到同一个量级
- DRAM latency太大，us级别
- NCCL 通信开销大，all reduce 慢

## 针对kernel launch
cuda graph 无法做调度
PDL 

都不能解决问题

kernel fusion

硬件tensor core快了很多，带宽也大了很多。kernel 分布式导致每个kernel latency降低，导致kernel fusion的重要性起来了。
MegaKernel。


## 打破kernel 概念，直接调度tile任务
MTP
300%
计算 存储 通信

## QA
1. 吞吐 vs latency？
latency 优先，吞吐量不重要

2. 分布式，卡挂了怎么办？
换

3. 调度是动态的还是静态的？
静态的，编译期就做好了

4. 现有算子怎么办？
仅支持tilelang

5. 仅支持nv卡吗？
不一定，warp spec

6. runtime层面支持吗？
不支持，也不支持多用户

7. 跨SM同步
DRAM级别的同步，在这里进行了优化，1us。

8. 大batch
大batch，计算高，tilert收益不大

9. 开源计划
后面有计划开源

