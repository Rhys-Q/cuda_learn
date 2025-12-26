
# CuTe DSL Ampere Tensor Core GEMM 实战拆解

# 1 为什么要写这篇博客
## 现有资料的问题
## 本文目标
## 读者假设


# cute 基础
## layout
## layout composition
## layout partition
## layout local_tile
## Copy 抽象
## MMA 抽象

# ampere gemm 自顶向下流程
## gemm规格描述
## kernel grid、block大小
## grid rasterization
## ctx 层
## serial k
## main loop
## 双重 pipeline parallelism
## global memorey -> shared memory: cp.async
## shared memory -> register: ldmatrix
## shared memory bank conflict: swizzle
## mma
## predict
## epilogue
## register -> shared memory
## shared memory -> global memory

# QA
## mma
### tiled mma的permutation_mnk参数的含义是什么？为什么要设置为这个大小？可以进行性能调优吗？
### tile mma 和mma指令是怎样对应的？tiled mma的工作原理?
### 一个 tiled_mma 内部到底会发射多少条 mma？这些 mma 是串行发射，还是 warp 内可并行？CuTe 是否会重排 mma 发射顺序？
### mma 的指令吞吐和 latency 是如何被 pipeline 隐藏的？单条 mma 的 latency 大约是多少？一个 warp 连续发射 mma 是否会 stall？需要多少 independent mma 才能吃满 Tensor Core？




## cp.async
### 为什么copy bits是128 bit？
### cp.async 是否一定比普通 ld/st 快？在什么情况下 cp.async 反而不划算？小 tile / 小 K 是否值得用 cp.async？L2 hit 时 cp.async 还有优势吗？
### cp.async 的 stage 数量如何选择？stage 多了为什么会“变慢”？
### cp.async 和 L2 cache 的关系？cp.async 是否绕过 L1？L2 miss / hit 对 latency 的真实影响有多大？prefetch 到 L2 vs prefetch 到 smem 的权衡？


## ldmatrix
### ldmatrix具体的工作机制是什么？
### ldmatrix性能会比普通的copy快吗？快多少？
### 为什么 ldmatrix 必须是 warp-synchronous？如果只有部分 thread 执行会发生什么？这是 ISA 约束还是硬件实现约束？
### ldmatrix 能否被普通 ld 指令替代？从“功能上”是否可行？从“性能上”为什么不可行？哪些场景（非 Tensor Core）不需要 ldmatrix？
### ldmatrix 对 shared memory layout 的约束有哪些？哪些 layout 是“ldmatrix-friendly”的？哪些 layout 会直接导致指令非法？swizzle 在这里起的是“必要条件”还是“优化条件”？
### ldmatrix 是否会产生 bank conflict？如果会，硬件如何处理？如果不会，为什么？swizzle 对 ldmatrix 的影响是 compile-time 还是 runtime？


## swizzle
### swizzle机制具体是怎样的？如何理解？
### swizzle的原理？为什么要使用xor操作？
### swizzle可以100%避免bank conflict吗？
### swizzle 的设计目标到底是什么？是减少 bank conflict？还是让 ldmatrix 可用？还是让 warp 内访存更均匀？
### swizzle 会影响 global memory 访问吗？swizzle 只作用于 shared memory 吗？是否存在 gmem → smem → reg 三段 swizzle 不一致的问题？
### swizzle 是否有“最佳形态”？XOR 是经验规则，还是理论最优？是否存在针对特定 tile 的定制 swizzle？CuTe 是否允许用户自定义 swizzle？

## register->shared memorey
### 为什么不直接 reg → gmem？

## Layout / CuTe DSL 抽象
### CuTe 的 layout 抽象会不会引入额外开销？是否完全在 compile-time 消解？是否存在 runtime address computation？和手写 pointer arithmetic 相比是否等价？

### layout 的表达能力是否有限？是否能表达所有 CUTLASS layout？是否能表达非矩形、非连续 tile？哪些 layout 在 CuTe 中“写得出来但跑不快”？

### CuTe 在做 layout 推导时，会进行哪些隐式优化？是否会自动 fuse layout？是否会消除中间 layout？用户能否控制这些行为？


## Pipeline / 并行性
### 为什么是“双重 pipeline”，而不是单一 pipeline？如果只有 cp.async pipeline 会发生什么？如果只有 mma pipeline 会发生什么？两者的 bottleneck 如何判断？

### pipeline 深度是否存在“理论最优值”？是否与 GPU SM 数量有关？是否与 Tensor Core 数量有关？不同 GPU（A100 vs RTX 30）是否不同？

### pipeline 和 occupancy 的权衡如何做？更深 pipeline vs 更多 block？register / smem 压力如何量化？实战中优先牺牲哪一个？

## Epilogue / Store
### 为什么 epilogue 经常成为 GEMM 的瓶颈？compute-bound → memory-bound 的转折点?activation / bias / scaling 的代价? 和 main loop 相比，哪些优化是“值得做的”？
### epilogue 是否可以和下一层 GEMM fuse？在 CuTe / CUTLASS 中是否可行？受限于哪些因素？现实中为什么很少这么做？

## 架构相关（Ampere 特有）
### 这些设计在 Hopper 上还成立吗？cp.async vs TMA? ldmatrix 是否还存在？tiled_mma 是否发生变化？
### Ampere 的哪些限制“塑造了”今天的 CuTe 设计？warp size, bank 数量, Tensor Core shape？
