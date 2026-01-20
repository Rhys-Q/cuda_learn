为了让profile结果可复现，ncu有多种方式，包括串行化kernel launch，清空gpu cache，调整gpu时钟。

# 1 Serialization
如果你的app有多个kernel，ncu会将这些kernel串行化执行，即便它们原本是并行执行的。即便是多个进程中的kernel，也会被串行化。
这是因为硬件计数器是“独占资源”不能被多个进程同时读，如果两个进程同时 profiling，计数器会混在一起，根本无法区分是哪个 kernel 贡献的。

ncu通过文件锁来实现互斥，每个 CUDA device 或 MIG instance 都有一个唯一 UUID，Nsight Compute 在临时目录创建锁文件。如果某个进程拿到了这个锁，其他进程 profiling 同一个 device → 必须等待。

同时，如果对串行要求更高，ncu提供全局锁的选项。NV_COMPUTE_PROFILER_DISABLE_CONCURRENT_PROFILING，开启后：所有 device 共享一个锁，连不同 GPU 都不能并发 profiling，适合CI / 自动化环境，避免多个 profiling 作业互相干扰。

# 2 Clock Control
许多指标的值，直接受到SM、memory的时钟频率影响。如果在profile前，时钟频率被之前的kernel调高了，那么这些指标测量就会失真。同样地，如果一个kernel是app的第一个kernel，则频率会低一些。
为了解决这些问题，ncu会限制时钟频率到一个基础值。但这种做法也会有负面效果，比如用户自己想控制时钟频率，这种情况下，ncu支持--clock-control选项来控制时钟。

# 3 Cache Control
ncu默认情况下，会在每个pass运行前清空cache。这个行为也可以通过--cache-control none来取消掉。

# 4 Persistence Mode
GPU并不是随时在线的，在第一次有程序访问GPU时，驱动会被加载，GPU会进行初始化。这一切是对用户透明的，但是是有代价的。

此外，当结束对GPU的访问后，GPU会销毁，这也是有开销的。

所以建议让GPU Persistence Mode保持开启状态，这样可以避免每次profile都要初始化GPU的开销。
``` shell
nvidia-smi -pm 1
```

