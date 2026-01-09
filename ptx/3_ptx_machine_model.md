## 3.1 A Set of SIMT Multiprocessors
nv gpu是建立在SMs集合上的。当用户的kernel grid被执行时，grid中的block会被分发到SMs上执行。一个block内所有的thread，都会在同一个SM上执行。当一个block执行中断时，会让出SM的执行权限，其他block可以在这个SM上执行。


一个multiprocessor的组成包括：
- 多个Scalar Processor（SP） cores
- 一个multithreaded instruction unit
- on-chip shared memory

SM 负责创建、管理并执行并行的线程，并且可以实现context切换zero scheduling overhead。
此外，SM还实现了single-instruction barrier sync，这使得SM可以在一个指令周期内完成多个线程的同步。

为了管理上百个线程，同时运行多个程序，multiprocessor采用了SIMT架构（Single-instruction，multiple-thread）。multiprocessor会将每个thread映射到一个SP core上执行。每个SP core会独立执行线程，有自己的指令地址和寄存器状态。
multiprocess 以warp为单位来创建、管理、调度、执行线程，每个warp包含32个thread。

每个multiprocessor在执行block的时候，会将block拆分为多个warp。block拆分为warp的规则是固定且相同的，每个warp包含连续的thread，它们的thread id是递增的，第一个warp的第一个thread的thread id是0。

![alt text](../media/images/image-23.png)

每条指令"issue time" 发射周期，SIMT单元选择一个已经ready的warp，并发射这个warp的下一条指令。每次会执行warp中的一条公共指令，所以当warp中的thread没有分叉时，效率最高。当有分叉时，warp会串行所有分支指令，直到所有分支都执行完毕。每个分支执行时，会disable掉不在这个分支的thread。

之前这样写是ok的。但是在volta以及后续架构中，这样写就不行了，thread0-15和thread16-31在if语句分支不是同步的。
``` python
// warp-synchronous code（老写法）
if (lane < 16)
    smem[lane] += smem[lane + 16];
// ❌ 没有 __syncwarp()
```
volta之后，正确姿势：
``` python
if (lane < 16) {
    smem[lane] += smem[lane + 16];
}
__syncwarp();

```

## 3.3 On-chip Shared Memory
从上图4中可以看到，每个multiprocessor有四种类型的on-chip memory：
- 32-bit的寄存器，每个thread私有的
- parallel data cache或者shared memory，是所有salar processor所共享的
- 一个read-only constant cache，它是被所有scalar processor共享的，用于加速constant memory space的读取，这是一个read only区域
- 一个read-only texture cache，它也是共享的，加速texture memory的读取，也是read-only的。

