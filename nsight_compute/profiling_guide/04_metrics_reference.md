# 1 Overview

可以使用ncu --query-metrics来查询指标，比如

```
ncu --query-metrics
```

就不展开了，后续遇到再看吧，这里将每一类参数过一下。


- Launch Metrics。启动参数，比如block size，grid size这些。
- Occupancy Metrics。理论占用情况。
- NVLINK Topology Metrics。NVLINK拓扑指标。
- NUMA Topology Metrics。NUMA拓扑指标。
- Device Attributes。这是硬件的属性，比如SM数量，架构代号等。
- Warp Stall Reasons。这是warp相关指标，使用warp scheduler state sampling来收集的。比如一个cycle，warp scheduler是否发射指令。
- Warp Stall Reasons(Not Issued)。仅在warp scheduler不发射指令的时候进行计数。
- Source Metrics：对SASS指令进行分析的指标，比如分支情况等。
- L2 Cache Eviction Metrics。L2 cache驱逐相关指令。
- Instruction Per Opcode Metrics。使用SASS-patching收集的指令。
- SASS Unit-Level Instructions Executed Metrics。指令执行次数。
- Metric Groups。
- Profiler Metrics。profile过程中收集的一些指标。
