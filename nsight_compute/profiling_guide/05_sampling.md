ncu可以以固定时间间隔来进行指标采样。
# 1 PM Sampling
ncu可以通过周期性采样Performance monitors(PM)来收集指标。
PM指标特点：
- 指标名称前缀为pmsampling
- 指标名称包含一个可用的Triage group。
- 指标在一个section的timeline中。

# 2 Warp Sampling

ncu支持周期性地采样warp program counter和warp scheduler state。它隔固定周期，从sm上active warp中随机选择一个warp，统计它的状态。
