# 1 Overview
![alt text](../../media/images/image-39.png)

这张图有两条roofline。包含下列组件：
- Vectical Axis：纵坐标表示算力，单位是FLOPS。处于可视化的目的，图中是log。
- Horizontal Axis：横坐标表示计算强度，单位是FLOPS/byte。
- Memory Bandwidth Boundary：表示内存带宽，是图中直线的斜率。
- Peak Performance Boundary：表示最大算力，是图片平着的部分。图中有两条平着的线。不知道啥意思。
- Ridge Point：算力和内存带宽的平衡点。
- Achieved Value：表示这个kernel达成的情况。

# 2 Analysis

可以分析是memory bound还是compute bound，以及优化空间。