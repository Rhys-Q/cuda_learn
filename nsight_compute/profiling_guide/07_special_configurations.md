# 1 Multi Instance GPU

MIG是一种将一个GPU分为多个GPU实例的技术。每个GPU实例有自己的资源，比如SM数量、memory数量等。
每个GPU实例又进一步可分为多个compute 实例。每个compute实例有自己的SM数量，但是共享memory。

因此在对compute进行profile时，就需要对memory的行为进行配置：memory隔离还是共享。

如果是隔离，那么这个compute实例会独占这个GPU实例的全部资源。如果是共享，那么对共享资源的profile会失败。

## Locking Clocks
ncu不能设置MIG compute 实例的时钟频率。如果你有权限，可以自行设置。

## MIG on NVIDIA vGPU

vGPU profile 会受到其他vm的影响。

# 2 CUDA Green Contexts
这是很新的功能。是一种对gpu进行划分的技术。可以将一个资源子集分配给一个cuda context。后面再看吧。

# 3 Multi-Process Service
略

