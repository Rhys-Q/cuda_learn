
# 3.1 使用GPU的好处

GPU: Graphics Processing Unit

![alt text](../media/images/image.png)

GPU的结构要简单不少。DRAM是指图形双倍数据率随机存取存储器。速度一般，是用于Global Memory。

# 3.2 弹性编程模型
![alt text](../media/images/image-1.png)
cuda编程会生成一系列block，也叫ctx，每个block会被调度到一个SM上进行运行。