PTX, Parallel Thread Execution ISA
ISA, Instruction Set Architecture

# 1 Introduction
本文描述PTX，一个low level的parallel thread execution virtual machine。在PTX的描述中，GPU是一个data-parallel的计算设备。

## 1.1 Scalable Data-Parallel Computing using GPUs

介绍了DP，PTX定义了一个虚拟机和一个指令集架构，来支持通用的parallel thread execution。PTX Program可以被翻译为硬件上的指令，这些指令可以在GPU上执行。

## 1.2 Goals of PTX

PTX为通用并行编程提供了一个稳定的编程模型和指令集，它的设计目标如下：
- 在多个GPU架构迭代中，提供一个稳定的ISA。
- 追求高性能。
- 为C/C++等高级语言提供一个机器独立的ISA。
- 为开发者提供一个code distribution ISA。
- 为代码生成以及翻译提供一个通用的、源码层面的ISA，将PTX映射为目标机器代码。
- 为手写高性能kernel提供一个指令集。
- 为一个GPU至多个GPU单元提供一个可伸缩的编程模型。

## 1.3 PTX ISA Version 9.1
略

## 1.4 Document Structure
- Programming Model 介绍编程模型
- PTX Machine Model 介绍PTY虚拟机
- Syntax 介绍PTX语法
- State Space，Types，and Variables 描述了PTX的状态空间、类型系统和变量。
- Instructions Operands 介绍指令操作
- Abstracting the ABI 介绍函数语法、调用规范，以及ABI。
- Instruction Set 描述了PTX的指令集。
- Special Registers 介绍了PTX的特殊寄存器。
- Directives 介绍PTX支持的二进制指令
- Release Note提供了PTX的版本更新信息。
