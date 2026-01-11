``` python
.version 8.7
.target sm_86
.address_size 64

	// .globl	vector_add_tilelang_kernel_kernel
.global .align 1 .b8 _ZN45_INTERNAL_f420aea2_14_tvm_kernels_cu_2d4dc2494cute7productE[1];
.global .align 1 .b8 _ZN45_INTERNAL_f420aea2_14_tvm_kernels_cu_2d4dc2494cute1_E[1];
.global .align 1 .b8 _ZN45_INTERNAL_f420aea2_14_tvm_kernels_cu_2d4dc2494cuda3std3__45__cpo5beginE[1];
.global .align 1 .b8 _ZN45_INTERNAL_f420aea2_14_tvm_kernels_cu_2d4dc2494cuda3std3__45__cpo3endE[1];
.global .align 1 .b8 _ZN45_INTERNAL_f420aea2_14_tvm_kernels_cu_2d4dc2494cuda3std3__45__cpo6cbeginE[1];
.global .align 1 .b8 _ZN45_INTERNAL_f420aea2_14_tvm_kernels_cu_2d4dc2494cuda3std3__45__cpo4cendE[1];
.global .align 1 .b8 _ZN45_INTERNAL_f420aea2_14_tvm_kernels_cu_2d4dc2494cuda3std3__419piecewise_constructE[1];
.global .align 1 .b8 _ZN45_INTERNAL_f420aea2_14_tvm_kernels_cu_2d4dc2494cuda3std3__48in_placeE[1];
.global .align 1 .b8 _ZN45_INTERNAL_f420aea2_14_tvm_kernels_cu_2d4dc2494cuda3std6ranges3__45__cpo4swapE[1];
.global .align 1 .b8 _ZN45_INTERNAL_f420aea2_14_tvm_kernels_cu_2d4dc2494cuda3std6ranges3__45__cpo9iter_moveE[1];

.visible .entry vector_add_tilelang_kernel_kernel(
	.param .u64 vector_add_tilelang_kernel_kernel_param_0,
	.param .u64 vector_add_tilelang_kernel_kernel_param_1,
	.param .u64 vector_add_tilelang_kernel_kernel_param_2
)
.maxntid 128, 1, 1
.minnctapersm 1
{
	.reg .f32 	%f<4>;
	.reg .b32 	%r<5>;
	.reg .b64 	%rd<11>;
	.loc	1 12 0


	ld.param.u64 	%rd1, [vector_add_tilelang_kernel_kernel_param_0];
	ld.param.u64 	%rd2, [vector_add_tilelang_kernel_kernel_param_1];
	ld.param.u64 	%rd3, [vector_add_tilelang_kernel_kernel_param_2];
	.loc	1 13 3
	cvta.to.global.u64 	%rd4, %rd3;
	cvta.to.global.u64 	%rd5, %rd2;
	cvta.to.global.u64 	%rd6, %rd1;
	mov.u32 	%r1, %ctaid.x;
	shl.b32 	%r2, %r1, 7;
	mov.u32 	%r3, %tid.x;
	add.s32 	%r4, %r2, %r3;
	mul.wide.s32 	%rd7, %r4, 4;
	add.s64 	%rd8, %rd6, %rd7;
	ld.global.nc.f32 	%f1, [%rd8];
	add.s64 	%rd9, %rd5, %rd7;
	ld.global.nc.f32 	%f2, [%rd9];
	add.f32 	%f3, %f1, %f2;
	add.s64 	%rd10, %rd4, %rd7;
	st.global.f32 	[%rd10], %f3;
	.loc	1 14 1
	ret;

}
```


# 解读
- .global .align 1 .b8 _ZN45_INTERNAL_f420aea2_14_tvm_kernels_cu_2d4dc2494cute7productE[1]; .global 声明这个变量在global memory。.align 1 声明这个变量的对齐方式为1字节。.b8 声明这个变量的类型为8bit。总结，这一行声明了一个global memory变量_ZNXX ，它是长度为1的list，每个元素都是8bit。
- .minnctapersm 1：表示每个SM只少能放下一个block，这是一个约束，表示cta必须要能够放到sm上，否则这个kernel非法，编译会报错。
- .loc	1 12 0：表示下一条指令是在源文件1的第12行第0列，用于定位指令在源文件中的位置。
- cvta.to.global.u64 	%rd4, %rd3; 类似于指针cast，将%rd3转换为global memory的指针，存储到%rd4中。%rd3本身是一个通用类型指针。
- shl.b32 	%r2, %r1, 7; shl是左移指令，.b32表示操作数类型为32bit，这一行表示将%r1数值左移7位，存储到%r2中。相当于%r2 = %r1 * 128。
- ld.global.nc.f32 	%f1, [%rd8]; L2 Cache pass load，不使用L2cache，这是性能好的主要原因。
