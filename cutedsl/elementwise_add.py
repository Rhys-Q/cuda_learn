import torch

from functools import partial
from typing import List

import cutlass
import cutlass.cute as cute

from cutlass.cute.runtime import from_dlpack


# Basic Kernel Implementation
# ---------------------------
# This is our first implementation of the elementwise add kernel.
# It follows a simple 1:1 mapping between threads and tensor elements.
@cute.kernel
def naive_elementwise_add_kernel(
    gA: cute.Tensor,
    gB: cute.Tensor,
    gC: cute.Tensor,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()
    
    thread_idx = bidx * bdim + tidx
    
    m, n = gA.shape
    
    ni = thread_idx % n
    mi = thread_idx // n
    
    a_val = gA[mi, ni]
    b_val = gB[mi, ni]
    
    gC[mi,ni] = a_val + b_val

@cute.jit
def naive_elementwise_add(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor,
):
    num_threads_per_block = 256
    
    m, n = mA.shape
    
    kernel = naive_elementwise_add_kernel(mA, mB, mC)
    
    kernel.launch(
        grid = (m*n // num_threads_per_block,1,1),
        block=(num_threads_per_block,1,1)
    )

M, N = 16384, 8192


a = torch.randn(M, N, device="cuda", dtype=torch.float16)  # Random input A
b = torch.randn(M, N, device="cuda", dtype=torch.float16)  # Random input B
c = torch.zeros(M, N, device="cuda", dtype=torch.float16)  # Output buffer
num_elements = sum([a.numel(), b.numel(), c.numel()])
a_ = from_dlpack(a, assumed_align=16)  # CuTe tensor A
b_ = from_dlpack(b, assumed_align=16)  # CuTe tensor B
c_ = from_dlpack(c, assumed_align=16)  # CuTe tensor C

naive_elementwise_add_ = cute.compile(naive_elementwise_add, a_, b_, c_)

naive_elementwise_add_(a_, b_, c_)
torch.testing.assert_close(c, a+b)

def benchmark(callable, a_, b_, c_):
    avg_time_us = cute.testing.benchmark(
        callable,
        kernel_arguments=cute.testing.JitArguments(a_, b_, c_),
        warmup_iterations=5,
        iterations=100,
    )

    # Calculate metrics
    # ----------------
    dtype = a_.element_type

    # Calculate total bytes transferred:
    # - 2 reads (A and B) + 1 write (C)
    # - Each element is dtype.width bits
    bytes_per_element = dtype.width // 8
    total_bytes = num_elements * bytes_per_element

    # Calculate achieved bandwidth
    achieved_bandwidth = total_bytes / (avg_time_us * 1000)  # GB/s

    # Print results
    # ------------
    print(f"Performance Metrics:")
    print(f"-------------------")
    print(f"Kernel execution time: {avg_time_us:.4f} us")
    print(f"Memory throughput: {achieved_bandwidth:.2f} GB/s")

benchmark(naive_elementwise_add_, a_, b_, c_)


@cute.kernel
def vectorized_elementwise_add_kernel(
    gA: cute.Tensor,
    gB: cute.Tensor,
    gC: cute.Tensor,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()
    
    thread_idx = bidx * bdim + tidx
    
    # Map thread index to logical index of input tensor in unit of vector
    m, n = gA.shape[1]
    ni = thread_idx % n
    mi = thread_idx // n
    
    a_val = gA[(None, (mi, ni))].load()
    b_val = gB[(None, (mi, ni))].load()
    
    gC[(None, (mi, ni))] = a_val + b_val

@cute.jit
def vectorized_elementwise_add(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):
    threads_per_block = 256
    print("[DSL INFO] Raw Tensors:")
    print(f"[DSL INFO]   mA = {mA}")
    print(f"[DSL INFO]   mB = {mB}")
    print(f"[DSL INFO]   mC = {mC}")
    # (16384,8192):(8192,1) zipped_divide (1, 8):(0, 1) = ((1,8),(16384,1024)):((0,1),(8192,8))
    # mode 0: (1,8):(0,1)
    # mode 1: (16384, 1024): (8192, 8)
    gA = cute.zipped_divide(mA, (1, 8))
    gB = cute.zipped_divide(mB, (1, 8))
    gC = cute.zipped_divide(mC, (1, 8))

    print("[DSL INFO] Tiled Tensors:")
    print(f"[DSL INFO]   gA = {gA}")
    print(f"[DSL INFO]   gB = {gB}")
    print(f"[DSL INFO]   gC = {gC}")

    vectorized_elementwise_add_kernel(gA, gB, gC).launch(
        grid=(cute.size(gC, mode=[1]) // threads_per_block, 1, 1),
        block=(threads_per_block, 1, 1),
    )


a = torch.randn(M, N, device="cuda", dtype=torch.float16)
b = torch.randn(M, N, device="cuda", dtype=torch.float16)
c = torch.zeros(M, N, device="cuda", dtype=torch.float16)

a_ = from_dlpack(a, assumed_align=16)
b_ = from_dlpack(b, assumed_align=16)
c_ = from_dlpack(c, assumed_align=16)

compiled_func = cute.compile(vectorized_elementwise_add, a_, b_, c_)
compiled_func(a_, b_, c_)

# verify correctness
torch.testing.assert_close(c, a + b)
benchmark(compiled_func, a_, b_, c_)

@cute.kernel
def elementwise_add_kernel(
    gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor, tv_layout: cute.Layout
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    
    blk_coord = ((None, None), bidx)
    
    blkA = gA[blk_coord]
    blkB = gB[blk_coord]
    blkC = gC[blk_coord]
    
    tidfrgA = cute.composition(blkA, tv_layout)
    tidfrgB = cute.composition(blkB, tv_layout)
    tidfrgC = cute.composition(blkC, tv_layout)
    
    print("Composed with TV layout:")
    print(f". tidfrgA: {tidfrgA.type}")
    
    thr_coord = (tidx, None)
    
    thrA = tidfrgA[thr_coord]
    thrB = tidfrgB[thr_coord]
    thrC = tidfrgC[thr_coord]
    
    thrC[None] = thrA.load() + thrB.load()

@cute.jit
def elementwise_add(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor,
):
    print("Raw Input Tensors:")
    print(f"[DSL INFO]   mA = {mA.type}")
    print(f"[DSL INFO]   mB = {mB.type}")
    print(f"[DSL INFO]   mC = {mC.type}")
    coalesced_ldst_bytes = 16
    assert all(t.element_type == mA.element_type for t in [mA, mB, mC])
    dtype = mA.element_type
    
    thr_layout = cute.make_ordered_layout((4, 64), order=(1, 0))
    val_layout = cute.make_ordered_layout((16, coalesced_ldst_bytes), order=(1, 0))
    val_layout = cute.recast_layout(dtype.width, 8, val_layout)
    tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)
    
    print(f"[DSL INFO] Tiler: {tiler_mn}")
    print(f"[DSL INFO] TV Layout: {tv_layout}")
    """
    Layout Shape : (M, N, L, ...)
    Tiler Shape  : <TileM, TileN>

    logical_divide : ((TileM,RestM), (TileN,RestN), L, ...)
    zipped_divide  : ((TileM,TileN), (RestM,RestN,L,...))
    tiled_divide   : ((TileM,TileN), RestM, RestN, L, ...)
    flat_divide    : (TileM, TileN, RestM, RestN, L, ...)
    """
    gA = cute.zipped_divide(mA, tiler_mn)
    gB = cute.zipped_divide(mB, tiler_mn)
    gC = cute.zipped_divide(mC, tiler_mn)
    print("Tiled Input Tensors:")
    print("[DSL INFO] Tiled Tensors:")
    print(f"[DSL INFO]   gA = {gA.type}")
    print(f"[DSL INFO]   gB = {gB.type}")
    print(f"[DSL INFO]   gC = {gC.type}")
    remap_block = cute.make_ordered_layout(cute.select(gA.shape[1], mode=[1, 0]), order=(1, 0))
    gA = cute.composition(gA, (None, remap_block))
    gB = cute.composition(gB, (None, remap_block))
    gC = cute.composition(gC, (None, remap_block))
    
    print("Tiled Input Tensors:")
    print("[DSL INFO] Tiled Tensors:")
    print(f"[DSL INFO]   gA = {gA.type}")
    print(f"[DSL INFO]   gB = {gB.type}")
    print(f"[DSL INFO]   gC = {gC.type}")
    
    elementwise_add_kernel(gA, gB, gC, tv_layout).launch(
        grid = [cute.size(gC, mode=[1]), 1, 1],
        block = [cute.size(tv_layout, mode=[0]), 1,1]
    )

a = torch.randn(M, N, device="cuda", dtype=torch.float16)
b = torch.randn(M, N, device="cuda", dtype=torch.float16)
c = torch.zeros(M, N, device="cuda", dtype=torch.float16)

a_ = from_dlpack(a, assumed_align=16)
b_ = from_dlpack(b, assumed_align=16)
c_ = from_dlpack(c, assumed_align=16)

elementwise_add_ = cute.compile(elementwise_add, a_, b_, c_)
elementwise_add_(a_, b_, c_)

# verify correctness
torch.testing.assert_close(c, a + b)
benchmark(elementwise_add_, a_, b_, c_)