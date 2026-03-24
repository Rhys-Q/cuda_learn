import argparse
import math
from typing import Type

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass.cute.runtime import from_dlpack


"""
Learning demo for Ampere-style non-bulk ``cp.async`` in CuTe DSL.

This file intentionally keeps the problem small and regular so the core ideas
stay visible:

1. Build a ``cp.async`` GMEM -> SMEM copy atom with ``num_bits_per_copy=128``.
2. Give the copy a tiled thread/value layout.
3. Stage data in shared memory with an explicit pipeline dimension.
4. Use ``cp_async_commit_group`` and ``cp_async_wait_group`` to manage the
   asynchronous copy queue.
5. Reuse a stage buffer after a CTA-wide synchronization.
6. Copy the staged tile back to global memory with a normal synchronous copy so
   the output can be validated easily.

What this demo does not try to teach yet:
- irregular tail predication
- ldmatrix / tensor core consumption
- swizzle tuning beyond the simplest useful explanation

Run example:

    export CUTE_DSL_ARCH=sm89
    /root/miniforge3/envs/py312/bin/python cutedsl/cp_async.py
"""


CACHE_MODE_MAP = {
    "always": cute.nvgpu.cpasync.LoadCacheMode.ALWAYS,
    "global": cute.nvgpu.cpasync.LoadCacheMode.GLOBAL,
    "streaming": cute.nvgpu.cpasync.LoadCacheMode.STREAMING,
    "last_use": cute.nvgpu.cpasync.LoadCacheMode.LAST_USE,
    "none": cute.nvgpu.cpasync.LoadCacheMode.NONE,
}


def make_row_major_mkl_tensor(
    m: int,
    k: int,
    l: int,
    dtype: Type[cutlass.Numeric],
    init: str,
):
    import torch
    import cutlass.torch as cutlass_torch

    shape = (l, m, k)
    torch_dtype = cutlass_torch.dtype(dtype)
    if init == "arange":
        torch_tensor = torch.arange(m * k * l, device="cuda", dtype=torch.int32)
        torch_tensor = torch_tensor.reshape(shape).to(dtype=torch_dtype)
    elif init == "zeros":
        torch_tensor = torch.zeros(shape, device="cuda", dtype=torch_dtype)
    else:
        torch_tensor = torch.empty(shape, device="cuda", dtype=torch.int32)
        torch_tensor.random_(-2, 3)
        torch_tensor = torch_tensor.to(dtype=torch_dtype)

    # (L, M, K) -> (M, K, L), K remains the contiguous mode of the logical
    # (M, K) matrix, so this is row-major over the matrix modes.
    torch_tensor = torch_tensor.permute(1, 2, 0)

    cute_tensor = (
        from_dlpack(torch_tensor, assumed_align=16)
        .mark_layout_dynamic(leading_dim=1)
        .mark_compact_shape_dynamic(
            mode=1,
            stride_order=(2, 0, 1),
            divisibility=(128 // dtype.width),
        )
    )
    return cute_tensor, torch_tensor


class CpAsyncTeachingDemo:
    def __init__(
        self,
        tile_m: int,
        tile_k: int,
        num_stages: int,
        copy_bits: int,
        cache_mode: cute.nvgpu.cpasync.LoadCacheMode,
        dtype: Type[cutlass.Numeric],
    ):
        self.tile_m = tile_m
        self.tile_k = tile_k
        self.num_stages = num_stages
        self.copy_bits = copy_bits
        self.cache_mode = cache_mode
        self.dtype = dtype

        self.copy_elems = self.copy_bits // self.dtype.width
        self.num_threads = self.tile_m

        assert self.dtype == cutlass.Float16, "demo is intentionally kept to fp16"
        assert self.copy_bits == 128, "demo assumes one 16B cp.async per thread"
        assert self.num_stages >= 2, "num_stages must be >= 2 to show a pipeline"
        assert self.tile_k == self.copy_elems, (
            "demo keeps one vector per thread per stage, so tile_k must equal copy_elems"
        )

    def _make_smem_layout(self, dtype, major_mode):
        major_mode_size = self.tile_k if major_mode == utils.LayoutEnum.ROW_MAJOR else self.tile_m
        major_mode_size = 64 if major_mode_size >= 64 else major_mode_size

        swizzle_bits = int(math.log2(major_mode_size * dtype.width // self.copy_bits))
        swizzle_bits = min(swizzle_bits, 3)

        layout_atom_outer = (
            cute.make_layout((8, major_mode_size), stride=(major_mode_size, 1))
            if major_mode == utils.LayoutEnum.ROW_MAJOR
            else cute.make_layout((major_mode_size, 8), stride=(1, major_mode_size))
        )
        layout_atom = cute.make_composed_layout(
            cute.make_swizzle(swizzle_bits, 3, 3),
            0,
            layout_atom_outer,
        )
        return cute.tile_to_shape(
            layout_atom,
            (self.tile_m, self.tile_k, self.num_stages),
            (0, 1, 2),
        )

    def _make_tiled_copy(self, atom, dtype, major_mode):
        thread_layout = cute.make_layout((self.num_threads, 1), stride=(1, self.num_threads))
        if major_mode != utils.LayoutEnum.ROW_MAJOR:
            raise NotImplementedError(
                "The teaching demo keeps the source tensor row-major to isolate cp.async."
            )
        value_layout = cute.make_layout((1, self.copy_elems))
        return cute.make_tiled_copy_tv(atom, thread_layout, value_layout)

    @cute.jit
    def __call__(self, m_src: cute.Tensor, m_dst: cute.Tensor):
        major_mode = utils.LayoutEnum.from_tensor(m_src)
        s_layout = self._make_smem_layout(m_src.element_type, major_mode)

        async_atom = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(cache_mode=self.cache_mode),
            m_src.element_type,
            num_bits_per_copy=self.copy_bits,
        )
        sync_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            m_dst.element_type,
            num_bits_per_copy=self.copy_bits,
        )
        tiled_copy_in = self._make_tiled_copy(async_atom, m_src.element_type, major_mode)
        tiled_copy_out = self._make_tiled_copy(sync_atom, m_dst.element_type, major_mode)

        grid_m = cute.ceil_div(cute.size(m_src, mode=[0]), self.tile_m)
        grid_dim = (cute.size(grid_m), 1, cute.size(m_src, mode=[2]))

        self.kernel(
            m_src,
            m_dst,
            s_layout,
            tiled_copy_in,
            tiled_copy_out,
        ).launch(
            grid=grid_dim,
            block=(self.num_threads, 1, 1),
            smem=cute.size_in_bytes(m_src.element_type, s_layout),
        )

    @cute.kernel
    def kernel(
        self,
        m_src: cute.Tensor,
        m_dst: cute.Tensor,
        s_layout: cute.ComposedLayout,
        tiled_copy_in: cute.TiledCopy,
        tiled_copy_out: cute.TiledCopy,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, bidz = cute.arch.block_idx()

        smem = cutlass.utils.SmemAllocator()
        s_src = smem.allocate_tensor(m_src.element_type, s_layout, 16)

        g_src = cute.local_tile(
            m_src[None, None, bidz],
            tiler=(self.tile_m, 1, self.tile_k),
            coord=(bidx, 0, None),
            proj=(1, None, 1),
        )
        g_dst = cute.local_tile(
            m_dst[None, None, bidz],
            tiler=(self.tile_m, 1, self.tile_k),
            coord=(bidx, 0, None),
            proj=(1, None, 1),
        )

        thr_copy_in = tiled_copy_in.get_slice(tidx)
        thr_copy_out = tiled_copy_out.get_slice(tidx)

        t_src_g = thr_copy_in.partition_S(g_src)
        t_src_s = thr_copy_in.partition_D(s_src)
        t_smem_for_store = thr_copy_out.partition_S(s_src)
        t_dst_g = thr_copy_out.partition_D(g_dst)

        k_tile_count = cute.size(t_src_g, mode=[3])
        num_prefetch = self.num_stages - 1
        if num_prefetch > k_tile_count:
            num_prefetch = k_tile_count

        # Prologue: issue the first N-1 async copies.
        k_tile_index = cutlass.Int32(0)
        for stage in range(num_prefetch):
            cute.copy(
                tiled_copy_in,
                t_src_g[None, None, None, k_tile_index],
                t_src_s[None, None, None, stage],
            )
            k_tile_index = k_tile_index + 1
            cute.arch.cp_async_commit_group()

        smem_pipe_read = 0
        smem_pipe_write = self.num_stages - 1

        for k_tile in range(k_tile_count):
            # Wait until the stage we are about to consume is visible in SMEM.
            if not (k_tile + self.num_stages - 1 < k_tile_count):
                # Once the pipeline stops issuing future copies, we are in the
                # drain phase. At that point the stage we are about to consume
                # may itself be the last in-flight group, so we must wait for
                # all remaining async copies.
                cute.arch.cp_async_wait_group(0)
            else:
                cute.arch.cp_async_wait_group(self.num_stages - 2)
            cute.arch.sync_threads()

            # Consume the current stage with a simple SMEM -> GMEM copy so the
            # result is easy to verify on the host.
            cute.copy(
                tiled_copy_out,
                t_smem_for_store[None, None, None, smem_pipe_read],
                t_dst_g[None, None, None, k_tile],
            )
            # Before reusing this stage buffer for the next async write, make
            # sure every thread has finished reading from it.
            cute.arch.sync_threads()

            if k_tile + self.num_stages - 1 < k_tile_count:
                cute.copy(
                    tiled_copy_in,
                    t_src_g[None, None, None, k_tile_index],
                    t_src_s[None, None, None, smem_pipe_write],
                )
                k_tile_index = k_tile_index + 1
                cute.arch.cp_async_commit_group()

            smem_pipe_write = smem_pipe_read
            smem_pipe_read = smem_pipe_read + 1
            if smem_pipe_read == self.num_stages:
                smem_pipe_read = 0

        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()


def print_learning_summary(args, demo: CpAsyncTeachingDemo):
    stage_bytes = demo.tile_m * demo.tile_k * (demo.dtype.width // 8)
    print("cp.async teaching demo")
    print(f"  tile_m={demo.tile_m}, tile_k={demo.tile_k}, num_stages={demo.num_stages}")
    print(f"  copy_bits={demo.copy_bits}, copy_elems={demo.copy_elems}")
    print(f"  one thread moves {demo.copy_bits // 8} bytes per cp.async")
    print(f"  one stage buffers {stage_bytes} bytes in shared memory")
    print(f"  cache_mode={args.cache_mode}")
    print("  learning points:")
    print("    1. cp.async is a GMEM -> SMEM async copy atom")
    print("    2. commit_group closes the current producer batch")
    print("    3. wait_group(N) waits until at most N groups remain in flight")
    print("    4. sync_threads is still needed before consuming or reusing a stage")
    print("    5. this demo pipelines copy over K-tiles and validates by copying back")


def run_demo(args):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run this demo")

    demo = CpAsyncTeachingDemo(
        tile_m=args.tile_m,
        tile_k=args.tile_k,
        num_stages=args.num_stages,
        copy_bits=args.copy_bits,
        cache_mode=CACHE_MODE_MAP[args.cache_mode],
        dtype=cutlass.Float16,
    )

    if args.m % args.tile_m != 0:
        raise ValueError("For the teaching demo, M must be a multiple of tile_m")
    if args.k % args.tile_k != 0:
        raise ValueError("For the teaching demo, K must be a multiple of tile_k")

    print_learning_summary(args, demo)

    m_src, src_torch = make_row_major_mkl_tensor(
        args.m,
        args.k,
        args.l,
        cutlass.Float16,
        init="arange",
    )
    m_dst, dst_torch = make_row_major_mkl_tensor(
        args.m,
        args.k,
        args.l,
        cutlass.Float16,
        init="zeros",
    )

    compiled = cute.compile(demo, m_src, m_dst)
    compiled(m_src, m_dst)

    torch.testing.assert_close(dst_torch.cpu(), src_torch.cpu())
    print("Validation passed.")
    print("Example values from the first tile:")
    print(dst_torch[: min(args.tile_m, 4), : min(args.tile_k, 8), 0].cpu())


def parse_args():
    parser = argparse.ArgumentParser(description="Minimal CuTe cp.async teaching demo")
    parser.add_argument("--m", type=int, default=128)
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--l", type=int, default=1)
    parser.add_argument("--tile_m", type=int, default=128)
    parser.add_argument("--tile_k", type=int, default=8)
    parser.add_argument("--num_stages", type=int, default=3)
    parser.add_argument("--copy_bits", type=int, default=128)
    parser.add_argument(
        "--cache_mode",
        choices=sorted(CACHE_MODE_MAP.keys()),
        default="global",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_demo(parse_args())
