import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import torch


# This demo keeps the problem intentionally small:
# - one warp per CTA
# - one MMA tile per CTA
# - one K tile only (no mainloop / no cp.async pipeline)
# The goal is to make the tiled_mma dataflow easy to follow:
#   gmem -> smem -> ldmatrix -> mma fragment -> cute.gemm -> smem -> gmem

CTA_TILE_MNK = (16, 16, 16)
MMA_INST_SHAPE = (16, 8, 16)
ATOM_LAYOUT_MNK = (1, 1, 1)
NUM_THREADS = 32
COPY_BITS = 128


def _make_row_major_smem_layout_ab(tile_m: int, tile_k: int):
    # Keep a single stage so the partition shapes match the larger GEMM demo.
    return cute.make_layout(
        (tile_m, tile_k, 1),
        stride=(tile_k, 1, tile_m * tile_k),
    )


def _make_row_major_smem_layout_c(tile_m: int, tile_n: int):
    return cute.make_layout((tile_m, tile_n), stride=(tile_n, 1))


def _make_row_major_gmem_tiled_copy(atom_copy, dtype, tile_shape):
    tile_rows, tile_cols = tile_shape
    copy_elems = COPY_BITS // dtype.width
    groups_in_contiguous_dim = tile_cols // copy_elems
    thread_layout = cute.make_layout(
        (NUM_THREADS // groups_in_contiguous_dim, groups_in_contiguous_dim),
        stride=(groups_in_contiguous_dim, 1),
    )
    value_layout = cute.make_layout((1, copy_elems))
    return cute.make_tiled_copy_tv(atom_copy, thread_layout, value_layout)


@cute.kernel
def mma_learning_kernel(
    tiled_copy_a: cute.TiledCopy,
    tiled_copy_b: cute.TiledCopy,
    tiled_copy_c: cute.TiledCopy,
    tiled_mma: cute.TiledMma,
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor,
    sA_layout,
    sB_layout,
    sC_layout,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()

    smem = cutlass.utils.SmemAllocator()
    sA = smem.allocate_tensor(mA.element_type, sA_layout, byte_alignment=16)
    sB = smem.allocate_tensor(mB.element_type, sB_layout, byte_alignment=16)
    sC = smem.allocate_tensor(mC.element_type, sC_layout, byte_alignment=16)

    # A is logically (M, K), B is logically (N, K), and C = A @ B^T.
    gA = cute.local_tile(mA, tiler=(CTA_TILE_MNK[0], CTA_TILE_MNK[2]), coord=(bidx, None))
    gB = cute.local_tile(mB, tiler=(CTA_TILE_MNK[1], CTA_TILE_MNK[2]), coord=(bidy, None))
    gC = cute.local_tile(mC, tiler=(CTA_TILE_MNK[0], CTA_TILE_MNK[1]), coord=(bidx, bidy))

    # Global -> shared copies.
    thr_copy_a = tiled_copy_a.get_slice(tidx)
    thr_copy_b = tiled_copy_b.get_slice(tidx)
    thr_copy_c = tiled_copy_c.get_slice(tidx)

    tAgA = thr_copy_a.partition_S(gA)
    tAsA = thr_copy_a.partition_D(sA)
    tBgB = thr_copy_b.partition_S(gB)
    tBsB = thr_copy_b.partition_D(sB)
    tCsC_store = thr_copy_c.partition_S(sC)
    tCgC_store = thr_copy_c.partition_D(gC)

    cute.copy(tiled_copy_a, tAgA[None, None, None, 0], tAsA[None, None, None, 0])
    cute.copy(tiled_copy_b, tBgB[None, None, None, 0], tBsB[None, None, None, 0])
    cute.arch.sync_threads()

    # Thread-level MMA partitions over shared/global tensors.
    thr_mma = tiled_mma.get_slice(tidx)
    tCsA = thr_mma.partition_A(sA)
    tCsB = thr_mma.partition_B(sB)
    tCsC = thr_mma.partition_C(sC)
    tCgC = thr_mma.partition_C(gC)

    # Register fragments that feed the warp MMA.
    tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
    tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])
    tCrC = tiled_mma.make_fragment_C(tCgC)
    tCrC.fill(0.0)

    # Shared -> register copies matching the tiled_mma TV layout.
    atom_copy_s2r_a = cute.make_copy_atom(
        cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 4),
        mA.element_type,
    )
    atom_copy_s2r_b = cute.make_copy_atom(
        cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 4),
        mB.element_type,
    )
    tiled_copy_s2r_a = cute.make_tiled_copy_A(atom_copy_s2r_a, tiled_mma)
    tiled_copy_s2r_b = cute.make_tiled_copy_B(atom_copy_s2r_b, tiled_mma)

    thr_copy_s2r_a = tiled_copy_s2r_a.get_slice(tidx)
    thr_copy_s2r_b = tiled_copy_s2r_b.get_slice(tidx)
    tCsA_copy_view = thr_copy_s2r_a.partition_S(sA)
    tCrA_copy_view = thr_copy_s2r_a.retile(tCrA)
    tCsB_copy_view = thr_copy_s2r_b.partition_S(sB)
    tCrB_copy_view = thr_copy_s2r_b.retile(tCrB)

    cute.copy(
        tiled_copy_s2r_a,
        tCsA_copy_view[None, None, None, 0],
        tCrA_copy_view[None, None, 0],
    )
    cute.copy(
        tiled_copy_s2r_b,
        tCsB_copy_view[None, None, None, 0],
        tCrB_copy_view[None, None, 0],
    )

    # One MMA step because this teaching demo fixes K == CTA_TILE_MNK[2].
    cute.gemm(
        tiled_mma,
        tCrC,
        tCrA[None, None, 0],
        tCrB[None, None, 0],
        tCrC,
    )

    # Convert the accumulator fragment back to fp16, then store it.
    tCrD = cute.make_fragment_like(tCrC, mC.element_type)
    tCrD[None] = tCrC.load().to(mC.element_type)
    cute.autovec_copy(tCrD, tCsC)
    cute.arch.sync_threads()

    cute.copy(tiled_copy_c, tCsC_store, tCgC_store)


@cute.jit
def mma_learning(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):
    tile_m, tile_n, tile_k = CTA_TILE_MNK
    assert len(mA.shape) == 2 and len(mB.shape) == 2 and len(mC.shape) == 2
    assert mA.shape[0] == mC.shape[0]
    assert mB.shape[0] == mC.shape[1]
    assert mA.shape[1] == tile_k
    assert mB.shape[1] == tile_k
    assert mC.shape[0] % tile_m == 0
    assert mC.shape[1] % tile_n == 0

    mma_op = cute.nvgpu.warp.MmaF16BF16Op(
        mA.element_type,
        cutlass.Float32,
        MMA_INST_SHAPE,
    )
    tiled_mma = cute.make_tiled_mma(
        mma_op,
        cute.make_layout(ATOM_LAYOUT_MNK),
        permutation_mnk=(
            ATOM_LAYOUT_MNK[0] * MMA_INST_SHAPE[0],
            ATOM_LAYOUT_MNK[1] * MMA_INST_SHAPE[1] * 2,
            ATOM_LAYOUT_MNK[2] * MMA_INST_SHAPE[2],
        ),
    )

    atom_g2s = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(),
        mA.element_type,
        num_bits_per_copy=COPY_BITS,
    )
    atom_s2g = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(),
        mC.element_type,
        num_bits_per_copy=COPY_BITS,
    )

    tiled_copy_a = _make_row_major_gmem_tiled_copy(atom_g2s, mA.element_type, (tile_m, tile_k))
    tiled_copy_b = _make_row_major_gmem_tiled_copy(atom_g2s, mB.element_type, (tile_n, tile_k))
    tiled_copy_c = _make_row_major_gmem_tiled_copy(atom_s2g, mC.element_type, (tile_m, tile_n))

    sA_layout = _make_row_major_smem_layout_ab(tile_m, tile_k)
    sB_layout = _make_row_major_smem_layout_ab(tile_n, tile_k)
    sC_layout = _make_row_major_smem_layout_c(tile_m, tile_n)

    smem_size = (
        cute.size_in_bytes(mA.element_type, sA_layout)
        + cute.size_in_bytes(mB.element_type, sB_layout)
        + cute.size_in_bytes(mC.element_type, sC_layout)
    )
    grid = (mC.shape[0] // tile_m, mC.shape[1] // tile_n)

    mma_learning_kernel(
        tiled_copy_a,
        tiled_copy_b,
        tiled_copy_c,
        tiled_mma,
        mA,
        mB,
        mC,
        sA_layout,
        sB_layout,
        sC_layout,
    ).launch(
        grid=grid,
        block=(NUM_THREADS, 1, 1),
        smem=smem_size,
    )


def print_demo_notes():
    tile_m, tile_n, tile_k = CTA_TILE_MNK
    print("Tiled MMA teaching demo")
    print(f"  CTA tile MNK      : {CTA_TILE_MNK}")
    print(f"  MMA inst shape    : {MMA_INST_SHAPE}")
    print(f"  Atom layout MNK   : {ATOM_LAYOUT_MNK}")
    print(f"  Threads per CTA   : {NUM_THREADS}")
    print(f"  One CTA computes  : C tile {tile_m} x {tile_n}")
    print(f"  A/B logical shapes: ({tile_m}, {tile_k}) and ({tile_n}, {tile_k}) per CTA")
    print("  This demo fixes K == 16, so there is no cp.async pipeline or K mainloop.")
    print("  B is stored as (N, K), so the math is C = A @ B^T.")


def run_demo():
    print_demo_notes()

    m, n, k = 32, 32, CTA_TILE_MNK[2]
    dtype = torch.float16

    a_torch = torch.randn((m, k), device="cuda", dtype=dtype)
    b_torch = torch.randn((n, k), device="cuda", dtype=dtype)
    c_torch = torch.zeros((m, n), device="cuda", dtype=dtype)

    a_cute = from_dlpack(a_torch, assumed_align=16)
    b_cute = from_dlpack(b_torch, assumed_align=16)
    c_cute = from_dlpack(c_torch, assumed_align=16)

    compiled = cute.compile(
        mma_learning,
        a_cute,
        b_cute,
        c_cute,
        options="--keep-ptx",
    )

    for _ in range(5):
        compiled(a_cute, b_cute, c_cute)

    ref = torch.matmul(a_torch.float(), b_torch.float().transpose(0, 1)).to(dtype)
    torch.testing.assert_close(c_torch, ref, atol=2e-2, rtol=2e-2)
    print("Verification passed.")


if __name__ == "__main__":
    run_demo()
