import cutlass 

import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import torch


# 我需要实现一个kernel，它负责将给的tensor ，copy到shared memory。
# 然后实现async copy和 sync copy 两个版本，比较ptx指令差异、sass指令差异，以及性能差异。

@cute.kernel
def copy_async_kernel(tiled_copy: cute.TiledCopy,sync_tiled_copy: cute.TiledCopy, mA: cute.Tensor, mB: cute.Tensor, s_layout):
    tx, _, _ = cute.arch.thread_idx()
    bx, by, _ = cute.arch.block_idx()
    
    
    smem  = cutlass.utils.SmemAllocator()
    s_src = smem.allocate_tensor(mA.element_type ,s_layout, byte_alignment=16)
    
    g_tensor = cute.local_tile(mA, tiler= (128, 8), coord=(bx, by))
    
    # get thr copy
    thr_copy = tiled_copy.get_slice(tx)
    
    thr_tensor = thr_copy.partition_S(g_tensor)
    smem_tensor = thr_copy.partition_D(s_src)
    
    # copy to mB
    dst_tensor = cute.local_tile(mB, tiler=(128, 8), coord=(bx, by))
    dst_thr_copy = sync_tiled_copy.get_slice(tx)
    thr_dst_tensor = dst_thr_copy.partition_D(dst_tensor)
    thr_src_tensor = dst_thr_copy.partition_S(s_src)

    cute.copy(tiled_copy, thr_tensor, smem_tensor)
    
    # sync
    cute.arch.cp_async_commit_group()
    cute.arch.cp_async_wait_group(0)
    cute.arch.sync_threads()
    

    
    cute.copy(sync_tiled_copy, thr_src_tensor, thr_dst_tensor)
    
    
    


@cute.jit
def copy_async(mA: cute.Tensor, mB: cute.Tensor, use_async: cutlass.Constexpr):
    tiler = [128, 8]
    assert len(tiler) == len(mA.shape)
    grid = [mA.shape[0] // tiler[0] , mA.shape[1] // tiler[1]]
    num_bits_per_copy = 128
    num_bits_element = mA.element_type.width
    num_thread = tiler[0] * tiler[1] // (num_bits_per_copy // num_bits_element)
    smem_size = tiler[0] * tiler[1] * num_bits_element // 8
    
    # create tiled copy
    # first, create tv layout
    atom_copy = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), mA.element_type ,num_bits_per_copy = num_bits_per_copy,
    )
    if cutlass.const_expr(use_async):
        atom_copy = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(), mA.element_type ,num_bits_per_copy = num_bits_per_copy,
        )

    thr_layout = cute.make_layout((num_thread, 1))
    value_layout = cute.make_layout((1, num_bits_per_copy // num_bits_element))
    tiled_copy = cute.make_tiled_copy_tv(atom=atom_copy,thr_layout=thr_layout, val_layout=value_layout)
    
    sync_atom_copy = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), mB.element_type,num_bits_per_copy = num_bits_per_copy
    )
    sync_tiled_copy = cute.make_tiled_copy_tv(
        atom=sync_atom_copy,
        thr_layout=thr_layout,
        val_layout=value_layout,
    )
    
    
    # prepare smem layout
    s_layout = cute.make_layout((128, 8), stride=(8,1))
    copy_async_kernel(tiled_copy, sync_tiled_copy, mA,mB, s_layout).launch(grid = grid, block = [num_thread, 1,1], smem = smem_size)


def copy_demo():
    shape = [1280, 80]
    dtype = torch.float16
    
    torch_tensor = torch.rand(shape, dtype=dtype).to("cuda")
    dst_tensor = torch.zeros(shape, dtype=dtype).to("cuda")
    cute_tensor = from_dlpack(torch_tensor, assumed_align=16)
    cute_dst_tensor = from_dlpack(dst_tensor, assumed_align=16)
    
    compiled = cute.compile(copy_async, cute_tensor, cute_dst_tensor, True,options="--keep-ptx")
    
    for _ in range(10):
        compiled(cute_tensor, cute_dst_tensor)
    
    torch.testing.assert_close(torch_tensor, dst_tensor)
    dst_tensor = torch.zeros(shape, dtype=dtype).to("cuda")
    cute_dst_tensor = from_dlpack(dst_tensor, assumed_align=16)
    compiled = cute.compile(copy_async, cute_tensor, cute_dst_tensor, False, options="--keep-ptx")
    for _ in range(10):
        compiled(cute_tensor, cute_dst_tensor)
    torch.testing.assert_close(torch_tensor, dst_tensor)
    
    
    
    pass


if __name__ == "__main__":
    copy_demo()