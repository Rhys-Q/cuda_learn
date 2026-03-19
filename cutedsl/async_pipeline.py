import torch
import cutlass
import cutlass.cute as cute

from cutlass.cute.runtime import from_dlpack

@cute.kernel
def synced_producer_consumer(SharedStorage: cutlass.Constexpr, res: cute.Tensor):
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)
    
    smem = cutlass.utils.SmemAllocator()
    storage = smem.allocate(SharedStorage, 64)
    
    staging_smem = storage.staging_buffer.get_tensor(cute.make_layout(1))
    staging_smem.fill(0)
    cute.arch.sync_threads()
    
    for i in cutlass.range(cute.size(res)):
        
        if warp_idx == 0:
            staging_smem[0] = i* 1.0
        
        cute.arch.sync_threads()
        
        if warp_idx == 1:
            res[i] = staging_smem[0]
        
        cute.arch.sync_threads()


@cute.jit
def run_synced_producer_consumer(res: cute.Tensor):
    @cute.struct
    class SharedStorage:
        staging_buffer: cute.struct.Align[
            cute.struct.MemRange[cutlass.Float32, 1], 1024
        ]
        
    synced_producer_consumer(SharedStorage,  res).launch(grid = (1,1,1), block = (64, 1, 1), smem = SharedStorage.size_in_bytes())

res = torch.zeros((8,) , device="cuda")
run_synced_producer_consumer(from_dlpack(res))
print(res)
