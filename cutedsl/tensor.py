import cutlass
import cutlass.cute as cute

@cute.jit
def create_tensor_from_ptr(ptr: cute.Pointer):
    layout = cute.make_layout((8, 5), stride=(5,1))
    tensor = cute.make_tensor(ptr, layout)
    tensor.fill(1)
    cute.print_tensor(tensor)
    
    

import torch
from cutlass.torch import dtype as torch_dtype
import cutlass.cute.runtime as cute_rt

a = torch.randn(8, 5, dtype=torch_dtype(cutlass.Float32))
ptr_a = cute_rt.make_ptr(cutlass.Float32, a.data_ptr())

create_tensor_from_ptr(ptr_a)


from cutlass.cute.runtime import from_dlpack

@cute.jit
def print_tensor_dlpack(src: cute.Tensor):
    print(src)
    cute.print_tensor(src)

print(from_dlpack(a))

import numpy as np

a = np.random.randn(8, 8).astype(np.float32)
print_tensor_dlpack(from_dlpack(a))

@cute.jit
def tensor_access_item(a : cute.Tensor):
    cute.print_tensor(a)
    cute.printf("a[1] = {} equivalent to a[{}]", a[1], cute.make_identity_tensor(a.layout.shape)[1],)
    cute.printf("a[2] = {} equivalent to a[{}]", a[2], cute.make_identity_tensor(a.layout.shape)[2],)
    
    cute.printf("a[2,0] = {}", a[2, 0])
    cute.printf("a[2,4] = {}", a[2, 4])
    cute.printf("a[(2,4)] = {}", a[2, 4])
    
    a[2, 3] = 100.0
    a[2,4] = 101.0
    cute.printf("a[2,3] = {}", a[2, 3])
    cute.printf("a[2,4] = {}", a[2, 4])
    

data = torch.arange(0, 8*5, dtype=torch.float32).reshape(8, 5)
tensor_access_item(from_dlpack(data))