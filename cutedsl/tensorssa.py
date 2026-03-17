
import cutlass
import cutlass.cute as cute

from cutlass.cute.runtime import from_dlpack

import numpy as np

@cute.jit
def load_and_store(res: cute.Tensor, a: cute.Tensor, b: cute.Tensor):
    a_vec = a.load()
    print(f"a_vec: {a_vec}")
    b_vec = b.load()
    print(f"b_vec: {b_vec}")
    
    res.store(a_vec + b_vec)
    cute.print_tensor(res)

a = np.ones(12).reshape((3,4)).astype(np.float32)
b = np.ones(12).reshape((3,4)).astype(np.float32)
c = np.zeros(12).reshape((3,4)).astype(np.float32)

load_and_store(from_dlpack(c), from_dlpack(a), from_dlpack(b))


@cute.jit
def apply_slice(src: cute.Tensor, dst: cute.Tensor, indices: cutlass.Constexpr):
    src_vec = src.load()
    dst_vec = src_vec[indices]
    print(f"{src_vec} -> {dst_vec}")
    if cutlass.const_expr(isinstance(dst_vec, cute.TensorSSA)):
        dst.store(dst_vec)
        cute.print_tensor(dst)
    else:
        dst[0] = dst_vec
        cute.print_tensor(dst)

def slice_1():
    src_shape = (4, 2, 3)
    dst_shape = (4, 3)
    indices = (None, 1, None)
    
    a = np.arange(np.prod(src_shape)).reshape(*src_shape).astype(np.float32)
    dst = np.random.randn(*dst_shape).astype(np.float32)
    apply_slice(from_dlpack(a), from_dlpack(dst), indices)

slice_1()

def slice_2():
    src_shape = (4,2,3)
    dst_shape = (1,)
    indices = 10
    a = np.arange(np.prod(src_shape)).reshape(*src_shape).astype(np.float32)
    dst = np.random.randn(*dst_shape).astype(np.float32)
    apply_slice(from_dlpack(a), from_dlpack(dst), indices)

slice_2()