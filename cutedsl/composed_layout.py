import cutlass
import cutlass.cute as cute

from cutlass.cute.runtime import from_dlpack, make_ptr

@cute.jit
def customized_layout():
    
    def inner(c):
        x, y = c
        return x, y+1
    
    layout = cute.make_composed_layout(inner, (1,0), cute.make_identity_layout(shape=(8,4)))
    print(layout)
    
    cute.printf(layout(0))

customized_layout()