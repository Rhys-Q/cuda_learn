import cutlass

import cutlass.cute as cute


@cute.kernel
def kernel():
    tidx, _, _ = cute.arch.thread_idx()
    if cutlass.dynamic_expr(tidx == 0):
        cute.printf("Hello world")
        

@cute.jit
def hello_world():
    cute.printf("hello world")
    
    kernel().launch(
        grid = (1,1,1),
        block = (32, 1,1)
    )
    

cutlass.cuda.initialize_cuda_context()

print("Running hello wolrd()...")
hello_world()
import time
time.sleep(1)
print("Compiling...")
hello_world_compiled = cute.compile(hello_world)

hello_world_compiled()