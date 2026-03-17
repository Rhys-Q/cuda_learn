import cutlass

import cutlass.cute as cute

@cute.jit
def coalesce_example():
    
    layout = cute.make_layout((2, (1, 6)), stride=(1, (cutlass.Int32(6), 2)))
    result = cute.coalesce(layout)
    
    print(">>> Original: ", layout)
    cute.printf(">?? Original: {}", layout)
    print(">>> Colasced:", result)
    cute.printf(">?? Coalesced: {}", result)

coalesce_example()

@cute.jit
def bymode_coalesce_example():
    
    layout = cute.make_layout((2, (1,6)), stride=(1,(6,2)))
    
    result = cute.coalesce(layout, target_profile=(1,1))
    
    print(">>> Original: ", layout)
    print(">>> Coalesced Result: ", result)

bymode_coalesce_example()

@cute.jit
def composition_example():
    A = cute.make_layout((6,2), stride=(cutlass.Int32(8), 2))
    B = cute.make_layout((4,3), stride=(3,1))
    R = cute.composition(A, B)
    
    print(">>> Layout A:", A)
    cute.printf(">?? Layout A: {}", A)
    print(">>> Layout B:", B)
    cute.printf(">?? Layout B: {}", B)
    print(">>> Composition R = A ◦ B:", R)
    cute.printf(">?? Composition R: {}", R)


composition_example()