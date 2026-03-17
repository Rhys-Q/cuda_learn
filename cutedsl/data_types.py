import cutlass
import cutlass.cute as cute

@cute.jit
def bar():
    a = cutlass.Float32(3.14)
    
    print("a(static) = ", a)
    cute.printf("a(dynamic) = {}", a)
    
    b = cutlass.Int32(5)
    print("b(static) =", b)
    
    cute.printf("b(dynamic) = {}", b)

bar()


@cute.jit
def type_conversion():
    x = cutlass.Int32(42)
    y = x.to(cutlass.Float32)
    cute.printf("Int32({}) => Float32({})", x, y)
    
    a = cutlass.Float32(3.14)
    b = a.to(cutlass.Int32)
    cute.printf("Float32({}) => Int32({})", a, b)
    
    c = cutlass.Int32(127)
    d = c.to(cutlass.Int8)
    cute.printf("Int32({}) => Int8({})", c, d)
    
    e = cutlass.Int32(300)
    f = e.to(cutlass.Int8)
    cute.printf("Int32({}) => Int8({}) (truncated due to range limitation)", e, f)

type_conversion()