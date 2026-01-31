import os

# Skip building the optional torch C DLPack extension that slows TVM import
os.environ.setdefault("TVM_FFI_DISABLE_TORCH_C_DLPACK", "1")
try:
    import tvm_ffi._optional_torch_c_dlpack as _ocd  # type: ignore

    _ocd.load_torch_c_dlpack_extension = lambda: None
except Exception:
    # If the helper module is unavailable we still proceed with the normal TVM import
    pass

import tvm
from tvm import te, tir
from tvm.script import ir as I
from tvm.script import relax as R


@I.ir_module
class Module:
    @R.function
    def main(
        x: R.Tensor((1024,), "float32"),
        y: R.Tensor((1024,), "float32"),
    ) -> R.Tensor((1024,), "float32"):
        with R.dataflow():
            z = R.add(x, y)
            R.output(z)
        return z


mod = Module


def build_te_add(n: int = 1024, dtype: str = "float32"):
    """Create a simple TE compute for elementwise add."""
    A = te.placeholder((n,), name="A", dtype=dtype)
    B = te.placeholder((n,), name="B", dtype=dtype)
    C = te.compute((n,), lambda i: A[i] + B[i], name="C")
    return A, B, C


def tir_from_te(A, B, C):
    """Convert TE compute to a naive (unscheduled) TIR PrimFunc wrapped in an IRModule."""
    prim_func = te.create_prim_func([A, B, C])
    return tvm.IRModule({"main": prim_func})


def schedule_cuda(tir_mod: tvm.IRModule, block_name: str = "C", threads: int = 256):
    """Apply a simple GPU schedule to the given TIR module."""
    sch = tvm.tir.Schedule(tir_mod)
    block = sch.get_block(block_name)
    i, = sch.get_loops(block)
    i0, i1 = sch.split(i, factors=[None, threads])
    sch.bind(i0, "blockIdx.x")
    sch.bind(i1, "threadIdx.x")
    return sch.mod


def try_codegen_cuda(scheduled: tvm.IRModule):
    """Build CUDA code and return the generated device source."""
    rt_mod = tvm.build(scheduled, target="cuda")
    # For CUDA targets, the imported module holds the device code (PTX/CUDA)
    if rt_mod.imported_modules:
        return rt_mod.imported_modules[0].get_source()
    return "<no device module generated>"


def main():
    print("=== Relax IR ===")
    print(mod.script())

    A, B, C = build_te_add()
    print("\n=== TE Compute ===")
    print(f"{C.name}[i] = {A.name}[i] + {B.name}[i], shape={tuple(int(s) for s in C.shape)}, dtype={C.dtype}")

    tir_mod = tir_from_te(A, B, C)
    print("\n=== TIR (before schedule) ===")
    print(tir_mod.script())

    scheduled = schedule_cuda(tir_mod)
    print("\n=== TIR (after GPU schedule) ===")
    print(scheduled.script())

    try:
        cuda_src = try_codegen_cuda(scheduled)
        print("\n=== Generated CUDA code ===")
        print(cuda_src)
    except Exception as err:  # pragma: no cover - environment dependent
        print("\n[CUDA codegen skipped]", err)


if __name__ == "__main__":
    main()
