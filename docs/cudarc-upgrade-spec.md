# cudarc 0.12 -> 0.19 Upgrade Specification

## New rvllm-gpu API Surface (what all agents code against)

### Core Type Change
```rust
// OLD: Arc<CudaDevice> everywhere
// NEW: Arc<CudaContext> + Arc<CudaStream>

// Wrapper in rvllm-gpu/src/device.rs:
pub type CudaDevice = Arc<cudarc::driver::CudaContext>;

// Every function that took Arc<CudaDevice> now takes:
//   device: &Arc<cudarc::driver::CudaContext>  (for module loading, device info)
//   stream: &Arc<cudarc::driver::CudaStream>   (for memory ops, kernel launch)
// OR a combined reference where both are needed
```

### Memory Operations (stream methods)
```rust
// OLD                              -> NEW
device.alloc_zeros::<T>(n)          -> stream.alloc_zeros::<T>(n)
device.htod_sync_copy(&data)        -> stream.clone_htod(&data)
device.dtoh_sync_copy(&slice)       -> stream.clone_dtoh(&slice)
device.htod_sync_copy_into(&h, &s)  -> stream.memcpy_htod(&h, &mut s)
device.dtod_copy(&src, &mut dst)    -> stream.memcpy_dtod(&src, &mut dst)
```

### PTX/Module Loading (context methods)
```rust
// OLD
device.load_ptx(Ptx::from_src(ptx), "module_name", &["fn1", "fn2"])
device.get_func("module_name", "fn_name") -> Option<CudaFunction>

// NEW
let module: Arc<CudaModule> = context.load_module(Ptx::from_src(ptx))?;
let func: CudaFunction = module.load_function("fn_name")?;
// KernelLoader stores HashMap<String, Arc<CudaModule>> internally
```

### Kernel Launch
```rust
// OLD (tuple args)
unsafe { func.launch(cfg, (arg1, arg2, arg3))? }
unsafe { func.launch_on_stream(&stream, cfg, (arg1, arg2, arg3))? }

// NEW (builder pattern)
unsafe {
    stream.launch_builder(&func)
        .arg(&arg1)
        .arg(&arg2)
        .arg(&arg3)
        .launch(cfg)?;
}
```

### cuBLAS
```rust
// OLD
CudaBlas::new(device: Arc<CudaDevice>)

// NEW
CudaBlas::new(stream: Arc<CudaStream>)
// Gemm trait signatures UNCHANGED
```

### Device Info
```rust
// OLD                                    -> NEW
CudaDevice::new(id)                       -> CudaContext::new(id)
CudaDevice::count()                       -> CudaContext::device_count()
cudarc::driver::result::mem_get_info()    -> context.mem_get_info()
result::device::total_mem(*dev.cu_device()) -> context.total_mem()
device.fork_default_stream()              -> context.new_stream()
device.wait_for(&stream)                  -> stream.synchronize()
device.bind_to_thread()                   -> context.bind_to_thread()
```

### Imports
```rust
// OLD
use cudarc::driver::{CudaDevice, CudaSlice, CudaStream, LaunchAsync, LaunchConfig};

// NEW
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, CudaModule, CudaFunction, LaunchConfig};
// LaunchAsync is REMOVED - use stream.launch_builder() instead
```

### Cargo.toml
```toml
# OLD
cudarc = { version = "0.12", features = ["driver", "cublas", "f16"] }

# NEW
cudarc = { version = "0.19", features = ["driver", "cublas", "f16"] }
```

## File Assignments (by agent number)

| Agent | Files | Phase |
|-------|-------|-------|
| 1 | rvllm-gpu: device.rs, lib.rs, mod exports | 1 |
| 2 | rvllm-gpu: kernel_loader.rs | 1 |
| 3 | rvllm-gpu: cublas.rs, cublas_ops.rs | 1 |
| 4 | rvllm-gpu: buffer.rs, stream.rs | 1 |
| 5 | rvllm-gpu: cuda_allocator.rs, pinned_memory.rs | 1 |
| 6 | rvllm-gpu: cuda_graph.rs | 1 |
| 7 | model-loader: gpu_loader.rs (inner mod) | 2 |
| 8 | model-loader: gpu_weights.rs | 2 |
| 9 | kv-cache: engine_cuda.rs | 2 |
| 10 | kv-cache: ops_cuda.rs | 2 |
| 11 | model-runner: gpu_runner.rs | 2 |
| 12 | model-runner: gpu_layer.rs | 2 |
| 13 | model-runner: layers/linear_cuda.rs | 2 |
| 14 | model-runner: layers/softmax_cuda.rs, norm_cuda.rs | 2 |
| 15 | model-runner: layers/activation_cuda.rs | 2 |
| 16 | model-runner: layers/rotary_cuda.rs, fused_ops.rs | 2 |
| 17 | attention: paged_attention_cuda.rs | 2 |
| 18 | attention: flash_attention_impl.rs, backend.rs | 2 |
| 19 | worker: gpu_worker.rs, config.rs | 2 |
| 20 | ALL Cargo.toml files (7) + review compilation | 3 |

## Rules
- Each agent edits ONLY their assigned files
- Use the API mappings above exactly
- Do NOT change function signatures visible to other crates unless you are agents 1-6
- Keep `#[cfg(feature = "cuda")]` gates exactly as they are
- The mock-gpu path (non-cuda) must still compile
- Run `cargo check` after changes if possible
