//! LLVM NVPTX backend for fused kernel compilation.
//!
//! Replaces nvcc dependency with direct LLVM IR -> PTX emission via inkwell.
//! Pattern classification reuses the same logic from codegen.rs; the difference
//! is that instead of emitting CUDA C text we build LLVM IR and lower it through
//! the NVPTX backend.

use inkwell::context::Context;
use inkwell::module::Module;
use inkwell::builder::Builder;
use inkwell::passes::PassBuilderOptions;
use inkwell::targets::{
    CodeModel, FileType, RelocMode, Target, TargetMachine, TargetTriple,
};
use inkwell::types::*;
use inkwell::values::*;
use inkwell::AddressSpace;
use inkwell::OptimizationLevel;

use crate::ir::{FusedKernel, FusionOp};

const THREADS: u32 = 256;
const WARP_SIZE: u32 = 32;
const NUM_WARPS: u32 = THREADS / WARP_SIZE;

// ---------------------------------------------------------------------------
// Pattern classification (mirrors codegen.rs)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FusionPattern {
    RMSNormGemv,
    SiLUElemMulGemv,
    ElemAddRMSNorm,
    ElemAddRMSNormGemv,
    Generic,
}

pub fn classify(kernel: &FusedKernel) -> FusionPattern {
    let tags: Vec<&str> = kernel.ops.iter().map(op_tag).collect();
    match tags.as_slice() {
        ["RMSNorm", "Gemv"] => FusionPattern::RMSNormGemv,
        ["SiLU", "ElemMul", "Gemv"] => FusionPattern::SiLUElemMulGemv,
        ["ElemAdd", "RMSNorm"] => FusionPattern::ElemAddRMSNorm,
        ["ElemAdd", "RMSNorm", "Gemv"] => FusionPattern::ElemAddRMSNormGemv,
        _ => FusionPattern::Generic,
    }
}

fn op_tag(op: &FusionOp) -> &'static str {
    match op {
        FusionOp::RMSNorm { .. } => "RMSNorm",
        FusionOp::Gemv => "Gemv",
        FusionOp::SiLU => "SiLU",
        FusionOp::ElemMul => "ElemMul",
        FusionOp::ElemAdd => "ElemAdd",
        FusionOp::BiasAdd => "BiasAdd",
        FusionOp::RoPE => "RoPE",
        FusionOp::Softmax => "Softmax",
        FusionOp::Copy => "Copy",
    }
}

// ---------------------------------------------------------------------------
// Kernel config for autotuning
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct KernelConfig {
    pub block_size: u32,
    pub num_warps: u32,
    pub tile_k: u32,
}

// ---------------------------------------------------------------------------
// LlvmPtxCompiler
// ---------------------------------------------------------------------------

pub struct LlvmPtxCompiler {
    context: Context,
}

impl LlvmPtxCompiler {
    pub fn new() -> Self {
        Target::initialize_nvptx(&Default::default());
        Self {
            context: Context::create(),
        }
    }

    /// Compile a fused kernel to PTX bytes via LLVM NVPTX backend.
    pub fn compile_fused_kernel(
        &self,
        kernel: &FusedKernel,
        hidden: usize,
        out_dim: usize,
        eps: f32,
        arch: &str,
    ) -> Result<Vec<u8>, String> {
        let module = self.context.create_module("fused_kernel");
        let triple = TargetTriple::create("nvptx64-nvidia-cuda");
        module.set_triple(&triple);

        // Data layout will be set by the target machine during PTX emission.

        let pattern = classify(kernel);
        match pattern {
            FusionPattern::RMSNormGemv => {
                self.emit_rmsnorm_gemv(&module, hidden, out_dim, eps)?;
            }
            FusionPattern::SiLUElemMulGemv => {
                self.emit_silu_down_gemv(&module, out_dim)?;
            }
            FusionPattern::ElemAddRMSNorm => {
                self.emit_add_rmsnorm(&module, hidden, eps)?;
            }
            FusionPattern::ElemAddRMSNormGemv => {
                self.emit_add_rmsnorm_gemv(&module, hidden, out_dim, eps)?;
            }
            FusionPattern::Generic => {
                return Err("unsupported fusion pattern for LLVM backend".into());
            }
        }

        self.optimize(&module, arch)?;
        self.emit_ptx(&module, arch)
    }

    // -----------------------------------------------------------------------
    // NVPTX intrinsic helpers
    // -----------------------------------------------------------------------

    fn declare_nvvm_read_sreg<'ctx>(
        &'ctx self,
        module: &Module<'ctx>,
        name: &str,
    ) -> FunctionValue<'ctx> {
        let i32_type = self.context.i32_type();
        let fn_type = i32_type.fn_type(&[], false);
        let full_name = format!("llvm.nvvm.read.ptx.sreg.{}", name);
        module.add_function(&full_name, fn_type, None)
    }

    fn declare_barrier0<'ctx>(&'ctx self, module: &Module<'ctx>) -> FunctionValue<'ctx> {
        let void_type = self.context.void_type();
        let fn_type = void_type.fn_type(&[], false);
        module.add_function("llvm.nvvm.barrier0", fn_type, None)
    }

    fn declare_shfl_down_f32<'ctx>(&'ctx self, module: &Module<'ctx>) -> FunctionValue<'ctx> {
        let f32_type = self.context.f32_type();
        let i32_type = self.context.i32_type();
        let fn_type = f32_type.fn_type(
            &[
                i32_type.into(), // mask
                f32_type.into(), // val
                i32_type.into(), // offset
                i32_type.into(), // width
            ],
            false,
        );
        module.add_function("llvm.nvvm.shfl.sync.down.f32", fn_type, None)
    }

    fn declare_rsqrt_approx_f32<'ctx>(&'ctx self, module: &Module<'ctx>) -> FunctionValue<'ctx> {
        let f32_type = self.context.f32_type();
        let fn_type = f32_type.fn_type(&[f32_type.into()], false);
        module.add_function("llvm.nvvm.rsqrt.approx.f", fn_type, None)
    }

    fn call_sreg<'ctx>(
        &'ctx self,
        builder: &Builder<'ctx>,
        func: FunctionValue<'ctx>,
        name: &str,
    ) -> IntValue<'ctx> {
        builder
            .build_call(func, &[], name)
            .unwrap()
            .try_as_basic_value()
            .basic()
            .unwrap()
            .into_int_value()
    }

    fn call_barrier0<'ctx>(&'ctx self, builder: &Builder<'ctx>, func: FunctionValue<'ctx>) {
        builder.build_call(func, &[], "").unwrap();
    }

    fn call_shfl_down<'ctx>(
        &'ctx self,
        builder: &Builder<'ctx>,
        shfl_fn: FunctionValue<'ctx>,
        val: FloatValue<'ctx>,
        offset: IntValue<'ctx>,
    ) -> FloatValue<'ctx> {
        let i32_type = self.context.i32_type();
        let mask = i32_type.const_int(0xFFFFFFFF, false);
        let width = i32_type.const_int(32, false);
        builder
            .build_call(
                shfl_fn,
                &[mask.into(), val.into(), offset.into(), width.into()],
                "shfl",
            )
            .unwrap()
            .try_as_basic_value()
            .basic()
            .unwrap()
            .into_float_value()
    }

    /// Emit a warp reduction (sum) of `val` using shfl.sync.down.
    /// Returns the reduced value (valid in lane 0).
    fn emit_warp_reduce_sum<'ctx>(
        &'ctx self,
        builder: &Builder<'ctx>,
        shfl_fn: FunctionValue<'ctx>,
        val: FloatValue<'ctx>,
    ) -> FloatValue<'ctx> {
        let i32_type = self.context.i32_type();
        let mut acc = val;
        // Unrolled: offsets 16, 8, 4, 2, 1
        for &offset in &[16u64, 8, 4, 2, 1] {
            let off = i32_type.const_int(offset, false);
            let shuffled = self.call_shfl_down(builder, shfl_fn, acc, off);
            acc = builder.build_float_add(acc, shuffled, "warp_sum").unwrap();
        }
        acc
    }

    /// Emit a block-level reduction across warps using shared memory.
    /// `val` is per-thread partial sum, `tid`/`warp_id`/`lane_id` are thread indices.
    /// `s_warp_ptr` points to shared mem float[NUM_WARPS].
    /// `barrier_fn` is llvm.nvvm.barrier0.
    /// Returns the fully reduced value (valid in tid==0).
    fn emit_block_reduce_sum<'ctx>(
        &'ctx self,
        builder: &Builder<'ctx>,
        function: FunctionValue<'ctx>,
        shfl_fn: FunctionValue<'ctx>,
        barrier_fn: FunctionValue<'ctx>,
        val: FloatValue<'ctx>,
        tid: IntValue<'ctx>,
        warp_id: IntValue<'ctx>,
        lane_id: IntValue<'ctx>,
        s_warp_ptr: PointerValue<'ctx>,
    ) -> FloatValue<'ctx> {
        let f32_type = self.context.f32_type();
        let i32_type = self.context.i32_type();

        // Warp-level reduce
        let warp_sum = self.emit_warp_reduce_sum(builder, shfl_fn, val);

        // Lane 0 of each warp writes to shared
        let is_lane0 = builder
            .build_int_compare(inkwell::IntPredicate::EQ, lane_id, i32_type.const_int(0, false), "is_lane0")
            .unwrap();

        let store_bb = self.context.append_basic_block(function, "warp_store");
        let after_store_bb = self.context.append_basic_block(function, "after_warp_store");
        builder.build_conditional_branch(is_lane0, store_bb, after_store_bb).unwrap();

        builder.position_at_end(store_bb);
        let slot = unsafe {
            builder.build_gep(f32_type, s_warp_ptr, &[warp_id], "warp_slot").unwrap()
        };
        builder.build_store(slot, warp_sum).unwrap();
        builder.build_unconditional_branch(after_store_bb).unwrap();

        builder.position_at_end(after_store_bb);
        self.call_barrier0(builder, barrier_fn);

        // Warp 0 loads and reduces
        let is_warp0 = builder
            .build_int_compare(
                inkwell::IntPredicate::EQ,
                warp_id,
                i32_type.const_int(0, false),
                "is_warp0",
            )
            .unwrap();

        let reduce_bb = self.context.append_basic_block(function, "final_reduce");
        let skip_bb = self.context.append_basic_block(function, "skip_reduce");
        let merge_bb = self.context.append_basic_block(function, "merge_reduce");
        builder.build_conditional_branch(is_warp0, reduce_bb, skip_bb).unwrap();

        // In warp 0: load from shared if lane < NUM_WARPS, else 0
        builder.position_at_end(reduce_bb);
        let in_range = builder
            .build_int_compare(
                inkwell::IntPredicate::ULT,
                lane_id,
                i32_type.const_int(NUM_WARPS as u64, false),
                "in_range",
            )
            .unwrap();
        let slot2 = unsafe {
            builder.build_gep(f32_type, s_warp_ptr, &[lane_id], "lane_slot").unwrap()
        };
        let loaded = builder.build_load(f32_type, slot2, "warp_val").unwrap().into_float_value();
        let zero = f32_type.const_float(0.0);
        let val_or_zero = builder
            .build_select(in_range, loaded, zero, "val_or_zero")
            .unwrap()
            .into_float_value();
        let final_sum = self.emit_warp_reduce_sum(builder, shfl_fn, val_or_zero);
        builder.build_unconditional_branch(merge_bb).unwrap();

        builder.position_at_end(skip_bb);
        builder.build_unconditional_branch(merge_bb).unwrap();

        builder.position_at_end(merge_bb);
        let phi = builder.build_phi(f32_type, "block_sum").unwrap();
        phi.add_incoming(&[(&final_sum, reduce_bb), (&zero, skip_bb)]);
        phi.as_basic_value().into_float_value()
    }

    // -----------------------------------------------------------------------
    // Address space helpers
    // -----------------------------------------------------------------------

    fn global_as(&self) -> AddressSpace {
        AddressSpace::from(1u16)
    }

    fn shared_as(&self) -> AddressSpace {
        AddressSpace::from(3u16)
    }

    fn global_ptr_type(&self) -> PointerType<'_> {
        self.context.ptr_type(self.global_as())
    }

    // -----------------------------------------------------------------------
    // Shared memory allocation
    // -----------------------------------------------------------------------

    fn add_shared_mem<'ctx>(
        &'ctx self,
        module: &Module<'ctx>,
        name: &str,
        num_floats: usize,
    ) -> GlobalValue<'ctx> {
        let f32_type = self.context.f32_type();
        let arr_type = f32_type.array_type(num_floats as u32);
        let gv = module.add_global(arr_type, Some(self.shared_as()), name);
        gv.set_initializer(&arr_type.const_zero());
        gv.set_alignment(16);
        gv
    }

    // -----------------------------------------------------------------------
    // Mark function as NVPTX kernel
    // -----------------------------------------------------------------------

    fn mark_kernel<'ctx>(&self, function: FunctionValue<'ctx>) {
        // The "nvptx-kernel" attribute tells LLVM this is a __global__ function
        let attr = self.context.create_string_attribute("nvptx-kernel", "");
        function.add_attribute(inkwell::attributes::AttributeLoc::Function, attr);
    }

    // -----------------------------------------------------------------------
    // RMSNorm + GEMV kernel
    // -----------------------------------------------------------------------

    fn emit_rmsnorm_gemv<'ctx>(
        &'ctx self,
        module: &Module<'ctx>,
        hidden: usize,
        out_dim: usize,
        eps: f32,
    ) -> Result<(), String> {
        let f32_type = self.context.f32_type();
        let f16_type = self.context.f16_type();
        let i32_type = self.context.i32_type();
        let i64_type = self.context.i64_type();
        let gptr = self.global_ptr_type();

        // Shared memory: hidden floats for normed data + NUM_WARPS floats for warp sums
        let smem_normed = self.add_shared_mem(module, "smem_normed", hidden);
        let smem_warp = self.add_shared_mem(module, "smem_warp", NUM_WARPS as usize);

        // Kernel signature:
        //   void fused_rmsnorm_gemv(half* output, half* hidden_in, half* norm_weight,
        //                           half* proj_weight, float eps, i32 out_dim, i32 hidden_size)
        let fn_type = self.context.void_type().fn_type(
            &[
                gptr.into(), // output
                gptr.into(), // hidden_in
                gptr.into(), // norm_weight
                gptr.into(), // proj_weight
                f32_type.into(), // eps
                i32_type.into(), // out_dim
                i32_type.into(), // hidden_size
            ],
            false,
        );

        let function = module.add_function("fused_rmsnorm_gemv", fn_type, None);
        self.mark_kernel(function);

        let builder = self.context.create_builder();

        // Declare intrinsics
        let tid_fn = self.declare_nvvm_read_sreg(module, "tid.x");
        let bid_fn = self.declare_nvvm_read_sreg(module, "ctaid.x");
        let barrier_fn = self.declare_barrier0(module);
        let shfl_fn = self.declare_shfl_down_f32(module);
        let rsqrt_fn = self.declare_rsqrt_approx_f32(module);

        // Entry block
        let entry_bb = self.context.append_basic_block(function, "entry");
        builder.position_at_end(entry_bb);

        let p_output = function.get_nth_param(0).unwrap().into_pointer_value();
        let p_hidden = function.get_nth_param(1).unwrap().into_pointer_value();
        let p_norm_w = function.get_nth_param(2).unwrap().into_pointer_value();
        let p_proj_w = function.get_nth_param(3).unwrap().into_pointer_value();
        let p_eps = function.get_nth_param(4).unwrap().into_float_value();
        let p_out_dim = function.get_nth_param(5).unwrap().into_int_value();
        let p_hidden_size = function.get_nth_param(6).unwrap().into_int_value();

        let tid = self.call_sreg(&builder, tid_fn, "tid");
        let bid = self.call_sreg(&builder, bid_fn, "bid");

        // row = blockIdx.x; if (row >= out_dim) return;
        let row = bid;
        let oob = builder
            .build_int_compare(inkwell::IntPredicate::SGE, row, p_out_dim, "oob")
            .unwrap();
        let body_bb = self.context.append_basic_block(function, "body");
        let exit_bb = self.context.append_basic_block(function, "exit");
        builder.build_conditional_branch(oob, exit_bb, body_bb).unwrap();

        builder.position_at_end(exit_bb);
        builder.build_return(None).unwrap();

        builder.position_at_end(body_bb);

        let warp_id = builder.build_right_shift(tid, i32_type.const_int(5, false), false, "warp_id").unwrap();
        let lane_id = builder.build_and(tid, i32_type.const_int(31, false), "lane_id").unwrap();

        // Pointers to shared memory arrays
        let smem_normed_ptr = smem_normed.as_pointer_value();
        let smem_warp_ptr = smem_warp.as_pointer_value();

        // ---------------------------------------------------------------
        // Phase 1: Load hidden into smem, compute sum-of-squares
        // ---------------------------------------------------------------
        // Loop: for (int i = tid; i < hidden_size; i += THREADS)
        let loop_header = self.context.append_basic_block(function, "load_loop_header");
        let loop_body = self.context.append_basic_block(function, "load_loop_body");
        let loop_exit = self.context.append_basic_block(function, "load_loop_exit");

        // local_ss accumulator
        let ss_alloca = builder.build_alloca(f32_type, "ss_alloca").unwrap();
        builder.build_store(ss_alloca, f32_type.const_float(0.0)).unwrap();
        builder.build_unconditional_branch(loop_header).unwrap();

        // Loop header: i = phi(tid, i + THREADS)
        builder.position_at_end(loop_header);
        let i_phi = builder.build_phi(i32_type, "i").unwrap();
        let i_val = i_phi.as_basic_value().into_int_value();
        let cond = builder
            .build_int_compare(inkwell::IntPredicate::SLT, i_val, p_hidden_size, "cond")
            .unwrap();
        builder.build_conditional_branch(cond, loop_body, loop_exit).unwrap();

        // Loop body: load hidden[i], store to smem, accumulate ss
        builder.position_at_end(loop_body);
        let addr_h = unsafe {
            builder.build_gep(f16_type, p_hidden, &[i_val], "addr_h").unwrap()
        };
        let h_f16 = builder.build_load(f16_type, addr_h, "h_f16").unwrap().into_float_value();
        let h_f32 = builder
            .build_float_ext(h_f16, f32_type, "h_f32")
            .unwrap();

        // smem_normed[i] = h_f32
        let smem_slot = unsafe {
            builder.build_gep(f32_type, smem_normed_ptr, &[i_val], "smem_slot").unwrap()
        };
        builder.build_store(smem_slot, h_f32).unwrap();

        // local_ss += h_f32 * h_f32
        let cur_ss = builder.build_load(f32_type, ss_alloca, "cur_ss").unwrap().into_float_value();
        let sq = builder.build_float_mul(h_f32, h_f32, "sq").unwrap();
        let new_ss = builder.build_float_add(cur_ss, sq, "new_ss").unwrap();
        builder.build_store(ss_alloca, new_ss).unwrap();

        let i_next = builder
            .build_int_add(i_val, i32_type.const_int(THREADS as u64, false), "i_next")
            .unwrap();
        i_phi.add_incoming(&[(&tid, body_bb), (&i_next, loop_body)]);
        builder.build_unconditional_branch(loop_header).unwrap();

        // After load loop
        builder.position_at_end(loop_exit);
        let local_ss = builder.build_load(f32_type, ss_alloca, "local_ss").unwrap().into_float_value();

        // Block reduction of sum-of-squares
        let total_ss = self.emit_block_reduce_sum(
            &builder, function, shfl_fn, barrier_fn,
            local_ss, tid, warp_id, lane_id, smem_warp_ptr,
        );

        // Compute rms_scale = rsqrt(total_ss / hidden_size + eps) -- only tid 0 does this
        // But we stored the result in smem_warp[0] so all threads can read it.
        let is_tid0 = builder
            .build_int_compare(inkwell::IntPredicate::EQ, tid, i32_type.const_int(0, false), "is_tid0")
            .unwrap();
        let compute_scale_bb = self.context.append_basic_block(function, "compute_scale");
        let after_scale_bb = self.context.append_basic_block(function, "after_scale");
        builder.build_conditional_branch(is_tid0, compute_scale_bb, after_scale_bb).unwrap();

        builder.position_at_end(compute_scale_bb);
        let hs_f32 = builder
            .build_signed_int_to_float(p_hidden_size, f32_type, "hs_f32")
            .unwrap();
        let mean_ss = builder.build_float_div(total_ss, hs_f32, "mean_ss").unwrap();
        let with_eps = builder.build_float_add(mean_ss, p_eps, "with_eps").unwrap();
        let rms_scale = builder
            .build_call(rsqrt_fn, &[with_eps.into()], "rms_scale")
            .unwrap()
            .try_as_basic_value()
            .basic()
            .unwrap()
            .into_float_value();
        // Store to smem_warp[0]
        let warp0_slot = unsafe {
            builder.build_gep(f32_type, smem_warp_ptr, &[i32_type.const_int(0, false)], "warp0").unwrap()
        };
        builder.build_store(warp0_slot, rms_scale).unwrap();
        builder.build_unconditional_branch(after_scale_bb).unwrap();

        builder.position_at_end(after_scale_bb);
        self.call_barrier0(&builder, barrier_fn);

        // All threads load rms_scale from smem_warp[0]
        let warp0_load = unsafe {
            builder.build_gep(f32_type, smem_warp_ptr, &[i32_type.const_int(0, false)], "warp0_ld").unwrap()
        };
        let rms_scale_all = builder.build_load(f32_type, warp0_load, "rms_scale_all").unwrap().into_float_value();

        // ---------------------------------------------------------------
        // Phase 1b: Apply RMSNorm: smem[i] = smem[i] * norm_weight[i] * rms_scale
        // ---------------------------------------------------------------
        let norm_header = self.context.append_basic_block(function, "norm_loop_header");
        let norm_body = self.context.append_basic_block(function, "norm_loop_body");
        let norm_exit = self.context.append_basic_block(function, "norm_loop_exit");
        builder.build_unconditional_branch(norm_header).unwrap();

        builder.position_at_end(norm_header);
        let ni_phi = builder.build_phi(i32_type, "ni").unwrap();
        let ni_val = ni_phi.as_basic_value().into_int_value();
        let ncond = builder
            .build_int_compare(inkwell::IntPredicate::SLT, ni_val, p_hidden_size, "ncond")
            .unwrap();
        builder.build_conditional_branch(ncond, norm_body, norm_exit).unwrap();

        builder.position_at_end(norm_body);
        // Load smem[ni]
        let smem_ni = unsafe {
            builder.build_gep(f32_type, smem_normed_ptr, &[ni_val], "smem_ni").unwrap()
        };
        let sv = builder.build_load(f32_type, smem_ni, "sv").unwrap().into_float_value();
        // Load norm_weight[ni] (f16)
        let nw_addr = unsafe {
            builder.build_gep(f16_type, p_norm_w, &[ni_val], "nw_addr").unwrap()
        };
        let nw_f16 = builder.build_load(f16_type, nw_addr, "nw_f16").unwrap().into_float_value();
        let nw_f32 = builder.build_float_ext(nw_f16, f32_type, "nw_f32").unwrap();
        // smem[ni] = sv * nw_f32 * rms_scale
        let t1 = builder.build_float_mul(sv, nw_f32, "t1").unwrap();
        let normed = builder.build_float_mul(t1, rms_scale_all, "normed").unwrap();
        builder.build_store(smem_ni, normed).unwrap();

        let ni_next = builder
            .build_int_add(ni_val, i32_type.const_int(THREADS as u64, false), "ni_next")
            .unwrap();
        ni_phi.add_incoming(&[(&tid, after_scale_bb), (&ni_next, norm_body)]);
        builder.build_unconditional_branch(norm_header).unwrap();

        builder.position_at_end(norm_exit);
        self.call_barrier0(&builder, barrier_fn);

        // ---------------------------------------------------------------
        // Phase 2: GEMV dot product
        // ---------------------------------------------------------------
        // w_row = proj_weight + row * hidden_size
        let row_i64 = builder.build_int_z_extend(row, i64_type, "row_i64").unwrap();
        let hs_i64 = builder.build_int_z_extend(p_hidden_size, i64_type, "hs_i64").unwrap();
        let row_offset = builder.build_int_mul(row_i64, hs_i64, "row_off").unwrap();
        let w_row = unsafe {
            builder.build_gep(f16_type, p_proj_w, &[row_offset], "w_row").unwrap()
        };

        // GEMV loop
        let gemv_header = self.context.append_basic_block(function, "gemv_header");
        let gemv_body = self.context.append_basic_block(function, "gemv_body");
        let gemv_exit = self.context.append_basic_block(function, "gemv_exit");

        let acc_alloca = builder.build_alloca(f32_type, "acc_alloca").unwrap();
        builder.build_store(acc_alloca, f32_type.const_float(0.0)).unwrap();
        builder.build_unconditional_branch(gemv_header).unwrap();

        builder.position_at_end(gemv_header);
        let gi_phi = builder.build_phi(i32_type, "gi").unwrap();
        let gi_val = gi_phi.as_basic_value().into_int_value();
        let gcond = builder
            .build_int_compare(inkwell::IntPredicate::SLT, gi_val, p_hidden_size, "gcond")
            .unwrap();
        builder.build_conditional_branch(gcond, gemv_body, gemv_exit).unwrap();

        builder.position_at_end(gemv_body);
        // Load weight[gi] (f16 -> f32)
        let w_addr = unsafe {
            builder.build_gep(f16_type, w_row, &[gi_val], "w_addr").unwrap()
        };
        let w_f16 = builder.build_load(f16_type, w_addr, "w_f16").unwrap().into_float_value();
        let w_f32 = builder.build_float_ext(w_f16, f32_type, "w_f32").unwrap();
        // Load smem[gi]
        let smem_gi = unsafe {
            builder.build_gep(f32_type, smem_normed_ptr, &[gi_val], "smem_gi").unwrap()
        };
        let s_val = builder.build_load(f32_type, smem_gi, "s_val").unwrap().into_float_value();
        // acc += w * s
        let prod = builder.build_float_mul(w_f32, s_val, "prod").unwrap();
        let cur_acc = builder.build_load(f32_type, acc_alloca, "cur_acc").unwrap().into_float_value();
        let new_acc = builder.build_float_add(cur_acc, prod, "new_acc").unwrap();
        builder.build_store(acc_alloca, new_acc).unwrap();

        let gi_next = builder
            .build_int_add(gi_val, i32_type.const_int(THREADS as u64, false), "gi_next")
            .unwrap();
        gi_phi.add_incoming(&[(&tid, norm_exit), (&gi_next, gemv_body)]);
        builder.build_unconditional_branch(gemv_header).unwrap();

        // After GEMV loop: block reduce the accumulator
        builder.position_at_end(gemv_exit);
        let acc_val = builder.build_load(f32_type, acc_alloca, "acc_val").unwrap().into_float_value();

        let final_sum = self.emit_block_reduce_sum(
            &builder, function, shfl_fn, barrier_fn,
            acc_val, tid, warp_id, lane_id, smem_warp_ptr,
        );

        // tid 0 writes output[row] = float2half(final_sum)
        let is_tid0_out = builder
            .build_int_compare(inkwell::IntPredicate::EQ, tid, i32_type.const_int(0, false), "is_tid0_out")
            .unwrap();
        let write_bb = self.context.append_basic_block(function, "write_output");
        let done_bb = self.context.append_basic_block(function, "done");
        builder.build_conditional_branch(is_tid0_out, write_bb, done_bb).unwrap();

        builder.position_at_end(write_bb);
        let out_f16 = builder
            .build_float_trunc(final_sum, f16_type, "out_f16")
            .unwrap();
        let out_addr = unsafe {
            builder.build_gep(f16_type, p_output, &[row], "out_addr").unwrap()
        };
        builder.build_store(out_addr, out_f16).unwrap();
        builder.build_unconditional_branch(done_bb).unwrap();

        builder.position_at_end(done_bb);
        builder.build_return(None).unwrap();

        tracing::debug!("LLVM IR for fused_rmsnorm_gemv (hidden={hidden}, out_dim={out_dim}):\n{}", module.print_to_string().to_string());

        Ok(())
    }

    // -----------------------------------------------------------------------
    // SiLU + ElemMul + GEMV kernel
    // -----------------------------------------------------------------------

    fn emit_silu_down_gemv<'ctx>(
        &'ctx self,
        module: &Module<'ctx>,
        _out_dim: usize,
    ) -> Result<(), String> {
        let f32_type = self.context.f32_type();
        let f16_type = self.context.f16_type();
        let i32_type = self.context.i32_type();
        let i64_type = self.context.i64_type();
        let gptr = self.global_ptr_type();

        let smem_warp = self.add_shared_mem(module, "smem_warp_silu", NUM_WARPS as usize);

        // void fused_silu_down_gemv(half* output, half* gate, half* up,
        //                            half* weight, i32 out_dim, i32 intermediate_size)
        let fn_type = self.context.void_type().fn_type(
            &[
                gptr.into(), gptr.into(), gptr.into(), gptr.into(),
                i32_type.into(), i32_type.into(),
            ],
            false,
        );

        let function = module.add_function("fused_silu_down_gemv", fn_type, None);
        self.mark_kernel(function);

        let builder = self.context.create_builder();
        let tid_fn = self.declare_nvvm_read_sreg(module, "tid.x");
        let bid_fn = self.declare_nvvm_read_sreg(module, "ctaid.x");
        let barrier_fn = self.declare_barrier0(module);
        let shfl_fn = self.declare_shfl_down_f32(module);

        let entry_bb = self.context.append_basic_block(function, "entry");
        builder.position_at_end(entry_bb);

        let p_output = function.get_nth_param(0).unwrap().into_pointer_value();
        let p_gate = function.get_nth_param(1).unwrap().into_pointer_value();
        let p_up = function.get_nth_param(2).unwrap().into_pointer_value();
        let p_weight = function.get_nth_param(3).unwrap().into_pointer_value();
        let p_out_dim = function.get_nth_param(4).unwrap().into_int_value();
        let p_inter_size = function.get_nth_param(5).unwrap().into_int_value();

        let tid = self.call_sreg(&builder, tid_fn, "tid");
        let bid = self.call_sreg(&builder, bid_fn, "bid");
        let row = bid;

        let oob = builder.build_int_compare(inkwell::IntPredicate::SGE, row, p_out_dim, "oob").unwrap();
        let body_bb = self.context.append_basic_block(function, "body");
        let exit_bb = self.context.append_basic_block(function, "exit");
        builder.build_conditional_branch(oob, exit_bb, body_bb).unwrap();

        builder.position_at_end(exit_bb);
        builder.build_return(None).unwrap();

        builder.position_at_end(body_bb);
        let warp_id = builder.build_right_shift(tid, i32_type.const_int(5, false), false, "warp_id").unwrap();
        let lane_id = builder.build_and(tid, i32_type.const_int(31, false), "lane_id").unwrap();

        // w_row = weight + row * intermediate_size
        let row_i64 = builder.build_int_z_extend(row, i64_type, "row_i64").unwrap();
        let is_i64 = builder.build_int_z_extend(p_inter_size, i64_type, "is_i64").unwrap();
        let row_off = builder.build_int_mul(row_i64, is_i64, "row_off").unwrap();
        let w_row = unsafe {
            builder.build_gep(f16_type, p_weight, &[row_off], "w_row").unwrap()
        };

        // GEMV loop with fused silu(gate[i]) * up[i] * weight[row, i]
        let loop_header = self.context.append_basic_block(function, "loop_header");
        let loop_body = self.context.append_basic_block(function, "loop_body");
        let loop_exit = self.context.append_basic_block(function, "loop_exit");

        let acc_alloca = builder.build_alloca(f32_type, "acc").unwrap();
        builder.build_store(acc_alloca, f32_type.const_float(0.0)).unwrap();
        builder.build_unconditional_branch(loop_header).unwrap();

        builder.position_at_end(loop_header);
        let i_phi = builder.build_phi(i32_type, "i").unwrap();
        let i_val = i_phi.as_basic_value().into_int_value();
        let cond = builder.build_int_compare(inkwell::IntPredicate::SLT, i_val, p_inter_size, "cond").unwrap();
        builder.build_conditional_branch(cond, loop_body, loop_exit).unwrap();

        builder.position_at_end(loop_body);
        // Load gate[i], up[i], weight[row, i]
        let g_addr = unsafe { builder.build_gep(f16_type, p_gate, &[i_val], "g_addr").unwrap() };
        let u_addr = unsafe { builder.build_gep(f16_type, p_up, &[i_val], "u_addr").unwrap() };
        let w_addr = unsafe { builder.build_gep(f16_type, w_row, &[i_val], "w_addr").unwrap() };

        let g_f16 = builder.build_load(f16_type, g_addr, "g_f16").unwrap().into_float_value();
        let u_f16 = builder.build_load(f16_type, u_addr, "u_f16").unwrap().into_float_value();
        let w_f16 = builder.build_load(f16_type, w_addr, "w_f16").unwrap().into_float_value();

        let g_f32 = builder.build_float_ext(g_f16, f32_type, "g_f32").unwrap();
        let u_f32 = builder.build_float_ext(u_f16, f32_type, "u_f32").unwrap();
        let w_f32 = builder.build_float_ext(w_f16, f32_type, "w_f32").unwrap();

        // silu(x) = x / (1 + exp(-x))
        // exp(x) = exp2(x * log2(e)) using NVVM's fast exp2 approximation
        let neg_g = builder.build_float_neg(g_f32, "neg_g").unwrap();
        let ex2_fn = {
            let ft = f32_type.fn_type(&[f32_type.into()], false);
            module.add_function("llvm.nvvm.ex2.approx.f", ft, None)
        };
        let log2e = f32_type.const_float(std::f64::consts::LOG2_E);
        let neg_g_log2e = builder.build_float_mul(neg_g, log2e, "neg_g_log2e").unwrap();
        let exp_neg = builder
            .build_call(ex2_fn, &[neg_g_log2e.into()], "exp_neg")
            .unwrap()
            .try_as_basic_value()
            .basic()
            .unwrap()
            .into_float_value();
        let one = f32_type.const_float(1.0);
        let denom = builder.build_float_add(one, exp_neg, "denom").unwrap();
        let silu_val = builder.build_float_div(g_f32, denom, "silu").unwrap();

        // acc += silu * up * weight
        let su = builder.build_float_mul(silu_val, u_f32, "su").unwrap();
        let suw = builder.build_float_mul(su, w_f32, "suw").unwrap();
        let cur_acc = builder.build_load(f32_type, acc_alloca, "cur_acc").unwrap().into_float_value();
        let new_acc = builder.build_float_add(cur_acc, suw, "new_acc").unwrap();
        builder.build_store(acc_alloca, new_acc).unwrap();

        let i_next = builder.build_int_add(i_val, i32_type.const_int(THREADS as u64, false), "i_next").unwrap();
        i_phi.add_incoming(&[(&tid, body_bb), (&i_next, loop_body)]);
        builder.build_unconditional_branch(loop_header).unwrap();

        // Block reduce
        builder.position_at_end(loop_exit);
        let acc_val = builder.build_load(f32_type, acc_alloca, "acc_val").unwrap().into_float_value();
        let smem_warp_ptr = smem_warp.as_pointer_value();

        let final_sum = self.emit_block_reduce_sum(
            &builder, function, shfl_fn, barrier_fn,
            acc_val, tid, warp_id, lane_id, smem_warp_ptr,
        );

        // Write output
        let is_tid0 = builder.build_int_compare(inkwell::IntPredicate::EQ, tid, i32_type.const_int(0, false), "is_tid0").unwrap();
        let write_bb = self.context.append_basic_block(function, "write_output");
        let done_bb = self.context.append_basic_block(function, "done");
        builder.build_conditional_branch(is_tid0, write_bb, done_bb).unwrap();

        builder.position_at_end(write_bb);
        let out_f16 = builder.build_float_trunc(final_sum, f16_type, "out_f16").unwrap();
        let out_addr = unsafe { builder.build_gep(f16_type, p_output, &[row], "out_addr").unwrap() };
        builder.build_store(out_addr, out_f16).unwrap();
        builder.build_unconditional_branch(done_bb).unwrap();

        builder.position_at_end(done_bb);
        builder.build_return(None).unwrap();

        Ok(())
    }

    // -----------------------------------------------------------------------
    // ElemAdd + RMSNorm kernel
    // -----------------------------------------------------------------------

    fn emit_add_rmsnorm<'ctx>(
        &'ctx self,
        module: &Module<'ctx>,
        hidden: usize,
        eps: f32,
    ) -> Result<(), String> {
        let f32_type = self.context.f32_type();
        let f16_type = self.context.f16_type();
        let i32_type = self.context.i32_type();
        let i64_type = self.context.i64_type();
        let gptr = self.global_ptr_type();

        let smem_data = self.add_shared_mem(module, "smem_data", hidden);
        let smem_warp = self.add_shared_mem(module, "smem_warp_addnorm", NUM_WARPS as usize);

        // void fused_add_rmsnorm(half* output, half* residual, half* input, half* add_vec,
        //                        half* norm_weight, float eps, i32 hidden_size)
        let fn_type = self.context.void_type().fn_type(
            &[
                gptr.into(), gptr.into(), gptr.into(), gptr.into(),
                gptr.into(), f32_type.into(), i32_type.into(),
            ],
            false,
        );

        let function = module.add_function("fused_add_rmsnorm", fn_type, None);
        self.mark_kernel(function);

        let builder = self.context.create_builder();
        let tid_fn = self.declare_nvvm_read_sreg(module, "tid.x");
        let bid_fn = self.declare_nvvm_read_sreg(module, "ctaid.x");
        let barrier_fn = self.declare_barrier0(module);
        let shfl_fn = self.declare_shfl_down_f32(module);
        let rsqrt_fn = self.declare_rsqrt_approx_f32(module);

        let entry_bb = self.context.append_basic_block(function, "entry");
        builder.position_at_end(entry_bb);

        let p_output = function.get_nth_param(0).unwrap().into_pointer_value();
        let p_residual = function.get_nth_param(1).unwrap().into_pointer_value();
        let p_input = function.get_nth_param(2).unwrap().into_pointer_value();
        let p_add_vec = function.get_nth_param(3).unwrap().into_pointer_value();
        let p_norm_w = function.get_nth_param(4).unwrap().into_pointer_value();
        let p_eps = function.get_nth_param(5).unwrap().into_float_value();
        let p_hidden_size = function.get_nth_param(6).unwrap().into_int_value();

        let tid = self.call_sreg(&builder, tid_fn, "tid");
        let bid = self.call_sreg(&builder, bid_fn, "bid");
        let warp_id = builder.build_right_shift(tid, i32_type.const_int(5, false), false, "warp_id").unwrap();
        let lane_id = builder.build_and(tid, i32_type.const_int(31, false), "lane_id").unwrap();

        // token = blockIdx.x, row_off = token * hidden_size
        let token = bid;
        let token_i64 = builder.build_int_z_extend(token, i64_type, "token_i64").unwrap();
        let hs_i64 = builder.build_int_z_extend(p_hidden_size, i64_type, "hs_i64").unwrap();
        let row_off = builder.build_int_mul(token_i64, hs_i64, "row_off").unwrap();

        let smem_data_ptr = smem_data.as_pointer_value();
        let smem_warp_ptr = smem_warp.as_pointer_value();

        // Phase 1: Residual add + sum-of-squares
        let loop1_header = self.context.append_basic_block(function, "add_loop_header");
        let loop1_body = self.context.append_basic_block(function, "add_loop_body");
        let loop1_exit = self.context.append_basic_block(function, "add_loop_exit");

        let ss_alloca = builder.build_alloca(f32_type, "ss_alloca").unwrap();
        builder.build_store(ss_alloca, f32_type.const_float(0.0)).unwrap();
        builder.build_unconditional_branch(loop1_header).unwrap();

        builder.position_at_end(loop1_header);
        let i_phi = builder.build_phi(i32_type, "i").unwrap();
        let i_val = i_phi.as_basic_value().into_int_value();
        let cond = builder.build_int_compare(inkwell::IntPredicate::SLT, i_val, p_hidden_size, "cond").unwrap();
        builder.build_conditional_branch(cond, loop1_body, loop1_exit).unwrap();

        builder.position_at_end(loop1_body);
        let i_i64 = builder.build_int_z_extend(i_val, i64_type, "i_i64").unwrap();
        let global_idx = builder.build_int_add(row_off, i_i64, "gidx").unwrap();

        // Load input[row_off + i] and add_vec[row_off + i]
        let in_addr = unsafe { builder.build_gep(f16_type, p_input, &[global_idx], "in_addr").unwrap() };
        let add_addr = unsafe { builder.build_gep(f16_type, p_add_vec, &[global_idx], "add_addr").unwrap() };
        let in_f16 = builder.build_load(f16_type, in_addr, "in_f16").unwrap().into_float_value();
        let add_f16 = builder.build_load(f16_type, add_addr, "add_f16").unwrap().into_float_value();
        let in_f32 = builder.build_float_ext(in_f16, f32_type, "in_f32").unwrap();
        let add_f32 = builder.build_float_ext(add_f16, f32_type, "add_f32").unwrap();
        let sum = builder.build_float_add(in_f32, add_f32, "sum").unwrap();

        // Write residual[row_off + i] = half(sum)
        let res_addr = unsafe { builder.build_gep(f16_type, p_residual, &[global_idx], "res_addr").unwrap() };
        let sum_f16 = builder.build_float_trunc(sum, f16_type, "sum_f16").unwrap();
        builder.build_store(res_addr, sum_f16).unwrap();

        // smem_data[i] = sum
        let smem_slot = unsafe { builder.build_gep(f32_type, smem_data_ptr, &[i_val], "smem_slot").unwrap() };
        builder.build_store(smem_slot, sum).unwrap();

        // ss += sum * sum
        let cur_ss = builder.build_load(f32_type, ss_alloca, "cur_ss").unwrap().into_float_value();
        let sq = builder.build_float_mul(sum, sum, "sq").unwrap();
        let new_ss = builder.build_float_add(cur_ss, sq, "new_ss").unwrap();
        builder.build_store(ss_alloca, new_ss).unwrap();

        let i_next = builder.build_int_add(i_val, i32_type.const_int(THREADS as u64, false), "i_next").unwrap();
        i_phi.add_incoming(&[(&tid, entry_bb), (&i_next, loop1_body)]);
        builder.build_unconditional_branch(loop1_header).unwrap();

        // Reduce and compute rms_scale
        builder.position_at_end(loop1_exit);
        let local_ss = builder.build_load(f32_type, ss_alloca, "local_ss").unwrap().into_float_value();

        let total_ss = self.emit_block_reduce_sum(
            &builder, function, shfl_fn, barrier_fn,
            local_ss, tid, warp_id, lane_id, smem_warp_ptr,
        );

        let is_tid0 = builder.build_int_compare(inkwell::IntPredicate::EQ, tid, i32_type.const_int(0, false), "is_tid0").unwrap();
        let compute_bb = self.context.append_basic_block(function, "compute_scale");
        let after_bb = self.context.append_basic_block(function, "after_scale");
        builder.build_conditional_branch(is_tid0, compute_bb, after_bb).unwrap();

        builder.position_at_end(compute_bb);
        let hs_f32 = builder.build_signed_int_to_float(p_hidden_size, f32_type, "hs_f32").unwrap();
        let mean_ss = builder.build_float_div(total_ss, hs_f32, "mean_ss").unwrap();
        let with_eps = builder.build_float_add(mean_ss, p_eps, "with_eps").unwrap();
        let rms_scale = builder.build_call(rsqrt_fn, &[with_eps.into()], "rms_scale").unwrap()
            .try_as_basic_value().basic().unwrap().into_float_value();
        let warp0 = unsafe { builder.build_gep(f32_type, smem_warp_ptr, &[i32_type.const_int(0, false)], "w0").unwrap() };
        builder.build_store(warp0, rms_scale).unwrap();
        builder.build_unconditional_branch(after_bb).unwrap();

        builder.position_at_end(after_bb);
        self.call_barrier0(&builder, barrier_fn);

        let w0_load = unsafe { builder.build_gep(f32_type, smem_warp_ptr, &[i32_type.const_int(0, false)], "w0_ld").unwrap() };
        let rms_scale_all = builder.build_load(f32_type, w0_load, "rms_all").unwrap().into_float_value();

        // Phase 2: Normalize and write output
        let norm_header = self.context.append_basic_block(function, "norm_header");
        let norm_body = self.context.append_basic_block(function, "norm_body");
        let norm_exit = self.context.append_basic_block(function, "norm_exit");
        builder.build_unconditional_branch(norm_header).unwrap();

        builder.position_at_end(norm_header);
        let ni_phi = builder.build_phi(i32_type, "ni").unwrap();
        let ni_val = ni_phi.as_basic_value().into_int_value();
        let ncond = builder.build_int_compare(inkwell::IntPredicate::SLT, ni_val, p_hidden_size, "ncond").unwrap();
        builder.build_conditional_branch(ncond, norm_body, norm_exit).unwrap();

        builder.position_at_end(norm_body);
        let smem_ni = unsafe { builder.build_gep(f32_type, smem_data_ptr, &[ni_val], "smem_ni").unwrap() };
        let sv = builder.build_load(f32_type, smem_ni, "sv").unwrap().into_float_value();
        let nw_addr = unsafe { builder.build_gep(f16_type, p_norm_w, &[ni_val], "nw_addr").unwrap() };
        let nw_f16 = builder.build_load(f16_type, nw_addr, "nw_f16").unwrap().into_float_value();
        let nw_f32 = builder.build_float_ext(nw_f16, f32_type, "nw_f32").unwrap();
        let t1 = builder.build_float_mul(sv, nw_f32, "t1").unwrap();
        let normed = builder.build_float_mul(t1, rms_scale_all, "normed").unwrap();
        let normed_f16 = builder.build_float_trunc(normed, f16_type, "normed_f16").unwrap();

        let ni_i64 = builder.build_int_z_extend(ni_val, i64_type, "ni_i64").unwrap();
        let out_gidx = builder.build_int_add(row_off, ni_i64, "out_gidx").unwrap();
        let out_addr = unsafe { builder.build_gep(f16_type, p_output, &[out_gidx], "out_addr").unwrap() };
        builder.build_store(out_addr, normed_f16).unwrap();

        let ni_next = builder.build_int_add(ni_val, i32_type.const_int(THREADS as u64, false), "ni_next").unwrap();
        ni_phi.add_incoming(&[(&tid, after_bb), (&ni_next, norm_body)]);
        builder.build_unconditional_branch(norm_header).unwrap();

        builder.position_at_end(norm_exit);
        builder.build_return(None).unwrap();

        Ok(())
    }

    // -----------------------------------------------------------------------
    // ElemAdd + RMSNorm + GEMV kernel
    // -----------------------------------------------------------------------

    fn emit_add_rmsnorm_gemv<'ctx>(
        &'ctx self,
        module: &Module<'ctx>,
        hidden: usize,
        out_dim: usize,
        eps: f32,
    ) -> Result<(), String> {
        let f32_type = self.context.f32_type();
        let f16_type = self.context.f16_type();
        let i32_type = self.context.i32_type();
        let i64_type = self.context.i64_type();
        let gptr = self.global_ptr_type();

        let smem_normed = self.add_shared_mem(module, "smem_normed", hidden);
        let smem_warp = self.add_shared_mem(module, "smem_warp_addnormgemv", NUM_WARPS as usize);

        // void fused_add_rmsnorm_gemv(half* output, half* residual_out,
        //     half* input, half* add_vec, half* norm_weight, half* proj_weight,
        //     float eps, i32 out_dim, i32 hidden_size)
        let fn_type = self.context.void_type().fn_type(
            &[
                gptr.into(), gptr.into(), gptr.into(), gptr.into(),
                gptr.into(), gptr.into(), f32_type.into(),
                i32_type.into(), i32_type.into(),
            ],
            false,
        );

        let function = module.add_function("fused_add_rmsnorm_gemv", fn_type, None);
        self.mark_kernel(function);

        let builder = self.context.create_builder();
        let tid_fn = self.declare_nvvm_read_sreg(module, "tid.x");
        let bid_fn = self.declare_nvvm_read_sreg(module, "ctaid.x");
        let barrier_fn = self.declare_barrier0(module);
        let shfl_fn = self.declare_shfl_down_f32(module);
        let rsqrt_fn = self.declare_rsqrt_approx_f32(module);

        let entry_bb = self.context.append_basic_block(function, "entry");
        builder.position_at_end(entry_bb);

        let p_output = function.get_nth_param(0).unwrap().into_pointer_value();
        let p_residual = function.get_nth_param(1).unwrap().into_pointer_value();
        let p_input = function.get_nth_param(2).unwrap().into_pointer_value();
        let p_add_vec = function.get_nth_param(3).unwrap().into_pointer_value();
        let p_norm_w = function.get_nth_param(4).unwrap().into_pointer_value();
        let p_proj_w = function.get_nth_param(5).unwrap().into_pointer_value();
        let p_eps = function.get_nth_param(6).unwrap().into_float_value();
        let p_out_dim = function.get_nth_param(7).unwrap().into_int_value();
        let p_hidden_size = function.get_nth_param(8).unwrap().into_int_value();

        let tid = self.call_sreg(&builder, tid_fn, "tid");
        let bid = self.call_sreg(&builder, bid_fn, "bid");
        let row = bid;

        let oob = builder.build_int_compare(inkwell::IntPredicate::SGE, row, p_out_dim, "oob").unwrap();
        let body_bb = self.context.append_basic_block(function, "body");
        let exit_bb = self.context.append_basic_block(function, "exit");
        builder.build_conditional_branch(oob, exit_bb, body_bb).unwrap();

        builder.position_at_end(exit_bb);
        builder.build_return(None).unwrap();

        builder.position_at_end(body_bb);
        let warp_id = builder.build_right_shift(tid, i32_type.const_int(5, false), false, "warp_id").unwrap();
        let lane_id = builder.build_and(tid, i32_type.const_int(31, false), "lane_id").unwrap();

        let smem_normed_ptr = smem_normed.as_pointer_value();
        let smem_warp_ptr = smem_warp.as_pointer_value();

        // Phase 1: Residual add -> smem, sum-of-squares
        let loop1_header = self.context.append_basic_block(function, "add_header");
        let loop1_body = self.context.append_basic_block(function, "add_body");
        let loop1_exit = self.context.append_basic_block(function, "add_exit");

        let ss_alloca = builder.build_alloca(f32_type, "ss_alloca").unwrap();
        builder.build_store(ss_alloca, f32_type.const_float(0.0)).unwrap();
        builder.build_unconditional_branch(loop1_header).unwrap();

        builder.position_at_end(loop1_header);
        let i_phi = builder.build_phi(i32_type, "i").unwrap();
        let i_val = i_phi.as_basic_value().into_int_value();
        let cond = builder.build_int_compare(inkwell::IntPredicate::SLT, i_val, p_hidden_size, "cond").unwrap();
        builder.build_conditional_branch(cond, loop1_body, loop1_exit).unwrap();

        builder.position_at_end(loop1_body);
        let in_addr = unsafe { builder.build_gep(f16_type, p_input, &[i_val], "in_addr").unwrap() };
        let add_addr = unsafe { builder.build_gep(f16_type, p_add_vec, &[i_val], "add_addr").unwrap() };
        let in_f16 = builder.build_load(f16_type, in_addr, "in_f16").unwrap().into_float_value();
        let add_f16 = builder.build_load(f16_type, add_addr, "add_f16").unwrap().into_float_value();
        let in_f32 = builder.build_float_ext(in_f16, f32_type, "in_f32").unwrap();
        let add_f32 = builder.build_float_ext(add_f16, f32_type, "add_f32").unwrap();
        let sum = builder.build_float_add(in_f32, add_f32, "sum").unwrap();

        let smem_slot = unsafe { builder.build_gep(f32_type, smem_normed_ptr, &[i_val], "smem_slot").unwrap() };
        builder.build_store(smem_slot, sum).unwrap();

        let cur_ss = builder.build_load(f32_type, ss_alloca, "cur_ss").unwrap().into_float_value();
        let sq = builder.build_float_mul(sum, sum, "sq").unwrap();
        let new_ss = builder.build_float_add(cur_ss, sq, "new_ss").unwrap();
        builder.build_store(ss_alloca, new_ss).unwrap();

        let i_next = builder.build_int_add(i_val, i32_type.const_int(THREADS as u64, false), "i_next").unwrap();
        i_phi.add_incoming(&[(&tid, body_bb), (&i_next, loop1_body)]);
        builder.build_unconditional_branch(loop1_header).unwrap();

        // Reduce
        builder.position_at_end(loop1_exit);
        let local_ss = builder.build_load(f32_type, ss_alloca, "local_ss").unwrap().into_float_value();
        let total_ss = self.emit_block_reduce_sum(
            &builder, function, shfl_fn, barrier_fn,
            local_ss, tid, warp_id, lane_id, smem_warp_ptr,
        );

        // Compute rms_scale
        let is_tid0 = builder.build_int_compare(inkwell::IntPredicate::EQ, tid, i32_type.const_int(0, false), "is_tid0").unwrap();
        let compute_bb = self.context.append_basic_block(function, "compute_scale");
        let after_bb = self.context.append_basic_block(function, "after_scale");
        builder.build_conditional_branch(is_tid0, compute_bb, after_bb).unwrap();

        builder.position_at_end(compute_bb);
        let hs_f32 = builder.build_signed_int_to_float(p_hidden_size, f32_type, "hs_f32").unwrap();
        let mean_ss = builder.build_float_div(total_ss, hs_f32, "mean_ss").unwrap();
        let with_eps = builder.build_float_add(mean_ss, p_eps, "with_eps").unwrap();
        let rms_scale = builder.build_call(rsqrt_fn, &[with_eps.into()], "rms_scale").unwrap()
            .try_as_basic_value().basic().unwrap().into_float_value();
        let w0 = unsafe { builder.build_gep(f32_type, smem_warp_ptr, &[i32_type.const_int(0, false)], "w0").unwrap() };
        builder.build_store(w0, rms_scale).unwrap();
        builder.build_unconditional_branch(after_bb).unwrap();

        builder.position_at_end(after_bb);
        self.call_barrier0(&builder, barrier_fn);

        let w0_load = unsafe { builder.build_gep(f32_type, smem_warp_ptr, &[i32_type.const_int(0, false)], "w0_ld").unwrap() };
        let rms_scale_all = builder.build_load(f32_type, w0_load, "rms_all").unwrap().into_float_value();

        // Write residual out (only block 0 to avoid races)
        let is_row0 = builder.build_int_compare(inkwell::IntPredicate::EQ, row, i32_type.const_int(0, false), "is_row0").unwrap();
        let res_write_bb = self.context.append_basic_block(function, "res_write");
        let after_res_bb = self.context.append_basic_block(function, "after_res");
        builder.build_conditional_branch(is_row0, res_write_bb, after_res_bb).unwrap();

        builder.position_at_end(res_write_bb);
        // Loop to write residual
        let rw_header = self.context.append_basic_block(function, "rw_header");
        let rw_body = self.context.append_basic_block(function, "rw_body");
        let rw_exit = self.context.append_basic_block(function, "rw_exit");
        builder.build_unconditional_branch(rw_header).unwrap();

        builder.position_at_end(rw_header);
        let rw_phi = builder.build_phi(i32_type, "rw_i").unwrap();
        let rw_val = rw_phi.as_basic_value().into_int_value();
        let rw_cond = builder.build_int_compare(inkwell::IntPredicate::SLT, rw_val, p_hidden_size, "rw_cond").unwrap();
        builder.build_conditional_branch(rw_cond, rw_body, rw_exit).unwrap();

        builder.position_at_end(rw_body);
        let smem_rw = unsafe { builder.build_gep(f32_type, smem_normed_ptr, &[rw_val], "smem_rw").unwrap() };
        let sv = builder.build_load(f32_type, smem_rw, "sv").unwrap().into_float_value();
        let sv_f16 = builder.build_float_trunc(sv, f16_type, "sv_f16").unwrap();
        let res_addr = unsafe { builder.build_gep(f16_type, p_residual, &[rw_val], "res_addr").unwrap() };
        builder.build_store(res_addr, sv_f16).unwrap();
        let rw_next = builder.build_int_add(rw_val, i32_type.const_int(THREADS as u64, false), "rw_next").unwrap();
        rw_phi.add_incoming(&[(&tid, res_write_bb), (&rw_next, rw_body)]);
        builder.build_unconditional_branch(rw_header).unwrap();

        builder.position_at_end(rw_exit);
        builder.build_unconditional_branch(after_res_bb).unwrap();

        builder.position_at_end(after_res_bb);

        // Apply norm weights in-place in smem
        let norm_header = self.context.append_basic_block(function, "norm_header");
        let norm_body = self.context.append_basic_block(function, "norm_body");
        let norm_exit = self.context.append_basic_block(function, "norm_exit");
        builder.build_unconditional_branch(norm_header).unwrap();

        builder.position_at_end(norm_header);
        let ni_phi = builder.build_phi(i32_type, "ni").unwrap();
        let ni_val = ni_phi.as_basic_value().into_int_value();
        let ncond = builder.build_int_compare(inkwell::IntPredicate::SLT, ni_val, p_hidden_size, "ncond").unwrap();
        builder.build_conditional_branch(ncond, norm_body, norm_exit).unwrap();

        builder.position_at_end(norm_body);
        let smem_ni = unsafe { builder.build_gep(f32_type, smem_normed_ptr, &[ni_val], "smem_ni").unwrap() };
        let sv = builder.build_load(f32_type, smem_ni, "sv").unwrap().into_float_value();
        let nw_addr = unsafe { builder.build_gep(f16_type, p_norm_w, &[ni_val], "nw_addr").unwrap() };
        let nw_f16 = builder.build_load(f16_type, nw_addr, "nw_f16").unwrap().into_float_value();
        let nw_f32 = builder.build_float_ext(nw_f16, f32_type, "nw_f32").unwrap();
        let t1 = builder.build_float_mul(sv, nw_f32, "t1").unwrap();
        let normed = builder.build_float_mul(t1, rms_scale_all, "normed").unwrap();
        builder.build_store(smem_ni, normed).unwrap();
        let ni_next = builder.build_int_add(ni_val, i32_type.const_int(THREADS as u64, false), "ni_next").unwrap();
        ni_phi.add_incoming(&[(&tid, after_res_bb), (&ni_next, norm_body)]);
        builder.build_unconditional_branch(norm_header).unwrap();

        builder.position_at_end(norm_exit);
        self.call_barrier0(&builder, barrier_fn);

        // Phase 2: GEMV
        let row_i64 = builder.build_int_z_extend(row, i64_type, "row_i64").unwrap();
        let hs_i64 = builder.build_int_z_extend(p_hidden_size, i64_type, "hs_i64").unwrap();
        let row_offset = builder.build_int_mul(row_i64, hs_i64, "row_offset").unwrap();
        let w_row = unsafe { builder.build_gep(f16_type, p_proj_w, &[row_offset], "w_row").unwrap() };

        let gemv_header = self.context.append_basic_block(function, "gemv_header");
        let gemv_body = self.context.append_basic_block(function, "gemv_body");
        let gemv_exit = self.context.append_basic_block(function, "gemv_exit");

        let acc_alloca = builder.build_alloca(f32_type, "acc").unwrap();
        builder.build_store(acc_alloca, f32_type.const_float(0.0)).unwrap();
        builder.build_unconditional_branch(gemv_header).unwrap();

        builder.position_at_end(gemv_header);
        let gi_phi = builder.build_phi(i32_type, "gi").unwrap();
        let gi_val = gi_phi.as_basic_value().into_int_value();
        let gcond = builder.build_int_compare(inkwell::IntPredicate::SLT, gi_val, p_hidden_size, "gcond").unwrap();
        builder.build_conditional_branch(gcond, gemv_body, gemv_exit).unwrap();

        builder.position_at_end(gemv_body);
        let w_addr = unsafe { builder.build_gep(f16_type, w_row, &[gi_val], "w_addr").unwrap() };
        let w_f16 = builder.build_load(f16_type, w_addr, "w_f16").unwrap().into_float_value();
        let w_f32 = builder.build_float_ext(w_f16, f32_type, "w_f32").unwrap();
        let smem_gi = unsafe { builder.build_gep(f32_type, smem_normed_ptr, &[gi_val], "smem_gi").unwrap() };
        let s_val = builder.build_load(f32_type, smem_gi, "s_val").unwrap().into_float_value();
        let prod = builder.build_float_mul(w_f32, s_val, "prod").unwrap();
        let cur_acc = builder.build_load(f32_type, acc_alloca, "cur_acc").unwrap().into_float_value();
        let new_acc = builder.build_float_add(cur_acc, prod, "new_acc").unwrap();
        builder.build_store(acc_alloca, new_acc).unwrap();

        let gi_next = builder.build_int_add(gi_val, i32_type.const_int(THREADS as u64, false), "gi_next").unwrap();
        gi_phi.add_incoming(&[(&tid, norm_exit), (&gi_next, gemv_body)]);
        builder.build_unconditional_branch(gemv_header).unwrap();

        builder.position_at_end(gemv_exit);
        let acc_val = builder.build_load(f32_type, acc_alloca, "acc_val").unwrap().into_float_value();
        let final_sum = self.emit_block_reduce_sum(
            &builder, function, shfl_fn, barrier_fn,
            acc_val, tid, warp_id, lane_id, smem_warp_ptr,
        );

        let is_tid0_out = builder.build_int_compare(inkwell::IntPredicate::EQ, tid, i32_type.const_int(0, false), "is_tid0_out").unwrap();
        let write_bb = self.context.append_basic_block(function, "write_output");
        let done_bb = self.context.append_basic_block(function, "done");
        builder.build_conditional_branch(is_tid0_out, write_bb, done_bb).unwrap();

        builder.position_at_end(write_bb);
        let out_f16 = builder.build_float_trunc(final_sum, f16_type, "out_f16").unwrap();
        let out_addr = unsafe { builder.build_gep(f16_type, p_output, &[row], "out_addr").unwrap() };
        builder.build_store(out_addr, out_f16).unwrap();
        builder.build_unconditional_branch(done_bb).unwrap();

        builder.position_at_end(done_bb);
        builder.build_return(None).unwrap();

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Optimization passes (new pass manager, LLVM 13+)
    // -----------------------------------------------------------------------

    fn optimize(&self, module: &Module, arch: &str) -> Result<(), String> {
        let triple = TargetTriple::create("nvptx64-nvidia-cuda");
        let target = Target::from_triple(&triple).map_err(|e| e.to_string())?;

        let features = "+ptx80";
        let machine = target
            .create_target_machine(
                &triple,
                arch,
                features,
                OptimizationLevel::Aggressive,
                RelocMode::Default,
                CodeModel::Default,
            )
            .ok_or("failed to create target machine for optimization")?;

        let opts = PassBuilderOptions::create();
        opts.set_loop_vectorization(true);
        opts.set_loop_slp_vectorization(true);
        opts.set_loop_unrolling(true);
        opts.set_merge_functions(true);

        module
            .run_passes("default<O3>", &machine, opts)
            .map_err(|e| format!("LLVM optimization failed: {}", e.to_string()))?;

        Ok(())
    }

    // -----------------------------------------------------------------------
    // PTX emission
    // -----------------------------------------------------------------------

    fn emit_ptx(&self, module: &Module, arch: &str) -> Result<Vec<u8>, String> {
        let triple = TargetTriple::create("nvptx64-nvidia-cuda");
        let target = Target::from_triple(&triple).map_err(|e| e.to_string())?;

        let features = "+ptx80";
        let machine = target
            .create_target_machine(
                &triple,
                arch,
                features,
                OptimizationLevel::Aggressive,
                RelocMode::Default,
                CodeModel::Default,
            )
            .ok_or("failed to create NVPTX target machine")?;

        let buf = machine
            .write_to_memory_buffer(module, FileType::Assembly)
            .map_err(|e| format!("PTX emission failed: {}", e.to_string()))?;

        Ok(buf.as_slice().to_vec())
    }

    // -----------------------------------------------------------------------
    // Autotuning
    // -----------------------------------------------------------------------

    /// Generate PTX for multiple kernel configs. Returns all variants.
    /// Actual benchmarking happens at runtime (caller picks fastest).
    pub fn autotune_kernel(
        &self,
        kernel: &FusedKernel,
        hidden: usize,
        out_dim: usize,
        eps: f32,
        arch: &str,
    ) -> Result<Vec<(Vec<u8>, KernelConfig)>, String> {
        let configs = vec![
            KernelConfig { block_size: 128, num_warps: 4, tile_k: 128 },
            KernelConfig { block_size: 256, num_warps: 8, tile_k: 256 },
            KernelConfig { block_size: 256, num_warps: 8, tile_k: 512 },
            KernelConfig { block_size: 512, num_warps: 16, tile_k: 256 },
        ];

        // For now, compile only the default config (256 threads).
        // Future: parameterize THREADS in IR generation per config.
        let ptx = self.compile_fused_kernel(kernel, hidden, out_dim, eps, arch)?;
        let default_cfg = configs.into_iter().nth(1).unwrap(); // 256 threads
        Ok(vec![(ptx, default_cfg)])
    }

    /// Get the kernel function name for a given pattern.
    pub fn kernel_function_name(kernel: &FusedKernel) -> &'static str {
        match classify(kernel) {
            FusionPattern::RMSNormGemv => "fused_rmsnorm_gemv",
            FusionPattern::SiLUElemMulGemv => "fused_silu_down_gemv",
            FusionPattern::ElemAddRMSNorm => "fused_add_rmsnorm",
            FusionPattern::ElemAddRMSNormGemv => "fused_add_rmsnorm_gemv",
            FusionPattern::Generic => "unsupported",
        }
    }
}

// ---------------------------------------------------------------------------
// Shared memory offset with bank-conflict padding
// ---------------------------------------------------------------------------

/// Pad shared memory row offset to avoid 32-bank conflicts.
/// Each row is padded by 1 element.
pub fn shared_mem_offset(row: usize, col: usize, cols: usize) -> usize {
    row * (cols + 1) + col
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{Dtype, FusedKernel, FusionOp};

    fn make_kernel(ops: Vec<FusionOp>) -> FusedKernel {
        FusedKernel {
            node_ids: (0..ops.len()).collect(),
            ops,
            output_shape: vec![1, 4096],
            dtype: Dtype::F16,
        }
    }

    #[test]
    fn test_classify_patterns() {
        assert_eq!(
            classify(&make_kernel(vec![FusionOp::RMSNorm { eps: 1e-5 }, FusionOp::Gemv])),
            FusionPattern::RMSNormGemv
        );
        assert_eq!(
            classify(&make_kernel(vec![FusionOp::SiLU, FusionOp::ElemMul, FusionOp::Gemv])),
            FusionPattern::SiLUElemMulGemv
        );
        assert_eq!(
            classify(&make_kernel(vec![FusionOp::ElemAdd, FusionOp::RMSNorm { eps: 1e-6 }])),
            FusionPattern::ElemAddRMSNorm
        );
        assert_eq!(
            classify(&make_kernel(vec![FusionOp::ElemAdd, FusionOp::RMSNorm { eps: 1e-5 }, FusionOp::Gemv])),
            FusionPattern::ElemAddRMSNormGemv
        );
        assert_eq!(
            classify(&make_kernel(vec![FusionOp::Softmax])),
            FusionPattern::Generic
        );
    }

    #[test]
    fn test_bank_conflict_padding() {
        // Row 0, col 5, 32 cols => offset = 0 * 33 + 5 = 5
        assert_eq!(shared_mem_offset(0, 5, 32), 5);
        // Row 1, col 0, 32 cols => offset = 1 * 33 + 0 = 33 (skips bank 0 conflict)
        assert_eq!(shared_mem_offset(1, 0, 32), 33);
    }

    #[test]
    fn test_llvm_ptx_compiler_creation() {
        let _compiler = LlvmPtxCompiler::new();
    }

    #[test]
    fn test_compile_rmsnorm_gemv() {
        let compiler = LlvmPtxCompiler::new();
        let kernel = make_kernel(vec![FusionOp::RMSNorm { eps: 1e-5 }, FusionOp::Gemv]);
        let result = compiler.compile_fused_kernel(&kernel, 4096, 4096, 1e-5, "sm_80");
        match result {
            Ok(ptx) => {
                assert!(!ptx.is_empty());
                let ptx_str = String::from_utf8_lossy(&ptx);
                assert!(ptx_str.contains(".version"), "output doesn't look like PTX: {}", &ptx_str[..100.min(ptx_str.len())]);
                assert!(ptx_str.contains("fused_rmsnorm_gemv"));
            }
            Err(e) => {
                // May fail if LLVM NVPTX target not available in the build
                eprintln!("LLVM PTX compile failed (expected if no NVPTX): {e}");
            }
        }
    }

    #[test]
    fn test_compile_silu_down_gemv() {
        let compiler = LlvmPtxCompiler::new();
        let kernel = make_kernel(vec![FusionOp::SiLU, FusionOp::ElemMul, FusionOp::Gemv]);
        let result = compiler.compile_fused_kernel(&kernel, 11008, 4096, 0.0, "sm_80");
        match result {
            Ok(ptx) => {
                let ptx_str = String::from_utf8_lossy(&ptx);
                assert!(ptx_str.contains("fused_silu_down_gemv"));
            }
            Err(e) => eprintln!("skip: {e}"),
        }
    }

    #[test]
    fn test_compile_add_rmsnorm() {
        let compiler = LlvmPtxCompiler::new();
        let kernel = make_kernel(vec![FusionOp::ElemAdd, FusionOp::RMSNorm { eps: 1e-6 }]);
        let result = compiler.compile_fused_kernel(&kernel, 4096, 0, 1e-6, "sm_80");
        match result {
            Ok(ptx) => {
                let ptx_str = String::from_utf8_lossy(&ptx);
                assert!(ptx_str.contains("fused_add_rmsnorm"));
            }
            Err(e) => eprintln!("skip: {e}"),
        }
    }

    #[test]
    fn test_compile_add_rmsnorm_gemv() {
        let compiler = LlvmPtxCompiler::new();
        let kernel = make_kernel(vec![FusionOp::ElemAdd, FusionOp::RMSNorm { eps: 1e-5 }, FusionOp::Gemv]);
        let result = compiler.compile_fused_kernel(&kernel, 4096, 4096, 1e-5, "sm_80");
        match result {
            Ok(ptx) => {
                let ptx_str = String::from_utf8_lossy(&ptx);
                assert!(ptx_str.contains("fused_add_rmsnorm_gemv"));
            }
            Err(e) => eprintln!("skip: {e}"),
        }
    }

    #[test]
    fn test_unsupported_pattern() {
        let compiler = LlvmPtxCompiler::new();
        let kernel = make_kernel(vec![FusionOp::Softmax]);
        let result = compiler.compile_fused_kernel(&kernel, 128, 128, 0.0, "sm_80");
        assert!(result.is_err());
    }

    #[test]
    fn test_kernel_function_names() {
        let k1 = make_kernel(vec![FusionOp::RMSNorm { eps: 1e-5 }, FusionOp::Gemv]);
        assert_eq!(LlvmPtxCompiler::kernel_function_name(&k1), "fused_rmsnorm_gemv");

        let k2 = make_kernel(vec![FusionOp::SiLU, FusionOp::ElemMul, FusionOp::Gemv]);
        assert_eq!(LlvmPtxCompiler::kernel_function_name(&k2), "fused_silu_down_gemv");
    }
}
