//! Direct PTX code emission for fused kernel patterns.
//!
//! Generates PTX assembly text loadable via cuModuleLoadData, replacing the
//! CUDA C -> nvcc pipeline. No external compiler dependency.

use std::collections::HashMap;
use std::fmt::Write;

use crate::ir::{FusedKernel, FusionOp};

const THREADS: u32 = 256;
const WARP_SIZE: u32 = 32;
const WARPS: u32 = THREADS / WARP_SIZE;

// ---------------------------------------------------------------------------
// PTX types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PtxType {
    Pred,
    U16,
    U32,
    U64,
    S32,
    S64,
    F16,
    F32,
    F64,
    B16,
    B32,
    B64,
}

impl PtxType {
    fn as_str(self) -> &'static str {
        match self {
            PtxType::Pred => ".pred",
            PtxType::U16 => ".u16",
            PtxType::U32 => ".u32",
            PtxType::U64 => ".u64",
            PtxType::S32 => ".s32",
            PtxType::S64 => ".s64",
            PtxType::F16 => ".f16",
            PtxType::F32 => ".f32",
            PtxType::F64 => ".f64",
            PtxType::B16 => ".b16",
            PtxType::B32 => ".b32",
            PtxType::B64 => ".b64",
        }
    }

    fn param_str(self) -> &'static str {
        match self {
            PtxType::Pred => ".pred",
            PtxType::U16 => ".u16",
            PtxType::U32 => ".u32",
            PtxType::U64 => ".u64",
            PtxType::S32 => ".s32",
            PtxType::S64 => ".s64",
            PtxType::F16 => ".b16",
            PtxType::F32 => ".f32",
            PtxType::F64 => ".f64",
            PtxType::B16 => ".b16",
            PtxType::B32 => ".b32",
            PtxType::B64 => ".b64",
        }
    }

    fn reg_prefix(self) -> &'static str {
        match self {
            PtxType::Pred => "%p",
            PtxType::U16 | PtxType::B16 | PtxType::F16 => "%h",
            PtxType::U32 | PtxType::S32 | PtxType::B32 => "%r",
            PtxType::U64 | PtxType::S64 | PtxType::B64 => "%rd",
            PtxType::F32 => "%f",
            PtxType::F64 => "%fd",
        }
    }
}

// ---------------------------------------------------------------------------
// PtxEmitter
// ---------------------------------------------------------------------------

pub struct PtxEmitter {
    buf: String,
    reg_counters: HashMap<&'static str, u32>,
    label_counter: u32,
    arch: String,
    indent: u32,
}

impl PtxEmitter {
    pub fn new(arch: &str) -> Self {
        Self {
            buf: String::with_capacity(8192),
            reg_counters: HashMap::new(),
            label_counter: 0,
            arch: arch.to_string(),
            indent: 0,
        }
    }

    fn ind(&self) -> String {
        "    ".repeat(self.indent as usize)
    }

    fn line(&mut self, s: &str) {
        let _ = writeln!(self.buf, "{}{}", self.ind(), s);
    }

    fn blank(&mut self) {
        self.buf.push('\n');
    }

    // -- Header / structure ---------------------------------------------------

    pub fn emit_header(&mut self) {
        let _ = writeln!(self.buf, ".version 8.0");
        let _ = writeln!(self.buf, ".target {}", self.arch);
        let _ = writeln!(self.buf, ".address_size 64");
        self.blank();
    }

    pub fn emit_kernel_start(&mut self, name: &str, params: &[(&str, PtxType)]) {
        let _ = write!(self.buf, ".visible .entry {}(\n", name);
        for (i, (pname, pty)) in params.iter().enumerate() {
            let comma = if i + 1 < params.len() { "," } else { "" };
            let _ = writeln!(self.buf, "    .param {} {}{}", pty.param_str(), pname, comma);
        }
        self.buf.push_str(")\n{\n");
        self.indent = 1;
    }

    pub fn emit_kernel_end(&mut self) {
        self.indent = 0;
        self.buf.push_str("}\n");
    }

    // -- Register allocation --------------------------------------------------

    pub fn alloc_reg(&mut self, ty: PtxType) -> String {
        let prefix = ty.reg_prefix();
        let counter = self.reg_counters.entry(prefix).or_insert(0);
        let id = *counter;
        *counter += 1;
        format!("{}{}", prefix, id)
    }

    pub fn alloc_pred(&mut self) -> String {
        self.alloc_reg(PtxType::Pred)
    }

    pub fn alloc_label(&mut self) -> String {
        let id = self.label_counter;
        self.label_counter += 1;
        format!("$L{}", id)
    }

    /// Insert a placeholder for register declarations. The actual declarations
    /// are resolved in `finish()` after all registers have been allocated.
    pub fn emit_reg_decls(&mut self) {
        self.line("// __REG_DECLS_PLACEHOLDER__");
        self.blank();
    }

    // -- Shared memory --------------------------------------------------------

    pub fn emit_shared_decl(&mut self, name: &str, bytes: usize, align: usize) {
        self.line(&format!(
            ".shared .align {} .b8 {}[{}];",
            align, name, bytes
        ));
    }

    // -- Parameter loads ------------------------------------------------------

    pub fn emit_ld_param(&mut self, dst: &str, param: &str, ty: PtxType) {
        self.line(&format!(
            "ld.param{} {}, [{}];",
            ty.param_str(),
            dst,
            param
        ));
    }

    // -- Global memory --------------------------------------------------------

    pub fn emit_ld_global(&mut self, dst: &str, addr: &str, ty: PtxType) {
        self.line(&format!(
            "ld.global{} {}, [{}];",
            ty.as_str(),
            dst,
            addr
        ));
    }

    pub fn emit_ld_global_v2_b32(&mut self, dst0: &str, dst1: &str, addr: &str) {
        self.line(&format!(
            "ld.global.v2.b32 {{{}, {}}}, [{}];",
            dst0, dst1, addr
        ));
    }

    pub fn emit_st_global(&mut self, addr: &str, src: &str, ty: PtxType) {
        self.line(&format!(
            "st.global{} [{}], {};",
            ty.as_str(),
            addr,
            src
        ));
    }

    pub fn emit_st_global_v2_b32(&mut self, addr: &str, src0: &str, src1: &str) {
        self.line(&format!(
            "st.global.v2.b32 [{}], {{{}, {}}};",
            addr, src0, src1
        ));
    }

    // -- Shared memory loads/stores -------------------------------------------

    pub fn emit_ld_shared(&mut self, dst: &str, addr: &str, ty: PtxType) {
        self.line(&format!(
            "ld.shared{} {}, [{}];",
            ty.as_str(),
            dst,
            addr
        ));
    }

    pub fn emit_st_shared(&mut self, addr: &str, src: &str, ty: PtxType) {
        self.line(&format!(
            "st.shared{} [{}], {};",
            ty.as_str(),
            addr,
            src
        ));
    }

    // -- Conversions ----------------------------------------------------------

    pub fn emit_cvt_f32_f16(&mut self, dst: &str, src: &str) {
        self.line(&format!("cvt.f32.f16 {}, {};", dst, src));
    }

    pub fn emit_cvt_f16_f32(&mut self, dst: &str, src: &str) {
        self.line(&format!("cvt.rn.f16.f32 {}, {};", dst, src));
    }

    pub fn emit_cvt(&mut self, dst: &str, src: &str, dst_ty: PtxType, src_ty: PtxType) {
        let rn = if matches!(dst_ty, PtxType::F16 | PtxType::F32) { ".rn" } else { "" };
        self.line(&format!(
            "cvt{}{}{} {}, {};",
            rn,
            dst_ty.as_str(),
            src_ty.as_str(),
            dst,
            src
        ));
    }

    // -- Arithmetic -----------------------------------------------------------

    pub fn emit_fma(&mut self, dst: &str, a: &str, b: &str, c: &str) {
        self.line(&format!("fma.rn.f32 {}, {}, {}, {};", dst, a, b, c));
    }

    pub fn emit_add(&mut self, dst: &str, a: &str, b: &str, ty: PtxType) {
        self.line(&format!("add{} {}, {}, {};", ty.as_str(), dst, a, b));
    }

    pub fn emit_mul(&mut self, dst: &str, a: &str, b: &str, ty: PtxType) {
        let rn = if matches!(ty, PtxType::F32 | PtxType::F64) { ".rn" } else { "" };
        self.line(&format!(
            "mul{}{} {}, {}, {};",
            rn,
            ty.as_str(),
            dst,
            a,
            b
        ));
    }

    pub fn emit_mul_lo(&mut self, dst: &str, a: &str, b: &str, ty: PtxType) {
        self.line(&format!("mul.lo{} {}, {}, {};", ty.as_str(), dst, a, b));
    }

    pub fn emit_mul_wide(&mut self, dst: &str, a: &str, b: &str, src_ty: PtxType) {
        self.line(&format!(
            "mul.wide{} {}, {}, {};",
            src_ty.as_str(),
            dst,
            a,
            b
        ));
    }

    pub fn emit_mad_lo(&mut self, dst: &str, a: &str, b: &str, c: &str, ty: PtxType) {
        self.line(&format!(
            "mad.lo{} {}, {}, {}, {};",
            ty.as_str(),
            dst,
            a,
            b,
            c
        ));
    }

    pub fn emit_mad_wide(&mut self, dst: &str, a: &str, b: &str, c: &str, src_ty: PtxType) {
        self.line(&format!(
            "mad.wide{} {}, {}, {}, {};",
            src_ty.as_str(),
            dst,
            a,
            b,
            c
        ));
    }

    pub fn emit_rsqrt(&mut self, dst: &str, src: &str) {
        self.line(&format!("rsqrt.approx.ftz.f32 {}, {};", dst, src));
    }

    pub fn emit_rcp(&mut self, dst: &str, src: &str) {
        self.line(&format!("rcp.approx.ftz.f32 {}, {};", dst, src));
    }

    pub fn emit_neg(&mut self, dst: &str, src: &str, ty: PtxType) {
        self.line(&format!("neg{} {}, {};", ty.as_str(), dst, src));
    }

    pub fn emit_ex2(&mut self, dst: &str, src: &str) {
        self.line(&format!("ex2.approx.ftz.f32 {}, {};", dst, src));
    }

    pub fn emit_div(&mut self, dst: &str, a: &str, b: &str, ty: PtxType) {
        let rn = if matches!(ty, PtxType::F32 | PtxType::F64) { ".rn" } else { "" };
        self.line(&format!(
            "div{}{} {}, {}, {};",
            rn,
            ty.as_str(),
            dst,
            a,
            b
        ));
    }

    pub fn emit_shr(&mut self, dst: &str, a: &str, b: &str, ty: PtxType) {
        self.line(&format!("shr{} {}, {}, {};", ty.as_str(), dst, a, b));
    }

    pub fn emit_shl(&mut self, dst: &str, a: &str, b: &str, ty: PtxType) {
        self.line(&format!("shl{} {}, {}, {};", ty.as_str(), dst, a, b));
    }

    pub fn emit_and(&mut self, dst: &str, a: &str, b: &str, ty: PtxType) {
        self.line(&format!("and{} {}, {}, {};", ty.as_str(), dst, a, b));
    }

    pub fn emit_or(&mut self, dst: &str, a: &str, b: &str, ty: PtxType) {
        self.line(&format!("or{} {}, {}, {};", ty.as_str(), dst, a, b));
    }

    // -- Shuffle / sync -------------------------------------------------------

    pub fn emit_shfl_down(&mut self, dst: &str, src: &str, offset: u32) {
        self.line(&format!(
            "shfl.sync.down.b32 {}, {}, {}, 0x1f, 0xffffffff;",
            dst, src, offset
        ));
    }

    pub fn emit_bar_sync(&mut self) {
        self.line("bar.sync 0;");
    }

    // -- Move / set / branch --------------------------------------------------

    pub fn emit_mov(&mut self, dst: &str, src: &str, ty: PtxType) {
        self.line(&format!("mov{} {}, {};", ty.as_str(), dst, src));
    }

    pub fn emit_mov_imm_f32(&mut self, dst: &str, val: f32) {
        self.line(&format!("mov.f32 {}, 0f{:08X};", dst, val.to_bits()));
    }

    pub fn emit_mov_imm_u32(&mut self, dst: &str, val: u32) {
        self.line(&format!("mov.u32 {}, {};", dst, val));
    }

    pub fn emit_mov_imm_u64(&mut self, dst: &str, val: u64) {
        self.line(&format!("mov.u64 {}, {};", dst, val));
    }

    pub fn emit_setp(&mut self, pred: &str, op: &str, a: &str, b: &str, ty: PtxType) {
        self.line(&format!(
            "setp.{}{} {}, {}, {};",
            op,
            ty.as_str(),
            pred,
            a,
            b
        ));
    }

    pub fn emit_selp(&mut self, dst: &str, a: &str, b: &str, pred: &str, ty: PtxType) {
        self.line(&format!(
            "selp{} {}, {}, {}, {};",
            ty.as_str(),
            dst,
            a,
            b,
            pred
        ));
    }

    pub fn emit_bra(&mut self, pred: Option<&str>, label: &str) {
        if let Some(p) = pred {
            self.line(&format!("@{} bra {};", p, label));
        } else {
            self.line(&format!("bra {};", label));
        }
    }

    pub fn emit_bra_neg(&mut self, pred: &str, label: &str) {
        self.line(&format!("@!{} bra {};", pred, label));
    }

    pub fn emit_label(&mut self, name: &str) {
        // Labels are not indented inside kernel body
        let _ = writeln!(self.buf, "{}:", name);
    }

    pub fn emit_ret(&mut self) {
        self.line("ret;");
    }

    // -- Special registers ----------------------------------------------------

    pub fn emit_thread_id(&mut self, dst: &str) {
        self.line(&format!("mov.u32 {}, %tid.x;", dst));
    }

    pub fn emit_block_id(&mut self, dst: &str) {
        self.line(&format!("mov.u32 {}, %ctaid.x;", dst));
    }

    pub fn emit_warp_id(&mut self, dst: &str, tid: &str) {
        self.line(&format!("shr.u32 {}, {}, 5;", dst, tid));
    }

    pub fn emit_lane_id(&mut self, dst: &str, tid: &str) {
        self.line(&format!("and.b32 {}, {}, 31;", dst, tid));
    }

    // -- High-level patterns --------------------------------------------------

    /// Emit a full warp-level reduce-sum using shfl.down.
    /// `val` register is overwritten with the reduction result (valid in lane 0).
    pub fn emit_warp_reduce_sum(&mut self, val: &str) {
        let tmp = self.alloc_reg(PtxType::F32);
        for offset in [16, 8, 4, 2, 1] {
            self.emit_shfl_down(&tmp, val, offset);
            self.emit_add(val, val, &tmp, PtxType::F32);
        }
    }

    /// Emit cross-warp reduction: each warp writes to smem, then warp 0 reduces.
    /// Expects: val = per-warp partial sum (valid in lane 0 of each warp).
    /// After: result_reg contains the block-wide sum (valid in lane 0 of warp 0).
    /// smem_base is the byte offset in shared memory for scratch space (needs WARPS*4 bytes).
    pub fn emit_block_reduce_sum(
        &mut self,
        val: &str,
        lane_id: &str,
        warp_id: &str,
        smem_name: &str,
        smem_offset: usize,
    ) {
        let p_lane0 = self.alloc_pred();
        let p_warp0 = self.alloc_pred();

        // lane 0 of each warp writes its partial sum
        self.emit_setp(&p_lane0, "eq", lane_id, "0", PtxType::U32);
        let wr_off = self.alloc_reg(PtxType::U32);
        self.emit_mul_lo(&wr_off, warp_id, "4", PtxType::U32);
        if smem_offset > 0 {
            self.emit_add(&wr_off, &wr_off, &format!("{}", smem_offset), PtxType::U32);
        }
        let wr_addr = self.alloc_reg(PtxType::U64);
        self.emit_cvt(&wr_addr, &wr_off, PtxType::U64, PtxType::U32);
        // add smem base
        let smem_base = self.alloc_reg(PtxType::U64);
        self.emit_mov(smem_base.as_str(), smem_name, PtxType::U64);
        self.emit_add(&wr_addr, &wr_addr, &smem_base, PtxType::U64);

        let skip_wr = self.alloc_label();
        self.emit_bra_neg(&p_lane0, &skip_wr);
        self.emit_st_shared(&wr_addr, val, PtxType::F32);
        self.emit_label(&skip_wr);
        self.emit_bar_sync();

        // Warp 0 loads all partial sums and reduces
        self.emit_setp(&p_warp0, "eq", warp_id, "0", PtxType::U32);
        let skip_reduce = self.alloc_label();
        self.emit_bra_neg(&p_warp0, &skip_reduce);

        let p_in_range = self.alloc_pred();
        self.emit_setp(
            &p_in_range,
            "lt",
            lane_id,
            &format!("{}", WARPS),
            PtxType::U32,
        );
        let rd_off = self.alloc_reg(PtxType::U32);
        self.emit_mul_lo(&rd_off, lane_id, "4", PtxType::U32);
        if smem_offset > 0 {
            self.emit_add(&rd_off, &rd_off, &format!("{}", smem_offset), PtxType::U32);
        }
        let rd_addr = self.alloc_reg(PtxType::U64);
        self.emit_cvt(&rd_addr, &rd_off, PtxType::U64, PtxType::U32);
        self.emit_add(&rd_addr, &rd_addr, &smem_base, PtxType::U64);

        let zero_f = self.alloc_reg(PtxType::F32);
        self.emit_mov_imm_f32(&zero_f, 0.0);
        self.emit_mov(val, &zero_f, PtxType::F32);
        let loaded = self.alloc_reg(PtxType::F32);
        let skip_load = self.alloc_label();
        self.emit_bra_neg(&p_in_range, &skip_load);
        self.emit_ld_shared(&loaded, &rd_addr, PtxType::F32);
        self.emit_mov(val, &loaded, PtxType::F32);
        self.emit_label(&skip_load);

        self.emit_warp_reduce_sum(val);

        self.emit_label(&skip_reduce);
        self.emit_bar_sync();
    }

    /// Convenience: emit early-exit if blockIdx.x >= limit.
    pub fn emit_bounds_check(&mut self, block_id: &str, limit: &str) {
        let p = self.alloc_pred();
        self.emit_setp(&p, "ge", block_id, limit, PtxType::U32);
        let skip = self.alloc_label();
        self.emit_bra_neg(&p, &skip);
        self.emit_ret();
        self.emit_label(&skip);
    }

    // -- Comment helper -------------------------------------------------------

    pub fn comment(&mut self, s: &str) {
        self.line(&format!("// {}", s));
    }

    // -- Finish ---------------------------------------------------------------

    pub fn finish(self) -> String {
        let mut decls = String::new();
        let mut sorted: Vec<_> = self.reg_counters.iter().map(|(k, v)| (*k, *v)).collect();
        sorted.sort_by_key(|(k, _)| *k);
        for (prefix, count) in sorted {
            let ty = match prefix {
                "%p" => ".pred",
                "%h" => ".b16",
                "%r" => ".b32",
                "%rd" => ".b64",
                "%f" => ".f32",
                "%fd" => ".f64",
                _ => ".b32",
            };
            let _ = writeln!(decls, "    .reg {} {}<{}>;", ty, prefix, count);
        }
        self.buf.replace("    // __REG_DECLS_PLACEHOLDER__\n", &decls)
    }
}

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

pub fn classify(kernel: &FusedKernel) -> FusionPattern {
    let tags: Vec<&str> = kernel.ops.iter().map(|op| op_tag(op)).collect();
    match tags.as_slice() {
        ["RMSNorm", "Gemv"] => FusionPattern::RMSNormGemv,
        ["SiLU", "ElemMul", "Gemv"] => FusionPattern::SiLUElemMulGemv,
        ["ElemAdd", "RMSNorm"] => FusionPattern::ElemAddRMSNorm,
        ["ElemAdd", "RMSNorm", "Gemv"] => FusionPattern::ElemAddRMSNormGemv,
        _ => FusionPattern::Generic,
    }
}

// ---------------------------------------------------------------------------
// Tiling configuration
// ---------------------------------------------------------------------------

fn compute_smem_bytes(hidden: usize) -> usize {
    // f32 normed vector + warp scratch
    hidden * 4 + WARPS as usize * 4
}

// ---------------------------------------------------------------------------
// Pattern: RMSNorm + Gemv
// ---------------------------------------------------------------------------

/// Generate PTX for fused RMSNorm -> Gemv kernel.
///
/// Grid: (out_dim, 1, 1), Block: (THREADS, 1, 1)
/// Each block redundantly normalizes the hidden vector in smem, then computes
/// one output element via dot product with proj_weight row.
///
/// Shared memory: hidden_size * 4 (f32 normed) + WARPS * 4 (scratch)
pub fn generate_rmsnorm_gemv_ptx(
    hidden: usize,
    out_dim: usize,
    _eps: f32,
    arch: &str,
) -> String {
    let name = format!("fused_rmsnorm_gemv_{}x{}", hidden, out_dim);
    let smem_bytes = compute_smem_bytes(hidden);
    let warp_scratch_off = hidden * 4;
    let h2 = hidden / 2;

    let mut e = PtxEmitter::new(arch);
    e.emit_header();

    let params: &[(&str, PtxType)] = &[
        ("output_ptr", PtxType::U64),
        ("hidden_ptr", PtxType::U64),
        ("norm_w_ptr", PtxType::U64),
        ("proj_w_ptr", PtxType::U64),
        ("eps", PtxType::F32),
        ("out_dim", PtxType::U32),
        ("hidden_size", PtxType::U32),
    ];
    e.emit_kernel_start(&name, params);

    // Pre-allocate all registers we need
    let tid = e.alloc_reg(PtxType::U32);
    let bid = e.alloc_reg(PtxType::U32);
    let warp_id = e.alloc_reg(PtxType::U32);
    let lane_id = e.alloc_reg(PtxType::U32);

    let r_out_dim = e.alloc_reg(PtxType::U32);
    let r_hidden = e.alloc_reg(PtxType::U32);
    let r_eps = e.alloc_reg(PtxType::F32);
    let rd_output = e.alloc_reg(PtxType::U64);
    let rd_hidden = e.alloc_reg(PtxType::U64);
    let rd_norm_w = e.alloc_reg(PtxType::U64);
    let rd_proj_w = e.alloc_reg(PtxType::U64);

    // Loop index / address regs
    let r_i = e.alloc_reg(PtxType::U32);
    let _r_i2 = e.alloc_reg(PtxType::U32); // reserved slot
    let rd_addr = e.alloc_reg(PtxType::U64);
    let _rd_addr2 = e.alloc_reg(PtxType::U64); // reserved slot
    let rd_smem_base = e.alloc_reg(PtxType::U64);
    let r_smem_off = e.alloc_reg(PtxType::U32);
    let rd_smem_addr = e.alloc_reg(PtxType::U64);

    // f16 load pair (as b32)
    let r_h2_lo = e.alloc_reg(PtxType::B32);
    let _r_h2_hi = e.alloc_reg(PtxType::B32); // reserved slot
    let h_val0 = e.alloc_reg(PtxType::B16);
    let h_val1 = e.alloc_reg(PtxType::B16);

    // f32 working regs
    let f_val = e.alloc_reg(PtxType::F32);
    let f_val2 = e.alloc_reg(PtxType::F32);
    let f_ss = e.alloc_reg(PtxType::F32);
    let f_rms = e.alloc_reg(PtxType::F32);
    let f_acc = e.alloc_reg(PtxType::F32);
    let f_w0 = e.alloc_reg(PtxType::F32);
    let f_w1 = e.alloc_reg(PtxType::F32);
    let f_n0 = e.alloc_reg(PtxType::F32);
    let f_n1 = e.alloc_reg(PtxType::F32);
    let _f_tmp = e.alloc_reg(PtxType::F32); // reserved slot
    let f_hidden_f = e.alloc_reg(PtxType::F32);

    // Norm weight h2 pair
    let r_nw_lo = e.alloc_reg(PtxType::B32);
    let _r_nw_hi = e.alloc_reg(PtxType::B32); // reserved slot
    let h_nw0 = e.alloc_reg(PtxType::B16);
    let h_nw1 = e.alloc_reg(PtxType::B16);

    // Proj weight h2 pair
    let r_pw_lo = e.alloc_reg(PtxType::B32);
    let _r_pw_hi = e.alloc_reg(PtxType::B32); // reserved slot
    let h_pw0 = e.alloc_reg(PtxType::B16);
    let h_pw1 = e.alloc_reg(PtxType::B16);

    // Row offset for proj_weight
    let rd_row_off = e.alloc_reg(PtxType::U64);
    let rd_w_row = e.alloc_reg(PtxType::U64);

    // h16 output
    let h_out = e.alloc_reg(PtxType::B16);

    // Write back address
    let rd_out_addr = e.alloc_reg(PtxType::U64);

    // Extra temps for address calc
    let r_byte_off = e.alloc_reg(PtxType::U32);
    let rd_byte_off = e.alloc_reg(PtxType::U64);

    e.emit_reg_decls();

    // Shared memory
    e.emit_shared_decl("smem", smem_bytes, 16);
    e.blank();

    // Load params
    e.emit_ld_param(&rd_output, "output_ptr", PtxType::U64);
    e.emit_ld_param(&rd_hidden, "hidden_ptr", PtxType::U64);
    e.emit_ld_param(&rd_norm_w, "norm_w_ptr", PtxType::U64);
    e.emit_ld_param(&rd_proj_w, "proj_w_ptr", PtxType::U64);
    e.emit_ld_param(&r_eps, "eps", PtxType::F32);
    e.emit_ld_param(&r_out_dim, "out_dim", PtxType::U32);
    e.emit_ld_param(&r_hidden, "hidden_size", PtxType::U32);
    e.blank();

    // Thread/block IDs
    e.emit_thread_id(&tid);
    e.emit_block_id(&bid);
    e.emit_warp_id(&warp_id, &tid);
    e.emit_lane_id(&lane_id, &tid);
    e.blank();

    // Bounds check
    e.emit_bounds_check(&bid, &r_out_dim);
    e.blank();

    // smem base address
    e.emit_mov(&rd_smem_base, "smem", PtxType::U64);
    e.blank();

    // ---- Phase 1: Load hidden -> smem, compute sum-of-squares ----
    e.comment("Phase 1: Load hidden into smem as f32, accumulate sum-of-squares");
    e.emit_mov_imm_f32(&f_ss, 0.0);

    // Loop: for i = tid; i < h2; i += THREADS
    // We load half2 pairs (4 bytes = 2 f16) and convert to f32
    e.emit_mov(&r_i, &tid, PtxType::U32);
    let loop_load = e.alloc_label();
    let end_load = e.alloc_label();
    e.emit_label(&loop_load);
    {
        let p_done = e.alloc_pred();
        e.emit_setp(&p_done, "ge", &r_i, &format!("{}", h2), PtxType::U32);
        e.emit_bra(Some(&p_done), &end_load);

        // Address: hidden_ptr + i * 4 (each half2 = 4 bytes)
        e.emit_mul_lo(&r_byte_off, &r_i, "4", PtxType::U32);
        e.emit_cvt(&rd_byte_off, &r_byte_off, PtxType::U64, PtxType::U32);
        e.emit_add(&rd_addr, &rd_hidden, &rd_byte_off, PtxType::U64);

        // Load half2 as b32
        e.emit_ld_global(&r_h2_lo, &rd_addr, PtxType::B32);

        // Unpack: lo 16 bits and hi 16 bits
        e.line(&format!("mov.b32 {{{}, {}}}, {};", h_val0, h_val1, r_h2_lo));

        // Convert to f32
        e.emit_cvt_f32_f16(&f_val, &h_val0);
        e.emit_cvt_f32_f16(&f_val2, &h_val1);

        // Store to smem as f32: smem[i*2] and smem[i*2+1]
        // smem offset = i * 2 * 4 = i * 8
        e.emit_mul_lo(&r_smem_off, &r_i, "8", PtxType::U32);
        e.emit_cvt(&rd_smem_addr, &r_smem_off, PtxType::U64, PtxType::U32);
        e.emit_add(&rd_smem_addr, &rd_smem_addr, &rd_smem_base, PtxType::U64);
        e.emit_st_shared(&rd_smem_addr, &f_val, PtxType::F32);
        // smem[i*2+1] = offset + 4
        e.emit_add(&rd_smem_addr, &rd_smem_addr, "4", PtxType::U64);
        e.emit_st_shared(&rd_smem_addr, &f_val2, PtxType::F32);

        // Accumulate sum-of-squares
        e.emit_fma(&f_ss, &f_val, &f_val, &f_ss);
        e.emit_fma(&f_ss, &f_val2, &f_val2, &f_ss);

        // i += THREADS
        e.emit_add(&r_i, &r_i, &format!("{}", THREADS), PtxType::U32);
        e.emit_bra(None, &loop_load);
    }
    e.emit_label(&end_load);
    e.blank();

    // Handle odd hidden_size (skip for even, which is the common case)
    if hidden % 2 != 0 {
        e.comment("Handle odd hidden_size tail");
        let p_tid0 = e.alloc_pred();
        let skip_odd = e.alloc_label();
        e.emit_setp(&p_tid0, "ne", &tid, "0", PtxType::U32);
        e.emit_bra(Some(&p_tid0), &skip_odd);
        // Load last element
        let last_off = (hidden - 1) * 2; // byte offset in f16 array
        e.emit_add(&rd_addr, &rd_hidden, &format!("{}", last_off), PtxType::U64);
        e.emit_ld_global(&h_val0, &rd_addr, PtxType::B16);
        e.emit_cvt_f32_f16(&f_val, &h_val0);
        let smem_last = (hidden - 1) * 4;
        e.emit_add(&rd_smem_addr, &rd_smem_base, &format!("{}", smem_last), PtxType::U64);
        e.emit_st_shared(&rd_smem_addr, &f_val, PtxType::F32);
        e.emit_fma(&f_ss, &f_val, &f_val, &f_ss);
        e.emit_label(&skip_odd);
    }

    // Warp reduce sum-of-squares
    e.comment("Warp reduction of sum-of-squares");
    e.emit_warp_reduce_sum(&f_ss);

    // Block-wide reduction via smem
    e.emit_block_reduce_sum(&f_ss, &lane_id, &warp_id, "smem", warp_scratch_off);

    // Lane 0 of warp 0 computes rsqrt(ss / hidden + eps) and writes to smem scratch[0]
    e.comment("Compute RMS scale = rsqrt(ss/hidden + eps)");
    {
        let p_lane0_warp0 = e.alloc_pred();
        let skip_rms = e.alloc_label();
        // Check tid == 0 (lane 0 of warp 0)
        e.emit_setp(&p_lane0_warp0, "ne", &tid, "0", PtxType::U32);
        e.emit_bra(Some(&p_lane0_warp0), &skip_rms);

        e.emit_cvt(&f_hidden_f, &r_hidden, PtxType::F32, PtxType::U32);
        e.emit_div(&f_ss, &f_ss, &f_hidden_f, PtxType::F32);
        e.emit_add(&f_ss, &f_ss, &r_eps, PtxType::F32);
        e.emit_rsqrt(&f_rms, &f_ss);

        // Store to smem scratch[0]
        let warp_scratch_addr = e.alloc_reg(PtxType::U64);
        e.emit_add(
            &warp_scratch_addr,
            &rd_smem_base,
            &format!("{}", warp_scratch_off),
            PtxType::U64,
        );
        e.emit_st_shared(&warp_scratch_addr, &f_rms, PtxType::F32);
        e.emit_label(&skip_rms);
    }
    e.emit_bar_sync();

    // All threads load rms_scale from smem
    {
        let warp_scratch_addr = e.alloc_reg(PtxType::U64);
        e.emit_add(
            &warp_scratch_addr,
            &rd_smem_base,
            &format!("{}", warp_scratch_off),
            PtxType::U64,
        );
        e.emit_ld_shared(&f_rms, &warp_scratch_addr, PtxType::F32);
    }
    e.blank();

    // ---- Phase 1b: Apply norm weights: smem[i] = smem[i] * norm_w[i] * rms_scale ----
    e.comment("Apply RMSNorm: smem[i] *= norm_weight[i] * rms_scale");
    e.emit_mov(&r_i, &tid, PtxType::U32);
    let loop_norm = e.alloc_label();
    let end_norm = e.alloc_label();
    e.emit_label(&loop_norm);
    {
        let p_done = e.alloc_pred();
        e.emit_setp(&p_done, "ge", &r_i, &format!("{}", h2), PtxType::U32);
        e.emit_bra(Some(&p_done), &end_norm);

        // Load norm_weight half2
        e.emit_mul_lo(&r_byte_off, &r_i, "4", PtxType::U32);
        e.emit_cvt(&rd_byte_off, &r_byte_off, PtxType::U64, PtxType::U32);
        e.emit_add(&rd_addr, &rd_norm_w, &rd_byte_off, PtxType::U64);
        e.emit_ld_global(&r_nw_lo, &rd_addr, PtxType::B32);
        e.line(&format!("mov.b32 {{{}, {}}}, {};", h_nw0, h_nw1, r_nw_lo));
        e.emit_cvt_f32_f16(&f_w0, &h_nw0);
        e.emit_cvt_f32_f16(&f_w1, &h_nw1);

        // Load from smem
        e.emit_mul_lo(&r_smem_off, &r_i, "8", PtxType::U32);
        e.emit_cvt(&rd_smem_addr, &r_smem_off, PtxType::U64, PtxType::U32);
        e.emit_add(&rd_smem_addr, &rd_smem_addr, &rd_smem_base, PtxType::U64);
        e.emit_ld_shared(&f_val, &rd_smem_addr, PtxType::F32);
        e.emit_add(&rd_smem_addr, &rd_smem_addr, "4", PtxType::U64);
        e.emit_ld_shared(&f_val2, &rd_smem_addr, PtxType::F32);

        // val * weight * rms_scale
        e.emit_mul(&f_val, &f_val, &f_rms, PtxType::F32);
        e.emit_mul(&f_val, &f_val, &f_w0, PtxType::F32);
        e.emit_mul(&f_val2, &f_val2, &f_rms, PtxType::F32);
        e.emit_mul(&f_val2, &f_val2, &f_w1, PtxType::F32);

        // Store back
        e.emit_mul_lo(&r_smem_off, &r_i, "8", PtxType::U32);
        e.emit_cvt(&rd_smem_addr, &r_smem_off, PtxType::U64, PtxType::U32);
        e.emit_add(&rd_smem_addr, &rd_smem_addr, &rd_smem_base, PtxType::U64);
        e.emit_st_shared(&rd_smem_addr, &f_val, PtxType::F32);
        e.emit_add(&rd_smem_addr, &rd_smem_addr, "4", PtxType::U64);
        e.emit_st_shared(&rd_smem_addr, &f_val2, PtxType::F32);

        e.emit_add(&r_i, &r_i, &format!("{}", THREADS), PtxType::U32);
        e.emit_bra(None, &loop_norm);
    }
    e.emit_label(&end_norm);
    e.emit_bar_sync();
    e.blank();

    // ---- Phase 2: GEMV dot product ----
    e.comment("Phase 2: GEMV - dot(proj_weight[row], normed)");

    // Compute w_row = proj_w_ptr + row * hidden_size * 2 (f16 bytes)
    e.emit_mul_wide(&rd_row_off, &bid, &format!("{}", hidden * 2), PtxType::U32);
    e.emit_add(&rd_w_row, &rd_proj_w, &rd_row_off, PtxType::U64);

    e.emit_mov_imm_f32(&f_acc, 0.0);
    e.emit_mov(&r_i, &tid, PtxType::U32);
    let loop_gemv = e.alloc_label();
    let end_gemv = e.alloc_label();
    e.emit_label(&loop_gemv);
    {
        let p_done = e.alloc_pred();
        e.emit_setp(&p_done, "ge", &r_i, &format!("{}", h2), PtxType::U32);
        e.emit_bra(Some(&p_done), &end_gemv);

        // Load proj_weight half2
        e.emit_mul_lo(&r_byte_off, &r_i, "4", PtxType::U32);
        e.emit_cvt(&rd_byte_off, &r_byte_off, PtxType::U64, PtxType::U32);
        e.emit_add(&rd_addr, &rd_w_row, &rd_byte_off, PtxType::U64);
        e.emit_ld_global(&r_pw_lo, &rd_addr, PtxType::B32);
        e.line(&format!("mov.b32 {{{}, {}}}, {};", h_pw0, h_pw1, r_pw_lo));
        e.emit_cvt_f32_f16(&f_w0, &h_pw0);
        e.emit_cvt_f32_f16(&f_w1, &h_pw1);

        // Load normed from smem
        e.emit_mul_lo(&r_smem_off, &r_i, "8", PtxType::U32);
        e.emit_cvt(&rd_smem_addr, &r_smem_off, PtxType::U64, PtxType::U32);
        e.emit_add(&rd_smem_addr, &rd_smem_addr, &rd_smem_base, PtxType::U64);
        e.emit_ld_shared(&f_n0, &rd_smem_addr, PtxType::F32);
        e.emit_add(&rd_smem_addr, &rd_smem_addr, "4", PtxType::U64);
        e.emit_ld_shared(&f_n1, &rd_smem_addr, PtxType::F32);

        // FMA: acc += w0*n0 + w1*n1
        e.emit_fma(&f_acc, &f_w0, &f_n0, &f_acc);
        e.emit_fma(&f_acc, &f_w1, &f_n1, &f_acc);

        e.emit_add(&r_i, &r_i, &format!("{}", THREADS), PtxType::U32);
        e.emit_bra(None, &loop_gemv);
    }
    e.emit_label(&end_gemv);
    e.blank();

    // Reduce acc across block
    e.comment("Reduce GEMV accumulator across block");
    e.emit_warp_reduce_sum(&f_acc);
    e.emit_block_reduce_sum(&f_acc, &lane_id, &warp_id, "smem", warp_scratch_off);

    // Thread 0 writes output
    e.comment("Thread 0 writes output[row]");
    {
        let p_tid0 = e.alloc_pred();
        let skip_write = e.alloc_label();
        e.emit_setp(&p_tid0, "ne", &tid, "0", PtxType::U32);
        e.emit_bra(Some(&p_tid0), &skip_write);

        // Convert f32 -> f16
        e.emit_cvt_f16_f32(&h_out, &f_acc);

        // output_ptr + bid * 2
        e.emit_mul_wide(&rd_byte_off, &bid, "2", PtxType::U32);
        e.emit_add(&rd_out_addr, &rd_output, &rd_byte_off, PtxType::U64);
        e.emit_st_global(&rd_out_addr, &h_out, PtxType::B16);

        e.emit_label(&skip_write);
    }
    e.blank();

    e.emit_ret();
    e.emit_kernel_end();

    e.finish()
}

// ---------------------------------------------------------------------------
// Pattern: SiLU + ElemMul + Gemv
// ---------------------------------------------------------------------------

/// Generate PTX for fused SiLU -> ElemMul -> Gemv kernel.
///
/// Grid: (out_dim, 1, 1), Block: (THREADS, 1, 1)
/// Each block computes silu(gate[i]) * up[i] * weight[row, i] on the fly.
/// No shared memory needed -- all in registers.
pub fn generate_silu_elemmul_gemv_ptx(
    intermediate: usize,
    out_dim: usize,
    arch: &str,
) -> String {
    let name = format!("fused_silu_mul_gemv_{}x{}", intermediate, out_dim);
    let warp_scratch = WARPS as usize * 4;
    let k2 = intermediate / 2;

    let mut e = PtxEmitter::new(arch);
    e.emit_header();

    let params: &[(&str, PtxType)] = &[
        ("output_ptr", PtxType::U64),
        ("gate_ptr", PtxType::U64),
        ("up_ptr", PtxType::U64),
        ("weight_ptr", PtxType::U64),
        ("out_dim", PtxType::U32),
        ("intermediate_size", PtxType::U32),
    ];
    e.emit_kernel_start(&name, params);

    let tid = e.alloc_reg(PtxType::U32);
    let bid = e.alloc_reg(PtxType::U32);
    let warp_id = e.alloc_reg(PtxType::U32);
    let lane_id = e.alloc_reg(PtxType::U32);

    let r_out_dim = e.alloc_reg(PtxType::U32);
    let r_intermediate = e.alloc_reg(PtxType::U32);
    let rd_output = e.alloc_reg(PtxType::U64);
    let rd_gate = e.alloc_reg(PtxType::U64);
    let rd_up = e.alloc_reg(PtxType::U64);
    let rd_weight = e.alloc_reg(PtxType::U64);

    let r_i = e.alloc_reg(PtxType::U32);
    let rd_addr = e.alloc_reg(PtxType::U64);
    let r_byte_off = e.alloc_reg(PtxType::U32);
    let rd_byte_off = e.alloc_reg(PtxType::U64);
    let rd_row_off = e.alloc_reg(PtxType::U64);
    let rd_w_row = e.alloc_reg(PtxType::U64);

    // half2 load temps
    let r_g_h2 = e.alloc_reg(PtxType::B32);
    let r_u_h2 = e.alloc_reg(PtxType::B32);
    let r_w_h2 = e.alloc_reg(PtxType::B32);
    let h_g0 = e.alloc_reg(PtxType::B16);
    let h_g1 = e.alloc_reg(PtxType::B16);
    let h_u0 = e.alloc_reg(PtxType::B16);
    let h_u1 = e.alloc_reg(PtxType::B16);
    let h_w0 = e.alloc_reg(PtxType::B16);
    let h_w1 = e.alloc_reg(PtxType::B16);

    let f_g0 = e.alloc_reg(PtxType::F32);
    let f_g1 = e.alloc_reg(PtxType::F32);
    let f_u0 = e.alloc_reg(PtxType::F32);
    let f_u1 = e.alloc_reg(PtxType::F32);
    let f_w0 = e.alloc_reg(PtxType::F32);
    let f_w1 = e.alloc_reg(PtxType::F32);
    let f_acc = e.alloc_reg(PtxType::F32);

    // SiLU temps
    let f_neg = e.alloc_reg(PtxType::F32);
    let f_exp = e.alloc_reg(PtxType::F32);
    let f_one = e.alloc_reg(PtxType::F32);
    let f_denom = e.alloc_reg(PtxType::F32);
    let f_sig = e.alloc_reg(PtxType::F32);
    let f_silu = e.alloc_reg(PtxType::F32);

    let h_out = e.alloc_reg(PtxType::B16);
    let rd_out_addr = e.alloc_reg(PtxType::U64);
    let rd_smem_base = e.alloc_reg(PtxType::U64);

    e.emit_reg_decls();

    // Shared memory for warp reduction scratch only
    e.emit_shared_decl("smem", warp_scratch, 16);
    e.blank();

    // Load params
    e.emit_ld_param(&rd_output, "output_ptr", PtxType::U64);
    e.emit_ld_param(&rd_gate, "gate_ptr", PtxType::U64);
    e.emit_ld_param(&rd_up, "up_ptr", PtxType::U64);
    e.emit_ld_param(&rd_weight, "weight_ptr", PtxType::U64);
    e.emit_ld_param(&r_out_dim, "out_dim", PtxType::U32);
    e.emit_ld_param(&r_intermediate, "intermediate_size", PtxType::U32);
    e.blank();

    e.emit_thread_id(&tid);
    e.emit_block_id(&bid);
    e.emit_warp_id(&warp_id, &tid);
    e.emit_lane_id(&lane_id, &tid);
    e.blank();

    e.emit_bounds_check(&bid, &r_out_dim);
    e.blank();

    e.emit_mov(&rd_smem_base, "smem", PtxType::U64);

    // w_row = weight_ptr + row * intermediate * 2
    e.emit_mul_wide(&rd_row_off, &bid, &format!("{}", intermediate * 2), PtxType::U32);
    e.emit_add(&rd_w_row, &rd_weight, &rd_row_off, PtxType::U64);

    e.emit_mov_imm_f32(&f_acc, 0.0);
    e.emit_mov_imm_f32(&f_one, 1.0);
    e.blank();

    // Main loop: SiLU(gate) * up * weight, vectorized half2
    e.comment("Main loop: acc += silu(gate[i]) * up[i] * weight[row, i]");
    e.emit_mov(&r_i, &tid, PtxType::U32);
    let loop_main = e.alloc_label();
    let end_main = e.alloc_label();
    e.emit_label(&loop_main);
    {
        let p_done = e.alloc_pred();
        e.emit_setp(&p_done, "ge", &r_i, &format!("{}", k2), PtxType::U32);
        e.emit_bra(Some(&p_done), &end_main);

        // byte offset = i * 4
        e.emit_mul_lo(&r_byte_off, &r_i, "4", PtxType::U32);
        e.emit_cvt(&rd_byte_off, &r_byte_off, PtxType::U64, PtxType::U32);

        // Load gate half2
        e.emit_add(&rd_addr, &rd_gate, &rd_byte_off, PtxType::U64);
        e.emit_ld_global(&r_g_h2, &rd_addr, PtxType::B32);
        e.line(&format!("mov.b32 {{{}, {}}}, {};", h_g0, h_g1, r_g_h2));
        e.emit_cvt_f32_f16(&f_g0, &h_g0);
        e.emit_cvt_f32_f16(&f_g1, &h_g1);

        // Load up half2
        e.emit_add(&rd_addr, &rd_up, &rd_byte_off, PtxType::U64);
        e.emit_ld_global(&r_u_h2, &rd_addr, PtxType::B32);
        e.line(&format!("mov.b32 {{{}, {}}}, {};", h_u0, h_u1, r_u_h2));
        e.emit_cvt_f32_f16(&f_u0, &h_u0);
        e.emit_cvt_f32_f16(&f_u1, &h_u1);

        // Load weight half2
        e.emit_add(&rd_addr, &rd_w_row, &rd_byte_off, PtxType::U64);
        e.emit_ld_global(&r_w_h2, &rd_addr, PtxType::B32);
        e.line(&format!("mov.b32 {{{}, {}}}, {};", h_w0, h_w1, r_w_h2));
        e.emit_cvt_f32_f16(&f_w0, &h_w0);
        e.emit_cvt_f32_f16(&f_w1, &h_w1);

        // SiLU(g0) = g0 * sigmoid(g0) = g0 / (1 + exp(-g0))
        // Using ex2 for fast exp: exp(x) = 2^(x * log2(e))
        // log2(e) = 1.4426950408889634
        // silu(x) = x / (1 + exp2(-x * 1.4426950408889634))
        e.comment("silu(g0)");
        e.emit_neg(&f_neg, &f_g0, PtxType::F32);
        e.emit_mul(&f_neg, &f_neg, "0f3FB8AA3B", PtxType::F32); // * log2(e)
        e.emit_ex2(&f_exp, &f_neg);
        e.emit_add(&f_denom, &f_one, &f_exp, PtxType::F32);
        e.emit_rcp(&f_sig, &f_denom);
        e.emit_mul(&f_silu, &f_g0, &f_sig, PtxType::F32);

        // acc += silu * up * weight
        e.emit_mul(&f_silu, &f_silu, &f_u0, PtxType::F32);
        e.emit_fma(&f_acc, &f_silu, &f_w0, &f_acc);

        // silu(g1)
        e.comment("silu(g1)");
        e.emit_neg(&f_neg, &f_g1, PtxType::F32);
        e.emit_mul(&f_neg, &f_neg, "0f3FB8AA3B", PtxType::F32);
        e.emit_ex2(&f_exp, &f_neg);
        e.emit_add(&f_denom, &f_one, &f_exp, PtxType::F32);
        e.emit_rcp(&f_sig, &f_denom);
        e.emit_mul(&f_silu, &f_g1, &f_sig, PtxType::F32);
        e.emit_mul(&f_silu, &f_silu, &f_u1, PtxType::F32);
        e.emit_fma(&f_acc, &f_silu, &f_w1, &f_acc);

        e.emit_add(&r_i, &r_i, &format!("{}", THREADS), PtxType::U32);
        e.emit_bra(None, &loop_main);
    }
    e.emit_label(&end_main);
    e.blank();

    // Block-wide reduction
    e.comment("Reduce accumulator across block");
    e.emit_warp_reduce_sum(&f_acc);
    e.emit_block_reduce_sum(&f_acc, &lane_id, &warp_id, "smem", 0);

    // Thread 0 writes output
    {
        let p_tid0 = e.alloc_pred();
        let skip_write = e.alloc_label();
        e.emit_setp(&p_tid0, "ne", &tid, "0", PtxType::U32);
        e.emit_bra(Some(&p_tid0), &skip_write);
        e.emit_cvt_f16_f32(&h_out, &f_acc);
        e.emit_mul_wide(&rd_byte_off, &bid, "2", PtxType::U32);
        e.emit_add(&rd_out_addr, &rd_output, &rd_byte_off, PtxType::U64);
        e.emit_st_global(&rd_out_addr, &h_out, PtxType::B16);
        e.emit_label(&skip_write);
    }
    e.blank();

    e.emit_ret();
    e.emit_kernel_end();

    e.finish()
}

// ---------------------------------------------------------------------------
// Pattern: ElemAdd + RMSNorm
// ---------------------------------------------------------------------------

/// Generate PTX for fused ElemAdd -> RMSNorm kernel.
///
/// Grid: (num_tokens, 1, 1), Block: (THREADS, 1, 1)
/// Each block handles one token row: residual = input + add, then RMSNorm.
/// Outputs both the residual and the normed result.
pub fn generate_elemadd_rmsnorm_ptx(hidden: usize, _eps: f32, arch: &str) -> String {
    let name = format!("fused_elemadd_rmsnorm_{}", hidden);
    let smem_bytes = compute_smem_bytes(hidden);
    let warp_scratch_off = hidden * 4;
    let h2 = hidden / 2;

    let mut e = PtxEmitter::new(arch);
    e.emit_header();

    let params: &[(&str, PtxType)] = &[
        ("output_ptr", PtxType::U64),
        ("residual_ptr", PtxType::U64),
        ("input_ptr", PtxType::U64),
        ("add_ptr", PtxType::U64),
        ("norm_w_ptr", PtxType::U64),
        ("eps", PtxType::F32),
        ("hidden_size", PtxType::U32),
    ];
    e.emit_kernel_start(&name, params);

    let tid = e.alloc_reg(PtxType::U32);
    let bid = e.alloc_reg(PtxType::U32);
    let warp_id = e.alloc_reg(PtxType::U32);
    let lane_id = e.alloc_reg(PtxType::U32);

    let r_hidden = e.alloc_reg(PtxType::U32);
    let r_eps = e.alloc_reg(PtxType::F32);
    let rd_output = e.alloc_reg(PtxType::U64);
    let rd_residual = e.alloc_reg(PtxType::U64);
    let rd_input = e.alloc_reg(PtxType::U64);
    let rd_add = e.alloc_reg(PtxType::U64);
    let rd_norm_w = e.alloc_reg(PtxType::U64);

    // Row offset base pointers
    let rd_in_row = e.alloc_reg(PtxType::U64);
    let rd_add_row = e.alloc_reg(PtxType::U64);
    let rd_res_row = e.alloc_reg(PtxType::U64);
    let rd_out_row = e.alloc_reg(PtxType::U64);
    let rd_row_off = e.alloc_reg(PtxType::U64);

    let r_i = e.alloc_reg(PtxType::U32);
    let rd_addr = e.alloc_reg(PtxType::U64);
    let rd_addr2 = e.alloc_reg(PtxType::U64);
    let rd_smem_base = e.alloc_reg(PtxType::U64);
    let r_smem_off = e.alloc_reg(PtxType::U32);
    let rd_smem_addr = e.alloc_reg(PtxType::U64);
    let r_byte_off = e.alloc_reg(PtxType::U32);
    let rd_byte_off = e.alloc_reg(PtxType::U64);

    // half2 loads
    let r_in_h2 = e.alloc_reg(PtxType::B32);
    let r_add_h2 = e.alloc_reg(PtxType::B32);
    let h_in0 = e.alloc_reg(PtxType::B16);
    let h_in1 = e.alloc_reg(PtxType::B16);
    let h_add0 = e.alloc_reg(PtxType::B16);
    let h_add1 = e.alloc_reg(PtxType::B16);

    let f_in0 = e.alloc_reg(PtxType::F32);
    let f_in1 = e.alloc_reg(PtxType::F32);
    let f_add0 = e.alloc_reg(PtxType::F32);
    let f_add1 = e.alloc_reg(PtxType::F32);
    let f_v0 = e.alloc_reg(PtxType::F32);
    let f_v1 = e.alloc_reg(PtxType::F32);
    let f_ss = e.alloc_reg(PtxType::F32);
    let f_rms = e.alloc_reg(PtxType::F32);
    let f_hidden_f = e.alloc_reg(PtxType::F32);

    // Norm weight
    let r_nw_h2 = e.alloc_reg(PtxType::B32);
    let h_nw0 = e.alloc_reg(PtxType::B16);
    let h_nw1 = e.alloc_reg(PtxType::B16);
    let f_nw0 = e.alloc_reg(PtxType::F32);
    let f_nw1 = e.alloc_reg(PtxType::F32);

    // Output half2
    let h_out0 = e.alloc_reg(PtxType::B16);
    let h_out1 = e.alloc_reg(PtxType::B16);
    let r_out_h2 = e.alloc_reg(PtxType::B32);
    // Residual half2
    let h_res0 = e.alloc_reg(PtxType::B16);
    let h_res1 = e.alloc_reg(PtxType::B16);
    let r_res_h2 = e.alloc_reg(PtxType::B32);

    e.emit_reg_decls();
    e.emit_shared_decl("smem", smem_bytes, 16);
    e.blank();

    // Load params
    e.emit_ld_param(&rd_output, "output_ptr", PtxType::U64);
    e.emit_ld_param(&rd_residual, "residual_ptr", PtxType::U64);
    e.emit_ld_param(&rd_input, "input_ptr", PtxType::U64);
    e.emit_ld_param(&rd_add, "add_ptr", PtxType::U64);
    e.emit_ld_param(&rd_norm_w, "norm_w_ptr", PtxType::U64);
    e.emit_ld_param(&r_eps, "eps", PtxType::F32);
    e.emit_ld_param(&r_hidden, "hidden_size", PtxType::U32);
    e.blank();

    e.emit_thread_id(&tid);
    e.emit_block_id(&bid);
    e.emit_warp_id(&warp_id, &tid);
    e.emit_lane_id(&lane_id, &tid);
    e.blank();

    e.emit_mov(&rd_smem_base, "smem", PtxType::U64);

    // Row base: token * hidden_size * 2 (f16 bytes)
    e.emit_mul_wide(&rd_row_off, &bid, &format!("{}", hidden * 2), PtxType::U32);
    e.emit_add(&rd_in_row, &rd_input, &rd_row_off, PtxType::U64);
    e.emit_add(&rd_add_row, &rd_add, &rd_row_off, PtxType::U64);
    e.emit_add(&rd_res_row, &rd_residual, &rd_row_off, PtxType::U64);
    e.emit_add(&rd_out_row, &rd_output, &rd_row_off, PtxType::U64);
    e.blank();

    // Phase 1: residual add -> smem + sum-of-squares
    e.comment("Phase 1: residual = input + add, store f32 to smem, accumulate SS");
    e.emit_mov_imm_f32(&f_ss, 0.0);
    e.emit_mov(&r_i, &tid, PtxType::U32);
    let loop_add = e.alloc_label();
    let end_add = e.alloc_label();
    e.emit_label(&loop_add);
    {
        let p_done = e.alloc_pred();
        e.emit_setp(&p_done, "ge", &r_i, &format!("{}", h2), PtxType::U32);
        e.emit_bra(Some(&p_done), &end_add);

        e.emit_mul_lo(&r_byte_off, &r_i, "4", PtxType::U32);
        e.emit_cvt(&rd_byte_off, &r_byte_off, PtxType::U64, PtxType::U32);

        // Load input half2
        e.emit_add(&rd_addr, &rd_in_row, &rd_byte_off, PtxType::U64);
        e.emit_ld_global(&r_in_h2, &rd_addr, PtxType::B32);
        e.line(&format!("mov.b32 {{{}, {}}}, {};", h_in0, h_in1, r_in_h2));
        e.emit_cvt_f32_f16(&f_in0, &h_in0);
        e.emit_cvt_f32_f16(&f_in1, &h_in1);

        // Load add half2
        e.emit_add(&rd_addr2, &rd_add_row, &rd_byte_off, PtxType::U64);
        e.emit_ld_global(&r_add_h2, &rd_addr2, PtxType::B32);
        e.line(&format!("mov.b32 {{{}, {}}}, {};", h_add0, h_add1, r_add_h2));
        e.emit_cvt_f32_f16(&f_add0, &h_add0);
        e.emit_cvt_f32_f16(&f_add1, &h_add1);

        // v = input + add
        e.emit_add(&f_v0, &f_in0, &f_add0, PtxType::F32);
        e.emit_add(&f_v1, &f_in1, &f_add1, PtxType::F32);

        // Write residual (f16)
        e.emit_cvt_f16_f32(&h_res0, &f_v0);
        e.emit_cvt_f16_f32(&h_res1, &f_v1);
        e.line(&format!("mov.b32 {}, {{{}, {}}};", r_res_h2, h_res0, h_res1));
        e.emit_add(&rd_addr, &rd_res_row, &rd_byte_off, PtxType::U64);
        e.emit_st_global(&rd_addr, &r_res_h2, PtxType::B32);

        // Store f32 to smem
        e.emit_mul_lo(&r_smem_off, &r_i, "8", PtxType::U32);
        e.emit_cvt(&rd_smem_addr, &r_smem_off, PtxType::U64, PtxType::U32);
        e.emit_add(&rd_smem_addr, &rd_smem_addr, &rd_smem_base, PtxType::U64);
        e.emit_st_shared(&rd_smem_addr, &f_v0, PtxType::F32);
        e.emit_add(&rd_smem_addr, &rd_smem_addr, "4", PtxType::U64);
        e.emit_st_shared(&rd_smem_addr, &f_v1, PtxType::F32);

        // SS
        e.emit_fma(&f_ss, &f_v0, &f_v0, &f_ss);
        e.emit_fma(&f_ss, &f_v1, &f_v1, &f_ss);

        e.emit_add(&r_i, &r_i, &format!("{}", THREADS), PtxType::U32);
        e.emit_bra(None, &loop_add);
    }
    e.emit_label(&end_add);
    e.blank();

    // Reduce SS
    e.comment("Reduce sum-of-squares");
    e.emit_warp_reduce_sum(&f_ss);
    e.emit_block_reduce_sum(&f_ss, &lane_id, &warp_id, "smem", warp_scratch_off);

    // Compute rms_scale
    {
        let p_tid0 = e.alloc_pred();
        let skip_rms = e.alloc_label();
        e.emit_setp(&p_tid0, "ne", &tid, "0", PtxType::U32);
        e.emit_bra(Some(&p_tid0), &skip_rms);
        e.emit_cvt(&f_hidden_f, &r_hidden, PtxType::F32, PtxType::U32);
        e.emit_div(&f_ss, &f_ss, &f_hidden_f, PtxType::F32);
        e.emit_add(&f_ss, &f_ss, &r_eps, PtxType::F32);
        e.emit_rsqrt(&f_rms, &f_ss);
        let wa = e.alloc_reg(PtxType::U64);
        e.emit_add(&wa, &rd_smem_base, &format!("{}", warp_scratch_off), PtxType::U64);
        e.emit_st_shared(&wa, &f_rms, PtxType::F32);
        e.emit_label(&skip_rms);
    }
    e.emit_bar_sync();

    // Load rms_scale
    {
        let wa = e.alloc_reg(PtxType::U64);
        e.emit_add(&wa, &rd_smem_base, &format!("{}", warp_scratch_off), PtxType::U64);
        e.emit_ld_shared(&f_rms, &wa, PtxType::F32);
    }
    e.blank();

    // Phase 2: normalize and write output
    e.comment("Phase 2: output = smem[i] * norm_weight[i] * rms_scale");
    e.emit_mov(&r_i, &tid, PtxType::U32);
    let loop_norm = e.alloc_label();
    let end_norm = e.alloc_label();
    e.emit_label(&loop_norm);
    {
        let p_done = e.alloc_pred();
        e.emit_setp(&p_done, "ge", &r_i, &format!("{}", h2), PtxType::U32);
        e.emit_bra(Some(&p_done), &end_norm);

        // Load from smem
        e.emit_mul_lo(&r_smem_off, &r_i, "8", PtxType::U32);
        e.emit_cvt(&rd_smem_addr, &r_smem_off, PtxType::U64, PtxType::U32);
        e.emit_add(&rd_smem_addr, &rd_smem_addr, &rd_smem_base, PtxType::U64);
        e.emit_ld_shared(&f_v0, &rd_smem_addr, PtxType::F32);
        e.emit_add(&rd_smem_addr, &rd_smem_addr, "4", PtxType::U64);
        e.emit_ld_shared(&f_v1, &rd_smem_addr, PtxType::F32);

        // Load norm_weight half2
        e.emit_mul_lo(&r_byte_off, &r_i, "4", PtxType::U32);
        e.emit_cvt(&rd_byte_off, &r_byte_off, PtxType::U64, PtxType::U32);
        e.emit_add(&rd_addr, &rd_norm_w, &rd_byte_off, PtxType::U64);
        e.emit_ld_global(&r_nw_h2, &rd_addr, PtxType::B32);
        e.line(&format!("mov.b32 {{{}, {}}}, {};", h_nw0, h_nw1, r_nw_h2));
        e.emit_cvt_f32_f16(&f_nw0, &h_nw0);
        e.emit_cvt_f32_f16(&f_nw1, &h_nw1);

        // normed = val * weight * rms_scale
        e.emit_mul(&f_v0, &f_v0, &f_rms, PtxType::F32);
        e.emit_mul(&f_v0, &f_v0, &f_nw0, PtxType::F32);
        e.emit_mul(&f_v1, &f_v1, &f_rms, PtxType::F32);
        e.emit_mul(&f_v1, &f_v1, &f_nw1, PtxType::F32);

        // Write output as half2
        e.emit_cvt_f16_f32(&h_out0, &f_v0);
        e.emit_cvt_f16_f32(&h_out1, &f_v1);
        e.line(&format!("mov.b32 {}, {{{}, {}}};", r_out_h2, h_out0, h_out1));
        e.emit_add(&rd_addr, &rd_out_row, &rd_byte_off, PtxType::U64);
        e.emit_st_global(&rd_addr, &r_out_h2, PtxType::B32);

        e.emit_add(&r_i, &r_i, &format!("{}", THREADS), PtxType::U32);
        e.emit_bra(None, &loop_norm);
    }
    e.emit_label(&end_norm);
    e.blank();

    e.emit_ret();
    e.emit_kernel_end();

    e.finish()
}

// ---------------------------------------------------------------------------
// Pattern: ElemAdd + RMSNorm + Gemv
// ---------------------------------------------------------------------------

/// Generate PTX for fused ElemAdd -> RMSNorm -> Gemv kernel.
///
/// Grid: (out_dim, 1, 1), Block: (THREADS, 1, 1)
/// Each block redundantly computes residual add + norm in smem, then dots
/// with one row of proj_weight.
pub fn generate_elemadd_rmsnorm_gemv_ptx(
    hidden: usize,
    out_dim: usize,
    _eps: f32,
    arch: &str,
) -> String {
    let name = format!("fused_add_rmsnorm_gemv_{}x{}", hidden, out_dim);
    let smem_bytes = compute_smem_bytes(hidden);
    let warp_scratch_off = hidden * 4;
    let h2 = hidden / 2;

    let mut e = PtxEmitter::new(arch);
    e.emit_header();

    let params: &[(&str, PtxType)] = &[
        ("output_ptr", PtxType::U64),
        ("residual_out_ptr", PtxType::U64),
        ("input_ptr", PtxType::U64),
        ("add_ptr", PtxType::U64),
        ("norm_w_ptr", PtxType::U64),
        ("proj_w_ptr", PtxType::U64),
        ("eps", PtxType::F32),
        ("out_dim", PtxType::U32),
        ("hidden_size", PtxType::U32),
    ];
    e.emit_kernel_start(&name, params);

    let tid = e.alloc_reg(PtxType::U32);
    let bid = e.alloc_reg(PtxType::U32);
    let warp_id = e.alloc_reg(PtxType::U32);
    let lane_id = e.alloc_reg(PtxType::U32);

    let r_out_dim = e.alloc_reg(PtxType::U32);
    let r_hidden = e.alloc_reg(PtxType::U32);
    let r_eps = e.alloc_reg(PtxType::F32);
    let rd_output = e.alloc_reg(PtxType::U64);
    let rd_res_out = e.alloc_reg(PtxType::U64);
    let rd_input = e.alloc_reg(PtxType::U64);
    let rd_add = e.alloc_reg(PtxType::U64);
    let rd_norm_w = e.alloc_reg(PtxType::U64);
    let rd_proj_w = e.alloc_reg(PtxType::U64);

    let r_i = e.alloc_reg(PtxType::U32);
    let rd_addr = e.alloc_reg(PtxType::U64);
    let rd_addr2 = e.alloc_reg(PtxType::U64);
    let rd_smem_base = e.alloc_reg(PtxType::U64);
    let r_smem_off = e.alloc_reg(PtxType::U32);
    let rd_smem_addr = e.alloc_reg(PtxType::U64);
    let r_byte_off = e.alloc_reg(PtxType::U32);
    let rd_byte_off = e.alloc_reg(PtxType::U64);

    // half2 loads
    let r_in_h2 = e.alloc_reg(PtxType::B32);
    let r_add_h2 = e.alloc_reg(PtxType::B32);
    let h_in0 = e.alloc_reg(PtxType::B16);
    let h_in1 = e.alloc_reg(PtxType::B16);
    let h_add0 = e.alloc_reg(PtxType::B16);
    let h_add1 = e.alloc_reg(PtxType::B16);

    let f_in0 = e.alloc_reg(PtxType::F32);
    let f_in1 = e.alloc_reg(PtxType::F32);
    let f_add0 = e.alloc_reg(PtxType::F32);
    let f_add1 = e.alloc_reg(PtxType::F32);
    let f_v0 = e.alloc_reg(PtxType::F32);
    let f_v1 = e.alloc_reg(PtxType::F32);
    let f_ss = e.alloc_reg(PtxType::F32);
    let f_rms = e.alloc_reg(PtxType::F32);
    let f_hidden_f = e.alloc_reg(PtxType::F32);
    let f_acc = e.alloc_reg(PtxType::F32);

    // Norm weight
    let r_nw_h2 = e.alloc_reg(PtxType::B32);
    let h_nw0 = e.alloc_reg(PtxType::B16);
    let h_nw1 = e.alloc_reg(PtxType::B16);
    let f_nw0 = e.alloc_reg(PtxType::F32);
    let f_nw1 = e.alloc_reg(PtxType::F32);

    // Proj weight
    let r_pw_h2 = e.alloc_reg(PtxType::B32);
    let h_pw0 = e.alloc_reg(PtxType::B16);
    let h_pw1 = e.alloc_reg(PtxType::B16);
    let f_pw0 = e.alloc_reg(PtxType::F32);
    let f_pw1 = e.alloc_reg(PtxType::F32);

    let rd_row_off = e.alloc_reg(PtxType::U64);
    let rd_w_row = e.alloc_reg(PtxType::U64);

    // Residual write
    let h_res0 = e.alloc_reg(PtxType::B16);
    let h_res1 = e.alloc_reg(PtxType::B16);
    let r_res_h2 = e.alloc_reg(PtxType::B32);

    // Output
    let h_out = e.alloc_reg(PtxType::B16);
    let rd_out_addr = e.alloc_reg(PtxType::U64);

    // f32 normed temps
    let f_n0 = e.alloc_reg(PtxType::F32);
    let f_n1 = e.alloc_reg(PtxType::F32);

    e.emit_reg_decls();
    e.emit_shared_decl("smem", smem_bytes, 16);
    e.blank();

    // Load params
    e.emit_ld_param(&rd_output, "output_ptr", PtxType::U64);
    e.emit_ld_param(&rd_res_out, "residual_out_ptr", PtxType::U64);
    e.emit_ld_param(&rd_input, "input_ptr", PtxType::U64);
    e.emit_ld_param(&rd_add, "add_ptr", PtxType::U64);
    e.emit_ld_param(&rd_norm_w, "norm_w_ptr", PtxType::U64);
    e.emit_ld_param(&rd_proj_w, "proj_w_ptr", PtxType::U64);
    e.emit_ld_param(&r_eps, "eps", PtxType::F32);
    e.emit_ld_param(&r_out_dim, "out_dim", PtxType::U32);
    e.emit_ld_param(&r_hidden, "hidden_size", PtxType::U32);
    e.blank();

    e.emit_thread_id(&tid);
    e.emit_block_id(&bid);
    e.emit_warp_id(&warp_id, &tid);
    e.emit_lane_id(&lane_id, &tid);
    e.blank();

    e.emit_bounds_check(&bid, &r_out_dim);
    e.blank();

    e.emit_mov(&rd_smem_base, "smem", PtxType::U64);

    // Phase 1: residual add -> smem + SS
    e.comment("Phase 1: residual = input + add, store f32 to smem");
    e.emit_mov_imm_f32(&f_ss, 0.0);
    e.emit_mov(&r_i, &tid, PtxType::U32);
    let loop_add = e.alloc_label();
    let end_add = e.alloc_label();
    e.emit_label(&loop_add);
    {
        let p_done = e.alloc_pred();
        e.emit_setp(&p_done, "ge", &r_i, &format!("{}", h2), PtxType::U32);
        e.emit_bra(Some(&p_done), &end_add);

        e.emit_mul_lo(&r_byte_off, &r_i, "4", PtxType::U32);
        e.emit_cvt(&rd_byte_off, &r_byte_off, PtxType::U64, PtxType::U32);

        e.emit_add(&rd_addr, &rd_input, &rd_byte_off, PtxType::U64);
        e.emit_ld_global(&r_in_h2, &rd_addr, PtxType::B32);
        e.line(&format!("mov.b32 {{{}, {}}}, {};", h_in0, h_in1, r_in_h2));
        e.emit_cvt_f32_f16(&f_in0, &h_in0);
        e.emit_cvt_f32_f16(&f_in1, &h_in1);

        e.emit_add(&rd_addr2, &rd_add, &rd_byte_off, PtxType::U64);
        e.emit_ld_global(&r_add_h2, &rd_addr2, PtxType::B32);
        e.line(&format!("mov.b32 {{{}, {}}}, {};", h_add0, h_add1, r_add_h2));
        e.emit_cvt_f32_f16(&f_add0, &h_add0);
        e.emit_cvt_f32_f16(&f_add1, &h_add1);

        e.emit_add(&f_v0, &f_in0, &f_add0, PtxType::F32);
        e.emit_add(&f_v1, &f_in1, &f_add1, PtxType::F32);

        // Store f32 to smem
        e.emit_mul_lo(&r_smem_off, &r_i, "8", PtxType::U32);
        e.emit_cvt(&rd_smem_addr, &r_smem_off, PtxType::U64, PtxType::U32);
        e.emit_add(&rd_smem_addr, &rd_smem_addr, &rd_smem_base, PtxType::U64);
        e.emit_st_shared(&rd_smem_addr, &f_v0, PtxType::F32);
        e.emit_add(&rd_smem_addr, &rd_smem_addr, "4", PtxType::U64);
        e.emit_st_shared(&rd_smem_addr, &f_v1, PtxType::F32);

        e.emit_fma(&f_ss, &f_v0, &f_v0, &f_ss);
        e.emit_fma(&f_ss, &f_v1, &f_v1, &f_ss);

        e.emit_add(&r_i, &r_i, &format!("{}", THREADS), PtxType::U32);
        e.emit_bra(None, &loop_add);
    }
    e.emit_label(&end_add);
    e.blank();

    // Reduce SS + compute rms_scale
    e.emit_warp_reduce_sum(&f_ss);
    e.emit_block_reduce_sum(&f_ss, &lane_id, &warp_id, "smem", warp_scratch_off);

    {
        let p_tid0 = e.alloc_pred();
        let skip_rms = e.alloc_label();
        e.emit_setp(&p_tid0, "ne", &tid, "0", PtxType::U32);
        e.emit_bra(Some(&p_tid0), &skip_rms);
        e.emit_cvt(&f_hidden_f, &r_hidden, PtxType::F32, PtxType::U32);
        e.emit_div(&f_ss, &f_ss, &f_hidden_f, PtxType::F32);
        e.emit_add(&f_ss, &f_ss, &r_eps, PtxType::F32);
        e.emit_rsqrt(&f_rms, &f_ss);
        let wa = e.alloc_reg(PtxType::U64);
        e.emit_add(&wa, &rd_smem_base, &format!("{}", warp_scratch_off), PtxType::U64);
        e.emit_st_shared(&wa, &f_rms, PtxType::F32);
        e.emit_label(&skip_rms);
    }
    e.emit_bar_sync();
    {
        let wa = e.alloc_reg(PtxType::U64);
        e.emit_add(&wa, &rd_smem_base, &format!("{}", warp_scratch_off), PtxType::U64);
        e.emit_ld_shared(&f_rms, &wa, PtxType::F32);
    }
    e.blank();

    // Block 0 writes residual out
    e.comment("Block 0 writes residual_out (avoid race)");
    {
        let p_blk0 = e.alloc_pred();
        let skip_res = e.alloc_label();
        e.emit_setp(&p_blk0, "ne", &bid, "0", PtxType::U32);
        e.emit_bra(Some(&p_blk0), &skip_res);

        e.emit_mov(&r_i, &tid, PtxType::U32);
        let loop_res = e.alloc_label();
        let end_res = e.alloc_label();
        e.emit_label(&loop_res);
        {
            let p_done = e.alloc_pred();
            e.emit_setp(&p_done, "ge", &r_i, &format!("{}", h2), PtxType::U32);
            e.emit_bra(Some(&p_done), &end_res);

            e.emit_mul_lo(&r_smem_off, &r_i, "8", PtxType::U32);
            e.emit_cvt(&rd_smem_addr, &r_smem_off, PtxType::U64, PtxType::U32);
            e.emit_add(&rd_smem_addr, &rd_smem_addr, &rd_smem_base, PtxType::U64);
            e.emit_ld_shared(&f_v0, &rd_smem_addr, PtxType::F32);
            e.emit_add(&rd_smem_addr, &rd_smem_addr, "4", PtxType::U64);
            e.emit_ld_shared(&f_v1, &rd_smem_addr, PtxType::F32);

            e.emit_cvt_f16_f32(&h_res0, &f_v0);
            e.emit_cvt_f16_f32(&h_res1, &f_v1);
            e.line(&format!("mov.b32 {}, {{{}, {}}};", r_res_h2, h_res0, h_res1));

            e.emit_mul_lo(&r_byte_off, &r_i, "4", PtxType::U32);
            e.emit_cvt(&rd_byte_off, &r_byte_off, PtxType::U64, PtxType::U32);
            e.emit_add(&rd_addr, &rd_res_out, &rd_byte_off, PtxType::U64);
            e.emit_st_global(&rd_addr, &r_res_h2, PtxType::B32);

            e.emit_add(&r_i, &r_i, &format!("{}", THREADS), PtxType::U32);
            e.emit_bra(None, &loop_res);
        }
        e.emit_label(&end_res);
        e.emit_label(&skip_res);
    }
    e.blank();

    // Apply norm weights in smem
    e.comment("Apply norm weights: smem[i] *= norm_weight[i] * rms_scale");
    e.emit_mov(&r_i, &tid, PtxType::U32);
    let loop_norm = e.alloc_label();
    let end_norm = e.alloc_label();
    e.emit_label(&loop_norm);
    {
        let p_done = e.alloc_pred();
        e.emit_setp(&p_done, "ge", &r_i, &format!("{}", h2), PtxType::U32);
        e.emit_bra(Some(&p_done), &end_norm);

        // Load norm_weight
        e.emit_mul_lo(&r_byte_off, &r_i, "4", PtxType::U32);
        e.emit_cvt(&rd_byte_off, &r_byte_off, PtxType::U64, PtxType::U32);
        e.emit_add(&rd_addr, &rd_norm_w, &rd_byte_off, PtxType::U64);
        e.emit_ld_global(&r_nw_h2, &rd_addr, PtxType::B32);
        e.line(&format!("mov.b32 {{{}, {}}}, {};", h_nw0, h_nw1, r_nw_h2));
        e.emit_cvt_f32_f16(&f_nw0, &h_nw0);
        e.emit_cvt_f32_f16(&f_nw1, &h_nw1);

        // Load from smem
        e.emit_mul_lo(&r_smem_off, &r_i, "8", PtxType::U32);
        e.emit_cvt(&rd_smem_addr, &r_smem_off, PtxType::U64, PtxType::U32);
        e.emit_add(&rd_smem_addr, &rd_smem_addr, &rd_smem_base, PtxType::U64);
        e.emit_ld_shared(&f_v0, &rd_smem_addr, PtxType::F32);
        e.emit_add(&rd_smem_addr, &rd_smem_addr, "4", PtxType::U64);
        e.emit_ld_shared(&f_v1, &rd_smem_addr, PtxType::F32);

        e.emit_mul(&f_v0, &f_v0, &f_rms, PtxType::F32);
        e.emit_mul(&f_v0, &f_v0, &f_nw0, PtxType::F32);
        e.emit_mul(&f_v1, &f_v1, &f_rms, PtxType::F32);
        e.emit_mul(&f_v1, &f_v1, &f_nw1, PtxType::F32);

        // Store back
        e.emit_mul_lo(&r_smem_off, &r_i, "8", PtxType::U32);
        e.emit_cvt(&rd_smem_addr, &r_smem_off, PtxType::U64, PtxType::U32);
        e.emit_add(&rd_smem_addr, &rd_smem_addr, &rd_smem_base, PtxType::U64);
        e.emit_st_shared(&rd_smem_addr, &f_v0, PtxType::F32);
        e.emit_add(&rd_smem_addr, &rd_smem_addr, "4", PtxType::U64);
        e.emit_st_shared(&rd_smem_addr, &f_v1, PtxType::F32);

        e.emit_add(&r_i, &r_i, &format!("{}", THREADS), PtxType::U32);
        e.emit_bra(None, &loop_norm);
    }
    e.emit_label(&end_norm);
    e.emit_bar_sync();
    e.blank();

    // Phase 2: GEMV
    e.comment("Phase 2: GEMV dot(proj_weight[row], normed)");
    e.emit_mul_wide(&rd_row_off, &bid, &format!("{}", hidden * 2), PtxType::U32);
    e.emit_add(&rd_w_row, &rd_proj_w, &rd_row_off, PtxType::U64);

    e.emit_mov_imm_f32(&f_acc, 0.0);
    e.emit_mov(&r_i, &tid, PtxType::U32);
    let loop_gemv = e.alloc_label();
    let end_gemv = e.alloc_label();
    e.emit_label(&loop_gemv);
    {
        let p_done = e.alloc_pred();
        e.emit_setp(&p_done, "ge", &r_i, &format!("{}", h2), PtxType::U32);
        e.emit_bra(Some(&p_done), &end_gemv);

        e.emit_mul_lo(&r_byte_off, &r_i, "4", PtxType::U32);
        e.emit_cvt(&rd_byte_off, &r_byte_off, PtxType::U64, PtxType::U32);
        e.emit_add(&rd_addr, &rd_w_row, &rd_byte_off, PtxType::U64);
        e.emit_ld_global(&r_pw_h2, &rd_addr, PtxType::B32);
        e.line(&format!("mov.b32 {{{}, {}}}, {};", h_pw0, h_pw1, r_pw_h2));
        e.emit_cvt_f32_f16(&f_pw0, &h_pw0);
        e.emit_cvt_f32_f16(&f_pw1, &h_pw1);

        e.emit_mul_lo(&r_smem_off, &r_i, "8", PtxType::U32);
        e.emit_cvt(&rd_smem_addr, &r_smem_off, PtxType::U64, PtxType::U32);
        e.emit_add(&rd_smem_addr, &rd_smem_addr, &rd_smem_base, PtxType::U64);
        e.emit_ld_shared(&f_n0, &rd_smem_addr, PtxType::F32);
        e.emit_add(&rd_smem_addr, &rd_smem_addr, "4", PtxType::U64);
        e.emit_ld_shared(&f_n1, &rd_smem_addr, PtxType::F32);

        e.emit_fma(&f_acc, &f_pw0, &f_n0, &f_acc);
        e.emit_fma(&f_acc, &f_pw1, &f_n1, &f_acc);

        e.emit_add(&r_i, &r_i, &format!("{}", THREADS), PtxType::U32);
        e.emit_bra(None, &loop_gemv);
    }
    e.emit_label(&end_gemv);
    e.blank();

    // Reduce
    e.emit_warp_reduce_sum(&f_acc);
    e.emit_block_reduce_sum(&f_acc, &lane_id, &warp_id, "smem", warp_scratch_off);

    // Write output
    {
        let p_tid0 = e.alloc_pred();
        let skip_write = e.alloc_label();
        e.emit_setp(&p_tid0, "ne", &tid, "0", PtxType::U32);
        e.emit_bra(Some(&p_tid0), &skip_write);
        e.emit_cvt_f16_f32(&h_out, &f_acc);
        e.emit_mul_wide(&rd_byte_off, &bid, "2", PtxType::U32);
        e.emit_add(&rd_out_addr, &rd_output, &rd_byte_off, PtxType::U64);
        e.emit_st_global(&rd_out_addr, &h_out, PtxType::B16);
        e.emit_label(&skip_write);
    }
    e.blank();

    e.emit_ret();
    e.emit_kernel_end();

    e.finish()
}

// ---------------------------------------------------------------------------
// Unified entry point
// ---------------------------------------------------------------------------

use crate::dispatch::ModelShapes;

/// Generate PTX for a fused kernel pattern directly, no nvcc needed.
///
/// Returns `None` for unrecognized patterns.
pub fn compile_fused_kernel_ptx(
    pattern: &FusedKernel,
    model_dims: &ModelShapes,
    arch: &str,
) -> Option<String> {
    let pat = classify(pattern);
    match pat {
        FusionPattern::RMSNormGemv => {
            let out = pattern.output_shape.last().copied().unwrap_or(model_dims.hidden_size);
            Some(generate_rmsnorm_gemv_ptx(
                model_dims.hidden_size,
                out,
                model_dims.rms_norm_eps,
                arch,
            ))
        }
        FusionPattern::SiLUElemMulGemv => {
            let out = pattern.output_shape.last().copied().unwrap_or(model_dims.hidden_size);
            Some(generate_silu_elemmul_gemv_ptx(
                model_dims.intermediate_size,
                out,
                arch,
            ))
        }
        FusionPattern::ElemAddRMSNorm => Some(generate_elemadd_rmsnorm_ptx(
            model_dims.hidden_size,
            model_dims.rms_norm_eps,
            arch,
        )),
        FusionPattern::ElemAddRMSNormGemv => {
            let out = pattern.output_shape.last().copied().unwrap_or(model_dims.hidden_size);
            Some(generate_elemadd_rmsnorm_gemv_ptx(
                model_dims.hidden_size,
                out,
                model_dims.rms_norm_eps,
                arch,
            ))
        }
        FusionPattern::Generic => None,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{Dtype, FusedKernel, FusionOp};

    fn make_kernel(ops: Vec<FusionOp>, out_shape: Vec<usize>) -> FusedKernel {
        FusedKernel {
            node_ids: (0..ops.len()).collect(),
            ops,
            output_shape: out_shape,
            dtype: Dtype::F16,
        }
    }

    fn test_shapes() -> ModelShapes {
        ModelShapes {
            hidden_size: 3584,
            intermediate_size: 9216,
            num_heads: 28,
            num_kv_heads: 4,
            head_dim: 128,
            vocab_size: 32000,
            rms_norm_eps: 1e-5,
        }
    }

    #[test]
    fn rmsnorm_gemv_ptx_structure() {
        let ptx = generate_rmsnorm_gemv_ptx(3584, 3584, 1e-5, "sm_90");
        assert!(ptx.contains(".version 8.0"));
        assert!(ptx.contains(".target sm_90"));
        assert!(ptx.contains(".address_size 64"));
        assert!(ptx.contains(".visible .entry fused_rmsnorm_gemv_3584x3584"));
        assert!(ptx.contains(".param .u64 output_ptr"));
        assert!(ptx.contains(".param .f32 eps"));
        assert!(ptx.contains("ld.param.u64"));
        assert!(ptx.contains("ld.param.f32"));
        assert!(ptx.contains("mov.u32"));
        assert!(ptx.contains("%tid.x"));
        assert!(ptx.contains("%ctaid.x"));
        assert!(ptx.contains("shfl.sync.down.b32"));
        assert!(ptx.contains("bar.sync 0"));
        assert!(ptx.contains("rsqrt.approx.ftz.f32"));
        assert!(ptx.contains("fma.rn.f32"));
        assert!(ptx.contains("cvt.f32.f16"));
        assert!(ptx.contains("cvt.rn.f16.f32"));
        assert!(ptx.contains("ld.global.b32")); // half2 loads
        assert!(ptx.contains("st.global.b16")); // f16 output
        assert!(ptx.contains(".shared .align 16"));
        assert!(ptx.contains("ret;"));
    }

    #[test]
    fn rmsnorm_gemv_smem_size() {
        let ptx = generate_rmsnorm_gemv_ptx(3584, 3584, 1e-5, "sm_90");
        // smem = 3584 * 4 + 8 * 4 = 14368
        let expected = format!("smem[{}]", 3584 * 4 + 8 * 4);
        assert!(ptx.contains(&expected));
    }

    #[test]
    fn silu_elemmul_gemv_ptx_structure() {
        let ptx = generate_silu_elemmul_gemv_ptx(9216, 3584, "sm_90");
        assert!(ptx.contains(".visible .entry fused_silu_mul_gemv_9216x3584"));
        assert!(ptx.contains("ex2.approx.ftz.f32")); // fast exp via ex2
        assert!(ptx.contains("rcp.approx.ftz.f32")); // 1/x for sigmoid
        assert!(ptx.contains("shfl.sync.down.b32"));
        assert!(ptx.contains("ret;"));
    }

    #[test]
    fn elemadd_rmsnorm_ptx_structure() {
        let ptx = generate_elemadd_rmsnorm_ptx(3584, 1e-5, "sm_90");
        assert!(ptx.contains(".visible .entry fused_elemadd_rmsnorm_3584"));
        assert!(ptx.contains("residual_ptr"));
        assert!(ptx.contains("rsqrt.approx.ftz.f32"));
        assert!(ptx.contains("ret;"));
    }

    #[test]
    fn elemadd_rmsnorm_gemv_ptx_structure() {
        let ptx = generate_elemadd_rmsnorm_gemv_ptx(3584, 9216, 1e-5, "sm_90");
        assert!(ptx.contains(".visible .entry fused_add_rmsnorm_gemv_3584x9216"));
        assert!(ptx.contains("residual_out_ptr"));
        assert!(ptx.contains("proj_w_ptr"));
        assert!(ptx.contains("rsqrt.approx.ftz.f32"));
        assert!(ptx.contains("fma.rn.f32"));
        assert!(ptx.contains("ret;"));
    }

    #[test]
    fn compile_fused_kernel_ptx_routes_correctly() {
        let shapes = test_shapes();

        let k1 = make_kernel(
            vec![FusionOp::RMSNorm { eps: 1e-5 }, FusionOp::Gemv],
            vec![1, 3584],
        );
        let ptx = compile_fused_kernel_ptx(&k1, &shapes, "sm_90");
        assert!(ptx.is_some());
        assert!(ptx.unwrap().contains("fused_rmsnorm_gemv_3584x3584"));

        let k2 = make_kernel(
            vec![FusionOp::SiLU, FusionOp::ElemMul, FusionOp::Gemv],
            vec![1, 3584],
        );
        let ptx = compile_fused_kernel_ptx(&k2, &shapes, "sm_90");
        assert!(ptx.is_some());
        assert!(ptx.unwrap().contains("fused_silu_mul_gemv_9216x3584"));

        let k3 = make_kernel(
            vec![FusionOp::ElemAdd, FusionOp::RMSNorm { eps: 1e-6 }],
            vec![1, 3584],
        );
        let ptx = compile_fused_kernel_ptx(&k3, &shapes, "sm_90");
        assert!(ptx.is_some());
        assert!(ptx.unwrap().contains("fused_elemadd_rmsnorm_3584"));

        let k4 = make_kernel(
            vec![FusionOp::ElemAdd, FusionOp::RMSNorm { eps: 1e-5 }, FusionOp::Gemv],
            vec![1, 9216],
        );
        let ptx = compile_fused_kernel_ptx(&k4, &shapes, "sm_90");
        assert!(ptx.is_some());
        assert!(ptx.unwrap().contains("fused_add_rmsnorm_gemv_3584x9216"));

        // Unsupported pattern
        let k5 = make_kernel(vec![FusionOp::Softmax], vec![1, 128]);
        assert!(compile_fused_kernel_ptx(&k5, &shapes, "sm_90").is_none());
    }

    #[test]
    fn ptx_has_no_null_terminated_strings() {
        let ptx = generate_rmsnorm_gemv_ptx(4096, 4096, 1e-5, "sm_90");
        assert!(!ptx.contains('\0'));
    }

    #[test]
    fn ptx_register_decls_present() {
        let ptx = generate_rmsnorm_gemv_ptx(4096, 4096, 1e-5, "sm_90");
        assert!(ptx.contains(".reg .pred %p<"));
        assert!(ptx.contains(".reg .b32 %r<"));
        assert!(ptx.contains(".reg .b64 %rd<"));
        assert!(ptx.contains(".reg .f32 %f<"));
    }

    #[test]
    fn different_arch_targets() {
        let ptx80 = generate_rmsnorm_gemv_ptx(4096, 4096, 1e-5, "sm_80");
        assert!(ptx80.contains(".target sm_80"));
        let ptx90 = generate_rmsnorm_gemv_ptx(4096, 4096, 1e-5, "sm_90");
        assert!(ptx90.contains(".target sm_90"));
    }

    #[test]
    fn emitter_basic_api() {
        let mut e = PtxEmitter::new("sm_90");
        e.emit_header();

        let r0 = e.alloc_reg(PtxType::F32);
        assert_eq!(r0, "%f0");
        let r1 = e.alloc_reg(PtxType::F32);
        assert_eq!(r1, "%f1");
        let p0 = e.alloc_pred();
        assert_eq!(p0, "%p0");
        let rd0 = e.alloc_reg(PtxType::U64);
        assert_eq!(rd0, "%rd0");

        let l = e.alloc_label();
        assert_eq!(l, "$L0");
        let l2 = e.alloc_label();
        assert_eq!(l2, "$L1");

        let out = e.finish();
        assert!(out.contains(".version 8.0"));
    }
}
