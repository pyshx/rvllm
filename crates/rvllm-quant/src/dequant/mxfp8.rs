use half::f16;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mxfp8ElementType {
    E4M3,
    E5M2,
}

#[derive(Debug, Clone)]
pub struct Mxfp8Config {
    pub block_size: usize,
    pub element_type: Mxfp8ElementType,
}

impl Default for Mxfp8Config {
    fn default() -> Self {
        Self {
            block_size: 32,
            element_type: Mxfp8ElementType::E4M3,
        }
    }
}

/// E8M0 scale factor: 8-bit exponent, no mantissa, no sign.
/// Represents 2^(exp - 127) for exp in [0, 254]. 255 is NaN.
#[inline]
fn e8m0_to_f32(byte: u8) -> f32 {
    if byte == 255 {
        return f32::NAN;
    }
    f32::from_bits(((byte as u32) << 23) & 0x7F80_0000)
}

/// Convert a positive f32 to the nearest E8M0 scale (power of 2).
/// Returns the biased exponent byte.
#[inline]
fn f32_to_e8m0(val: f32) -> u8 {
    if val <= 0.0 || val.is_nan() {
        return 0;
    }
    let bits = val.to_bits();
    let biased_exp = ((bits >> 23) & 0xFF) as u8;
    // Round to nearest power of 2: check if mantissa >= 0.5 (bit 22 set)
    let mantissa_high = (bits >> 22) & 1;
    let rounded = biased_exp.saturating_add(mantissa_high as u8);
    rounded.min(254)
}

/// Precomputed LUT for FP8 E4M3 -> f32 (reused from fp8.rs logic).
const FP8_E4M3_LUT: [f32; 256] = build_e4m3_lut();
const FP8_E5M2_LUT: [f32; 256] = build_e5m2_lut();

const fn build_e4m3_lut() -> [f32; 256] {
    let mut lut = [0.0f32; 256];
    let mut i: u16 = 0;
    while i < 256 {
        let byte = i as u8;
        let sign = (byte >> 7) & 1;
        let exp = (byte >> 3) & 0x0F;
        let mantissa = byte & 0x07;

        let val = if exp == 0 && mantissa == 0 {
            0.0f32
        } else if exp == 0 {
            0.015625 * (mantissa as f32 / 8.0)
        } else {
            let n = exp as i32 - 7;
            let pow = const_pow2_e4m3(n);
            pow * (1.0 + mantissa as f32 / 8.0)
        };

        lut[i as usize] = if sign == 1 { -val } else { val };
        i += 1;
    }
    lut
}

const fn const_pow2_e4m3(n: i32) -> f32 {
    match n {
        -6 => 1.0 / 64.0,
        -5 => 1.0 / 32.0,
        -4 => 1.0 / 16.0,
        -3 => 1.0 / 8.0,
        -2 => 1.0 / 4.0,
        -1 => 1.0 / 2.0,
        0 => 1.0,
        1 => 2.0,
        2 => 4.0,
        3 => 8.0,
        4 => 16.0,
        5 => 32.0,
        6 => 64.0,
        7 => 128.0,
        8 => 256.0,
        _ => 1.0,
    }
}

const fn build_e5m2_lut() -> [f32; 256] {
    let mut lut = [0.0f32; 256];
    let mut i: u16 = 0;
    while i < 256 {
        let byte = i as u8;
        let sign = (byte >> 7) & 1;
        let exp = (byte >> 2) & 0x1F;
        let mantissa = byte & 0x03;

        let val = if exp == 0 && mantissa == 0 {
            0.0f32
        } else if exp == 0x1F {
            // Inf/NaN for E5M2
            if mantissa == 0 {
                f32::INFINITY
            } else {
                f32::NAN
            }
        } else if exp == 0 {
            // Subnormal: 2^(1-15) * (mantissa/4)
            let pow = const_pow2_e5m2(1 - 15);
            pow * (mantissa as f32 / 4.0)
        } else {
            let n = exp as i32 - 15;
            let pow = const_pow2_e5m2(n);
            pow * (1.0 + mantissa as f32 / 4.0)
        };

        lut[i as usize] = if sign == 1 { -val } else { val };
        i += 1;
    }
    lut
}

const fn const_pow2_e5m2(n: i32) -> f32 {
    match n {
        -14 => 1.0 / 16384.0,
        -13 => 1.0 / 8192.0,
        -12 => 1.0 / 4096.0,
        -11 => 1.0 / 2048.0,
        -10 => 1.0 / 1024.0,
        -9 => 1.0 / 512.0,
        -8 => 1.0 / 256.0,
        -7 => 1.0 / 128.0,
        -6 => 1.0 / 64.0,
        -5 => 1.0 / 32.0,
        -4 => 1.0 / 16.0,
        -3 => 1.0 / 8.0,
        -2 => 1.0 / 4.0,
        -1 => 1.0 / 2.0,
        0 => 1.0,
        1 => 2.0,
        2 => 4.0,
        3 => 8.0,
        4 => 16.0,
        5 => 32.0,
        6 => 64.0,
        7 => 128.0,
        8 => 256.0,
        9 => 512.0,
        10 => 1024.0,
        11 => 2048.0,
        12 => 4096.0,
        13 => 8192.0,
        14 => 16384.0,
        15 => 32768.0,
        _ => 1.0,
    }
}

#[inline]
fn fp8_to_f32(byte: u8, element_type: Mxfp8ElementType) -> f32 {
    match element_type {
        Mxfp8ElementType::E4M3 => FP8_E4M3_LUT[byte as usize],
        Mxfp8ElementType::E5M2 => FP8_E5M2_LUT[byte as usize],
    }
}

fn f32_to_fp8_e4m3(val: f32) -> u8 {
    if val == 0.0 {
        return if val.is_sign_negative() { 0x80 } else { 0x00 };
    }
    let sign = if val < 0.0 { 1u8 } else { 0u8 };
    let abs_val = val.abs();
    let bias = 7i32;
    let max_val = 448.0f32;
    let min_normal = 2.0f32.powi(1 - bias);
    let min_subnormal = min_normal / 8.0;

    if abs_val > max_val {
        return (sign << 7) | 0x7F;
    }
    if abs_val < min_subnormal {
        return sign << 7;
    }
    if abs_val < min_normal {
        let mantissa = (abs_val / min_normal * 8.0).round() as u8;
        return (sign << 7) | mantissa.min(7);
    }
    let log2 = abs_val.log2().floor() as i32;
    let exp = (log2 + bias).clamp(1, 15) as u8;
    let pow = 2.0f32.powi(exp as i32 - bias);
    let mantissa = ((abs_val / pow - 1.0) * 8.0).round().clamp(0.0, 7.0) as u8;
    (sign << 7) | (exp << 3) | mantissa
}

fn f32_to_fp8_e5m2(val: f32) -> u8 {
    if val == 0.0 {
        return if val.is_sign_negative() { 0x80 } else { 0x00 };
    }
    let sign = if val < 0.0 { 1u8 } else { 0u8 };
    let abs_val = val.abs();
    let bias = 15i32;
    let max_val = 57344.0f32;
    let min_normal = 2.0f32.powi(1 - bias);
    let min_subnormal = min_normal / 4.0;

    if abs_val > max_val {
        return (sign << 7) | 0x7B; // max finite: exp=30, mantissa=3
    }
    if abs_val < min_subnormal {
        return sign << 7;
    }
    if abs_val < min_normal {
        let mantissa = (abs_val / min_normal * 4.0).round() as u8;
        return (sign << 7) | mantissa.min(3);
    }
    let log2 = abs_val.log2().floor() as i32;
    let exp = (log2 + bias).clamp(1, 30) as u8;
    let pow = 2.0f32.powi(exp as i32 - bias);
    let mantissa = ((abs_val / pow - 1.0) * 4.0).round().clamp(0.0, 3.0) as u8;
    (sign << 7) | (exp << 2) | mantissa
}

#[inline]
fn f32_to_fp8(val: f32, element_type: Mxfp8ElementType) -> u8 {
    match element_type {
        Mxfp8ElementType::E4M3 => f32_to_fp8_e4m3(val),
        Mxfp8ElementType::E5M2 => f32_to_fp8_e5m2(val),
    }
}

/// Dequantize MXFP8 block-scaled data to f16.
///
/// Each `block_size` elements share one E8M0 scale factor.
/// `scales` contains one byte per block (E8M0 format).
pub fn dequantize_mxfp8(data: &[u8], scales: &[u8], config: &Mxfp8Config, out: &mut [f16]) {
    let block_size = config.block_size;
    let num_elements = data.len();
    assert!(out.len() >= num_elements);

    let num_blocks = (num_elements + block_size - 1) / block_size;
    assert!(
        scales.len() >= num_blocks,
        "expected at least {num_blocks} scale factors, got {}",
        scales.len()
    );

    for block_idx in 0..num_blocks {
        let start = block_idx * block_size;
        let end = (start + block_size).min(num_elements);
        let scale = e8m0_to_f32(scales[block_idx]);

        for i in start..end {
            let raw = fp8_to_f32(data[i], config.element_type);
            out[i] = f16::from_f32(raw * scale);
        }
    }
}

/// Quantize f16 data to MXFP8 with block-level E8M0 scaling.
///
/// Returns (quantized_data, scales) where scales has one E8M0 byte per block.
pub fn quantize_mxfp8(data: &[f16], block_size: usize, element_type: Mxfp8ElementType) -> (Vec<u8>, Vec<u8>) {
    let num_elements = data.len();
    let num_blocks = (num_elements + block_size - 1) / block_size;
    let mut quantized = vec![0u8; num_elements];
    let mut scales = vec![0u8; num_blocks];

    for block_idx in 0..num_blocks {
        let start = block_idx * block_size;
        let end = (start + block_size).min(num_elements);

        // Find max absolute value in block
        let mut max_abs: f32 = 0.0;
        for i in start..end {
            let v = data[i].to_f32().abs();
            if v > max_abs {
                max_abs = v;
            }
        }

        // Compute E8M0 scale: nearest power-of-2 that keeps values in FP8 range
        let fp8_max = match element_type {
            Mxfp8ElementType::E4M3 => 448.0f32,
            Mxfp8ElementType::E5M2 => 57344.0f32,
        };

        let scale_val = if max_abs == 0.0 {
            1.0
        } else {
            max_abs / fp8_max
        };

        let scale_byte = f32_to_e8m0(scale_val);
        scales[block_idx] = scale_byte;

        let scale_f32 = e8m0_to_f32(scale_byte);
        let inv_scale = if scale_f32 == 0.0 { 0.0 } else { 1.0 / scale_f32 };

        for i in start..end {
            let scaled = data[i].to_f32() * inv_scale;
            quantized[i] = f32_to_fp8(scaled, element_type);
        }
    }

    (quantized, scales)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn e8m0_round_trip_powers_of_two() {
        for exp in 0..=254u8 {
            let val = e8m0_to_f32(exp);
            if val > 0.0 && val.is_finite() {
                let back = f32_to_e8m0(val);
                assert_eq!(back, exp, "e8m0 round-trip failed for exp={exp}, val={val}");
            }
        }
    }

    #[test]
    fn e8m0_nan() {
        assert!(e8m0_to_f32(255).is_nan());
    }

    #[test]
    fn mxfp8_e4m3_round_trip() {
        let config = Mxfp8Config::default();
        let block_size = config.block_size;
        let original: Vec<f16> = (0..64)
            .map(|i| f16::from_f32((i as f32 - 32.0) * 0.5))
            .collect();

        let (quantized, scales) = quantize_mxfp8(&original, block_size, Mxfp8ElementType::E4M3);
        assert_eq!(quantized.len(), 64);
        assert_eq!(scales.len(), 2); // 64 / 32 = 2 blocks

        let mut restored = vec![f16::ZERO; 64];
        dequantize_mxfp8(&quantized, &scales, &config, &mut restored);

        for (i, (o, r)) in original.iter().zip(restored.iter()).enumerate() {
            let o_f32 = o.to_f32();
            let r_f32 = r.to_f32();
            let tol = o_f32.abs() * 0.3 + 0.5;
            assert!(
                (o_f32 - r_f32).abs() < tol,
                "e4m3 round-trip mismatch at {i}: original={o_f32}, restored={r_f32}"
            );
        }
    }

    #[test]
    fn mxfp8_e5m2_round_trip() {
        let config = Mxfp8Config {
            block_size: 32,
            element_type: Mxfp8ElementType::E5M2,
        };
        let original: Vec<f16> = (0..32)
            .map(|i| f16::from_f32((i as f32 - 16.0) * 2.0))
            .collect();

        let (quantized, scales) = quantize_mxfp8(&original, config.block_size, config.element_type);
        assert_eq!(quantized.len(), 32);
        assert_eq!(scales.len(), 1);

        let mut restored = vec![f16::ZERO; 32];
        dequantize_mxfp8(&quantized, &scales, &config, &mut restored);

        for (i, (o, r)) in original.iter().zip(restored.iter()).enumerate() {
            let o_f32 = o.to_f32();
            let r_f32 = r.to_f32();
            let tol = o_f32.abs() * 0.5 + 1.0;
            assert!(
                (o_f32 - r_f32).abs() < tol,
                "e5m2 round-trip mismatch at {i}: original={o_f32}, restored={r_f32}"
            );
        }
    }

    #[test]
    fn mxfp8_partial_block() {
        let config = Mxfp8Config::default();
        // 40 elements = 1 full block (32) + 1 partial block (8)
        let original: Vec<f16> = (0..40)
            .map(|i| f16::from_f32(i as f32 * 0.1))
            .collect();

        let (quantized, scales) = quantize_mxfp8(&original, config.block_size, config.element_type);
        assert_eq!(quantized.len(), 40);
        assert_eq!(scales.len(), 2);

        let mut restored = vec![f16::ZERO; 40];
        dequantize_mxfp8(&quantized, &scales, &config, &mut restored);

        for (i, (o, r)) in original.iter().zip(restored.iter()).enumerate() {
            let o_f32 = o.to_f32();
            let r_f32 = r.to_f32();
            let tol = o_f32.abs() * 0.3 + 0.5;
            assert!(
                (o_f32 - r_f32).abs() < tol,
                "partial block mismatch at {i}: original={o_f32}, restored={r_f32}"
            );
        }
    }

    #[test]
    fn mxfp8_zeros() {
        let config = Mxfp8Config::default();
        let original = vec![f16::ZERO; 32];
        let (quantized, scales) = quantize_mxfp8(&original, config.block_size, config.element_type);

        let mut restored = vec![f16::from_f32(999.0); 32];
        dequantize_mxfp8(&quantized, &scales, &config, &mut restored);

        for r in &restored {
            assert_eq!(r.to_f32(), 0.0);
        }
    }

    #[test]
    fn mxfp8_block_scaling_isolates_blocks() {
        let config = Mxfp8Config::default();
        // Block 0: small values, Block 1: large values
        let mut original = vec![f16::ZERO; 64];
        for i in 0..32 {
            original[i] = f16::from_f32(0.01 * (i as f32 + 1.0));
        }
        for i in 32..64 {
            original[i] = f16::from_f32(100.0 * ((i - 32) as f32 + 1.0));
        }

        let (quantized, scales) = quantize_mxfp8(&original, config.block_size, config.element_type);
        assert_eq!(scales.len(), 2);
        // The two blocks should have different scales
        assert_ne!(scales[0], scales[1]);

        let mut restored = vec![f16::ZERO; 64];
        dequantize_mxfp8(&quantized, &scales, &config, &mut restored);

        // Small values block should have reasonable precision
        for i in 0..32 {
            let o = original[i].to_f32();
            let r = restored[i].to_f32();
            let tol = o.abs() * 0.5 + 0.01;
            assert!(
                (o - r).abs() < tol,
                "small block mismatch at {i}: original={o}, restored={r}"
            );
        }
    }
}
