use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rvllm_core::prelude::SamplingParams;
use rvllm_sampling::sampler::Sampler;

fn bench_greedy(c: &mut Criterion) {
    let mut group = c.benchmark_group("greedy_sample");
    let sampler = Sampler::new();
    let params = SamplingParams {
        temperature: 0.0,
        ..Default::default()
    };

    for vocab_size in [32000, 65536, 128256] {
        let logits: Vec<f32> = (0..vocab_size).map(|i| (i as f32) * 0.001).collect();
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        group.bench_with_input(
            BenchmarkId::new("vocab", vocab_size),
            &vocab_size,
            |b, _| {
                b.iter(|| {
                    black_box(
                        sampler
                            .sample(&logits, vocab_size, &params, &[], &mut rng)
                            .unwrap(),
                    )
                })
            },
        );
    }
    group.finish();
}

fn bench_top_p(c: &mut Criterion) {
    let mut group = c.benchmark_group("top_p_sample");
    let sampler = Sampler::new();
    let params = SamplingParams {
        temperature: 0.8,
        top_p: 0.9,
        ..Default::default()
    };

    for vocab_size in [32000, 128256] {
        let logits: Vec<f32> = (0..vocab_size).map(|i| (i as f32) * 0.001).collect();

        group.bench_with_input(
            BenchmarkId::new("vocab", vocab_size),
            &vocab_size,
            |b, _| {
                let mut rng = ChaCha8Rng::seed_from_u64(42);
                b.iter(|| {
                    black_box(
                        sampler
                            .sample(&logits, vocab_size, &params, &[], &mut rng)
                            .unwrap(),
                    )
                })
            },
        );
    }
    group.finish();
}

fn bench_top_k(c: &mut Criterion) {
    let mut group = c.benchmark_group("top_k_sample");
    let sampler = Sampler::new();
    let params = SamplingParams {
        temperature: 0.8,
        top_k: 50,
        ..Default::default()
    };

    for vocab_size in [32000, 128256] {
        let logits: Vec<f32> = (0..vocab_size).map(|i| (i as f32) * 0.001).collect();

        group.bench_with_input(
            BenchmarkId::new("vocab", vocab_size),
            &vocab_size,
            |b, _| {
                let mut rng = ChaCha8Rng::seed_from_u64(42);
                b.iter(|| {
                    black_box(
                        sampler
                            .sample(&logits, vocab_size, &params, &[], &mut rng)
                            .unwrap(),
                    )
                })
            },
        );
    }
    group.finish();
}

fn bench_repetition_penalty(c: &mut Criterion) {
    let mut group = c.benchmark_group("repetition_penalty");
    let sampler = Sampler::new();

    let vocab_size = 128256;
    let logits: Vec<f32> = (0..vocab_size).map(|i| (i as f32) * 0.001).collect();
    let past_tokens: Vec<u32> = (0..512).collect();

    let params = SamplingParams {
        temperature: 0.0,
        repetition_penalty: 1.1,
        ..Default::default()
    };

    group.bench_function("512_past_tokens", |b| {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        b.iter(|| {
            black_box(
                sampler
                    .sample(&logits, vocab_size, &params, &past_tokens, &mut rng)
                    .unwrap(),
            )
        })
    });

    group.finish();
}

// -- Zig SIMD vs pure Rust comparisons -----------------------------------------

const VOCAB: usize = 128_000;

fn rust_softmax_ref(logits: &[f32], out: &mut [f32]) {
    let n = logits.len();
    let mut max = f32::NEG_INFINITY;
    for &v in logits { if v > max { max = v; } }
    let mut sum = 0.0f32;
    for (i, &l) in logits.iter().enumerate() {
        let e = (l - max).exp();
        out[i] = e;
        sum += e;
    }
    let inv = 1.0 / sum;
    for v in out[..n].iter_mut() { *v *= inv; }
}

fn rust_argmax_ref(logits: &[f32]) -> u32 {
    let mut best = f32::NEG_INFINITY;
    let mut idx = 0u32;
    for (i, &v) in logits.iter().enumerate() {
        if v > best { best = v; idx = i as u32; }
    }
    idx
}

fn rust_scale_ref(logits: &mut [f32], factor: f32) {
    for l in logits.iter_mut() { *l *= factor; }
}

fn bench_zig_vs_rust(c: &mut Criterion) {
    let logits: Vec<f32> = (0..VOCAB).map(|i| (i as f32 * 0.01) - 640.0).collect();
    let mut out = vec![0.0f32; VOCAB];

    c.bench_function("zig softmax 128k", |b| {
        b.iter(|| rvllm_zig::softmax_into(black_box(&logits), black_box(&mut out)))
    });
    c.bench_function("rust softmax 128k", |b| {
        b.iter(|| rust_softmax_ref(black_box(&logits), black_box(&mut out)))
    });

    c.bench_function("zig argmax 128k", |b| {
        b.iter(|| rvllm_zig::argmax(black_box(&logits)))
    });
    c.bench_function("rust argmax 128k", |b| {
        b.iter(|| rust_argmax_ref(black_box(&logits)))
    });

    // Fused argmax+logprob vs separate (4-pass Rust)
    c.bench_function("zig argmax_logprob 128k (fused 2-pass)", |b| {
        b.iter(|| rvllm_zig::argmax_logprob(black_box(&logits)))
    });
    c.bench_function("rust argmax+logprob 128k (4-pass)", |b| {
        b.iter(|| {
            let idx = rust_argmax_ref(black_box(&logits));
            let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let sum: f32 = logits.iter().map(|x| (x - max).exp()).sum();
            let lp = (logits[idx as usize] - max) - sum.ln();
            black_box((idx, lp))
        })
    });

    let mut logits_mut: Vec<f32> = logits.clone();
    c.bench_function("zig scale 128k", |b| {
        b.iter(|| rvllm_zig::temperature_scale(black_box(&mut logits_mut), black_box(0.5)))
    });
    let mut logits_mut2: Vec<f32> = logits.clone();
    c.bench_function("rust scale 128k", |b| {
        b.iter(|| rust_scale_ref(black_box(&mut logits_mut2), black_box(0.5)))
    });
}

fn bench_weight_convert(c: &mut Criterion) {
    let n = 4096 * 4096;
    let bf16_src: Vec<u16> = (0..n).map(|i| (i & 0xFFFF) as u16).collect();
    let f32_src: Vec<f32> = (0..n).map(|i| (i as f32) * 0.001 - 8.0).collect();
    let mut dst = vec![0u16; n];

    c.bench_function("zig bf16->f16 16M", |b| {
        b.iter(|| rvllm_zig::bf16_to_f16(black_box(&bf16_src), black_box(&mut dst)))
    });
    c.bench_function("zig f32->f16 16M", |b| {
        b.iter(|| rvllm_zig::f32_to_f16(black_box(&f32_src), black_box(&mut dst)))
    });
}

criterion_group!(
    benches,
    bench_greedy,
    bench_top_p,
    bench_top_k,
    bench_repetition_penalty,
    bench_zig_vs_rust,
    bench_weight_convert,
);
criterion_main!(benches);
