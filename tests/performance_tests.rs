//! CI-safe performance smoke tests.
//!
//! These tests are opt-in because OCR inference is much slower than normal unit tests.
//! Run locally with:
//!
//! OCR_RS_PERF_TESTS=1 cargo test --release --test performance_tests -- --nocapture --test-threads=1

use ocr_rs::{DetModel, DetOptions, OcrEngine, OcrEngineConfig, RecModel, RecOptions};
use std::hint::black_box;
use std::path::Path;
use std::time::{Duration, Instant};

const PERF_ENV: &str = "OCR_RS_PERF_TESTS";
const TEST_IMAGE: &str = "res/1.png";
const DET_MODEL: &str = "models/PP-OCRv6_tiny_det.mnn";
const REC_MODEL: &str = "models/PP-OCRv6_tiny_rec.mnn";
const CHARSET: &str = "models/ppocr_keys_v6_tiny.txt";

fn perf_enabled() -> bool {
    std::env::var(PERF_ENV)
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(false)
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(default)
}

fn assert_files_exist(paths: &[&str]) {
    for path in paths {
        assert!(
            Path::new(path).exists(),
            "performance test fixture is missing: {path}"
        );
    }
}

fn perf_config() -> OcrEngineConfig {
    OcrEngineConfig::fast().with_threads(4).with_parallel(false)
}

fn measure<T, F>(name: &str, iterations: usize, mut f: F) -> Vec<Duration>
where
    F: FnMut() -> T,
{
    let mut durations = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let started = Instant::now();
        let value = f();
        black_box(value);
        durations.push(started.elapsed());
    }
    print_metric(name, &durations);
    durations
}

fn print_metric(name: &str, durations: &[Duration]) {
    let total_secs: f64 = durations.iter().map(Duration::as_secs_f64).sum();
    let mean_ms = total_secs * 1000.0 / durations.len() as f64;
    let min_ms = durations
        .iter()
        .map(Duration::as_secs_f64)
        .fold(f64::INFINITY, f64::min)
        * 1000.0;
    let max_ms = durations
        .iter()
        .map(Duration::as_secs_f64)
        .fold(0.0, f64::max)
        * 1000.0;
    let samples_ms = durations
        .iter()
        .map(|duration| format!("{:.3}", duration.as_secs_f64() * 1000.0))
        .collect::<Vec<_>>()
        .join(",");

    println!(
        "PERF_METRIC name={name} iterations={} min_ms={min_ms:.3} mean_ms={mean_ms:.3} max_ms={max_ms:.3} samples_ms=[{samples_ms}]",
        durations.len()
    );
}

#[test]
fn performance_smoke_reports_v6_tiny_metrics() {
    if !perf_enabled() {
        eprintln!("skipping performance smoke; set {PERF_ENV}=1 to run");
        return;
    }

    assert!(
        !cfg!(debug_assertions),
        "performance smoke should be run in release mode"
    );
    assert_files_exist(&[TEST_IMAGE, DET_MODEL, REC_MODEL, CHARSET]);

    let iterations = env_usize("OCR_RS_PERF_ITERS", 3);
    let batch_size = env_usize("OCR_RS_PERF_BATCH_SIZE", 8).clamp(3, 32);
    let image = image::open(TEST_IMAGE).expect("failed to load test image");

    let det = DetModel::from_file(DET_MODEL, None)
        .expect("failed to load detection model")
        .with_options(DetOptions::fast());
    let rec_single = RecModel::from_file(REC_MODEL, CHARSET, None)
        .expect("failed to load recognition model")
        .with_options(RecOptions::new().with_batch(false));
    let rec_batch = RecModel::from_file(REC_MODEL, CHARSET, None)
        .expect("failed to load recognition model")
        .with_options(RecOptions::new().with_batch_size(batch_size));
    let engine = OcrEngine::new(DET_MODEL, REC_MODEL, CHARSET, Some(perf_config()))
        .expect("failed to create OCR engine");

    let warmup_results = engine.recognize(&image).expect("warmup OCR failed");
    assert!(
        !warmup_results.is_empty(),
        "warmup OCR should find text in {TEST_IMAGE}"
    );

    let crops = det
        .detect_and_crop(&image)
        .expect("failed to prepare recognition crops");
    assert!(!crops.is_empty(), "test image should produce text crops");

    let first_crop = crops[0].0.clone();
    let batch_crops = crops
        .iter()
        .map(|(crop, _)| crop.clone())
        .cycle()
        .take(batch_size)
        .collect::<Vec<_>>();

    measure("v6_tiny_detect", iterations, || {
        let boxes = det.detect(&image).expect("detection failed");
        assert!(!boxes.is_empty(), "detection should find boxes");
        boxes.len()
    });

    measure("v6_tiny_detect_and_crop", iterations, || {
        let crops = det.detect_and_crop(&image).expect("detection crop failed");
        assert!(!crops.is_empty(), "detection crop should find crops");
        crops.len()
    });

    measure("v6_tiny_recognize_single_line", iterations, || {
        let result = rec_single
            .recognize(&first_crop)
            .expect("single recognition failed");
        assert!(
            result.confidence >= 0.0 && result.confidence <= 1.0,
            "confidence should stay in [0, 1]"
        );
        result.text.len()
    });

    measure("v6_tiny_recognize_batch", iterations, || {
        let results = rec_batch
            .recognize_batch(&batch_crops)
            .expect("batch recognition failed");
        assert_eq!(results.len(), batch_crops.len());
        results.len()
    });

    measure("v6_tiny_full_pipeline", iterations, || {
        let results = engine.recognize(&image).expect("full OCR failed");
        assert!(!results.is_empty(), "full pipeline should find text");
        results.len()
    });
}
