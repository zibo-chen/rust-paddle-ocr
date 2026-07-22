//! CI-safe performance smoke tests.
//!
//! These tests are opt-in because OCR inference is much slower than normal unit tests.
//! Run locally with:
//!
//! OCR_RS_PERF_TESTS=1 cargo test --release --test performance_tests -- --nocapture --test-threads=1

use ocr_rs::{DetModel, DetOptions, OcrEngine, OcrEngineConfig, RecModel, RecOptions};
use rayon::prelude::*;
use std::hint::black_box;
use std::path::Path;
use std::time::{Duration, Instant};

const PERF_ENV: &str = "OCR_RS_PERF_TESTS";
const TEST_IMAGE: &str = "res/1.png";
const TEST_PAGE_IMAGE: &str = "res/2.png";
const DET_MODEL: &str = "models/PP-OCRv6_tiny_det.mnn";
const REC_MODEL: &str = "models/PP-OCRv6_tiny_rec.mnn";
const CHARSET: &str = "models/ppocr_keys_v6_tiny.txt";
const DET_MODEL_V5: &str = "models/PP-OCRv5_mobile_det_fp16.mnn";
const REC_MODEL_V5: &str = "models/PP-OCRv5_mobile_rec_fp16.mnn";
const CHARSET_V5: &str = "models/ppocr_keys_v5.txt";

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

fn env_f64(name: &str, default: f64) -> f64 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|value| value.is_finite() && *value > 0.0)
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

fn perf_config(enable_parallel: bool) -> OcrEngineConfig {
    OcrEngineConfig::fast()
        .with_threads(4)
        .with_parallel(enable_parallel)
}

fn mean_ms(durations: &[Duration]) -> f64 {
    durations.iter().map(Duration::as_secs_f64).sum::<f64>() * 1000.0 / durations.len() as f64
}

fn median_ms(durations: &[Duration]) -> f64 {
    let mut samples = durations
        .iter()
        .map(|duration| duration.as_secs_f64() * 1000.0)
        .collect::<Vec<_>>();
    samples.sort_by(f64::total_cmp);
    let midpoint = samples.len() / 2;

    if samples.len() % 2 == 0 {
        (samples[midpoint - 1] + samples[midpoint]) / 2.0
    } else {
        samples[midpoint]
    }
}

fn measure_comparison<T, U, F, G>(
    optimized_name: &str,
    legacy_name: &str,
    iterations: usize,
    mut optimized: F,
    mut legacy: G,
) -> (Vec<Duration>, Vec<Duration>)
where
    F: FnMut() -> T,
    G: FnMut() -> U,
{
    let mut optimized_durations = Vec::with_capacity(iterations);
    let mut legacy_durations = Vec::with_capacity(iterations);

    for iteration in 0..iterations {
        let mut run_optimized = || {
            let started = Instant::now();
            black_box(optimized());
            optimized_durations.push(started.elapsed());
        };
        let mut run_legacy = || {
            let started = Instant::now();
            black_box(legacy());
            legacy_durations.push(started.elapsed());
        };

        if iteration % 2 == 0 {
            run_optimized();
            run_legacy();
        } else {
            run_legacy();
            run_optimized();
        }
    }

    print_metric(optimized_name, &optimized_durations);
    print_metric(legacy_name, &legacy_durations);
    (optimized_durations, legacy_durations)
}

fn legacy_exact_width_pipeline(engine: &OcrEngine, image: &image::DynamicImage) -> usize {
    let detections = engine
        .det_model()
        .detect_and_crop(image)
        .expect("legacy detection crop failed");
    let images = detections
        .into_iter()
        .map(|(crop, _)| crop)
        .collect::<Vec<_>>();

    let results = if images.len() > 4 {
        images
            .par_iter()
            .map(|crop| engine.rec_model().recognize(crop))
            .collect::<ocr_rs::OcrResult<Vec<_>>>()
            .expect("legacy parallel recognition failed")
    } else {
        engine
            .rec_model()
            .recognize_batch(&images)
            .expect("legacy batch recognition failed")
    };

    results.len()
}

fn assert_parallel_pipeline_performance(
    label: &str,
    image: &image::DynamicImage,
    det_model: &str,
    rec_model: &str,
    charset: &str,
    iterations: usize,
) {
    let engine = OcrEngine::new(det_model, rec_model, charset, Some(perf_config(true)))
        .expect("failed to create OCR engine");

    let warmup_results = engine.recognize(image).expect("optimized warmup failed");
    assert!(
        warmup_results.len() > 4,
        "page fixture needs multiple lines"
    );
    assert_eq!(
        legacy_exact_width_pipeline(&engine, image),
        engine
            .det_model()
            .detect(image)
            .expect("detection count failed")
            .len(),
        "legacy warmup should recognize every detected region"
    );

    let optimized_name = format!("{label}_parallel_direct_pipeline");
    let legacy_name = format!("{label}_legacy_exact_width_pipeline");
    let (optimized, legacy) = measure_comparison(
        &optimized_name,
        &legacy_name,
        iterations,
        || {
            engine
                .recognize(image)
                .expect("optimized full pipeline failed")
                .len()
        },
        || legacy_exact_width_pipeline(&engine, image),
    );
    let optimized_mean_ms = mean_ms(&optimized);
    let legacy_mean_ms = mean_ms(&legacy);
    let optimized_median_ms = median_ms(&optimized);
    let legacy_median_ms = median_ms(&legacy);
    let ratio = optimized_median_ms / legacy_median_ms;
    let regression_limit = env_f64("OCR_RS_PERF_REGRESSION_LIMIT", 1.15);

    println!(
        "PERF_COMPARISON label={label} optimized_mean_ms={optimized_mean_ms:.3} legacy_mean_ms={legacy_mean_ms:.3} optimized_median_ms={optimized_median_ms:.3} legacy_median_ms={legacy_median_ms:.3} ratio={ratio:.3} limit={regression_limit:.3}"
    );
    assert!(
        ratio <= regression_limit,
        "{label} parallel direct pipeline regressed: optimized={optimized_median_ms:.3}ms legacy={legacy_median_ms:.3}ms ratio={ratio:.3} limit={regression_limit:.3}"
    );
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
    let median_ms = median_ms(durations);
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
        "PERF_METRIC name={name} iterations={} min_ms={min_ms:.3} median_ms={median_ms:.3} mean_ms={mean_ms:.3} max_ms={max_ms:.3} samples_ms=[{samples_ms}]",
        durations.len()
    );
}

#[test]
fn performance_smoke_reports_v6_tiny_metrics() {
    if !perf_enabled() {
        eprintln!("skipping performance smoke; set {PERF_ENV}=1 to run");
        return;
    }

    if cfg!(debug_assertions) {
        panic!("performance smoke should be run in release mode");
    }
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
    let parallel_engine = OcrEngine::new(DET_MODEL, REC_MODEL, CHARSET, Some(perf_config(true)))
        .expect("failed to create OCR engine");
    let batch_engine = OcrEngine::new(DET_MODEL, REC_MODEL, CHARSET, Some(perf_config(false)))
        .expect("failed to create batch OCR engine");

    let warmup_results = parallel_engine
        .recognize(&image)
        .expect("parallel warmup OCR failed");
    assert!(
        !warmup_results.is_empty(),
        "warmup OCR should find text in {TEST_IMAGE}"
    );
    batch_engine
        .recognize(&image)
        .expect("batch warmup OCR failed");

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

    measure("v6_tiny_full_pipeline_parallel", iterations, || {
        let results = parallel_engine
            .recognize(&image)
            .expect("parallel full OCR failed");
        assert!(!results.is_empty(), "full pipeline should find text");
        results.len()
    });

    measure("v6_tiny_full_pipeline_batch", iterations, || {
        let results = batch_engine
            .recognize(&image)
            .expect("batch full OCR failed");
        assert!(!results.is_empty(), "batch pipeline should find text");
        results.len()
    });
}

#[test]
fn parallel_pipeline_does_not_regress_against_legacy_exact_width_dispatch() {
    if !perf_enabled() {
        eprintln!("skipping performance regression guard; set {PERF_ENV}=1 to run");
        return;
    }

    if cfg!(debug_assertions) {
        panic!("performance regression guard should be run in release mode");
    }
    assert_files_exist(&[
        TEST_PAGE_IMAGE,
        DET_MODEL,
        REC_MODEL,
        CHARSET,
        DET_MODEL_V5,
        REC_MODEL_V5,
        CHARSET_V5,
    ]);

    let iterations = env_usize("OCR_RS_PERF_ITERS", 3).max(3);
    let image = image::open(TEST_PAGE_IMAGE).expect("failed to load page test image");
    assert_parallel_pipeline_performance(
        "v6_tiny", &image, DET_MODEL, REC_MODEL, CHARSET, iterations,
    );
    assert_parallel_pipeline_performance(
        "v5_mobile_fp16",
        &image,
        DET_MODEL_V5,
        REC_MODEL_V5,
        CHARSET_V5,
        iterations,
    );
}
