use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use image::{DynamicImage, GenericImageView};
use ocr_rs::preprocess::{
    preprocess_batch_for_rec, preprocess_for_det, resize_to_max_side, NormalizeParams,
};
use ocr_rs::{DetModel, DetOptions, OcrEngine, OcrEngineConfig, RecModel, RecOptions};
use std::path::Path;
use std::time::Duration;

const TEST_IMAGE: &str = "res/1.png";
const DET_V5: &str = "models/PP-OCRv5_mobile_det.mnn";
const REC_V5: &str = "models/PP-OCRv5_mobile_rec.mnn";
const CHARSET_V5: &str = "models/ppocr_keys_v5.txt";
const DET_V6_TINY: &str = "models/PP-OCRv6_tiny_det.mnn";
const REC_V6_TINY: &str = "models/PP-OCRv6_tiny_rec.mnn";
const CHARSET_V6_TINY: &str = "models/ppocr_keys_v6_tiny.txt";
const DET_V6_SMALL: &str = "models/PP-OCRv6_small_det.mnn";
const REC_V6_SMALL: &str = "models/PP-OCRv6_small_rec.mnn";
const CHARSET_V6_SMALL: &str = "models/ppocr_keys_v6_small.txt";
const DET_V6_MEDIUM: &str = "models/PP-OCRv6_medium_det.mnn";
const REC_V6_MEDIUM: &str = "models/PP-OCRv6_medium_rec.mnn";
const CHARSET_V6_MEDIUM: &str = "models/ppocr_keys_v6_medium.txt";

#[derive(Clone, Copy)]
struct ModelSuite {
    label: &'static str,
    det_path: &'static str,
    rec_path: &'static str,
    charset_path: &'static str,
}

const MODEL_SUITES: &[ModelSuite] = &[
    ModelSuite {
        label: "v5_mobile",
        det_path: DET_V5,
        rec_path: REC_V5,
        charset_path: CHARSET_V5,
    },
    ModelSuite {
        label: "v6_tiny",
        det_path: DET_V6_TINY,
        rec_path: REC_V6_TINY,
        charset_path: CHARSET_V6_TINY,
    },
    ModelSuite {
        label: "v6_small",
        det_path: DET_V6_SMALL,
        rec_path: REC_V6_SMALL,
        charset_path: CHARSET_V6_SMALL,
    },
    ModelSuite {
        label: "v6_medium",
        det_path: DET_V6_MEDIUM,
        rec_path: REC_V6_MEDIUM,
        charset_path: CHARSET_V6_MEDIUM,
    },
];

fn has_files(paths: &[&str]) -> bool {
    let missing: Vec<_> = paths
        .iter()
        .copied()
        .filter(|path| !Path::new(path).exists())
        .collect();

    if !missing.is_empty() {
        eprintln!("skipping benchmark case; missing files: {missing:?}");
        return false;
    }

    true
}

fn load_image() -> Option<DynamicImage> {
    if !has_files(&[TEST_IMAGE]) {
        return None;
    }

    Some(image::open(TEST_IMAGE).expect("failed to load benchmark image"))
}

fn bench_config() -> OcrEngineConfig {
    OcrEngineConfig::fast().with_parallel(false).with_threads(4)
}

fn load_detector(det_path: &str) -> DetModel {
    DetModel::from_file(det_path, None)
        .expect("failed to load detection model")
        .with_options(DetOptions::fast())
}

fn load_recognizer(rec_path: &str, charset_path: &str, batch_size: usize) -> RecModel {
    RecModel::from_file(rec_path, charset_path, None)
        .expect("failed to load recognition model")
        .with_options(RecOptions::new().with_batch_size(batch_size))
}

fn crops_for_suite(suite: ModelSuite, image: &DynamicImage) -> Option<Vec<DynamicImage>> {
    if !has_files(&[suite.det_path]) {
        return None;
    }

    let det = load_detector(suite.det_path);
    let crops = det
        .detect_and_crop(image)
        .expect("failed to prepare recognition crops");
    if crops.is_empty() {
        eprintln!(
            "skipping recognition benchmark for {}; no crops detected",
            suite.label
        );
        return None;
    }

    Some(crops.into_iter().map(|(crop, _)| crop).collect())
}

fn repeated_crops(crops: &[DynamicImage], target_len: usize) -> Vec<DynamicImage> {
    crops.iter().cloned().cycle().take(target_len).collect()
}

fn bench_preprocessing(c: &mut Criterion) {
    let Some(image) = load_image() else {
        return;
    };

    let (width, height) = image.dimensions();
    let det_params = NormalizeParams::paddle_det();
    let rec_params = NormalizeParams::paddle_rec();
    let line_images = vec![
        DynamicImage::new_rgb8(160, 48),
        DynamicImage::new_rgb8(240, 48),
        DynamicImage::new_rgb8(360, 64),
        DynamicImage::new_rgb8(480, 96),
    ];

    let mut group = c.benchmark_group("preprocessing");
    group.throughput(Throughput::Elements((width as u64) * (height as u64)));

    group.bench_function("resize_to_max_side_960", |b| {
        b.iter(|| {
            let resized = resize_to_max_side(black_box(&image), black_box(960))
                .expect("resize_to_max_side failed");
            black_box(resized);
        });
    });

    group.bench_function("det_tensor", |b| {
        b.iter(|| {
            let tensor = preprocess_for_det(black_box(&image), black_box(&det_params))
                .expect("det preprocess failed");
            black_box(tensor);
        });
    });

    group.bench_function("rec_batch_tensor_4", |b| {
        b.iter(|| {
            let tensor = preprocess_batch_for_rec(
                black_box(&line_images),
                black_box(48),
                black_box(&rec_params),
            )
            .expect("rec batch preprocess failed");
            black_box(tensor);
        });
    });

    group.finish();
}

fn bench_detection(c: &mut Criterion) {
    let Some(image) = load_image() else {
        return;
    };

    let mut group = c.benchmark_group("text_detection");
    group.throughput(Throughput::Elements(1));

    for suite in MODEL_SUITES {
        if !has_files(&[suite.det_path]) {
            continue;
        }

        let det = load_detector(suite.det_path);
        group.bench_with_input(BenchmarkId::new("detect", suite.label), suite, |b, _| {
            b.iter(|| {
                let boxes = det.detect(black_box(&image)).expect("detection failed");
                black_box(boxes);
            });
        });

        group.bench_with_input(
            BenchmarkId::new("detect_and_crop", suite.label),
            suite,
            |b, _| {
                b.iter(|| {
                    let crops = det
                        .detect_and_crop(black_box(&image))
                        .expect("detection crop failed");
                    black_box(crops);
                });
            },
        );
    }

    group.finish();
}

fn bench_recognition(c: &mut Criterion) {
    let Some(image) = load_image() else {
        return;
    };

    let mut group = c.benchmark_group("text_recognition");

    for suite in MODEL_SUITES {
        if !has_files(&[suite.det_path, suite.rec_path, suite.charset_path]) {
            continue;
        }

        let Some(crops) = crops_for_suite(*suite, &image) else {
            continue;
        };
        let first_crop = crops[0].clone();
        let batch_crops = repeated_crops(&crops, 8);
        let rec = load_recognizer(suite.rec_path, suite.charset_path, 8);

        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::new("single_line", suite.label),
            suite,
            |b, _| {
                b.iter(|| {
                    let result = rec
                        .recognize(black_box(&first_crop))
                        .expect("recognition failed");
                    black_box(result);
                });
            },
        );

        group.throughput(Throughput::Elements(batch_crops.len() as u64));
        group.bench_with_input(
            BenchmarkId::new("batch_8_lines", suite.label),
            suite,
            |b, _| {
                b.iter(|| {
                    let results = rec
                        .recognize_batch(black_box(&batch_crops))
                        .expect("batch recognition failed");
                    black_box(results);
                });
            },
        );
    }

    group.finish();
}

fn bench_recognition_batch_scaling(c: &mut Criterion) {
    let Some(image) = load_image() else {
        return;
    };

    let suite = ModelSuite {
        label: "v6_tiny",
        det_path: DET_V6_TINY,
        rec_path: REC_V6_TINY,
        charset_path: CHARSET_V6_TINY,
    };

    if !has_files(&[suite.det_path, suite.rec_path, suite.charset_path]) {
        return;
    }

    let Some(crops) = crops_for_suite(suite, &image) else {
        return;
    };
    let rec = load_recognizer(suite.rec_path, suite.charset_path, 8);

    let mut group = c.benchmark_group("recognition_batch_scaling");
    for batch_size in [1usize, 2, 4, 8, 16] {
        let batch_crops = repeated_crops(&crops, batch_size);
        group.throughput(Throughput::Elements(batch_crops.len() as u64));
        group.bench_with_input(
            BenchmarkId::new("v6_tiny_batch", batch_size),
            &batch_crops,
            |b, images| {
                b.iter(|| {
                    let results = rec
                        .recognize_batch(black_box(images))
                        .expect("batch recognition failed");
                    black_box(results);
                });
            },
        );
    }
    group.finish();
}

fn bench_full_pipeline(c: &mut Criterion) {
    let Some(image) = load_image() else {
        return;
    };

    let mut group = c.benchmark_group("ocr_pipeline");
    group.throughput(Throughput::Elements(1));

    for suite in MODEL_SUITES {
        if !has_files(&[suite.det_path, suite.rec_path, suite.charset_path]) {
            continue;
        }

        let engine = OcrEngine::new(
            suite.det_path,
            suite.rec_path,
            suite.charset_path,
            Some(bench_config()),
        )
        .expect("failed to create OCR engine");

        group.bench_with_input(BenchmarkId::new("recognize", suite.label), suite, |b, _| {
            b.iter(|| {
                let results = engine.recognize(black_box(&image)).expect("OCR failed");
                black_box(results);
            });
        });
    }

    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(10)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(5));
    targets =
        bench_preprocessing,
        bench_detection,
        bench_recognition,
        bench_recognition_batch_scaling,
        bench_full_pipeline
}
criterion_main!(benches);
