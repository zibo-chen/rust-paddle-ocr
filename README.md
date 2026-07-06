# Rust PaddleOCR

[English](README.md) | [中文](./docs/README.zh.md) | [日本語](./docs/README.ja.md) | [한국어](./docs/README.ko.md)

A lightweight Rust OCR library based on PaddleOCR models and the MNN inference runtime. It provides text detection, text recognition, and end-to-end OCR with file or in-memory model loading.

Related projects:
- CLI: [newbee-ocr-cli](https://github.com/zibo-chen/newbee-ocr-cli)
- C API bindings: [paddle-ocr-capi](https://github.com/zibo-chen/paddle-ocr-capi)
- HTTP service: `newbee_ocr_service` is local-only and is not published as a public repository.

## Supported Models

All runtime model files should be placed under `models/`.

| Family | Detection | Recognition | Notes |
|---|---|---|---|
| PP-OCRv4 | `ch_PP-OCRv4_det_infer.mnn` | `ch_PP-OCRv4_rec_infer.mnn` | Legacy CN/EN model |
| PP-OCRv5 | `PP-OCRv5_mobile_det.mnn` or `PP-OCRv5_mobile_det_fp16.mnn` | `PP-OCRv5_mobile_rec*.mnn` | Default CN/EN/JP plus script-specific models |
| PP-OCRv6 tiny | `PP-OCRv6_tiny_det.mnn` | `PP-OCRv6_tiny_rec.mnn` | Lightweight v6 tier; Japanese is not supported |
| PP-OCRv6 small | `PP-OCRv6_small_det.mnn` | `PP-OCRv6_small_rec.mnn` | Balanced v6 tier |
| PP-OCRv6 medium | `PP-OCRv6_medium_det.mnn` | `PP-OCRv6_medium_rec.mnn` | Accuracy-first v6 tier |

PP-OCRv6 `small` and `medium` support the official 50 v6 recognition languages: Simplified Chinese, Traditional Chinese, English, Japanese, and 46 Latin-script languages. PP-OCRv6 `tiny` follows the same set except Japanese. Korean, Cyrillic, Arabic, Devanagari, Thai, Greek, Tamil, and Telugu should continue using the PP-OCRv5 script-specific recognition models.

V6 charset files are tier-specific:

```text
models/ppocr_keys_v6_tiny.txt
models/ppocr_keys_v6_small.txt
models/ppocr_keys_v6_medium.txt
```

## Convert Paddle Models To MNN

The converter defaults to MNN FP16 to reduce model size. Use `--install-dir ./models` to copy converted runtime files into the expected directory and filenames.

```bash
python script/convert_paddle_to_mnn.py \
  --ocr-dir /path/to/paddle/inference/models \
  --install-dir ./models
```

Use `--no-fp16` only when full precision MNN output is required.

## Rust Usage

```rust
use ocr_rs::OcrEngine;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let engine = OcrEngine::new(
        "models/PP-OCRv6_small_det.mnn",
        "models/PP-OCRv6_small_rec.mnn",
        "models/ppocr_keys_v6_small.txt",
        None,
    )?;

    let image = image::open("test.jpg")?;
    let results = engine.recognize(&image)?;

    for item in results {
        println!("{:.2}: {}", item.confidence, item.text);
    }

    Ok(())
}
```

Detection-only and recognition-only engines are also available:

```rust
let det = ocr_rs::OcrEngine::det_only("models/PP-OCRv6_small_det.mnn", None)?;
let rec = ocr_rs::OcrEngine::rec_only(
    "models/PP-OCRv6_small_rec.mnn",
    "models/ppocr_keys_v6_small.txt",
    None,
)?;
```

## Build

```bash
cargo build --release
cargo test
```

## Performance Checks

Run Criterion benchmarks locally:

```bash
cargo bench --bench bench_metrics
```

Run the CI-style performance smoke test:

```bash
OCR_RS_PERF_TESTS=1 cargo test --release --test performance_tests -- --nocapture --test-threads=1
```

GitHub Actions runs the release smoke test on Ubuntu and compiles the Criterion benchmarks. The smoke test prints `PERF_METRIC` lines, but does not fail on fixed latency thresholds because hosted runners vary.

Prebuilt MNN libraries are used automatically when available. For custom MNN builds:

```bash
cargo build --features build-mnn-from-source
```

GPU backends are selected through `OcrEngineConfig`:

```rust
use ocr_rs::{Backend, OcrEngineConfig};

let config = OcrEngineConfig::new().with_backend(Backend::Metal);
```

## License

Apache-2.0
