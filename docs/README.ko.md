# Rust PaddleOCR

[English](../README.md) | [中文](README.zh.md) | [日本語](README.ja.md) | [한국어](README.ko.md)

PaddleOCR 모델과 MNN 추론 런타임을 사용하는 경량 Rust OCR 라이브러리입니다. 텍스트 검출, 텍스트 인식, 엔드투엔드 OCR, 파일 또는 메모리 바이트에서의 모델 로딩을 지원합니다.

관련 프로젝트:
- CLI: [newbee-ocr-cli](https://github.com/zibo-chen/newbee-ocr-cli)
- C API 바인딩: [paddle-ocr-capi](https://github.com/zibo-chen/paddle-ocr-capi)
- HTTP 서비스: `newbee_ocr_service`는 로컬 전용이며 공개 저장소로 게시하지 않았습니다.

## 지원 모델

런타임 MNN 모델 파일은 모두 `models/` 아래에 두어야 합니다.

| 계열 | 검출 모델 | 인식 모델 | 설명 |
|---|---|---|---|
| PP-OCRv4 | `ch_PP-OCRv4_det_infer.mnn` | `ch_PP-OCRv4_rec_infer.mnn` | 구형 중/영 모델 |
| PP-OCRv5 | `PP-OCRv5_mobile_det.mnn` 또는 `PP-OCRv5_mobile_det_fp16.mnn` | `PP-OCRv5_mobile_rec*.mnn` | 기본 중/영/일 모델과 스크립트별 모델 |
| PP-OCRv6 tiny | `PP-OCRv6_tiny_det.mnn` | `PP-OCRv6_tiny_rec.mnn` | 경량 v6 티어. 일본어는 지원하지 않음 |
| PP-OCRv6 small | `PP-OCRv6_small_det.mnn` | `PP-OCRv6_small_rec.mnn` | 균형형 v6 티어 |
| PP-OCRv6 medium | `PP-OCRv6_medium_det.mnn` | `PP-OCRv6_medium_rec.mnn` | 정확도 우선 v6 티어 |

PP-OCRv6 `small` 과 `medium` 은 중국어 간체, 중국어 번체, 영어, 일본어, 46개 라틴 문자 언어를 포함한 공식 v6 인식 언어 50개를 지원합니다. PP-OCRv6 `tiny` 는 같은 범위에서 일본어를 제외합니다. 한국어, 키릴 문자, 아랍 문자, 데바나가리, 태국어, 그리스어, 타밀어, 텔루구어는 PP-OCRv5 스크립트별 인식 모델을 사용해야 합니다.

v6 문자셋 파일은 티어별로 분리되어 있습니다.

```text
models/ppocr_keys_v6_tiny.txt
models/ppocr_keys_v6_small.txt
models/ppocr_keys_v6_medium.txt
```

## Paddle 모델을 MNN 으로 변환

변환 스크립트는 기본적으로 MNN FP16 을 사용해 모델 크기를 줄입니다. `--install-dir ./models` 를 지정하면 런타임에서 쓰는 표준 파일명으로 `models/` 에 복사합니다.

```bash
python script/convert_paddle_to_mnn.py \
  --ocr-dir /path/to/paddle/inference/models \
  --install-dir ./models
```

전체 정밀도 MNN 이 필요할 때만 `--no-fp16` 을 사용하세요.

## Rust 사용법

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

검출 전용 또는 인식 전용 엔진도 만들 수 있습니다.

```rust
let det = ocr_rs::OcrEngine::det_only("models/PP-OCRv6_small_det.mnn", None)?;
let rec = ocr_rs::OcrEngine::rec_only(
    "models/PP-OCRv6_small_rec.mnn",
    "models/ppocr_keys_v6_small.txt",
    None,
)?;
```

## 빌드

```bash
cargo build --release
cargo test
```

## 성능 확인

로컬에서 Criterion 벤치마크를 실행합니다.

```bash
cargo bench --bench bench_metrics
```

CI 방식의 짧은 성능 smoke 테스트를 실행합니다.

```bash
OCR_RS_PERF_TESTS=1 cargo test --release --test performance_tests -- --nocapture --test-threads=1
```

GitHub Actions 는 release 모드 테스트를 직렬로 실행하고 `PERF_METRIC` 로그를 artifact 로 저장합니다. 회귀 검사는 동일한 runner 에서 direct exact-width 파이프라인과 기존 crop 파이프라인을 비교하며, 중앙값 비율이 `OCR_RS_PERF_REGRESSION_LIMIT`(기본값 `1.15`)을 초과하면 실패하므로 불안정한 절대 지연 시간에 의존하지 않습니다.

호환되는 경우 CPU 또는 Apple Metal 사전 빌드 MNN 이 자동으로 사용됩니다. 사전 빌드 패키지에 없는 GPU feature 를 활성화하면 MNN 을 소스에서 자동으로 빌드합니다.

```bash
cargo build --features build-mnn-from-source
cargo build --release --features cuda
cargo build --release --features vulkan
```

빌드하기 전에 선택한 백엔드의 SDK 와 개발 라이브러리를 설치해야 합니다. GPU 백엔드는 `OcrEngineConfig` 로 선택합니다.

```rust
use ocr_rs::{Backend, OcrEngineConfig};

let config = OcrEngineConfig::new().with_backend(Backend::Metal);
assert!(Backend::Metal.is_available());
```

링크된 MNN 에 요청한 백엔드가 등록되어 있지 않으면 CPU 로 조용히 폴백하지 않고 엔진 생성 시 `MnnError::BackendUnavailable` 을 반환합니다.

`x86_64-pc-windows-gnu` 는 MNN 을 소스에서 빌드하므로 MinGW C/C++ 툴체인이 필요합니다. 기본적으로 애플리케이션과 함께 일치하는 MinGW 런타임 DLL을 배포해야 합니다. `static-cpp-runtime` 을 활성화하면 libstdc++, libgcc 및 winpthreads를 정적으로 링크하여 생성된 바이너리의 MinGW 런타임 DLL 의존성을 제거할 수 있습니다.

```bash
cargo build --release --target x86_64-pc-windows-gnu --features static-cpp-runtime
```

NVIDIA Windows CUDA 툴체인은 MSVC 를 요구합니다. CUDA 소스 빌드에는 `x86_64-pc-windows-msvc` 를 사용하거나 `mnn-dynamic`/`mnn-static` 으로 호환 MNN 라이브러리를 제공하세요.

이 feature는 `ocr-rs` 자체가 링크하는 런타임만 제어합니다. `mnn-dynamic` 으로 제공한 DLL에는 자체 MinGW 런타임 의존성이 남아 있을 수 있습니다.

## License

Apache-2.0
