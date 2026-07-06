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

GitHub Actions 는 Ubuntu 에서 release 모드 smoke 테스트를 실행하고 Criterion 벤치마크를 컴파일합니다. smoke 테스트는 `PERF_METRIC` 행을 출력하지만, 호스팅 runner 의 성능 차이가 크기 때문에 고정 지연 시간 임계값으로 실패시키지는 않습니다.

사용 가능한 경우 사전 빌드된 MNN 라이브러리가 자동으로 사용됩니다. MNN 을 소스에서 빌드하려면:

```bash
cargo build --features build-mnn-from-source
```

GPU 백엔드는 `OcrEngineConfig` 로 선택합니다.

```rust
use ocr_rs::{Backend, OcrEngineConfig};

let config = OcrEngineConfig::new().with_backend(Backend::Metal);
```

## License

Apache-2.0
