# Rust PaddleOCR

[English](../README.md) | [中文](README.zh.md) | [日本語](README.ja.md) | [한국어](README.ko.md)

PaddleOCR モデルと MNN 推論ランタイムを利用する軽量な Rust OCR ライブラリです。テキスト検出、テキスト認識、エンドツーエンド OCR、ファイルまたはメモリバイトからのモデル読み込みをサポートします。

関連プロジェクト：
- CLI：[newbee-ocr-cli](https://github.com/zibo-chen/newbee-ocr-cli)
- C API バインディング：[paddle-ocr-capi](https://github.com/zibo-chen/paddle-ocr-capi)
- HTTP サービス：`newbee_ocr_service` はローカル専用で、公開リポジトリとしては公開していません。

## 対応モデル

実行時に使うモデルファイルはすべて `models/` に配置してください。

| ファミリー | 検出モデル | 認識モデル | 備考 |
|---|---|---|---|
| PP-OCRv4 | `ch_PP-OCRv4_det_infer.mnn` | `ch_PP-OCRv4_rec_infer.mnn` | 旧世代の中英モデル |
| PP-OCRv5 | `PP-OCRv5_mobile_det.mnn` または `PP-OCRv5_mobile_det_fp16.mnn` | `PP-OCRv5_mobile_rec*.mnn` | 標準の中/英/日モデルとスクリプト別モデル |
| PP-OCRv6 tiny | `PP-OCRv6_tiny_det.mnn` | `PP-OCRv6_tiny_rec.mnn` | 軽量 v6 ティア。日本語は非対応 |
| PP-OCRv6 small | `PP-OCRv6_small_det.mnn` | `PP-OCRv6_small_rec.mnn` | バランス重視の v6 ティア |
| PP-OCRv6 medium | `PP-OCRv6_medium_det.mnn` | `PP-OCRv6_medium_rec.mnn` | 精度重視の v6 ティア |

PP-OCRv6 `small` と `medium` は、簡体字中国語、繁体字中国語、英語、日本語、46 種類のラテン文字系言語を含む公式 50 言語をサポートします。PP-OCRv6 `tiny` は同じ範囲から日本語を除いたものです。韓国語、キリル文字、アラビア文字、デーヴァナーガリー、タイ語、ギリシャ語、タミル語、テルグ語は PP-OCRv5 のスクリプト別認識モデルを使用してください。

v6 の文字セットファイルはティアごとに分かれています。

```text
models/ppocr_keys_v6_tiny.txt
models/ppocr_keys_v6_small.txt
models/ppocr_keys_v6_medium.txt
```

## Paddle モデルから MNN への変換

変換スクリプトは既定で MNN FP16 を有効にし、モデルサイズを削減します。`--install-dir ./models` を指定すると、実行時に必要な標準ファイル名で `models/` にコピーします。

```bash
python script/convert_paddle_to_mnn.py \
  --ocr-dir /path/to/paddle/inference/models \
  --install-dir ./models
```

フル精度の MNN が必要な場合のみ `--no-fp16` を指定してください。

## Rust での利用

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

検出のみ、認識のみのエンジンも作成できます。

```rust
let det = ocr_rs::OcrEngine::det_only("models/PP-OCRv6_small_det.mnn", None)?;
let rec = ocr_rs::OcrEngine::rec_only(
    "models/PP-OCRv6_small_rec.mnn",
    "models/ppocr_keys_v6_small.txt",
    None,
)?;
```

## ビルド

```bash
cargo build --release
cargo test
```

## パフォーマンス確認

ローカルで Criterion ベンチマークを実行します。

```bash
cargo bench --bench bench_metrics
```

CI と同じ短いパフォーマンス smoke テストを実行します。

```bash
OCR_RS_PERF_TESTS=1 cargo test --release --test performance_tests -- --nocapture --test-threads=1
```

GitHub Actions は Ubuntu 上で release モードの smoke テストを実行し、Criterion ベンチマークをコンパイルします。smoke テストは `PERF_METRIC` 行を出力しますが、ホスト runner の性能差が大きいため固定の遅延しきい値では失敗させません。

利用可能な場合は事前ビルド済み MNN ライブラリが自動的に使われます。MNN をソースからビルドする場合：

```bash
cargo build --features build-mnn-from-source
```

GPU バックエンドは `OcrEngineConfig` で指定します。

```rust
use ocr_rs::{Backend, OcrEngineConfig};

let config = OcrEngineConfig::new().with_backend(Backend::Metal);
```

## License

Apache-2.0
