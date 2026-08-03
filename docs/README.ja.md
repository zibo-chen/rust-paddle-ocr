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

GitHub Actions は release モードのテストを直列実行し、`PERF_METRIC` ログを artifact として保存します。回帰ガードは同じ runner 上で direct exact-width パイプラインと従来の crop パイプラインを比較し、中央値の比率が `OCR_RS_PERF_REGRESSION_LIMIT`（既定値 `1.15`）を超えた場合に失敗するため、不安定な絶対時間には依存しません。

互換性がある場合は CPU または Apple Metal の事前ビルド済み MNN が自動的に使われます。事前ビルドに含まれない GPU feature を有効にすると、MNN は自動的にソースからビルドされます。

```bash
cargo build --features build-mnn-from-source
cargo build --release --features cuda
cargo build --release --features vulkan
```

ビルド前に対象バックエンドの SDK と開発ライブラリをインストールしてください。GPU バックエンドは `OcrEngineConfig` で指定します。

```rust
use ocr_rs::{Backend, OcrEngineConfig};

let config = OcrEngineConfig::new().with_backend(Backend::Metal);
assert!(Backend::Metal.is_available());
```

リンクされた MNN に要求したバックエンドが登録されていない場合、CPU に暗黙でフォールバックせず、エンジン作成時に `MnnError::BackendUnavailable` を返します。

`x86_64-pc-windows-gnu` は MNN をソースからビルドするため、MinGW C/C++ ツールチェーンが必要です。デフォルトでは対応する MinGW ランタイム DLL の配布が必要です。`static-cpp-runtime` を有効にすると libstdc++、libgcc、winpthreads が静的リンクされ、生成されたバイナリは MinGW ランタイム DLL に依存しません。

```bash
cargo build --release --target x86_64-pc-windows-gnu --features static-cpp-runtime
```

NVIDIA の Windows CUDA ツールチェーンには MSVC が必要です。CUDA のソースビルドには `x86_64-pc-windows-msvc` を使うか、`mnn-dynamic`/`mnn-static` で互換 MNN を指定してください。

この feature が制御するのは `ocr-rs` 自身がリンクするランタイムです。`mnn-dynamic` で指定した DLL には、独自の MinGW ランタイム依存関係が残る場合があります。

## License

Apache-2.0
