# Rust PaddleOCR

[English](../README.md) | [中文](README.zh.md) | [日本語](README.ja.md) | [한국어](README.ko.md)

一个基于 PaddleOCR 模型和 MNN 推理运行时的轻量级 Rust OCR 库。支持文本检测、文本识别、端到端 OCR，以及从文件或内存字节加载模型。

相关项目：
- CLI：[newbee-ocr-cli](https://github.com/zibo-chen/newbee-ocr-cli)
- C API 绑定：[paddle-ocr-capi](https://github.com/zibo-chen/paddle-ocr-capi)
- HTTP 服务：`newbee_ocr_service` 仅在本地仓库中，未发布为公开项目。

## 支持的模型

所有运行时模型文件都应放在 `models/` 下。

| 系列 | 检测模型 | 识别模型 | 说明 |
|---|---|---|---|
| PP-OCRv4 | `ch_PP-OCRv4_det_infer.mnn` | `ch_PP-OCRv4_rec_infer.mnn` | 旧版中英文模型 |
| PP-OCRv5 | `PP-OCRv5_mobile_det.mnn` 或 `PP-OCRv5_mobile_det_fp16.mnn` | `PP-OCRv5_mobile_rec*.mnn` | 默认中/英/日，另有脚本专用模型 |
| PP-OCRv6 tiny | `PP-OCRv6_tiny_det.mnn` | `PP-OCRv6_tiny_rec.mnn` | 轻量 v6 档位；不支持日文 |
| PP-OCRv6 small | `PP-OCRv6_small_det.mnn` | `PP-OCRv6_small_rec.mnn` | 平衡 v6 档位 |
| PP-OCRv6 medium | `PP-OCRv6_medium_det.mnn` | `PP-OCRv6_medium_rec.mnn` | 准确率优先 v6 档位 |

PP-OCRv6 `small` 和 `medium` 支持官方 50 种 v6 识别语言：简体中文、繁体中文、英文、日文，以及 46 种拉丁字母语言。PP-OCRv6 `tiny` 支持同一组语言但不支持日文。韩语、西里尔、阿拉伯、天城文、泰语、希腊语、泰米尔语、泰卢固语应继续使用 PP-OCRv5 脚本专用识别模型。

v6 字符集文件按档位区分：

```text
models/ppocr_keys_v6_tiny.txt
models/ppocr_keys_v6_small.txt
models/ppocr_keys_v6_medium.txt
```

## Paddle 模型转 MNN

转换脚本默认启用 MNN FP16 以减小模型大小。使用 `--install-dir ./models` 可把转换后的运行时文件复制到标准目录和文件名。

```bash
python script/convert_paddle_to_mnn.py \
  --ocr-dir /path/to/paddle/inference/models \
  --install-dir ./models
```

只有需要全精度 MNN 输出时才使用 `--no-fp16`。

## Rust 用法

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

也可以只创建检测或识别引擎：

```rust
let det = ocr_rs::OcrEngine::det_only("models/PP-OCRv6_small_det.mnn", None)?;
let rec = ocr_rs::OcrEngine::rec_only(
    "models/PP-OCRv6_small_rec.mnn",
    "models/ppocr_keys_v6_small.txt",
    None,
)?;
```

## 构建

```bash
cargo build --release
cargo test
```

## 性能检查

本地运行 Criterion 基准：

```bash
cargo bench --bench bench_metrics
```

运行 CI 风格的性能 smoke 测试：

```bash
OCR_RS_PERF_TESTS=1 cargo test --release --test performance_tests -- --nocapture --test-threads=1
```

GitHub Actions 会在 Ubuntu 上运行 release 模式 smoke 测试，并编译 Criterion 基准。smoke 测试会输出 `PERF_METRIC` 行，但不会用固定耗时阈值失败任务，因为托管 runner 的性能波动较大。

默认会自动使用可用的预构建 MNN 库。如需自定义构建 MNN：

```bash
cargo build --features build-mnn-from-source
```

GPU 后端通过 `OcrEngineConfig` 选择：

```rust
use ocr_rs::{Backend, OcrEngineConfig};

let config = OcrEngineConfig::new().with_backend(Backend::Metal);
```

## License

Apache-2.0
