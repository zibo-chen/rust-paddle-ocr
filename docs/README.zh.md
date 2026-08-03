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

### 横向与竖向文字混排

原有 `recognize` 方法保持单次检测的既有行为。图片中同时包含横向文字和旋转
90°/270° 的文字时，可以只为本次调用显式启用 Robust 模式：

```rust
use ocr_rs::{RecognizeOptions, RotatedTextMode};

let options = RecognizeOptions::new()
    .with_rotated_text_mode(RotatedTextMode::Robust);
let results = engine.recognize_with_options(&image, &options)?;
```

Robust 模式会额外检测旋转 90° 和 270° 的图像，将找回的文本框映射到输入图坐标，
并且只识别竖向候选框。原有 `recognize` 调用不会增加任何推理开销。如果普通检测
已经能找到竖向框，可以使用开销更低的 `RotatedTextMode::DetectedOnly`。

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

GitHub Actions 会串行运行 release 模式测试，并将 `PERF_METRIC` 日志保存为 artifact。回归门禁会在同一 runner 上比较直通 exact-width 流水线与旧 crop 流水线；中位数比值超过 `OCR_RS_PERF_REGRESSION_LIMIT`（默认 `1.15`）时失败，因此不依赖不稳定的绝对耗时。

兼容时会自动使用 CPU 预构建包或 Apple Metal 预构建包。启用预构建包未包含的 GPU feature 时，会自动从源码构建 MNN：

```bash
cargo build --features build-mnn-from-source
cargo build --release --features cuda
cargo build --release --features vulkan
```

构建前需要安装对应后端的 SDK 和开发库。GPU 后端通过 `OcrEngineConfig` 选择：

```rust
use ocr_rs::{Backend, OcrEngineConfig};

let config = OcrEngineConfig::new().with_backend(Backend::Metal);
assert!(Backend::Metal.is_available());
```

如果链接的 MNN 没有注册所请求的后端，创建引擎会返回 `MnnError::BackendUnavailable`，不再静默回退到 CPU。

`x86_64-pc-windows-gnu` 会从源码构建 MNN，需要 MinGW C/C++ 工具链。默认情况下，应用需要携带匹配的 MinGW 运行时 DLL。启用 `static-cpp-runtime` 可静态链接 libstdc++、libgcc 和 winpthreads，使生成的二进制文件不再依赖 MinGW 运行时 DLL：

```bash
cargo build --release --target x86_64-pc-windows-gnu --features static-cpp-runtime
```

NVIDIA 的 Windows CUDA 工具链要求 MSVC；源码构建 CUDA 请使用 `x86_64-pc-windows-msvc`，或通过 `mnn-dynamic`/`mnn-static` 提供兼容的 MNN 库。

该 feature 只控制 `ocr-rs` 自身链接的运行时；通过 `mnn-dynamic` 提供的第三方 DLL 仍可能带有自己的 MinGW 运行时依赖。

## License

Apache-2.0
