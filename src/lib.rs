//! # Rust PaddleOCR
//!
//! A high-performance OCR library based on PaddleOCR models, using the MNN inference framework.
//!
//! ## Version 2.0 New Features
//!
//! - **New API Design**: Complete layered API from low-level models to high-level pipeline
//! - **Flexible Model Loading**: Support loading models from file paths or memory bytes
//! - **Configurable Detection Parameters**: Support custom detection thresholds, resolution, etc.
//! - **GPU Acceleration**: Support multiple GPU backends including Metal, OpenCL, Vulkan
//! - **Batch Processing**: Support batch text recognition for improved throughput
//!
//! ## Quick Start
//!
//! ### Simple Usage - Using High-Level API (Recommended)
//!
//! ```ignore
//! use ocr_rs::{OcrEngine, OcrEngineConfig};
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Create OCR engine
//!     let engine = OcrEngine::new(
//!         "models/det_model.mnn",
//!         "models/rec_model.mnn",
//!         "models/ppocr_keys.txt",
//!         None, // Use default config
//!     )?;
//!
//!     // Open and recognize image
//!     let image = image::open("test.jpg")?;
//!     let results = engine.recognize(&image)?;
//!
//!     for result in results {
//!         println!("Text: {}, Confidence: {:.2}%", result.text, result.confidence * 100.0);
//!         println!("Position: ({}, {})", result.bbox.rect.left(), result.bbox.rect.top());
//!     }
//!
//!     Ok(())
//! }
//! ```
//!
//! ### Advanced Usage - Using Low-Level API
//!
//! ```ignore
//! use ocr_rs::{DetModel, RecModel, DetOptions, DetPrecisionMode};
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Create detection model
//!     let det = DetModel::from_file("models/det_model.mnn", None)?
//!         .with_options(DetOptions::fast());
//!
//!     // Create recognition model
//!     let rec = RecModel::from_file("models/rec_model.mnn", "models/ppocr_keys.txt", None)?;
//!
//!     // Load image
//!     let image = image::open("test.jpg")?;
//!
//!     // Detect and crop text regions
//!     let detections = det.detect_and_crop(&image)?;
//!
//!     // Recognize each text region
//!     for (cropped_img, bbox) in detections {
//!         let result = rec.recognize(&cropped_img)?;
//!         println!("Position: ({}, {}), Text: {}",
//!             bbox.rect.left(), bbox.rect.top(), result.text);
//!     }
//!
//!     Ok(())
//! }
//! ```
//!
//! ### GPU Acceleration
//!
//! ```ignore
//! use ocr_rs::{OcrEngine, OcrEngineConfig, Backend};
//!
//! let config = OcrEngineConfig::new()
//!     .with_backend(Backend::Metal);  // macOS/iOS
//!     // .with_backend(Backend::OpenCL);  // Cross-platform
//!
//! let engine = OcrEngine::new(det_path, rec_path, charset_path, Some(config))?;
//! ```
//!
//! ## Module Structure
//!
//! - [`mnn`]: MNN inference engine wrapper, provides low-level inference capabilities
//! - [`det`]: Text detection module ([`DetModel`]), detects text regions in images
//! - [`rec`]: Text recognition module ([`RecModel`]), recognizes text content
//! - [`engine`]: High-level OCR pipeline ([`OcrEngine`]), all-in-one OCR solution
//! - [`preprocess`]: Image preprocessing utilities, including normalization, scaling, etc.
//! - [`postprocess`]: Post-processing utilities, including NMS, box merging, sorting, etc.
//! - [`error`]: Error types [`OcrError`]
//!
//! ## API Hierarchy
//!
//! ```text
//! ┌─────────────────────────────────────────┐
//! │        OcrEngine (High-Level API)       │
//! │   Complete detection and recognition    │
//! ├─────────────────────────────────────────┤
//! │     DetModel      │      RecModel       │
//! │  Detection Model  │  Recognition Model  │
//! ├─────────────────────────────────────────┤
//! │          InferenceEngine (MNN)          │
//! │         Low-level inference engine      │
//! └─────────────────────────────────────────┘
//! ```
//!
//! ## Supported Models
//!
//! - **PP-OCRv4**: Stable version, good compatibility
//! - **PP-OCRv5**: Recommended version, supports multiple languages, higher accuracy
//! - **PP-OCRv5 FP16**: Efficient version, faster inference, lower memory usage

// Core modules
pub mod det;
pub mod engine;
pub mod error;
pub mod mnn;
mod ori;
pub mod postprocess;
pub mod preprocess;
pub mod rec;

// Re-export commonly used types
pub use det::{DetModel, DetOptions, DetPrecisionMode};
pub use engine::{
    ocr_file, DetOnlyEngine, OcrEngine, OcrEngineBuilder, OcrEngineConfig, OcrResult_,
    RecOnlyEngine, RecognizeOptions, RotatedTextMode,
};
pub use error::{OcrError, OcrResult};
pub use mnn::{Backend, InferenceConfig, InferenceEngine, PrecisionMode};
pub use ori::{OriModel, OriOptions, OriPreprocessMode, OrientationResult};
pub use postprocess::TextBox;
pub use rec::{RecModel, RecOptions, RecognitionResult};

/// Get library version
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Get MNN version
pub fn mnn_version() -> String {
    mnn::get_version()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        let v = version();
        assert!(!v.is_empty());
    }
}
