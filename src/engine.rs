//! OCR Engine
//!
//! Provides complete OCR pipeline encapsulation, performs detection and recognition in one call

use image::DynamicImage;
use std::path::{Path, PathBuf};

use crate::det::{DetModel, DetOptions};
use crate::error::{OcrError, OcrResult};
use crate::mnn::{Backend, InferenceConfig, PrecisionMode};
use crate::ori::{OriModel, OriOptions};
use crate::postprocess::TextBox;
use crate::rec::{RecModel, RecOptions, RecognitionResult};

/// OCR result
#[derive(Debug, Clone)]
pub struct OcrResult_ {
    /// Recognized text
    pub text: String,
    /// Confidence score
    pub confidence: f32,
    /// Bounding box
    pub bbox: TextBox,
}

impl OcrResult_ {
    /// Create a new OCR result
    pub fn new(text: String, confidence: f32, bbox: TextBox) -> Self {
        Self {
            text,
            confidence,
            bbox,
        }
    }
}

/// OCR engine configuration
#[derive(Debug, Clone)]
pub struct OcrEngineConfig {
    /// Inference backend
    pub backend: Backend,
    /// Thread count
    pub thread_count: i32,
    /// Precision mode
    pub precision_mode: PrecisionMode,
    /// Detection options
    pub det_options: DetOptions,
    /// Recognition options
    pub rec_options: RecOptions,
    /// Orientation options (used when orientation model is enabled)
    pub ori_options: OriOptions,
    /// Whether to enable parallel recognition (use rayon to process multiple text regions in parallel)
    pub enable_parallel: bool,
    /// Minimum confidence threshold at result level (recognition results below this value will be filtered)
    pub min_result_confidence: f32,
    /// Minimum confidence threshold for orientation correction
    pub ori_min_confidence: f32,
}

impl Default for OcrEngineConfig {
    fn default() -> Self {
        Self {
            backend: Backend::CPU,
            thread_count: 4,
            precision_mode: PrecisionMode::Normal,
            det_options: DetOptions::default(),
            rec_options: RecOptions::default(),
            ori_options: OriOptions::default(),
            enable_parallel: true,
            min_result_confidence: 0.5,
            ori_min_confidence: 0.3,
        }
    }
}

impl OcrEngineConfig {
    /// Create new configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Set inference backend
    pub fn with_backend(mut self, backend: Backend) -> Self {
        self.backend = backend;
        self
    }

    /// Set thread count
    pub fn with_threads(mut self, threads: i32) -> Self {
        self.thread_count = threads;
        self
    }

    /// Set precision mode
    pub fn with_precision(mut self, precision: PrecisionMode) -> Self {
        self.precision_mode = precision;
        self
    }

    /// Set detection options
    pub fn with_det_options(mut self, options: DetOptions) -> Self {
        self.det_options = options;
        self
    }

    /// Set recognition options
    pub fn with_rec_options(mut self, options: RecOptions) -> Self {
        self.rec_options = options;
        self
    }

    /// Set orientation options
    pub fn with_ori_options(mut self, options: OriOptions) -> Self {
        self.ori_options = options;
        self
    }

    /// Enable/disable parallel processing
    ///
    /// Note: When multiple text regions are detected, use rayon for parallel recognition.
    /// If MNN is already set to multi-threading, enabling this option may cause thread contention.
    pub fn with_parallel(mut self, enable: bool) -> Self {
        self.enable_parallel = enable;
        self
    }

    /// Set minimum confidence threshold at result level
    ///
    /// Recognition results below this threshold will be filtered out.
    /// Recommended values: 0.5 (lenient), 0.7 (balanced), 0.9 (strict)
    pub fn with_min_result_confidence(mut self, threshold: f32) -> Self {
        self.min_result_confidence = threshold;
        self
    }

    /// Set minimum confidence threshold for orientation correction
    pub fn with_ori_min_confidence(mut self, threshold: f32) -> Self {
        self.ori_min_confidence = threshold;
        self
    }

    /// Fast mode preset
    pub fn fast() -> Self {
        Self {
            precision_mode: PrecisionMode::Low,
            det_options: DetOptions::fast(),
            ..Default::default()
        }
    }

    /// GPU mode preset (Metal)
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    pub fn gpu() -> Self {
        Self {
            backend: Backend::Metal,
            ..Default::default()
        }
    }

    /// GPU mode preset (OpenCL)
    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
    pub fn gpu() -> Self {
        Self {
            backend: Backend::OpenCL,
            ..Default::default()
        }
    }

    fn to_inference_config(&self) -> InferenceConfig {
        InferenceConfig {
            thread_count: self.thread_count,
            precision_mode: self.precision_mode,
            backend: self.backend,
            ..Default::default()
        }
    }
}

/// OCR engine
///
/// Encapsulates complete OCR pipeline, including text detection and recognition
///
/// # Example
///
/// ```ignore
/// use ocr_rs::{OcrEngine, OcrEngineConfig};
///
/// // Create engine
/// let engine = OcrEngine::new(
///     "det_model.mnn",
///     "rec_model.mnn",
///     "ppocr_keys.txt",
///     None,
/// )?;
///
/// // Recognize image
/// let image = image::open("test.jpg")?;
/// let results = engine.recognize(&image)?;
///
/// for result in results {
///     println!("{}: {:.2}", result.text, result.confidence);
/// }
/// ```
pub struct OcrEngine {
    det_model: DetModel,
    rec_model: RecModel,
    ori_model: Option<OriModel>,
    config: OcrEngineConfig,
}

impl OcrEngine {
    fn build_with_paths(
        det_model_path: &Path,
        rec_model_path: &Path,
        charset_path: &Path,
        ori_model_path: Option<&Path>,
        config: Option<OcrEngineConfig>,
    ) -> OcrResult<Self> {
        let config = config.unwrap_or_default();
        let inference_config = config.to_inference_config();

        // Optimization: Directly move the configuration to avoid multiple clones
        let det_options = config.det_options.clone();
        let rec_options = config.rec_options.clone();
        let ori_options = config.ori_options.clone();

        let det_model = DetModel::from_file(det_model_path, Some(inference_config.clone()))?
            .with_options(det_options);

        let rec_model =
            RecModel::from_file(rec_model_path, charset_path, Some(inference_config.clone()))?
                .with_options(rec_options);

        let ori_model = match ori_model_path {
            Some(path) => {
                Some(OriModel::from_file(path, Some(inference_config))?.with_options(ori_options))
            }
            None => None,
        };

        Ok(Self {
            det_model,
            rec_model,
            ori_model,
            config,
        })
    }

    /// Create OCR engine from model files
    ///
    /// # Parameters
    /// - `det_model_path`: Detection model file path
    /// - `rec_model_path`: Recognition model file path
    /// - `charset_path`: Charset file path
    /// - `config`: Optional engine configuration
    pub fn new(
        det_model_path: impl AsRef<Path>,
        rec_model_path: impl AsRef<Path>,
        charset_path: impl AsRef<Path>,
        config: Option<OcrEngineConfig>,
    ) -> OcrResult<Self> {
        Self::build_with_paths(
            det_model_path.as_ref(),
            rec_model_path.as_ref(),
            charset_path.as_ref(),
            None,
            config,
        )
    }

    /// Create OCR engine from model files with orientation model
    pub fn new_with_ori(
        det_model_path: impl AsRef<Path>,
        rec_model_path: impl AsRef<Path>,
        charset_path: impl AsRef<Path>,
        ori_model_path: impl AsRef<Path>,
        config: Option<OcrEngineConfig>,
    ) -> OcrResult<Self> {
        Self::build_with_paths(
            det_model_path.as_ref(),
            rec_model_path.as_ref(),
            charset_path.as_ref(),
            Some(ori_model_path.as_ref()),
            config,
        )
    }

    /// Create OCR engine from model bytes
    pub fn from_bytes(
        det_model_bytes: &[u8],
        rec_model_bytes: &[u8],
        charset_bytes: &[u8],
        config: Option<OcrEngineConfig>,
    ) -> OcrResult<Self> {
        let config = config.unwrap_or_default();
        let inference_config = config.to_inference_config();

        // Optimization: Directly move the configuration to avoid multiple clones
        let det_options = config.det_options.clone();
        let rec_options = config.rec_options.clone();

        let det_model = DetModel::from_bytes(det_model_bytes, Some(inference_config.clone()))?
            .with_options(det_options);

        let rec_model = RecModel::from_bytes_with_charset(
            rec_model_bytes,
            charset_bytes,
            Some(inference_config.clone()),
        )?
        .with_options(rec_options);

        Ok(Self {
            det_model,
            rec_model,
            ori_model: None,
            config,
        })
    }

    /// Create OCR engine from model bytes with orientation model
    pub fn from_bytes_with_ori(
        det_model_bytes: &[u8],
        rec_model_bytes: &[u8],
        charset_bytes: &[u8],
        ori_model_bytes: &[u8],
        config: Option<OcrEngineConfig>,
    ) -> OcrResult<Self> {
        let config = config.unwrap_or_default();
        let inference_config = config.to_inference_config();

        let det_options = config.det_options.clone();
        let rec_options = config.rec_options.clone();
        let ori_options = config.ori_options.clone();

        let det_model = DetModel::from_bytes(det_model_bytes, Some(inference_config.clone()))?
            .with_options(det_options);

        let rec_model = RecModel::from_bytes_with_charset(
            rec_model_bytes,
            charset_bytes,
            Some(inference_config.clone()),
        )?
        .with_options(rec_options);

        let ori_model = OriModel::from_bytes(ori_model_bytes, Some(inference_config))?
            .with_options(ori_options);

        Ok(Self {
            det_model,
            rec_model,
            ori_model: Some(ori_model),
            config,
        })
    }

    /// Create detection-only engine
    pub fn det_only(
        det_model_path: impl AsRef<Path>,
        config: Option<OcrEngineConfig>,
    ) -> OcrResult<DetOnlyEngine> {
        let config = config.unwrap_or_default();
        let inference_config = config.to_inference_config();

        let det_model = DetModel::from_file(det_model_path, Some(inference_config))?
            .with_options(config.det_options);

        Ok(DetOnlyEngine { det_model })
    }

    /// Create recognition-only engine
    pub fn rec_only(
        rec_model_path: impl AsRef<Path>,
        charset_path: impl AsRef<Path>,
        config: Option<OcrEngineConfig>,
    ) -> OcrResult<RecOnlyEngine> {
        let config = config.unwrap_or_default();
        let inference_config = config.to_inference_config();

        let rec_model = RecModel::from_file(rec_model_path, charset_path, Some(inference_config))?
            .with_options(config.rec_options);

        Ok(RecOnlyEngine { rec_model })
    }

    /// Perform complete OCR recognition
    ///
    /// # Parameters
    /// - `image`: Input image
    ///
    /// # Returns
    /// List of OCR results, each result contains text, confidence and bounding box
    pub fn recognize(&self, image: &DynamicImage) -> OcrResult<Vec<OcrResult_>> {
        // 0. Orientation correction for full image (optional)
        let corrected_image = if let Some(ori_model) = self.ori_model.as_ref() {
            self.correct_orientation_with_model(ori_model, image.clone())
        } else {
            image.clone()
        };

        // 1. Detect text regions
        let boxes = self.det_model.detect_expanded(&corrected_image)?;

        if boxes.is_empty() {
            return Ok(Vec::new());
        }

        // 2. Recognize directly from source image regions. This avoids materializing
        // DynamicImage crops and a second resize before recognition preprocessing.
        let rec_results = self.rec_model.recognize_regions(&corrected_image, &boxes)?;

        // 3. Combine results and filter low confidence
        let results: Vec<OcrResult_> = rec_results
            .into_iter()
            .zip(boxes)
            .filter(|(rec, _)| {
                !rec.text.is_empty() && rec.confidence >= self.config.min_result_confidence
            })
            .map(|(rec, bbox)| OcrResult_::new(rec.text, rec.confidence, bbox))
            .collect();

        Ok(results)
    }

    /// Perform detection only
    pub fn detect(&self, image: &DynamicImage) -> OcrResult<Vec<TextBox>> {
        self.det_model.detect(image)
    }

    /// Perform recognition only (requires pre-cropped text line images)
    pub fn recognize_text(&self, image: &DynamicImage) -> OcrResult<RecognitionResult> {
        self.rec_model.recognize(image)
    }

    /// Batch recognize text line images
    pub fn recognize_batch(&self, images: &[DynamicImage]) -> OcrResult<Vec<RecognitionResult>> {
        self.rec_model.recognize_batch(images)
    }

    /// Get orientation model reference (if enabled)
    pub fn ori_model(&self) -> Option<&OriModel> {
        self.ori_model.as_ref()
    }

    /// Get detection model reference
    pub fn det_model(&self) -> &DetModel {
        &self.det_model
    }

    /// Get recognition model reference
    pub fn rec_model(&self) -> &RecModel {
        &self.rec_model
    }

    /// Get configuration
    pub fn config(&self) -> &OcrEngineConfig {
        &self.config
    }

    fn correct_orientation_with_model(
        &self,
        ori_model: &OriModel,
        image: DynamicImage,
    ) -> DynamicImage {
        let result = match ori_model.classify(&image) {
            Ok(result) => result,
            Err(_) => return image,
        };

        if !result.is_valid(self.config.ori_min_confidence) {
            return image;
        }

        if result.angle.rem_euclid(360) == 0 {
            return image;
        }

        rotate_by_angle(&image, result.angle)
    }
}

/// Builder for OCR engine
pub struct OcrEngineBuilder {
    det_model_path: Option<PathBuf>,
    rec_model_path: Option<PathBuf>,
    charset_path: Option<PathBuf>,
    ori_model_path: Option<PathBuf>,
    config: Option<OcrEngineConfig>,
}

impl OcrEngineBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            det_model_path: None,
            rec_model_path: None,
            charset_path: None,
            ori_model_path: None,
            config: None,
        }
    }

    /// Set detection model path
    pub fn with_det_model_path(mut self, path: impl AsRef<Path>) -> Self {
        self.det_model_path = Some(path.as_ref().to_path_buf());
        self
    }

    /// Set recognition model path
    pub fn with_rec_model_path(mut self, path: impl AsRef<Path>) -> Self {
        self.rec_model_path = Some(path.as_ref().to_path_buf());
        self
    }

    /// Set charset path
    pub fn with_charset_path(mut self, path: impl AsRef<Path>) -> Self {
        self.charset_path = Some(path.as_ref().to_path_buf());
        self
    }

    /// Set orientation model path
    pub fn with_ori_model_path(mut self, path: impl AsRef<Path>) -> Self {
        self.ori_model_path = Some(path.as_ref().to_path_buf());
        self
    }

    /// Set engine configuration
    pub fn with_config(mut self, config: OcrEngineConfig) -> Self {
        self.config = Some(config);
        self
    }

    /// Build OCR engine
    pub fn build(self) -> OcrResult<OcrEngine> {
        let det_model_path = self
            .det_model_path
            .ok_or_else(|| OcrError::InvalidParameter("Missing det_model_path".to_string()))?;
        let rec_model_path = self
            .rec_model_path
            .ok_or_else(|| OcrError::InvalidParameter("Missing rec_model_path".to_string()))?;
        let charset_path = self
            .charset_path
            .ok_or_else(|| OcrError::InvalidParameter("Missing charset_path".to_string()))?;

        OcrEngine::build_with_paths(
            det_model_path.as_path(),
            rec_model_path.as_path(),
            charset_path.as_path(),
            self.ori_model_path.as_deref(),
            self.config,
        )
    }
}

/// Detection-only engine
pub struct DetOnlyEngine {
    det_model: DetModel,
}

impl DetOnlyEngine {
    /// Detect text regions in image
    pub fn detect(&self, image: &DynamicImage) -> OcrResult<Vec<TextBox>> {
        self.det_model.detect(image)
    }

    /// Detect and return cropped images
    pub fn detect_and_crop(&self, image: &DynamicImage) -> OcrResult<Vec<(DynamicImage, TextBox)>> {
        self.det_model.detect_and_crop(image)
    }

    /// Get detection model reference
    pub fn model(&self) -> &DetModel {
        &self.det_model
    }
}

/// Recognition-only engine
pub struct RecOnlyEngine {
    rec_model: RecModel,
}

impl RecOnlyEngine {
    /// Recognize a single image
    pub fn recognize(&self, image: &DynamicImage) -> OcrResult<RecognitionResult> {
        self.rec_model.recognize(image)
    }

    /// Return text only
    pub fn recognize_text(&self, image: &DynamicImage) -> OcrResult<String> {
        self.rec_model.recognize_text(image)
    }

    /// Batch recognition
    pub fn recognize_batch(&self, images: &[DynamicImage]) -> OcrResult<Vec<RecognitionResult>> {
        self.rec_model.recognize_batch(images)
    }

    /// Get recognition model reference
    pub fn model(&self) -> &RecModel {
        &self.rec_model
    }
}

/// Convenience function: recognize from file
///
/// # Example
///
/// ```ignore
/// let results = ocr_rs::ocr_file(
///     "test.jpg",
///     "det_model.mnn",
///     "rec_model.mnn",
///     "ppocr_keys.txt",
/// )?;
/// ```
pub fn ocr_file(
    image_path: impl AsRef<Path>,
    det_model_path: impl AsRef<Path>,
    rec_model_path: impl AsRef<Path>,
    charset_path: impl AsRef<Path>,
) -> OcrResult<Vec<OcrResult_>> {
    let image = image::open(image_path)?;
    let engine = OcrEngine::new(det_model_path, rec_model_path, charset_path, None)?;
    engine.recognize(&image)
}

/// Convenience function: recognize from file with orientation model
pub fn ocr_file_with_ori(
    image_path: impl AsRef<Path>,
    det_model_path: impl AsRef<Path>,
    rec_model_path: impl AsRef<Path>,
    charset_path: impl AsRef<Path>,
    ori_model_path: impl AsRef<Path>,
) -> OcrResult<Vec<OcrResult_>> {
    let image = image::open(image_path)?;
    let engine = OcrEngine::new_with_ori(
        det_model_path,
        rec_model_path,
        charset_path,
        ori_model_path,
        None,
    )?;
    engine.recognize(&image)
}

fn rotate_by_angle(image: &DynamicImage, angle: i32) -> DynamicImage {
    // The model reports rotation from horizontal; rotate back to correct.
    match angle.rem_euclid(360) {
        90 => DynamicImage::ImageRgb8(image::imageops::rotate270(&image.to_rgb8())),
        180 => DynamicImage::ImageRgb8(image::imageops::rotate180(&image.to_rgb8())),
        270 => DynamicImage::ImageRgb8(image::imageops::rotate90(&image.to_rgb8())),
        _ => image.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ocr_result() {
        let bbox = TextBox::new(imageproc::rect::Rect::at(0, 0).of_size(100, 20), 0.9);
        let result = OcrResult_::new("Hello".to_string(), 0.95, bbox);

        assert_eq!(result.text, "Hello");
        assert_eq!(result.confidence, 0.95);
    }
}
