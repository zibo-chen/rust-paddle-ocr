//! OCR Engine
//!
//! Provides complete OCR pipeline encapsulation, performs detection and recognition in one call

use image::{DynamicImage, GenericImageView};
use imageproc::point::Point;
use imageproc::rect::Rect;
use std::path::{Path, PathBuf};

use crate::det::{DetModel, DetOptions};
use crate::error::{OcrError, OcrResult};
use crate::mnn::{Backend, InferenceConfig, PrecisionMode};
use crate::ori::{OriModel, OriOptions};
use crate::postprocess::{compute_iou, TextBox};
use crate::rec::{RecModel, RecOptions, RecognitionResult};

const PARALLEL_RECOGNITION_MIN_REGIONS: usize = 5;
const ROTATED_RESULT_IOU_THRESHOLD: f32 = 0.5;

/// Strategy for recognizing text whose baseline is rotated by 90 or 270 degrees.
///
/// This is independent of the optional document-orientation model. The latter
/// rotates a whole page, while this mode handles mixed horizontal and vertical
/// text regions in the same image.
#[non_exhaustive]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum RotatedTextMode {
    /// Keep the original single-pass OCR pipeline.
    #[default]
    Disabled,
    /// Re-orient tall regions already found by the normal detection pass.
    DetectedOnly,
    /// Also detect on 90° and 270° copies to recover regions missed by the
    /// normal detection pass.
    Robust,
}

/// Per-call recognition options.
///
/// Engine configuration and the existing [`OcrEngine::recognize`] API remain
/// unchanged. Rotated-text support is opt-in so existing users pay no extra
/// detection or recognition cost.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RecognizeOptions {
    rotated_text_mode: RotatedTextMode,
    vertical_aspect_ratio: f32,
}

impl Default for RecognizeOptions {
    fn default() -> Self {
        Self {
            rotated_text_mode: RotatedTextMode::Disabled,
            vertical_aspect_ratio: 1.5,
        }
    }
}

impl RecognizeOptions {
    /// Create default per-call recognition options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Select how rotated text is handled.
    pub fn with_rotated_text_mode(mut self, mode: RotatedTextMode) -> Self {
        self.rotated_text_mode = mode;
        self
    }

    /// Set the minimum long-side/short-side ratio for a vertical-text candidate.
    ///
    /// Invalid values are ignored and the current value is retained.
    pub fn with_vertical_aspect_ratio(mut self, ratio: f32) -> Self {
        if ratio.is_finite() && ratio >= 1.0 {
            self.vertical_aspect_ratio = ratio;
        }
        self
    }

    /// Return the configured rotated-text strategy.
    pub fn rotated_text_mode(&self) -> RotatedTextMode {
        self.rotated_text_mode
    }

    /// Return the vertical candidate aspect-ratio threshold.
    pub fn vertical_aspect_ratio(&self) -> f32 {
        self.vertical_aspect_ratio
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RegionRecognitionStrategy {
    Batch,
    ExactWidthParallel,
}

fn select_region_recognition_strategy(
    enable_parallel: bool,
    region_count: usize,
) -> RegionRecognitionStrategy {
    if enable_parallel && region_count >= PARALLEL_RECOGNITION_MIN_REGIONS {
        RegionRecognitionStrategy::ExactWidthParallel
    } else {
        RegionRecognitionStrategy::Batch
    }
}

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
    /// Whether to enable exact-width parallel recognition for multi-line images
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
    /// When at least five text regions are detected, preprocessing is parallelized and each
    /// region keeps its exact tensor width to avoid padded batch inference.
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
        self.recognize_with_options(image, &RecognizeOptions::default())
    }

    /// Perform OCR recognition with per-call options.
    ///
    /// [`RotatedTextMode::Disabled`] is the default and runs the same one-pass
    /// pipeline as [`Self::recognize`]. [`RotatedTextMode::Robust`] performs two
    /// additional detection passes and only recognizes horizontal candidates
    /// from those rotated images, keeping the added recognition work small.
    pub fn recognize_with_options(
        &self,
        image: &DynamicImage,
        options: &RecognizeOptions,
    ) -> OcrResult<Vec<OcrResult_>> {
        // 0. Orientation correction for full image (optional)
        let corrected_image = if let Some(ori_model) = self.ori_model.as_ref() {
            self.correct_orientation_with_model(ori_model, image.clone())
        } else {
            image.clone()
        };

        match options.rotated_text_mode {
            RotatedTextMode::Disabled => self.recognize_single_pass(&corrected_image),
            RotatedTextMode::DetectedOnly => {
                self.recognize_detected_text(&corrected_image, options.vertical_aspect_ratio)
            }
            RotatedTextMode::Robust => {
                self.recognize_robust(&corrected_image, options.vertical_aspect_ratio)
            }
        }
    }

    fn recognize_single_pass(&self, image: &DynamicImage) -> OcrResult<Vec<OcrResult_>> {
        let boxes = self.det_model.detect_expanded(image)?;
        if boxes.is_empty() {
            return Ok(Vec::new());
        }

        let rec_results = self.recognize_regions(image, &boxes)?;
        Ok(self.combine_results(rec_results, boxes))
    }

    fn recognize_detected_text(
        &self,
        image: &DynamicImage,
        vertical_aspect_ratio: f32,
    ) -> OcrResult<Vec<OcrResult_>> {
        let boxes = self.det_model.detect_expanded(image)?;

        if boxes.is_empty() {
            return Ok(Vec::new());
        }

        let mut rec_results = self.recognize_regions(image, &boxes)?;

        for (index, text_box) in boxes.iter().enumerate() {
            if !is_vertical_box(text_box, vertical_aspect_ratio) {
                continue;
            }

            let variants = vertical_recognition_variants(text_box);
            let variant_results = self.rec_model.recognize_regions(image, &variants)?;
            if let Some(best) = variant_results
                .into_iter()
                .max_by(|a, b| a.confidence.total_cmp(&b.confidence))
            {
                if best.confidence > rec_results[index].confidence {
                    rec_results[index] = best;
                }
            }
        }

        Ok(self.combine_results(rec_results, boxes))
    }

    fn recognize_robust(
        &self,
        image: &DynamicImage,
        vertical_aspect_ratio: f32,
    ) -> OcrResult<Vec<OcrResult_>> {
        let mut results = self.recognize_detected_text(image, vertical_aspect_ratio)?;
        let (original_width, original_height) = image.dimensions();

        for turn in [QuarterTurn::Clockwise90, QuarterTurn::Clockwise270] {
            let rotated_image = turn.rotate(image);
            let rotated_boxes = self
                .det_model
                .detect_expanded(&rotated_image)?
                .into_iter()
                .filter(|text_box| is_horizontal_box(text_box, vertical_aspect_ratio))
                .collect::<Vec<_>>();

            if rotated_boxes.is_empty() {
                continue;
            }

            let rec_results = self.recognize_regions(&rotated_image, &rotated_boxes)?;
            let rotated_results = rec_results
                .into_iter()
                .zip(rotated_boxes)
                .filter(|(rec, _)| {
                    !rec.text.is_empty() && rec.confidence >= self.config.min_result_confidence
                })
                .map(|(rec, text_box)| {
                    OcrResult_::new(
                        rec.text,
                        rec.confidence,
                        turn.map_box_to_original(&text_box, original_width, original_height),
                    )
                });

            for result in rotated_results {
                merge_spatial_result(&mut results, result);
            }
        }

        sort_results_by_reading_order(&mut results);
        Ok(results)
    }

    fn recognize_regions(
        &self,
        image: &DynamicImage,
        boxes: &[TextBox],
    ) -> OcrResult<Vec<RecognitionResult>> {
        // Render regions directly into recognition tensors. Large pages use
        // exact-width tensors to avoid padding every line to the widest region.
        match select_region_recognition_strategy(self.config.enable_parallel, boxes.len()) {
            RegionRecognitionStrategy::Batch => self.rec_model.recognize_regions(image, boxes),
            RegionRecognitionStrategy::ExactWidthParallel => self
                .rec_model
                .recognize_regions_exact_parallel(image, boxes),
        }
    }

    fn combine_results(
        &self,
        rec_results: Vec<RecognitionResult>,
        boxes: Vec<TextBox>,
    ) -> Vec<OcrResult_> {
        rec_results
            .into_iter()
            .zip(boxes)
            .filter(|(rec, _)| {
                !rec.text.is_empty() && rec.confidence >= self.config.min_result_confidence
            })
            .map(|(rec, bbox)| OcrResult_::new(rec.text, rec.confidence, bbox))
            .collect()
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

#[derive(Debug, Clone, Copy)]
enum QuarterTurn {
    Clockwise90,
    Clockwise270,
}

impl QuarterTurn {
    fn rotate(self, image: &DynamicImage) -> DynamicImage {
        match self {
            Self::Clockwise90 => image.rotate90(),
            Self::Clockwise270 => image.rotate270(),
        }
    }

    fn map_box_to_original(
        self,
        text_box: &TextBox,
        original_width: u32,
        original_height: u32,
    ) -> TextBox {
        let points = text_box_points(text_box).map(|point| {
            let mapped = match self {
                Self::Clockwise90 => {
                    Point::new(point.y, original_height.saturating_sub(1) as f32 - point.x)
                }
                Self::Clockwise270 => {
                    Point::new(original_width.saturating_sub(1) as f32 - point.y, point.x)
                }
            };
            Point::new(
                mapped.x.clamp(0.0, original_width.saturating_sub(1) as f32),
                mapped
                    .y
                    .clamp(0.0, original_height.saturating_sub(1) as f32),
            )
        });
        let points = order_box_points(points);
        let rect = rect_for_points(&points, original_width, original_height)
            .unwrap_or_else(|| Rect::at(0, 0).of_size(1, 1));

        TextBox::with_points(rect, text_box.score, points)
    }
}

fn box_dimensions(text_box: &TextBox) -> (f32, f32) {
    if let Some(points) = text_box.points {
        let width = point_distance(points[0], points[1]).max(point_distance(points[3], points[2]));
        let height = point_distance(points[0], points[3]).max(point_distance(points[1], points[2]));
        (width.max(1.0), height.max(1.0))
    } else {
        (
            text_box.rect.width().max(1) as f32,
            text_box.rect.height().max(1) as f32,
        )
    }
}

fn is_vertical_box(text_box: &TextBox, aspect_ratio: f32) -> bool {
    let (width, height) = box_dimensions(text_box);
    height / width.max(1.0) >= aspect_ratio
}

fn is_horizontal_box(text_box: &TextBox, aspect_ratio: f32) -> bool {
    let (width, height) = box_dimensions(text_box);
    width / height.max(1.0) >= aspect_ratio
}

fn vertical_recognition_variants(text_box: &TextBox) -> [TextBox; 2] {
    let [top_left, top_right, bottom_right, bottom_left] = text_box_points(text_box);
    [
        TextBox::with_points(
            text_box.rect,
            text_box.score,
            [bottom_left, top_left, top_right, bottom_right],
        ),
        TextBox::with_points(
            text_box.rect,
            text_box.score,
            [top_right, bottom_right, bottom_left, top_left],
        ),
    ]
}

fn text_box_points(text_box: &TextBox) -> [Point<f32>; 4] {
    text_box.points.unwrap_or_else(|| {
        let left = text_box.rect.left() as f32;
        let top = text_box.rect.top() as f32;
        let right = left + text_box.rect.width().saturating_sub(1) as f32;
        let bottom = top + text_box.rect.height().saturating_sub(1) as f32;
        [
            Point::new(left, top),
            Point::new(right, top),
            Point::new(right, bottom),
            Point::new(left, bottom),
        ]
    })
}

fn order_box_points(points: [Point<f32>; 4]) -> [Point<f32>; 4] {
    let mut top_left = points[0];
    let mut top_right = points[0];
    let mut bottom_right = points[0];
    let mut bottom_left = points[0];

    for point in points {
        if point.x + point.y < top_left.x + top_left.y {
            top_left = point;
        }
        if point.x + point.y > bottom_right.x + bottom_right.y {
            bottom_right = point;
        }
        if point.x - point.y > top_right.x - top_right.y {
            top_right = point;
        }
        if point.x - point.y < bottom_left.x - bottom_left.y {
            bottom_left = point;
        }
    }

    [top_left, top_right, bottom_right, bottom_left]
}

fn rect_for_points(points: &[Point<f32>; 4], image_width: u32, image_height: u32) -> Option<Rect> {
    let min_x = points
        .iter()
        .map(|point| point.x)
        .fold(f32::INFINITY, f32::min)
        .floor()
        .max(0.0) as u32;
    let min_y = points
        .iter()
        .map(|point| point.y)
        .fold(f32::INFINITY, f32::min)
        .floor()
        .max(0.0) as u32;
    let max_x = points
        .iter()
        .map(|point| point.x)
        .fold(f32::NEG_INFINITY, f32::max)
        .ceil()
        .min(image_width as f32) as u32;
    let max_y = points
        .iter()
        .map(|point| point.y)
        .fold(f32::NEG_INFINITY, f32::max)
        .ceil()
        .min(image_height as f32) as u32;

    if max_x <= min_x || max_y <= min_y {
        return None;
    }

    Some(Rect::at(min_x as i32, min_y as i32).of_size(max_x - min_x, max_y - min_y))
}

fn point_distance(a: Point<f32>, b: Point<f32>) -> f32 {
    let dx = a.x - b.x;
    let dy = a.y - b.y;
    (dx * dx + dy * dy).sqrt()
}

fn merge_spatial_result(results: &mut Vec<OcrResult_>, candidate: OcrResult_) {
    if let Some(existing) = results.iter_mut().find(|result| {
        compute_iou(&result.bbox.rect, &candidate.bbox.rect) >= ROTATED_RESULT_IOU_THRESHOLD
    }) {
        if candidate.confidence > existing.confidence {
            *existing = candidate;
        }
    } else {
        results.push(candidate);
    }
}

fn sort_results_by_reading_order(results: &mut [OcrResult_]) {
    results.sort_by(|a, b| {
        a.bbox
            .rect
            .top()
            .cmp(&b.bbox.rect.top())
            .then_with(|| a.bbox.rect.left().cmp(&b.bbox.rect.left()))
    });
}

/// Builder for OCR engine
pub struct OcrEngineBuilder {
    det_model_path: Option<PathBuf>,
    rec_model_path: Option<PathBuf>,
    charset_path: Option<PathBuf>,
    ori_model_path: Option<PathBuf>,
    config: Option<OcrEngineConfig>,
}

impl Default for OcrEngineBuilder {
    fn default() -> Self {
        Self::new()
    }
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
    fn parallel_strategy_uses_exact_width_only_for_multi_line_pages() {
        assert_eq!(
            select_region_recognition_strategy(true, 5),
            RegionRecognitionStrategy::ExactWidthParallel
        );
        assert_eq!(
            select_region_recognition_strategy(true, 4),
            RegionRecognitionStrategy::Batch
        );
        assert_eq!(
            select_region_recognition_strategy(false, 12),
            RegionRecognitionStrategy::Batch
        );
    }

    #[test]
    fn test_ocr_result() {
        let bbox = TextBox::new(imageproc::rect::Rect::at(0, 0).of_size(100, 20), 0.9);
        let result = OcrResult_::new("Hello".to_string(), 0.95, bbox);

        assert_eq!(result.text, "Hello");
        assert_eq!(result.confidence, 0.95);
    }

    #[test]
    fn recognize_options_are_disabled_by_default() {
        let options = RecognizeOptions::default();

        assert_eq!(options.rotated_text_mode(), RotatedTextMode::Disabled);
        assert_eq!(options.vertical_aspect_ratio(), 1.5);
    }

    #[test]
    fn recognize_options_ignore_invalid_aspect_ratios() {
        let options = RecognizeOptions::new()
            .with_vertical_aspect_ratio(2.0)
            .with_vertical_aspect_ratio(f32::NAN)
            .with_vertical_aspect_ratio(0.5);

        assert_eq!(options.vertical_aspect_ratio(), 2.0);
    }

    #[test]
    fn quarter_turns_map_boxes_back_to_original_coordinates() {
        let clockwise_90_box = TextBox::with_points(
            Rect::at(30, 10).of_size(9, 29),
            0.9,
            [
                Point::new(30.0, 10.0),
                Point::new(39.0, 10.0),
                Point::new(39.0, 39.0),
                Point::new(30.0, 39.0),
            ],
        );
        let clockwise_270_box = TextBox::with_points(
            Rect::at(20, 60).of_size(9, 29),
            0.9,
            [
                Point::new(20.0, 60.0),
                Point::new(29.0, 60.0),
                Point::new(29.0, 89.0),
                Point::new(20.0, 89.0),
            ],
        );

        let from_90 = QuarterTurn::Clockwise90.map_box_to_original(&clockwise_90_box, 100, 60);
        let from_270 = QuarterTurn::Clockwise270.map_box_to_original(&clockwise_270_box, 100, 60);

        for mapped in [from_90, from_270] {
            assert_eq!(mapped.rect.left(), 10);
            assert_eq!(mapped.rect.top(), 20);
            assert_eq!(mapped.rect.width(), 29);
            assert_eq!(mapped.rect.height(), 9);
            assert!(is_horizontal_box(&mapped, 1.5));
        }
    }

    #[test]
    fn vertical_variants_turn_a_tall_box_into_horizontal_regions() {
        let text_box = TextBox::new(Rect::at(10, 20).of_size(20, 80), 0.9);

        let variants = vertical_recognition_variants(&text_box);

        assert!(variants
            .iter()
            .all(|variant| is_horizontal_box(variant, 1.5)));
    }

    #[test]
    fn merge_spatial_result_keeps_only_the_highest_confidence() {
        let bbox = TextBox::new(Rect::at(10, 20).of_size(20, 80), 0.9);
        let mut results = vec![OcrResult_::new("低".into(), 0.6, bbox.clone())];

        merge_spatial_result(&mut results, OcrResult_::new("高".into(), 0.9, bbox));

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].text, "高");
        assert_eq!(results[0].confidence, 0.9);
    }
}
