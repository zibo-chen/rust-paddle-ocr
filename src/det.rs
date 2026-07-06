//! Text Detection Model
//!
//! Provides text region detection functionality based on PaddleOCR detection models

use image::{DynamicImage, GenericImageView, Rgb, RgbImage};
use imageproc::geometric_transformations::{warp_into, Interpolation, Projection};
use imageproc::point::Point;
use ndarray::ArrayD;
use std::path::Path;

use crate::error::{OcrError, OcrResult};
use crate::mnn::{InferenceConfig, InferenceEngine};
use crate::postprocess::{extract_boxes_with_unclip, TextBox};
use crate::preprocess::{preprocess_for_det, resize_to_max_side, NormalizeParams};

/// Detection precision mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DetPrecisionMode {
    /// Fast mode - single detection
    #[default]
    Fast,
}

/// Detection options
#[derive(Debug, Clone)]
pub struct DetOptions {
    /// Maximum image side length limit (will be scaled if exceeded)
    pub max_side_len: u32,
    /// Bounding box binarization threshold (0.0 - 1.0)
    pub box_threshold: f32,
    /// Text box expansion ratio
    pub unclip_ratio: f32,
    /// Pixel-level segmentation threshold
    pub score_threshold: f32,
    /// Minimum bounding box area
    pub min_area: u32,
    /// Bounding box border expansion
    pub box_border: u32,
    /// Whether to merge adjacent text boxes
    pub merge_boxes: bool,
    /// Merge distance threshold
    pub merge_threshold: i32,
    /// Precision mode
    pub precision_mode: DetPrecisionMode,
    /// Scale ratios for multi-scale detection (high precision mode only)
    pub multi_scales: Vec<f32>,
    /// Block size for block detection (high precision mode only)
    pub block_size: u32,
    /// Overlap area for block detection
    pub block_overlap: u32,
    /// NMS IoU threshold
    pub nms_threshold: f32,
}

impl Default for DetOptions {
    fn default() -> Self {
        Self {
            max_side_len: 960,
            box_threshold: 0.5,
            unclip_ratio: 1.5,
            score_threshold: 0.3,
            min_area: 16,
            box_border: 5,
            merge_boxes: false,
            merge_threshold: 10,
            precision_mode: DetPrecisionMode::Fast,
            multi_scales: vec![0.5, 1.0, 1.5],
            block_size: 640,
            block_overlap: 100,
            nms_threshold: 0.3,
        }
    }
}

impl DetOptions {
    /// Create new detection options
    pub fn new() -> Self {
        Self::default()
    }

    /// Set maximum side length
    pub fn with_max_side_len(mut self, len: u32) -> Self {
        self.max_side_len = len;
        self
    }

    /// Set bounding box threshold
    pub fn with_box_threshold(mut self, threshold: f32) -> Self {
        self.box_threshold = threshold;
        self
    }

    /// Set segmentation threshold
    pub fn with_score_threshold(mut self, threshold: f32) -> Self {
        self.score_threshold = threshold;
        self
    }

    /// Set minimum area
    pub fn with_min_area(mut self, area: u32) -> Self {
        self.min_area = area;
        self
    }

    /// Set box border expansion
    pub fn with_box_border(mut self, border: u32) -> Self {
        self.box_border = border;
        self
    }

    /// Enable box merging
    pub fn with_merge_boxes(mut self, merge: bool) -> Self {
        self.merge_boxes = merge;
        self
    }

    /// Set merge threshold
    pub fn with_merge_threshold(mut self, threshold: i32) -> Self {
        self.merge_threshold = threshold;
        self
    }

    /// Set precision mode
    pub fn with_precision_mode(mut self, mode: DetPrecisionMode) -> Self {
        self.precision_mode = mode;
        self
    }

    /// Set multi-scale ratios
    pub fn with_multi_scales(mut self, scales: Vec<f32>) -> Self {
        self.multi_scales = scales;
        self
    }

    /// Set block size
    pub fn with_block_size(mut self, size: u32) -> Self {
        self.block_size = size;
        self
    }

    /// Fast mode preset
    pub fn fast() -> Self {
        Self {
            max_side_len: 960,
            precision_mode: DetPrecisionMode::Fast,
            ..Default::default()
        }
    }
}

/// Text detection model
pub struct DetModel {
    engine: InferenceEngine,
    options: DetOptions,
    normalize_params: NormalizeParams,
}

impl DetModel {
    /// Create detector from model file
    ///
    /// # Parameters
    /// - `model_path`: Model file path (.mnn format)
    /// - `config`: Optional inference config
    pub fn from_file(
        model_path: impl AsRef<Path>,
        config: Option<InferenceConfig>,
    ) -> OcrResult<Self> {
        let engine = InferenceEngine::from_file(model_path, config)?;
        Ok(Self {
            engine,
            options: DetOptions::default(),
            normalize_params: NormalizeParams::paddle_det(),
        })
    }

    /// Create detector from model bytes
    pub fn from_bytes(model_bytes: &[u8], config: Option<InferenceConfig>) -> OcrResult<Self> {
        let engine = InferenceEngine::from_buffer(model_bytes, config)?;
        Ok(Self {
            engine,
            options: DetOptions::default(),
            normalize_params: NormalizeParams::paddle_det(),
        })
    }

    /// Set detection options
    pub fn with_options(mut self, options: DetOptions) -> Self {
        self.options = options;
        self
    }

    /// Get current detection options
    pub fn options(&self) -> &DetOptions {
        &self.options
    }

    /// Modify detection options
    pub fn options_mut(&mut self) -> &mut DetOptions {
        &mut self.options
    }

    /// Detect text regions in image
    ///
    /// # Parameters
    /// - `image`: Input image
    ///
    /// # Returns
    /// List of detected text bounding boxes
    pub fn detect(&self, image: &DynamicImage) -> OcrResult<Vec<TextBox>> {
        self.detect_fast(image)
    }

    /// Detect and return cropped text images
    ///
    /// # Parameters
    /// - `image`: Input image
    ///
    /// # Returns
    /// List of (text image, corresponding bounding box)
    pub fn detect_and_crop(&self, image: &DynamicImage) -> OcrResult<Vec<(DynamicImage, TextBox)>> {
        let boxes = self.detect_expanded(image)?;
        let rotated_source = if boxes.iter().any(|text_box| text_box.points.is_some()) {
            Some(image.to_rgb8())
        } else {
            None
        };

        let mut results = Vec::with_capacity(boxes.len());

        for text_box in boxes {
            // Crop image
            let cropped = crop_text_region(image, rotated_source.as_ref(), &text_box);

            results.push((cropped, text_box));
        }

        Ok(results)
    }

    pub(crate) fn detect_expanded(&self, image: &DynamicImage) -> OcrResult<Vec<TextBox>> {
        let boxes = self.detect(image)?;
        let (width, height) = image.dimensions();

        Ok(boxes
            .into_iter()
            .map(|text_box| text_box.expand(self.options.box_border, width, height))
            .collect())
    }

    /// Fast detection (single inference)
    fn detect_fast(&self, image: &DynamicImage) -> OcrResult<Vec<TextBox>> {
        let (original_width, original_height) = image.dimensions();

        // Scale image
        let scaled = self.scale_image(image)?;
        let (scaled_width, scaled_height) = scaled.dimensions();

        // Preprocess
        let input = preprocess_for_det(&scaled, &self.normalize_params)?;

        // Inference (using dynamic shape)
        let output = self.engine.run_dynamic(input.view().into_dyn())?;

        // Post-processing - output shape matches input (including padding)
        let output_shape = output.shape();
        let out_w = output_shape[3] as u32;
        let out_h = output_shape[2] as u32;

        let boxes = self.postprocess_output(
            &output,
            out_w,
            out_h,
            scaled_width,
            scaled_height,
            original_width,
            original_height,
        )?;

        Ok(boxes)
    }

    /// Balanced mode detection (multi-scale)
    /// Scale image to maximum side length limit
    fn scale_image(&self, image: &DynamicImage) -> OcrResult<DynamicImage> {
        resize_to_max_side(image, self.options.max_side_len)
    }

    /// Post-process inference output
    fn postprocess_output(
        &self,
        output: &ArrayD<f32>,
        out_w: u32,
        out_h: u32,
        scaled_width: u32,
        scaled_height: u32,
        original_width: u32,
        original_height: u32,
    ) -> OcrResult<Vec<TextBox>> {
        // Retrieve output data
        let output_shape = output.shape();
        if output_shape.len() < 3 {
            return Err(OcrError::PostprocessError(
                "Detection model output shape invalid".to_string(),
            ));
        }

        // Binarization
        let binary_mask: Vec<u8> = output
            .iter()
            .map(|&v| {
                if v > self.options.score_threshold {
                    255u8
                } else {
                    0u8
                }
            })
            .collect();

        // Extract bounding boxes (with unclip expansion)
        // DB algorithm needs to expand detected contours because model output segmentation mask is usually smaller than actual text region
        let boxes = extract_boxes_with_unclip(
            &binary_mask,
            out_w,
            out_h,
            scaled_width,
            scaled_height,
            original_width,
            original_height,
            self.options.min_area,
            self.options.unclip_ratio,
        );

        Ok(boxes)
    }
}

fn crop_text_region(
    image: &DynamicImage,
    rotated_source: Option<&RgbImage>,
    text_box: &TextBox,
) -> DynamicImage {
    if let (Some(points), Some(source)) = (text_box.points, rotated_source) {
        if let Some(cropped) = crop_rotated_region(source, points) {
            return cropped;
        }
    }

    crop_axis_aligned_region(image, text_box)
}

fn crop_axis_aligned_region(image: &DynamicImage, text_box: &TextBox) -> DynamicImage {
    let (image_width, image_height) = image.dimensions();
    let x = text_box.rect.left().max(0) as u32;
    let y = text_box.rect.top().max(0) as u32;
    let width = text_box
        .rect
        .width()
        .min(image_width.saturating_sub(x))
        .max(1);
    let height = text_box
        .rect
        .height()
        .min(image_height.saturating_sub(y))
        .max(1);

    image.crop_imm(x, y, width, height)
}

fn crop_rotated_region(source: &RgbImage, points: [Point<f32>; 4]) -> Option<DynamicImage> {
    let crop_width = distance(points[0], points[1])
        .max(distance(points[3], points[2]))
        .round()
        .max(1.0) as u32;
    let crop_height = distance(points[0], points[3])
        .max(distance(points[1], points[2]))
        .round()
        .max(1.0) as u32;

    if crop_width <= 1 || crop_height <= 1 {
        return None;
    }

    let source_points = points.map(|point| (point.x, point.y));
    let target_points = [
        (0.0, 0.0),
        (crop_width.saturating_sub(1) as f32, 0.0),
        (
            crop_width.saturating_sub(1) as f32,
            crop_height.saturating_sub(1) as f32,
        ),
        (0.0, crop_height.saturating_sub(1) as f32),
    ];

    let projection = Projection::from_control_points(source_points, target_points)?;
    let mut output = RgbImage::new(crop_width, crop_height);
    warp_into(
        source,
        &projection,
        Interpolation::Bilinear,
        Rgb([255, 255, 255]),
        &mut output,
    );

    Some(DynamicImage::ImageRgb8(output))
}

fn distance(a: Point<f32>, b: Point<f32>) -> f32 {
    let dx = a.x - b.x;
    let dy = a.y - b.y;
    (dx * dx + dy * dy).sqrt()
}

/// Low-level detection API
impl DetModel {
    /// Raw inference interface
    ///
    /// Execute model inference directly without preprocessing and postprocessing
    ///
    /// # Parameters
    /// - `input`: Preprocessed input tensor [1, 3, H, W]
    ///
    /// # Returns
    /// Model raw output
    pub fn run_raw(&self, input: ndarray::ArrayViewD<f32>) -> OcrResult<ArrayD<f32>> {
        Ok(self.engine.run_dynamic(input)?)
    }

    /// Get model input shape
    pub fn input_shape(&self) -> &[usize] {
        self.engine.input_shape()
    }

    /// Get model output shape
    pub fn output_shape(&self) -> &[usize] {
        self.engine.output_shape()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_det_options_default() {
        let opts = DetOptions::default();
        assert_eq!(opts.max_side_len, 960);
        assert_eq!(opts.box_threshold, 0.5);
        assert_eq!(opts.unclip_ratio, 1.5);
        assert_eq!(opts.score_threshold, 0.3);
        assert_eq!(opts.min_area, 16);
        assert_eq!(opts.box_border, 5);
        assert!(!opts.merge_boxes);
        assert_eq!(opts.merge_threshold, 10);
        assert_eq!(opts.precision_mode, DetPrecisionMode::Fast);
        assert_eq!(opts.nms_threshold, 0.3);
    }

    #[test]
    fn test_det_options_fast() {
        let opts = DetOptions::fast();
        assert_eq!(opts.max_side_len, 960);
        assert_eq!(opts.precision_mode, DetPrecisionMode::Fast);
    }

    #[test]
    fn test_det_options_builder() {
        let opts = DetOptions::new()
            .with_max_side_len(1280)
            .with_box_threshold(0.6)
            .with_score_threshold(0.4)
            .with_min_area(32)
            .with_box_border(10)
            .with_merge_boxes(true)
            .with_merge_threshold(20)
            .with_precision_mode(DetPrecisionMode::Fast)
            .with_multi_scales(vec![0.5, 1.0, 1.5])
            .with_block_size(800);

        assert_eq!(opts.max_side_len, 1280);
        assert_eq!(opts.box_threshold, 0.6);
        assert_eq!(opts.score_threshold, 0.4);
        assert_eq!(opts.min_area, 32);
        assert_eq!(opts.box_border, 10);
        assert!(opts.merge_boxes);
        assert_eq!(opts.merge_threshold, 20);
        assert_eq!(opts.precision_mode, DetPrecisionMode::Fast);
        assert_eq!(opts.multi_scales, vec![0.5, 1.0, 1.5]);
        assert_eq!(opts.block_size, 800);
    }

    #[test]
    fn test_det_precision_mode_default() {
        let mode = DetPrecisionMode::default();
        assert_eq!(mode, DetPrecisionMode::Fast);
    }

    #[test]
    fn test_det_precision_mode_equality() {
        assert_eq!(DetPrecisionMode::Fast, DetPrecisionMode::Fast);
    }

    #[test]
    fn test_det_options_chaining() {
        // Test that chaining calls do not lose previous settings
        let opts = DetOptions::new()
            .with_max_side_len(1000)
            .with_box_threshold(0.7);

        assert_eq!(opts.max_side_len, 1000);
        assert_eq!(opts.box_threshold, 0.7);
        // Other values should be default values
        assert_eq!(opts.score_threshold, 0.3);
    }

    #[test]
    fn test_det_options_presets_are_valid() {
        // Ensure preset parameter values are within valid ranges
        let fast = DetOptions::fast();
        assert!(fast.box_threshold >= 0.0 && fast.box_threshold <= 1.0);
        assert!(fast.score_threshold >= 0.0 && fast.score_threshold <= 1.0);
        assert!(fast.nms_threshold >= 0.0 && fast.nms_threshold <= 1.0);
    }
}
