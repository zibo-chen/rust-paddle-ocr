//! Text Recognition Model
//!
//! Provides text recognition functionality based on PaddleOCR recognition models

use image::{DynamicImage, RgbImage};
use imageproc::geometric_transformations::Projection;
use imageproc::point::Point;
use ndarray::{Array4, ArrayD, ArrayViewD, Axis};
use std::{borrow::Cow, path::Path};

use crate::error::{OcrError, OcrResult};
use crate::mnn::{InferenceConfig, InferenceEngine};
use crate::postprocess::TextBox;
use crate::preprocess::{preprocess_for_rec, NormalizeParams};

/// Recognition result
#[derive(Debug, Clone)]
pub struct RecognitionResult {
    /// Recognized text
    pub text: String,
    /// Confidence score (0.0 - 1.0)
    pub confidence: f32,
    /// Confidence score for each character
    pub char_scores: Vec<(char, f32)>,
}

impl RecognitionResult {
    /// Create a new recognition result
    pub fn new(text: String, confidence: f32, char_scores: Vec<(char, f32)>) -> Self {
        Self {
            text,
            confidence,
            char_scores,
        }
    }

    /// Check if the result is valid (confidence above threshold)
    pub fn is_valid(&self, threshold: f32) -> bool {
        self.confidence >= threshold
    }
}

/// Recognition options
#[derive(Debug, Clone)]
pub struct RecOptions {
    /// Target height (recognition model input height)
    pub target_height: u32,
    /// Minimum confidence threshold (characters below this value will be filtered)
    pub min_score: f32,
    /// Minimum confidence threshold for punctuation
    pub punct_min_score: f32,
    /// Batch size
    pub batch_size: usize,
    /// Whether to enable batch processing
    pub enable_batch: bool,
}

impl Default for RecOptions {
    fn default() -> Self {
        Self {
            target_height: 48,
            min_score: 0.3, // Lower threshold, model output is raw logit
            punct_min_score: 0.1,
            batch_size: 8,
            enable_batch: true,
        }
    }
}

impl RecOptions {
    /// Create new recognition options
    pub fn new() -> Self {
        Self::default()
    }

    /// Set target height
    pub fn with_target_height(mut self, height: u32) -> Self {
        self.target_height = height;
        self
    }

    /// Set minimum confidence
    pub fn with_min_score(mut self, score: f32) -> Self {
        self.min_score = score;
        self
    }

    /// Set punctuation minimum confidence
    pub fn with_punct_min_score(mut self, score: f32) -> Self {
        self.punct_min_score = score;
        self
    }

    /// Set batch size
    pub fn with_batch_size(mut self, size: usize) -> Self {
        self.batch_size = size;
        self
    }

    /// Enable/disable batch processing
    pub fn with_batch(mut self, enable: bool) -> Self {
        self.enable_batch = enable;
        self
    }
}

/// Text recognition model
pub struct RecModel {
    engine: InferenceEngine,
    /// Character set (index to character mapping)
    charset: Vec<char>,
    options: RecOptions,
    normalize_params: NormalizeParams,
}

/// Common punctuation marks
const PUNCTUATIONS: [char; 49] = [
    ',', '.', '!', '?', ';', ':', '"', '\'', '(', ')', '[', ']', '{', '}', '-', '_', '/', '\\',
    '|', '@', '#', '$', '%', '&', '*', '+', '=', '~', '，', '。', '！', '？', '；', '：', '、',
    '「', '」', '『', '』', '（', '）', '【', '】', '《', '》', '—', '…', '·', '～',
];

impl RecModel {
    /// Create recognizer from model file and charset file
    ///
    /// # Parameters
    /// - `model_path`: Model file path (.mnn format)
    /// - `charset_path`: Charset file path (one character per line)
    /// - `config`: Optional inference config
    pub fn from_file(
        model_path: impl AsRef<Path>,
        charset_path: impl AsRef<Path>,
        config: Option<InferenceConfig>,
    ) -> OcrResult<Self> {
        let engine = InferenceEngine::from_file(model_path, config)?;
        let charset = Self::load_charset_from_file(charset_path)?;

        Ok(Self {
            engine,
            charset,
            options: RecOptions::default(),
            normalize_params: NormalizeParams::paddle_rec(),
        })
    }

    /// Create recognizer from model bytes and charset file
    pub fn from_bytes(
        model_bytes: &[u8],
        charset_path: impl AsRef<Path>,
        config: Option<InferenceConfig>,
    ) -> OcrResult<Self> {
        let engine = InferenceEngine::from_buffer(model_bytes, config)?;
        let charset = Self::load_charset_from_file(charset_path)?;

        Ok(Self {
            engine,
            charset,
            options: RecOptions::default(),
            normalize_params: NormalizeParams::paddle_rec(),
        })
    }

    /// Create recognizer from model bytes and charset bytes
    pub fn from_bytes_with_charset(
        model_bytes: &[u8],
        charset_bytes: &[u8],
        config: Option<InferenceConfig>,
    ) -> OcrResult<Self> {
        let engine = InferenceEngine::from_buffer(model_bytes, config)?;
        let charset = Self::parse_charset(charset_bytes)?;

        Ok(Self {
            engine,
            charset,
            options: RecOptions::default(),
            normalize_params: NormalizeParams::paddle_rec(),
        })
    }

    /// Load charset from file
    fn load_charset_from_file(path: impl AsRef<Path>) -> OcrResult<Vec<char>> {
        let content = std::fs::read_to_string(path)?;
        Self::parse_charset(content.as_bytes())
    }

    /// Parse charset data
    fn parse_charset(data: &[u8]) -> OcrResult<Vec<char>> {
        let content = std::str::from_utf8(data)
            .map_err(|e| OcrError::CharsetError(format!("UTF-8 decode error: {}", e)))?;

        // Charset format: one character per line
        // Add space at beginning and end as blank and padding
        let mut charset: Vec<char> = vec![' ']; // blank token at start

        for ch in content.chars() {
            if ch != '\n' && ch != '\r' {
                charset.push(ch);
            }
        }

        charset.push(' '); // padding token at end

        if charset.len() < 3 {
            return Err(OcrError::CharsetError("Charset too small".to_string()));
        }

        Ok(charset)
    }

    /// Set recognition options
    pub fn with_options(mut self, options: RecOptions) -> Self {
        self.options = options;
        self
    }

    /// Get current recognition options
    pub fn options(&self) -> &RecOptions {
        &self.options
    }

    /// Modify recognition options
    pub fn options_mut(&mut self) -> &mut RecOptions {
        &mut self.options
    }

    /// Get charset size
    pub fn charset_size(&self) -> usize {
        self.charset.len()
    }

    /// Recognize a single image
    ///
    /// # Parameters
    /// - `image`: Input image (text line image)
    ///
    /// # Returns
    /// Recognition result
    pub fn recognize(&self, image: &DynamicImage) -> OcrResult<RecognitionResult> {
        // Preprocess
        let input = preprocess_for_rec(image, self.options.target_height, &self.normalize_params)?;

        // Inference (using dynamic shape)
        let output = self.engine.run_dynamic(input.view().into_dyn())?;

        // Decode
        self.decode_output_view(output.view())
    }

    /// Recognize a single image, return text only
    pub fn recognize_text(&self, image: &DynamicImage) -> OcrResult<String> {
        let result = self.recognize(image)?;
        Ok(result.text)
    }

    /// Batch recognize images
    ///
    /// # Parameters
    /// - `images`: List of input images
    ///
    /// # Returns
    /// List of recognition results
    pub fn recognize_batch(&self, images: &[DynamicImage]) -> OcrResult<Vec<RecognitionResult>> {
        if images.is_empty() {
            return Ok(Vec::new());
        }

        // For small number of images, process individually
        if images.len() <= 2 || !self.options.enable_batch {
            return images.iter().map(|img| self.recognize(img)).collect();
        }

        // Batch processing
        let mut results = Vec::with_capacity(images.len());

        for chunk in images.chunks(self.options.batch_size) {
            let batch_results = self.recognize_batch_internal(chunk)?;
            results.extend(batch_results);
        }

        Ok(results)
    }

    /// Batch recognize images (borrowed version, avoid cloning)
    ///
    /// # Parameters
    /// - `images`: List of input image references
    ///
    /// # Returns
    /// List of recognition results
    pub fn recognize_batch_ref(
        &self,
        images: &[&DynamicImage],
    ) -> OcrResult<Vec<RecognitionResult>> {
        if images.is_empty() {
            return Ok(Vec::new());
        }

        // For small number of images, process individually
        if images.len() <= 2 || !self.options.enable_batch {
            return images.iter().map(|img| self.recognize(img)).collect();
        }

        // Batch processing
        let mut results = Vec::with_capacity(images.len());

        for chunk in images.chunks(self.options.batch_size) {
            // Dereference and convert to Vec<DynamicImage>
            let chunk_owned: Vec<DynamicImage> = chunk.iter().map(|img| (*img).clone()).collect();
            let batch_results = self.recognize_batch_internal(&chunk_owned)?;
            results.extend(batch_results);
        }

        Ok(results)
    }

    pub(crate) fn recognize_regions(
        &self,
        image: &DynamicImage,
        boxes: &[TextBox],
    ) -> OcrResult<Vec<RecognitionResult>> {
        if boxes.is_empty() {
            return Ok(Vec::new());
        }

        let source = image.to_rgb8();
        let mut results = Vec::with_capacity(boxes.len());
        let batch_size = self.options.batch_size.max(1);

        for chunk in boxes.chunks(batch_size) {
            let batch_input = self.preprocess_regions_batch(&source, chunk)?;
            let batch_output = self.engine.run_dynamic(batch_input.view().into_dyn())?;

            let shape = batch_output.shape();
            if shape.len() != 3 {
                return Err(OcrError::PostprocessError(format!(
                    "Region batch inference output shape error: {:?}",
                    shape
                )));
            }

            for i in 0..shape[0] {
                let sample_output = batch_output.index_axis(Axis(0), i).into_dyn();
                results.push(self.decode_output_view(sample_output)?);
            }
        }

        Ok(results)
    }

    fn preprocess_regions_batch(
        &self,
        source: &RgbImage,
        boxes: &[TextBox],
    ) -> OcrResult<Array4<f32>> {
        if boxes.is_empty() {
            return Ok(Array4::<f32>::zeros((
                0,
                3,
                self.options.target_height as usize,
                0,
            )));
        }

        if self.options.target_height == 0 {
            return Err(OcrError::InvalidParameter(
                "Recognition target height must be greater than 0".into(),
            ));
        }

        let target_height = self.options.target_height;
        let target_widths = boxes
            .iter()
            .map(|text_box| region_target_width(text_box, target_height))
            .collect::<Vec<_>>();
        let max_width = target_widths.iter().copied().max().unwrap_or(1) as usize;
        let batch_size = boxes.len();
        let target_height_usize = target_height as usize;
        let sample_size = 3 * target_height_usize * max_width;
        let plane_size = target_height_usize * max_width;

        let mut batch = Array4::<f32>::zeros((batch_size, 3, target_height_usize, max_width));
        let data = batch
            .as_slice_mut()
            .expect("Array4 created by zeros should be contiguous");
        let scales = [
            1.0 / (255.0 * self.normalize_params.std[0]),
            1.0 / (255.0 * self.normalize_params.std[1]),
            1.0 / (255.0 * self.normalize_params.std[2]),
        ];
        let offsets = [
            -self.normalize_params.mean[0] / self.normalize_params.std[0],
            -self.normalize_params.mean[1] / self.normalize_params.std[1],
            -self.normalize_params.mean[2] / self.normalize_params.std[2],
        ];

        for (i, (text_box, &target_width)) in boxes.iter().zip(target_widths.iter()).enumerate() {
            let projection =
                target_to_source_projection(source, text_box, target_width, target_height)
                    .ok_or_else(|| {
                        OcrError::PreprocessError(format!(
                            "Failed to render recognition region: {:?}",
                            text_box.rect
                        ))
                    })?;
            let target_width = target_width as usize;
            let sample_offset = i * sample_size;

            write_projected_region_to_tensor(
                source,
                projection,
                target_width,
                target_height_usize,
                max_width,
                sample_offset,
                plane_size,
                data,
                &scales,
                &offsets,
            );
        }

        Ok(batch)
    }

    /// Internal batch recognition
    fn recognize_batch_internal(
        &self,
        images: &[DynamicImage],
    ) -> OcrResult<Vec<RecognitionResult>> {
        if images.is_empty() {
            return Ok(Vec::new());
        }

        // If only one image, process individually
        if images.len() == 1 {
            return Ok(vec![self.recognize(&images[0])?]);
        }

        // Batch preprocessing
        let batch_input = crate::preprocess::preprocess_batch_for_rec(
            images,
            self.options.target_height,
            &self.normalize_params,
        )?;

        // Batch inference
        let batch_output = self.engine.run_dynamic(batch_input.view().into_dyn())?;

        // Decode output for each sample
        let shape = batch_output.shape();
        if shape.len() != 3 {
            return Err(OcrError::PostprocessError(format!(
                "Batch inference output shape error: {:?}",
                shape
            )));
        }

        let batch_size = shape[0];
        let mut results = Vec::with_capacity(batch_size);

        for i in 0..batch_size {
            let sample_output = batch_output.index_axis(Axis(0), i).into_dyn();
            let result = self.decode_output_view(sample_output)?;
            results.push(result);
        }

        Ok(results)
    }

    fn decode_output_view(&self, output: ArrayViewD<'_, f32>) -> OcrResult<RecognitionResult> {
        let shape = output.shape();
        let output_data = match output.as_slice_memory_order() {
            Some(slice) => Cow::Borrowed(slice),
            None => Cow::Owned(output.iter().copied().collect()),
        };

        // Output shape should be [batch, seq_len, num_classes] or [seq_len, num_classes]
        let (seq_len, num_classes) = if shape.len() == 3 {
            (shape[1], shape[2])
        } else if shape.len() == 2 {
            (shape[0], shape[1])
        } else {
            return Err(OcrError::PostprocessError(format!(
                "Invalid output shape: {:?}",
                shape
            )));
        };

        if num_classes == 0 {
            return Err(OcrError::PostprocessError(
                "Invalid output shape with zero classes".into(),
            ));
        }

        // CTC decoding
        let mut char_scores = Vec::with_capacity(seq_len.min(32));
        let mut text = String::new();
        let mut score_sum = 0.0f32;
        let mut prev_idx = 0usize;

        for t in 0..seq_len {
            // Find character with maximum probability at current time step
            let start = t * num_classes;
            let end = start + num_classes;
            let probs = &output_data[start..end];

            let mut max_idx = 0usize;
            let mut max_prob = f32::NEG_INFINITY;
            for (idx, &prob) in probs.iter().enumerate() {
                if prob > max_prob {
                    max_idx = idx;
                    max_prob = prob;
                }
            }

            // CTC decoding rule: skip blank (index 0) and duplicate characters
            if max_idx != 0 && max_idx != prev_idx {
                if max_idx < self.charset.len() {
                    let ch = self.charset[max_idx];

                    // Use raw logit value as confidence (model output is already softmax probability)
                    // For large character sets, softmax scores can be very small, so use max_prob directly
                    let score = max_prob;

                    // Only filter out very low confidence characters
                    let threshold = if Self::is_punctuation(ch) {
                        self.options.punct_min_score
                    } else {
                        self.options.min_score
                    };

                    if score >= threshold {
                        text.push(ch);
                        score_sum += score;
                        char_scores.push((ch, score));
                    }
                }
            }

            prev_idx = max_idx;
        }

        // Calculate average confidence
        let confidence = if char_scores.is_empty() {
            0.0
        } else {
            score_sum / char_scores.len() as f32
        };

        Ok(RecognitionResult::new(text, confidence, char_scores))
    }

    /// Check if character is punctuation
    fn is_punctuation(ch: char) -> bool {
        PUNCTUATIONS.contains(&ch)
    }
}

fn region_target_width(text_box: &TextBox, target_height: u32) -> u32 {
    let (width, height) = region_dimensions(text_box);
    ((width / height.max(1.0)) * target_height as f32)
        .round()
        .max(2.0) as u32
}

fn target_to_source_projection(
    source: &RgbImage,
    text_box: &TextBox,
    target_width: u32,
    target_height: u32,
) -> Option<Projection> {
    if target_width < 2 || target_height < 2 {
        return None;
    }

    if let Some(source_points) =
        source_points_for_text_box(text_box, source.width(), source.height())
    {
        if let Some(projection) =
            build_target_to_source_projection(source_points, target_width, target_height)
        {
            return Some(projection);
        }
    }

    let source_points = rect_source_points_for_text_box(text_box, source.width(), source.height())?;
    build_target_to_source_projection(source_points, target_width, target_height)
}

fn build_target_to_source_projection(
    source_points: [(f32, f32); 4],
    target_width: u32,
    target_height: u32,
) -> Option<Projection> {
    let target_points = [
        (0.0, 0.0),
        (target_width.saturating_sub(1) as f32, 0.0),
        (
            target_width.saturating_sub(1) as f32,
            target_height.saturating_sub(1) as f32,
        ),
        (0.0, target_height.saturating_sub(1) as f32),
    ];

    Projection::from_control_points(source_points, target_points)
        .map(|projection| projection.invert())
}

#[allow(clippy::too_many_arguments)]
fn write_projected_region_to_tensor(
    source: &RgbImage,
    target_to_source: Projection,
    target_width: usize,
    target_height: usize,
    max_width: usize,
    sample_offset: usize,
    plane_size: usize,
    data: &mut [f32],
    scales: &[f32; 3],
    offsets: &[f32; 3],
) {
    let source_width = source.width() as usize;
    let source_height = source.height() as usize;
    let source_data = source.as_raw();

    for y in 0..target_height {
        let dst_row = y * max_width;

        for x in 0..target_width {
            let (source_x, source_y) = target_to_source * (x as f32, y as f32);
            let dst = sample_offset + dst_row + x;
            write_normalized_sample(
                source_data,
                source_width,
                source_height,
                source_x,
                source_y,
                data,
                dst,
                sample_offset + plane_size + dst_row + x,
                sample_offset + plane_size * 2 + dst_row + x,
                scales,
                offsets,
            );
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn write_normalized_sample(
    source_data: &[u8],
    source_width: usize,
    source_height: usize,
    x: f32,
    y: f32,
    data: &mut [f32],
    dst_r: usize,
    dst_g: usize,
    dst_b: usize,
    scales: &[f32; 3],
    offsets: &[f32; 3],
) {
    let left = x.floor();
    let right = left + 1.0;
    let top = y.floor();
    let bottom = top + 1.0;

    if !(left >= 0.0 && right < source_width as f32 && top >= 0.0 && bottom < source_height as f32)
    {
        data[dst_r] = 255.0 * scales[0] + offsets[0];
        data[dst_g] = 255.0 * scales[1] + offsets[1];
        data[dst_b] = 255.0 * scales[2] + offsets[2];
        return;
    }

    let right_weight = x - left;
    let bottom_weight = y - top;
    let left = left as usize;
    let right = right as usize;
    let top = top as usize;
    let bottom = bottom as usize;
    let top_left = (top * source_width + left) * 3;
    let top_right = (top * source_width + right) * 3;
    let bottom_left = (bottom * source_width + left) * 3;
    let bottom_right = (bottom * source_width + right) * 3;

    let r = bilinear_channel(
        source_data[top_left],
        source_data[top_right],
        source_data[bottom_left],
        source_data[bottom_right],
        right_weight,
        bottom_weight,
    );
    let g = bilinear_channel(
        source_data[top_left + 1],
        source_data[top_right + 1],
        source_data[bottom_left + 1],
        source_data[bottom_right + 1],
        right_weight,
        bottom_weight,
    );
    let b = bilinear_channel(
        source_data[top_left + 2],
        source_data[top_right + 2],
        source_data[bottom_left + 2],
        source_data[bottom_right + 2],
        right_weight,
        bottom_weight,
    );

    data[dst_r] = r as f32 * scales[0] + offsets[0];
    data[dst_g] = g as f32 * scales[1] + offsets[1];
    data[dst_b] = b as f32 * scales[2] + offsets[2];
}

#[inline(always)]
fn bilinear_channel(
    top_left: u8,
    top_right: u8,
    bottom_left: u8,
    bottom_right: u8,
    right_weight: f32,
    bottom_weight: f32,
) -> u8 {
    let top = lerp(top_left as f32, top_right as f32, right_weight);
    let bottom = lerp(bottom_left as f32, bottom_right as f32, right_weight);
    clamp_to_u8(lerp(top, bottom, bottom_weight))
}

#[inline]
fn lerp(left: f32, right: f32, weight: f32) -> f32 {
    (1.0 - weight) * left + weight * right
}

#[inline]
fn clamp_to_u8(value: f32) -> u8 {
    if value < u8::MAX as f32 {
        if value > u8::MIN as f32 {
            value as u8
        } else {
            u8::MIN
        }
    } else {
        u8::MAX
    }
}

fn source_points_for_text_box(
    text_box: &TextBox,
    image_width: u32,
    image_height: u32,
) -> Option<[(f32, f32); 4]> {
    if let Some(points) = text_box.points {
        let max_x = image_width.saturating_sub(1) as f32;
        let max_y = image_height.saturating_sub(1) as f32;
        return Some(points.map(|point| (point.x.clamp(0.0, max_x), point.y.clamp(0.0, max_y))));
    }

    rect_source_points_for_text_box(text_box, image_width, image_height)
}

fn rect_source_points_for_text_box(
    text_box: &TextBox,
    image_width: u32,
    image_height: u32,
) -> Option<[(f32, f32); 4]> {
    let left = text_box.rect.left().max(0) as u32;
    let top = text_box.rect.top().max(0) as u32;
    let right = left
        .saturating_add(text_box.rect.width())
        .min(image_width)
        .saturating_sub(1);
    let bottom = top
        .saturating_add(text_box.rect.height())
        .min(image_height)
        .saturating_sub(1);

    if right <= left || bottom <= top {
        return None;
    }

    Some([
        (left as f32, top as f32),
        (right as f32, top as f32),
        (right as f32, bottom as f32),
        (left as f32, bottom as f32),
    ])
}

fn region_dimensions(text_box: &TextBox) -> (f32, f32) {
    if let Some(points) = text_box.points {
        let width = distance(points[0], points[1]).max(distance(points[3], points[2]));
        let height = distance(points[0], points[3]).max(distance(points[1], points[2]));
        (width.max(1.0), height.max(1.0))
    } else {
        (
            text_box.rect.width().max(1) as f32,
            text_box.rect.height().max(1) as f32,
        )
    }
}

fn distance(a: Point<f32>, b: Point<f32>) -> f32 {
    let dx = a.x - b.x;
    let dy = a.y - b.y;
    (dx * dx + dy * dy).sqrt()
}

/// Low-level recognition API
impl RecModel {
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

    /// Get charset
    pub fn charset(&self) -> &[char] {
        &self.charset
    }

    /// Get character by index
    pub fn get_char(&self, index: usize) -> Option<char> {
        self.charset.get(index).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rec_options_default() {
        let opts = RecOptions::default();
        assert_eq!(opts.target_height, 48);
        assert_eq!(opts.min_score, 0.3);
        assert_eq!(opts.punct_min_score, 0.1);
        assert_eq!(opts.batch_size, 8);
        assert!(opts.enable_batch);
    }

    #[test]
    fn test_rec_options_builder() {
        let opts = RecOptions::new()
            .with_target_height(32)
            .with_min_score(0.6)
            .with_punct_min_score(0.2)
            .with_batch_size(16)
            .with_batch(false);

        assert_eq!(opts.target_height, 32);
        assert_eq!(opts.min_score, 0.6);
        assert_eq!(opts.punct_min_score, 0.2);
        assert_eq!(opts.batch_size, 16);
        assert!(!opts.enable_batch);
    }

    #[test]
    fn test_recognition_result_new() {
        let char_scores = vec![
            ('H', 0.99),
            ('e', 0.94),
            ('l', 0.93),
            ('l', 0.95),
            ('o', 0.94),
        ];
        let result = RecognitionResult::new("Hello".to_string(), 0.95, char_scores.clone());

        assert_eq!(result.text, "Hello");
        assert_eq!(result.confidence, 0.95);
        assert_eq!(result.char_scores.len(), 5);
        assert_eq!(result.char_scores[0].0, 'H');
        assert_eq!(result.char_scores[0].1, 0.99);
    }

    #[test]
    fn test_recognition_result_is_valid() {
        let result = RecognitionResult::new(
            "Hello".to_string(),
            0.95,
            vec![
                ('H', 0.99),
                ('e', 0.94),
                ('l', 0.93),
                ('l', 0.95),
                ('o', 0.94),
            ],
        );

        assert!(result.is_valid(0.9));
        assert!(result.is_valid(0.95));
        assert!(!result.is_valid(0.96));
        assert!(!result.is_valid(0.99));
    }

    #[test]
    fn test_recognition_result_empty() {
        let result = RecognitionResult::new(String::new(), 0.0, vec![]);

        assert!(result.text.is_empty());
        assert_eq!(result.confidence, 0.0);
        assert!(!result.is_valid(0.1));
    }

    #[test]
    fn test_region_target_width_avoids_projection_degenerate_width() {
        let text_box = TextBox::with_points(
            imageproc::rect::Rect::at(747, 14).of_size(61, 1695),
            0.9,
            [
                Point::new(747.0, 14.0),
                Point::new(747.4, 14.0),
                Point::new(747.4, 1709.0),
                Point::new(747.0, 1709.0),
            ],
        );

        assert_eq!(region_target_width(&text_box, 48), 2);
    }

    #[test]
    fn test_is_punctuation_common() {
        // English punctuation
        assert!(RecModel::is_punctuation(','));
        assert!(RecModel::is_punctuation('.'));
        assert!(RecModel::is_punctuation('!'));
        assert!(RecModel::is_punctuation('?'));
        assert!(RecModel::is_punctuation(';'));
        assert!(RecModel::is_punctuation(':'));
        assert!(RecModel::is_punctuation('"'));
        assert!(RecModel::is_punctuation('\''));
    }

    #[test]
    fn test_is_punctuation_chinese() {
        // Chinese punctuation
        assert!(RecModel::is_punctuation('，'));
        assert!(RecModel::is_punctuation('。'));
        assert!(RecModel::is_punctuation('！'));
        assert!(RecModel::is_punctuation('？'));
        assert!(RecModel::is_punctuation('；'));
        assert!(RecModel::is_punctuation('：'));
        assert!(RecModel::is_punctuation('、'));
        assert!(RecModel::is_punctuation('—'));
        assert!(RecModel::is_punctuation('…'));
    }

    #[test]
    fn test_is_punctuation_brackets() {
        assert!(RecModel::is_punctuation('('));
        assert!(RecModel::is_punctuation(')'));
        assert!(RecModel::is_punctuation('['));
        assert!(RecModel::is_punctuation(']'));
        assert!(RecModel::is_punctuation('{'));
        assert!(RecModel::is_punctuation('}'));
        assert!(RecModel::is_punctuation('「'));
        assert!(RecModel::is_punctuation('」'));
        assert!(RecModel::is_punctuation('《'));
        assert!(RecModel::is_punctuation('》'));
    }

    #[test]
    fn test_is_punctuation_false() {
        // Non-punctuation characters
        assert!(!RecModel::is_punctuation('A'));
        assert!(!RecModel::is_punctuation('z'));
        assert!(!RecModel::is_punctuation('0'));
        assert!(!RecModel::is_punctuation('中'));
        assert!(!RecModel::is_punctuation('文'));
        assert!(!RecModel::is_punctuation(' '));
    }
}
