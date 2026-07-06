//! Image Preprocessing Utilities
//!
//! Provides various image preprocessing functions required for OCR

use image::{DynamicImage, GenericImageView, RgbImage};
use ndarray::{Array4, ArrayBase, Dim, OwnedRepr};

use crate::error::{OcrError, OcrResult};

/// Image normalization parameters
#[derive(Debug, Clone)]
pub struct NormalizeParams {
    /// RGB channel means
    pub mean: [f32; 3],
    /// RGB channel standard deviations
    pub std: [f32; 3],
}

impl Default for NormalizeParams {
    fn default() -> Self {
        // ImageNet normalization parameters
        Self {
            mean: [0.485, 0.456, 0.406],
            std: [0.229, 0.224, 0.225],
        }
    }
}

impl NormalizeParams {
    /// Normalization parameters for PaddleOCR detection model
    pub fn paddle_det() -> Self {
        Self {
            mean: [0.485, 0.456, 0.406],
            std: [0.229, 0.224, 0.225],
        }
    }

    /// Normalization parameters for PaddleOCR recognition model
    pub fn paddle_rec() -> Self {
        Self {
            mean: [0.5, 0.5, 0.5],
            std: [0.5, 0.5, 0.5],
        }
    }
}

/// Calculate size to pad to (multiple of 32)
#[inline]
pub fn get_padded_size(size: u32) -> u32 {
    ((size + 31) / 32) * 32
}

/// Scale image to specified maximum side length
///
/// Maintains aspect ratio, scales longest side to max_side_len
pub fn resize_to_max_side(img: &DynamicImage, max_side_len: u32) -> OcrResult<DynamicImage> {
    let (w, h) = img.dimensions();
    let max_dim = w.max(h);

    if max_dim <= max_side_len {
        return Ok(img.clone());
    }

    let scale = max_side_len as f64 / max_dim as f64;
    let new_w = (w as f64 * scale).round() as u32;
    let new_h = (h as f64 * scale).round() as u32;

    fast_resize(img, new_w, new_h)
}

/// Scale image to specified height (for recognition model)
///
/// Scales maintaining aspect ratio
pub fn resize_to_height(img: &DynamicImage, target_height: u32) -> OcrResult<DynamicImage> {
    let (w, h) = img.dimensions();

    if h == target_height {
        return Ok(img.clone());
    }

    let scale = target_height as f64 / h as f64;
    let new_w = (w as f64 * scale).round() as u32;

    fast_resize(img, new_w, target_height)
}

/// Fast image resizing using fast_image_resize
/// Can pass DynamicImage directly when "image" feature is enabled
fn fast_resize(img: &DynamicImage, new_w: u32, new_h: u32) -> OcrResult<DynamicImage> {
    use fast_image_resize::{images::Image, IntoImageView, PixelType, Resizer};

    // Only U8x3 (RGB) and U8x4 (RGBA) are handled end-to-end.
    // Grayscale (U8), 16-bit, and other formats must be converted to RGB first;
    // otherwise the output buffer byte count would not match the expected channel count.
    let converted: DynamicImage;
    let (src, pixel_type) = match img.pixel_type() {
        Some(PixelType::U8x3) => (img, PixelType::U8x3),
        Some(PixelType::U8x4) => (img, PixelType::U8x4),
        _ => {
            converted = DynamicImage::ImageRgb8(img.to_rgb8());
            (&converted, PixelType::U8x3)
        }
    };

    // Create destination image container
    let mut dst_image = Image::new(new_w, new_h, pixel_type);

    // Resize using Resizer
    let mut resizer = Resizer::new();
    resizer
        .resize(src, &mut dst_image, None)
        .map_err(|e| OcrError::PreprocessError(format!("Image resize failed: {e}")))?;

    // Convert result back to DynamicImage
    match pixel_type {
        PixelType::U8x3 => RgbImage::from_raw(new_w, new_h, dst_image.into_vec())
            .map(DynamicImage::ImageRgb8)
            .ok_or_else(|| {
                OcrError::PreprocessError("RGB buffer size mismatch after resize".into())
            }),
        PixelType::U8x4 => image::RgbaImage::from_raw(new_w, new_h, dst_image.into_vec())
            .map(DynamicImage::ImageRgba8)
            .ok_or_else(|| {
                OcrError::PreprocessError("RGBA buffer size mismatch after resize".into())
            }),
        _ => unreachable!("pixel_type is constrained to U8x3 or U8x4 above"),
    }
}

/// Convert image to detection model input tensor
///
/// Output format: [1, 3, H, W] (NCHW)
pub fn preprocess_for_det(
    img: &DynamicImage,
    params: &NormalizeParams,
) -> OcrResult<ArrayBase<OwnedRepr<f32>, Dim<[usize; 4]>>> {
    let (w, h) = img.dimensions();
    let pad_w = get_padded_size(w) as usize;
    let pad_h = get_padded_size(h) as usize;
    let (w, h) = (w as usize, h as usize);

    let mut input = Array4::<f32>::zeros((1, 3, pad_h, pad_w));
    let rgb_img = img.to_rgb8();
    let rgb = rgb_img.as_raw();

    let plane_size = pad_h * pad_w;
    let data = input
        .as_slice_mut()
        .expect("Array4 created by zeros should be contiguous");
    let scales = [
        1.0 / (255.0 * params.std[0]),
        1.0 / (255.0 * params.std[1]),
        1.0 / (255.0 * params.std[2]),
    ];
    let offsets = [
        -params.mean[0] / params.std[0],
        -params.mean[1] / params.std[1],
        -params.mean[2] / params.std[2],
    ];

    // Normalize and pad
    for y in 0..h {
        let src_row = &rgb[y * w * 3..(y + 1) * w * 3];
        let dst_row = y * pad_w;

        for (x, pixel) in src_row.chunks_exact(3).enumerate() {
            let dst = dst_row + x;
            data[dst] = pixel[0] as f32 * scales[0] + offsets[0];
            data[plane_size + dst] = pixel[1] as f32 * scales[1] + offsets[1];
            data[plane_size * 2 + dst] = pixel[2] as f32 * scales[2] + offsets[2];
        }
    }

    Ok(input)
}

/// Convert image to recognition model input tensor
///
/// Output format: [1, 3, H, W] (NCHW)
/// Height is fixed at 48 (or specified value), width scaled proportionally
pub fn preprocess_for_rec(
    img: &DynamicImage,
    target_height: u32,
    params: &NormalizeParams,
) -> OcrResult<ArrayBase<OwnedRepr<f32>, Dim<[usize; 4]>>> {
    let (_, h) = img.dimensions();

    let resized = if h == target_height {
        img.clone()
    } else {
        resize_to_height(img, target_height)?
    };

    let rgb_img = resized.to_rgb8();
    let rgb = rgb_img.as_raw();
    let (w, h) = (rgb_img.width() as usize, rgb_img.height() as usize);

    let mut input = Array4::<f32>::zeros((1, 3, h, w));
    let plane_size = h * w;
    let data = input
        .as_slice_mut()
        .expect("Array4 created by zeros should be contiguous");
    let scales = [
        1.0 / (255.0 * params.std[0]),
        1.0 / (255.0 * params.std[1]),
        1.0 / (255.0 * params.std[2]),
    ];
    let offsets = [
        -params.mean[0] / params.std[0],
        -params.mean[1] / params.std[1],
        -params.mean[2] / params.std[2],
    ];

    for y in 0..h {
        let src_row = &rgb[y * w * 3..(y + 1) * w * 3];
        let dst_row = y * w;

        for (x, pixel) in src_row.chunks_exact(3).enumerate() {
            let dst = dst_row + x;
            data[dst] = pixel[0] as f32 * scales[0] + offsets[0];
            data[plane_size + dst] = pixel[1] as f32 * scales[1] + offsets[1];
            data[plane_size * 2 + dst] = pixel[2] as f32 * scales[2] + offsets[2];
        }
    }

    Ok(input)
}

/// Batch preprocess recognition images
///
/// Process multiple images into batch tensor, all images padded to same width
pub fn preprocess_batch_for_rec(
    images: &[DynamicImage],
    target_height: u32,
    params: &NormalizeParams,
) -> OcrResult<ArrayBase<OwnedRepr<f32>, Dim<[usize; 4]>>> {
    if images.is_empty() {
        return Ok(Array4::<f32>::zeros((0, 3, target_height as usize, 0)));
    }

    // Calculate scaled width for all images
    let widths: Vec<u32> = images
        .iter()
        .map(|img| {
            let (w, h) = img.dimensions();
            let scale = target_height as f64 / h as f64;
            (w as f64 * scale).round() as u32
        })
        .collect();

    // widths is non-empty because images is non-empty (checked above)
    let max_width = *widths.iter().max().unwrap() as usize;
    let batch_size = images.len();

    let mut batch = Array4::<f32>::zeros((batch_size, 3, target_height as usize, max_width));
    let sample_size = 3 * target_height as usize * max_width;
    let plane_size = target_height as usize * max_width;
    let data = batch
        .as_slice_mut()
        .expect("Array4 created by zeros should be contiguous");
    let scales = [
        1.0 / (255.0 * params.std[0]),
        1.0 / (255.0 * params.std[1]),
        1.0 / (255.0 * params.std[2]),
    ];
    let offsets = [
        -params.mean[0] / params.std[0],
        -params.mean[1] / params.std[1],
        -params.mean[2] / params.std[2],
    ];

    for (i, img) in images.iter().enumerate() {
        let resized = resize_to_height(img, target_height)?;
        let rgb_img = resized.to_rgb8();
        let rgb = rgb_img.as_raw();
        let w = rgb_img.width() as usize;
        let h = target_height as usize;
        let sample_offset = i * sample_size;

        for y in 0..h {
            let src_row = &rgb[y * w * 3..(y + 1) * w * 3];
            let dst_row = y * max_width;

            for (x, pixel) in src_row.chunks_exact(3).enumerate() {
                let dst = sample_offset + dst_row + x;
                data[dst] = pixel[0] as f32 * scales[0] + offsets[0];
                data[sample_offset + plane_size + dst_row + x] =
                    pixel[1] as f32 * scales[1] + offsets[1];
                data[sample_offset + plane_size * 2 + dst_row + x] =
                    pixel[2] as f32 * scales[2] + offsets[2];
            }
        }
    }

    Ok(batch)
}

/// Crop image region
pub fn crop_image(img: &DynamicImage, x: u32, y: u32, width: u32, height: u32) -> DynamicImage {
    img.crop_imm(x, y, width, height)
}

/// Split image into blocks (for high precision mode)
///
/// # Parameters
/// - `img`: Input image
/// - `block_size`: Block size
/// - `overlap`: Overlap region size
///
/// # Returns
/// List of block images and their positions in original image (x, y)
pub fn split_into_blocks(
    img: &DynamicImage,
    block_size: u32,
    overlap: u32,
) -> Vec<(DynamicImage, u32, u32)> {
    let (width, height) = img.dimensions();
    let mut blocks = Vec::new();

    let step = block_size - overlap;

    let mut y = 0u32;
    while y < height {
        let mut x = 0u32;
        while x < width {
            let block_w = (block_size).min(width - x);
            let block_h = (block_size).min(height - y);

            let block = img.crop_imm(x, y, block_w, block_h);
            blocks.push((block, x, y));

            x += step;
            if x + overlap >= width && x < width {
                break;
            }
        }

        y += step;
        if y + overlap >= height && y < height {
            break;
        }
    }

    blocks
}

/// Convert grayscale mask to binary mask
pub fn threshold_mask(mask: &[f32], threshold: f32) -> Vec<u8> {
    mask.iter()
        .map(|&v| if v > threshold { 255u8 } else { 0u8 })
        .collect()
}

/// Create grayscale image
pub fn create_gray_image(data: &[u8], width: u32, height: u32) -> image::GrayImage {
    image::GrayImage::from_raw(width, height, data.to_vec())
        .unwrap_or_else(|| image::GrayImage::new(width, height))
}

/// Convert image to RGB
pub fn to_rgb(img: &DynamicImage) -> RgbImage {
    img.to_rgb8()
}

/// Create image from RGB data
pub fn rgb_to_image(data: &[u8], width: u32, height: u32) -> DynamicImage {
    let rgb = RgbImage::from_raw(width, height, data.to_vec())
        .unwrap_or_else(|| RgbImage::new(width, height));
    DynamicImage::ImageRgb8(rgb)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_padded_size() {
        assert_eq!(get_padded_size(100), 128);
        assert_eq!(get_padded_size(32), 32);
        assert_eq!(get_padded_size(33), 64);
        assert_eq!(get_padded_size(0), 0);
        assert_eq!(get_padded_size(1), 32);
        assert_eq!(get_padded_size(31), 32);
        assert_eq!(get_padded_size(64), 64);
        assert_eq!(get_padded_size(65), 96);
    }

    #[test]
    fn test_normalize_params() {
        let params = NormalizeParams::default();
        assert_eq!(params.mean[0], 0.485);

        let paddle = NormalizeParams::paddle_det();
        assert_eq!(paddle.mean[0], 0.485);
        assert_eq!(paddle.std[0], 0.229);
    }

    #[test]
    fn test_normalize_params_paddle_rec() {
        let params = NormalizeParams::paddle_rec();
        assert_eq!(params.mean[0], 0.5);
        assert_eq!(params.mean[1], 0.5);
        assert_eq!(params.mean[2], 0.5);
        assert_eq!(params.std[0], 0.5);
        assert_eq!(params.std[1], 0.5);
        assert_eq!(params.std[2], 0.5);
    }

    #[test]
    fn test_resize_to_max_side_no_resize() {
        let img = DynamicImage::new_rgb8(100, 50);
        let resized = resize_to_max_side(&img, 200).unwrap();

        // 图像已经小于最大边，不应该缩放
        assert_eq!(resized.width(), 100);
        assert_eq!(resized.height(), 50);
    }

    #[test]
    fn test_resize_to_max_side_width_limited() {
        let img = DynamicImage::new_rgb8(1000, 500);
        let resized = resize_to_max_side(&img, 500).unwrap();

        // 宽度是最大边，应该缩放到 500
        assert_eq!(resized.width(), 500);
        assert_eq!(resized.height(), 250);
    }

    #[test]
    fn test_resize_to_max_side_height_limited() {
        let img = DynamicImage::new_rgb8(500, 1000);
        let resized = resize_to_max_side(&img, 500).unwrap();

        // 高度是最大边，应该缩放到 500
        assert_eq!(resized.width(), 250);
        assert_eq!(resized.height(), 500);
    }

    #[test]
    fn test_resize_to_height() {
        let img = DynamicImage::new_rgb8(200, 100);
        let resized = resize_to_height(&img, 48).unwrap();

        assert_eq!(resized.height(), 48);
        // 宽度应该按比例缩放: 200 * 48/100 = 96
        assert_eq!(resized.width(), 96);
    }

    #[test]
    fn test_resize_to_height_no_resize() {
        let img = DynamicImage::new_rgb8(200, 48);
        let resized = resize_to_height(&img, 48).unwrap();

        // 高度已经是目标高度，不应该缩放
        assert_eq!(resized.height(), 48);
        assert_eq!(resized.width(), 200);
    }

    #[test]
    fn test_preprocess_for_det_shape() {
        let img = DynamicImage::new_rgb8(100, 50);
        let params = NormalizeParams::paddle_det();
        let tensor = preprocess_for_det(&img, &params).unwrap();

        // 输出形状应该是 [1, 3, H, W]，H 和 W 是 32 的倍数
        assert_eq!(tensor.shape()[0], 1);
        assert_eq!(tensor.shape()[1], 3);
        assert_eq!(tensor.shape()[2], 64); // 50 向上取整到 64
        assert_eq!(tensor.shape()[3], 128); // 100 向上取整到 128
    }

    #[test]
    fn test_preprocess_for_rec_shape() {
        let img = DynamicImage::new_rgb8(200, 100);
        let params = NormalizeParams::paddle_rec();
        let tensor = preprocess_for_rec(&img, 48, &params).unwrap();

        // 输出高度应该是 48
        assert_eq!(tensor.shape()[0], 1);
        assert_eq!(tensor.shape()[1], 3);
        assert_eq!(tensor.shape()[2], 48);
        // 宽度应该按比例缩放: 200 * 48/100 = 96
        assert_eq!(tensor.shape()[3], 96);
    }

    #[test]
    fn test_preprocess_batch_for_rec_empty() {
        let images: Vec<DynamicImage> = vec![];
        let params = NormalizeParams::paddle_rec();
        let tensor = preprocess_batch_for_rec(&images, 48, &params).unwrap();

        assert_eq!(tensor.shape()[0], 0);
    }

    #[test]
    fn test_preprocess_batch_for_rec_single() {
        let images = vec![DynamicImage::new_rgb8(200, 100)];
        let params = NormalizeParams::paddle_rec();
        let tensor = preprocess_batch_for_rec(&images, 48, &params).unwrap();

        assert_eq!(tensor.shape()[0], 1);
        assert_eq!(tensor.shape()[1], 3);
        assert_eq!(tensor.shape()[2], 48);
    }

    #[test]
    fn test_preprocess_batch_for_rec_multiple() {
        let images = vec![
            DynamicImage::new_rgb8(200, 100),
            DynamicImage::new_rgb8(300, 100),
        ];
        let params = NormalizeParams::paddle_rec();
        let tensor = preprocess_batch_for_rec(&images, 48, &params).unwrap();

        assert_eq!(tensor.shape()[0], 2);
        assert_eq!(tensor.shape()[1], 3);
        assert_eq!(tensor.shape()[2], 48);
        // 宽度应该是最大宽度: max(96, 144) = 144
        assert_eq!(tensor.shape()[3], 144);
    }

    #[test]
    fn test_crop_image() {
        let img = DynamicImage::new_rgb8(200, 100);
        let cropped = crop_image(&img, 50, 25, 100, 50);

        assert_eq!(cropped.width(), 100);
        assert_eq!(cropped.height(), 50);
    }

    #[test]
    fn test_split_into_blocks() {
        let img = DynamicImage::new_rgb8(500, 500);
        let blocks = split_into_blocks(&img, 200, 50);

        // 应该有多个块
        assert!(!blocks.is_empty());

        // 每个块的位置应该记录正确
        for (block, x, y) in &blocks {
            assert!(block.width() <= 200);
            assert!(block.height() <= 200);
            assert!(*x < 500);
            assert!(*y < 500);
        }
    }

    #[test]
    fn test_split_into_blocks_small_image() {
        let img = DynamicImage::new_rgb8(100, 100);
        let blocks = split_into_blocks(&img, 200, 50);

        // 图像小于块大小，应该只有一个块
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].1, 0); // x offset
        assert_eq!(blocks[0].2, 0); // y offset
    }

    #[test]
    fn test_threshold_mask() {
        let mask = vec![0.1, 0.3, 0.5, 0.7, 0.9];
        let binary = threshold_mask(&mask, 0.5);

        assert_eq!(binary, vec![0, 0, 0, 255, 255]);
    }

    #[test]
    fn test_threshold_mask_all_below() {
        let mask = vec![0.1, 0.2, 0.3, 0.4];
        let binary = threshold_mask(&mask, 0.5);

        assert_eq!(binary, vec![0, 0, 0, 0]);
    }

    #[test]
    fn test_threshold_mask_all_above() {
        let mask = vec![0.6, 0.7, 0.8, 0.9];
        let binary = threshold_mask(&mask, 0.5);

        assert_eq!(binary, vec![255, 255, 255, 255]);
    }

    #[test]
    fn test_create_gray_image() {
        let data = vec![128u8; 100];
        let gray = create_gray_image(&data, 10, 10);

        assert_eq!(gray.width(), 10);
        assert_eq!(gray.height(), 10);
    }

    #[test]
    fn test_to_rgb() {
        let img = DynamicImage::new_rgb8(100, 50);
        let rgb = to_rgb(&img);

        assert_eq!(rgb.width(), 100);
        assert_eq!(rgb.height(), 50);
    }

    #[test]
    fn test_rgb_to_image() {
        let data = vec![128u8; 300]; // 10x10 RGB
        let img = rgb_to_image(&data, 10, 10);

        assert_eq!(img.width(), 10);
        assert_eq!(img.height(), 10);
    }
}
