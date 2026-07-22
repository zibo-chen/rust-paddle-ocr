//! 所有模型集成测试
//!
//! 使用 res/1.png（纯英文图片）统一测试所有模型的推理能力

use ocr_rs::{DetModel, OcrEngine, OcrEngineConfig, OriModel, RecModel};

const TEST_IMAGE: &str = "res/1.png";
/// 5.png 是 1.png 顺时针旋转 90° 的图片，用于测试方向检测
const TEST_IMAGE_ROTATED_90: &str = "res/5.png";
const TEST_IMAGES: &[&str] = &["res/1.png", "res/2.png", "res/3.png", "res/4.png"];

// ============================================================
// 检测模型路径
// ============================================================
const DET_V5: &str = "models/PP-OCRv5_mobile_det.mnn";
const DET_V5_FP16: &str = "models/PP-OCRv5_mobile_det_fp16.mnn";
const DET_V4: &str = "models/ch_PP-OCRv4_det_infer.mnn";
const DET_V6_TINY: &str = "models/PP-OCRv6_tiny_det.mnn";
const DET_V6_SMALL: &str = "models/PP-OCRv6_small_det.mnn";
const DET_V6_MEDIUM: &str = "models/PP-OCRv6_medium_det.mnn";

// ============================================================
// 识别模型 + 字符集路径
// ============================================================
const REC_V5: &str = "models/PP-OCRv5_mobile_rec.mnn";
const REC_V5_FP16: &str = "models/PP-OCRv5_mobile_rec_fp16.mnn";
const CHARSET_V5: &str = "models/ppocr_keys_v5.txt";

const REC_V4: &str = "models/ch_PP-OCRv4_rec_infer.mnn";
const CHARSET_V4: &str = "models/ppocr_keys_v4.txt";

const REC_V6_TINY: &str = "models/PP-OCRv6_tiny_rec.mnn";
const CHARSET_V6_TINY: &str = "models/ppocr_keys_v6_tiny.txt";

const REC_V6_SMALL: &str = "models/PP-OCRv6_small_rec.mnn";
const CHARSET_V6_SMALL: &str = "models/ppocr_keys_v6_small.txt";

const REC_V6_MEDIUM: &str = "models/PP-OCRv6_medium_rec.mnn";
const CHARSET_V6_MEDIUM: &str = "models/ppocr_keys_v6_medium.txt";

const REC_EN: &str = "models/en_PP-OCRv5_mobile_rec_infer.mnn";
const CHARSET_EN: &str = "models/ppocr_keys_en.txt";

const REC_ARABIC: &str = "models/arabic_PP-OCRv5_mobile_rec_infer.mnn";
const CHARSET_ARABIC: &str = "models/ppocr_keys_arabic.txt";

const REC_CYRILLIC: &str = "models/cyrillic_PP-OCRv5_mobile_rec_infer.mnn";
const CHARSET_CYRILLIC: &str = "models/ppocr_keys_cyrillic.txt";

const REC_DEVANAGARI: &str = "models/devanagari_PP-OCRv5_mobile_rec_infer.mnn";
const CHARSET_DEVANAGARI: &str = "models/ppocr_keys_devanagari.txt";

const REC_EL: &str = "models/el_PP-OCRv5_mobile_rec_infer.mnn";
const CHARSET_EL: &str = "models/ppocr_keys_el.txt";

const REC_ESLAV: &str = "models/eslav_PP-OCRv5_mobile_rec_infer.mnn";
const CHARSET_ESLAV: &str = "models/ppocr_keys_eslav.txt";

const REC_KOREAN: &str = "models/korean_PP-OCRv5_mobile_rec_infer.mnn";
const CHARSET_KOREAN: &str = "models/ppocr_keys_korean.txt";

const REC_LATIN: &str = "models/latin_PP-OCRv5_mobile_rec_infer.mnn";
const CHARSET_LATIN: &str = "models/ppocr_keys_latin.txt";

const REC_TA: &str = "models/ta_PP-OCRv5_mobile_rec_infer.mnn";
const CHARSET_TA: &str = "models/ppocr_keys_ta.txt";

const REC_TE: &str = "models/te_PP-OCRv5_mobile_rec_infer.mnn";
const CHARSET_TE: &str = "models/ppocr_keys_te.txt";

const REC_TH: &str = "models/th_PP-OCRv5_mobile_rec_infer.mnn";
const CHARSET_TH: &str = "models/ppocr_keys_th.txt";

// ============================================================
// 方向分类模型路径
// ============================================================
const ORI_MODEL: &str = "models/PP-LCNet_x1_0_doc_ori.mnn";

// ============================================================
// 辅助函数
// ============================================================
fn require_file(path: &str) -> bool {
    if !std::path::Path::new(path).exists() {
        eprintln!("跳过测试：文件不存在 {}", path);
        false
    } else {
        true
    }
}

fn load_test_image() -> image::DynamicImage {
    image::open(TEST_IMAGE).expect("无法打开测试图片 res/1.png")
}

/// 通用：检测模型 + 识别模型 完整 pipeline 测试
fn run_full_pipeline(det_path: &str, rec_path: &str, charset_path: &str, label: &str) {
    run_full_pipeline_on_image(det_path, rec_path, charset_path, TEST_IMAGE, label);
}

fn run_full_pipeline_on_image(
    det_path: &str,
    rec_path: &str,
    charset_path: &str,
    image_path: &str,
    label: &str,
) {
    if !require_file(det_path)
        || !require_file(rec_path)
        || !require_file(charset_path)
        || !require_file(image_path)
    {
        return;
    }

    let config = OcrEngineConfig::fast().with_min_result_confidence(0.0);
    let engine = OcrEngine::new(det_path, rec_path, charset_path, Some(config))
        .unwrap_or_else(|e| panic!("[{}] 引擎创建失败: {:?}", label, e));

    let image = image::open(image_path)
        .unwrap_or_else(|e| panic!("[{}] 无法打开 {}: {:?}", label, image_path, e));
    let results = engine
        .recognize(&image)
        .unwrap_or_else(|e| panic!("[{}] 识别失败: {:?}", label, e));

    println!(
        "[{}:{}] 识别到 {} 个文本区域",
        label,
        image_path,
        results.len()
    );
    for r in &results {
        println!("  text={:?}  confidence={:.4}", r.text, r.confidence);
    }

    assert!(!results.is_empty(), "[{}] 测试图片应该检测到文本", label);

    for r in &results {
        assert!(
            r.confidence >= 0.0 && r.confidence <= 1.0,
            "[{}] 置信度应在 [0,1] 范围内，实际: {}",
            label,
            r.confidence
        );
    }
}

fn assert_fixture_recognizes_text(
    det_path: &str,
    rec_path: &str,
    charset_path: &str,
    image_path: &str,
    label: &str,
) {
    if !require_file(det_path)
        || !require_file(rec_path)
        || !require_file(charset_path)
        || !require_file(image_path)
    {
        return;
    }

    let config = OcrEngineConfig::fast().with_min_result_confidence(0.0);
    let engine = OcrEngine::new(det_path, rec_path, charset_path, Some(config))
        .unwrap_or_else(|e| panic!("[{}] 引擎创建失败: {:?}", label, e));
    let image = image::open(image_path).expect("无法打开多语言测试图片");
    let results = engine
        .recognize(&image)
        .unwrap_or_else(|e| panic!("[{}] 多语言 fixture 识别失败: {:?}", label, e));

    println!(
        "[{}] fixture={} results={}",
        label,
        image_path,
        results.len()
    );
    for result in &results {
        println!(
            "  text={:?} confidence={:.4}",
            result.text, result.confidence
        );
        assert!(
            result.confidence >= 0.0 && result.confidence <= 1.0,
            "[{}] 置信度应在 [0,1] 范围内，实际: {}",
            label,
            result.confidence
        );
        assert!(result.bbox.area() > 0, "[{}] bbox 面积应大于 0", label);
    }

    let joined = results
        .iter()
        .map(|result| result.text.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    assert!(
        !joined.trim().is_empty(),
        "[{}] 多语言 fixture 应至少识别出一段文本",
        label
    );
    assert!(
        joined.chars().any(|ch| ch.is_ascii_digit()),
        "[{}] 多语言 fixture 应识别出图片中的数字，实际: {:?}",
        label,
        joined
    );
}

fn legacy_crop_pipeline(engine: &OcrEngine, image: &image::DynamicImage) -> Vec<(String, usize)> {
    let detections = engine
        .det_model()
        .detect_and_crop(image)
        .expect("legacy detect_and_crop failed");
    let (images, boxes): (Vec<_>, Vec<_>) = detections.into_iter().unzip();
    let rec_results = engine
        .rec_model()
        .recognize_batch(&images)
        .expect("legacy recognize_batch failed");

    rec_results
        .into_iter()
        .zip(boxes)
        .filter(|(rec, _)| !rec.text.is_empty())
        .map(|(rec, bbox)| (rec.text, bbox.area() as usize))
        .collect()
}

/// 通用：仅检测模型测试
fn run_det_only(det_path: &str, label: &str) {
    if !require_file(det_path) || !require_file(TEST_IMAGE) {
        return;
    }

    let det = DetModel::from_file(det_path, None)
        .unwrap_or_else(|e| panic!("[{}] 检测模型创建失败: {:?}", label, e));

    let image = load_test_image();
    let boxes = det
        .detect(&image)
        .unwrap_or_else(|e| panic!("[{}] 检测失败: {:?}", label, e));

    println!("[{}] 检测到 {} 个文本框", label, boxes.len());
    assert!(!boxes.is_empty(), "[{}] 测试图片应该检测到文本框", label);
}

/// 通用：仅识别模型测试（使用默认 v5 det 做前置检测）
fn run_rec_only(rec_path: &str, charset_path: &str, label: &str) {
    if !require_file(DET_V5)
        || !require_file(rec_path)
        || !require_file(charset_path)
        || !require_file(TEST_IMAGE)
    {
        return;
    }

    let det = DetModel::from_file(DET_V5, None)
        .unwrap_or_else(|e| panic!("[{}] 检测模型创建失败: {:?}", label, e));
    let rec = RecModel::from_file(rec_path, charset_path, None)
        .unwrap_or_else(|e| panic!("[{}] 识别模型创建失败: {:?}", label, e));

    let image = load_test_image();
    let detections = det
        .detect_and_crop(&image)
        .unwrap_or_else(|e| panic!("[{}] 检测裁剪失败: {:?}", label, e));

    assert!(!detections.is_empty(), "[{}] 应该检测到文本区域", label);

    let images: Vec<_> = detections.iter().map(|(img, _)| img.clone()).collect();
    let results = rec
        .recognize_batch(&images)
        .unwrap_or_else(|e| panic!("[{}] 批量识别失败: {:?}", label, e));

    println!("[{}] 批量识别 {} 个区域", label, results.len());
    for r in &results {
        println!("  text={:?}  confidence={:.4}", r.text, r.confidence);
    }

    assert_eq!(
        results.len(),
        images.len(),
        "[{}] 结果数量应与输入一致",
        label
    );
}

// ============================================================
// 检测模型测试
// ============================================================

#[test]
fn test_det_v5() {
    run_det_only(DET_V5, "det-v5");
}

#[test]
fn test_det_v5_fp16() {
    run_det_only(DET_V5_FP16, "det-v5-fp16");
}

#[test]
fn test_det_v4() {
    run_det_only(DET_V4, "det-v4");
}

#[test]
fn test_det_v6_tiny() {
    run_det_only(DET_V6_TINY, "det-v6-tiny");
}

#[test]
fn test_det_v6_small() {
    run_det_only(DET_V6_SMALL, "det-v6-small");
}

#[test]
fn test_det_v6_medium() {
    run_det_only(DET_V6_MEDIUM, "det-v6-medium");
}

// ============================================================
// 识别模型测试（仅识别，使用 v5 det 做前置检测）
// ============================================================

#[test]
fn test_rec_v5() {
    run_rec_only(REC_V5, CHARSET_V5, "rec-v5");
}

#[test]
fn test_rec_v5_fp16() {
    run_rec_only(REC_V5_FP16, CHARSET_V5, "rec-v5-fp16");
}

#[test]
fn test_rec_v4() {
    run_rec_only(REC_V4, CHARSET_V4, "rec-v4");
}

#[test]
fn test_rec_v6_tiny() {
    run_rec_only(REC_V6_TINY, CHARSET_V6_TINY, "rec-v6-tiny");
}

#[test]
fn test_rec_v6_small() {
    run_rec_only(REC_V6_SMALL, CHARSET_V6_SMALL, "rec-v6-small");
}

#[test]
fn test_rec_v6_medium() {
    run_rec_only(REC_V6_MEDIUM, CHARSET_V6_MEDIUM, "rec-v6-medium");
}

#[test]
fn test_rec_en() {
    run_rec_only(REC_EN, CHARSET_EN, "rec-en");
}

#[test]
fn test_rec_arabic() {
    run_rec_only(REC_ARABIC, CHARSET_ARABIC, "rec-arabic");
}

#[test]
fn test_rec_cyrillic() {
    run_rec_only(REC_CYRILLIC, CHARSET_CYRILLIC, "rec-cyrillic");
}

#[test]
fn test_rec_devanagari() {
    run_rec_only(REC_DEVANAGARI, CHARSET_DEVANAGARI, "rec-devanagari");
}

#[test]
fn test_rec_el() {
    run_rec_only(REC_EL, CHARSET_EL, "rec-el");
}

#[test]
fn test_rec_eslav() {
    run_rec_only(REC_ESLAV, CHARSET_ESLAV, "rec-eslav");
}

#[test]
fn test_rec_korean() {
    run_rec_only(REC_KOREAN, CHARSET_KOREAN, "rec-korean");
}

#[test]
fn test_rec_latin() {
    run_rec_only(REC_LATIN, CHARSET_LATIN, "rec-latin");
}

#[test]
fn test_rec_ta() {
    run_rec_only(REC_TA, CHARSET_TA, "rec-ta");
}

#[test]
fn test_rec_te() {
    run_rec_only(REC_TE, CHARSET_TE, "rec-te");
}

#[test]
fn test_rec_th() {
    run_rec_only(REC_TH, CHARSET_TH, "rec-th");
}

// ============================================================
// 完整 OCR Pipeline 测试（det + rec 全组合）
// ============================================================

#[test]
fn test_pipeline_v5_det_v5_rec() {
    run_full_pipeline(DET_V5, REC_V5, CHARSET_V5, "pipeline-v5+v5");
}

#[test]
fn test_pipeline_v5_det_v5_rec_fp16() {
    run_full_pipeline(DET_V5, REC_V5_FP16, CHARSET_V5, "pipeline-v5+v5fp16");
}

#[test]
fn test_pipeline_v5fp16_det_v5_rec() {
    run_full_pipeline(DET_V5_FP16, REC_V5, CHARSET_V5, "pipeline-v5fp16+v5");
}

#[test]
fn test_pipeline_v4_det_v4_rec() {
    run_full_pipeline(DET_V4, REC_V4, CHARSET_V4, "pipeline-v4+v4");
}

#[test]
fn test_pipeline_v6_tiny() {
    run_full_pipeline(
        DET_V6_TINY,
        REC_V6_TINY,
        CHARSET_V6_TINY,
        "pipeline-v6-tiny",
    );
}

#[test]
fn test_pipeline_v6_small() {
    run_full_pipeline(
        DET_V6_SMALL,
        REC_V6_SMALL,
        CHARSET_V6_SMALL,
        "pipeline-v6-small",
    );
}

#[test]
fn test_pipeline_v6_medium() {
    run_full_pipeline(
        DET_V6_MEDIUM,
        REC_V6_MEDIUM,
        CHARSET_V6_MEDIUM,
        "pipeline-v6-medium",
    );
}

#[test]
fn test_pipeline_v5_det_en_rec() {
    run_full_pipeline(DET_V5, REC_EN, CHARSET_EN, "pipeline-v5+en");
}

#[test]
fn test_pipeline_v5_det_arabic_rec() {
    run_full_pipeline(DET_V5, REC_ARABIC, CHARSET_ARABIC, "pipeline-v5+arabic");
}

#[test]
fn test_pipeline_v5_det_cyrillic_rec() {
    run_full_pipeline(
        DET_V5,
        REC_CYRILLIC,
        CHARSET_CYRILLIC,
        "pipeline-v5+cyrillic",
    );
}

#[test]
fn test_pipeline_v5_det_devanagari_rec() {
    run_full_pipeline(
        DET_V5,
        REC_DEVANAGARI,
        CHARSET_DEVANAGARI,
        "pipeline-v5+devanagari",
    );
}

#[test]
fn test_pipeline_v5_det_el_rec() {
    run_full_pipeline(DET_V5, REC_EL, CHARSET_EL, "pipeline-v5+el");
}

#[test]
fn test_pipeline_v5_det_eslav_rec() {
    run_full_pipeline(DET_V5, REC_ESLAV, CHARSET_ESLAV, "pipeline-v5+eslav");
}

#[test]
fn test_pipeline_v5_det_korean_rec() {
    run_full_pipeline(DET_V5, REC_KOREAN, CHARSET_KOREAN, "pipeline-v5+korean");
}

#[test]
fn test_pipeline_v5_det_latin_rec() {
    run_full_pipeline(DET_V5, REC_LATIN, CHARSET_LATIN, "pipeline-v5+latin");
}

#[test]
fn test_pipeline_v5_det_ta_rec() {
    run_full_pipeline(DET_V5, REC_TA, CHARSET_TA, "pipeline-v5+ta");
}

#[test]
fn test_pipeline_v5_det_te_rec() {
    run_full_pipeline(DET_V5, REC_TE, CHARSET_TE, "pipeline-v5+te");
}

#[test]
fn test_pipeline_v5_det_th_rec() {
    run_full_pipeline(DET_V5, REC_TH, CHARSET_TH, "pipeline-v5+th");
}

#[test]
fn test_v6_tiers_on_multiple_document_images() {
    for image_path in TEST_IMAGES {
        run_full_pipeline_on_image(
            DET_V6_TINY,
            REC_V6_TINY,
            CHARSET_V6_TINY,
            image_path,
            "pipeline-v6-tiny-multi-image",
        );
        run_full_pipeline_on_image(
            DET_V6_SMALL,
            REC_V6_SMALL,
            CHARSET_V6_SMALL,
            image_path,
            "pipeline-v6-small-multi-image",
        );
    }
}

#[test]
fn test_multilingual_fixture_pipelines() {
    let cases = [
        (
            DET_V6_SMALL,
            REC_V6_SMALL,
            CHARSET_V6_SMALL,
            "tests/fixtures/multilingual/latin.png",
            "fixture-v6-latin",
        ),
        (
            DET_V6_SMALL,
            REC_V6_SMALL,
            CHARSET_V6_SMALL,
            "tests/fixtures/multilingual/cjk.png",
            "fixture-v6-cjk",
        ),
        (
            DET_V5,
            REC_KOREAN,
            CHARSET_KOREAN,
            "tests/fixtures/multilingual/korean.png",
            "fixture-korean",
        ),
        (
            DET_V5,
            REC_ARABIC,
            CHARSET_ARABIC,
            "tests/fixtures/multilingual/arabic.png",
            "fixture-arabic",
        ),
        (
            DET_V5,
            REC_CYRILLIC,
            CHARSET_CYRILLIC,
            "tests/fixtures/multilingual/cyrillic.png",
            "fixture-cyrillic",
        ),
        (
            DET_V5,
            REC_DEVANAGARI,
            CHARSET_DEVANAGARI,
            "tests/fixtures/multilingual/devanagari.png",
            "fixture-devanagari",
        ),
        (
            DET_V5,
            REC_TH,
            CHARSET_TH,
            "tests/fixtures/multilingual/thai.png",
            "fixture-thai",
        ),
        (
            DET_V5,
            REC_EL,
            CHARSET_EL,
            "tests/fixtures/multilingual/greek.png",
            "fixture-greek",
        ),
        (
            DET_V5,
            REC_TA,
            CHARSET_TA,
            "tests/fixtures/multilingual/tamil.png",
            "fixture-tamil",
        ),
        (
            DET_V5,
            REC_TE,
            CHARSET_TE,
            "tests/fixtures/multilingual/telugu.png",
            "fixture-telugu",
        ),
    ];

    for (det_path, rec_path, charset_path, image_path, label) in cases {
        assert_fixture_recognizes_text(det_path, rec_path, charset_path, image_path, label);
    }
}

#[test]
fn test_engine_pipeline_matches_public_crop_pipeline() {
    if !require_file(DET_V6_TINY)
        || !require_file(REC_V6_TINY)
        || !require_file(CHARSET_V6_TINY)
        || !require_file(TEST_IMAGE)
    {
        return;
    }

    let config = OcrEngineConfig::fast()
        .with_parallel(false)
        .with_min_result_confidence(0.0);
    let engine = OcrEngine::new(DET_V6_TINY, REC_V6_TINY, CHARSET_V6_TINY, Some(config))
        .expect("OCR 引擎创建失败");
    let image = load_test_image();

    let engine_results = engine.recognize(&image).expect("engine recognize failed");
    let legacy_results = legacy_crop_pipeline(&engine, &image);

    let engine_texts = engine_results
        .iter()
        .filter(|result| !result.text.is_empty())
        .map(|result| (result.text.clone(), result.bbox.area() as usize))
        .collect::<Vec<_>>();

    assert_eq!(
        engine_texts, legacy_results,
        "OcrEngine::recognize 应与公开 detect_and_crop + recognize_batch 路径保持一致"
    );
}

#[test]
fn test_parallel_engine_pipeline_preserves_page_goldens_and_order() {
    if !require_file(DET_V6_TINY) || !require_file(REC_V6_TINY) || !require_file(CHARSET_V6_TINY) {
        return;
    }

    let config = OcrEngineConfig::fast()
        .with_parallel(true)
        .with_min_result_confidence(0.0);
    let parallel_engine = OcrEngine::new(DET_V6_TINY, REC_V6_TINY, CHARSET_V6_TINY, Some(config))
        .expect("OCR 引擎创建失败");
    let batch_config = OcrEngineConfig::fast()
        .with_parallel(false)
        .with_min_result_confidence(0.0);
    let batch_engine = OcrEngine::new(
        DET_V6_TINY,
        REC_V6_TINY,
        CHARSET_V6_TINY,
        Some(batch_config),
    )
    .expect("batch OCR 引擎创建失败");
    let golden_fragments = [
        (
            "res/1.png",
            [
                "The dominant sequence transduction models",
                "Transformer generalizes",
            ],
        ),
        (
            "res/2.png",
            ["Attention Is All You Need", "Submission history"],
        ),
        ("res/3.png", ["镀膜技术暗战", "2025年初上海眼镜展"]),
        ("res/4.png", ["FLD", "PP-LiteSeg"]),
    ];

    for image_path in TEST_IMAGES {
        if !require_file(image_path) {
            continue;
        }

        let image = image::open(image_path).expect("无法加载页面测试图片");
        let parallel_results = parallel_engine
            .recognize(&image)
            .expect("parallel recognize failed");
        let batch_results = batch_engine
            .recognize(&image)
            .expect("batch recognize failed");
        let batch_boxes = batch_results
            .iter()
            .map(|result| {
                (
                    result.bbox.rect.left(),
                    result.bbox.rect.top(),
                    result.bbox.rect.width(),
                    result.bbox.rect.height(),
                )
            })
            .collect::<Vec<_>>();
        let parallel_boxes = parallel_results
            .iter()
            .map(|result| {
                (
                    result.bbox.rect.left(),
                    result.bbox.rect.top(),
                    result.bbox.rect.width(),
                    result.bbox.rect.height(),
                )
            })
            .collect::<Vec<_>>();

        assert_eq!(
            parallel_boxes, batch_boxes,
            "parallel exact-width dispatch changed result order or boxes for {image_path}"
        );

        let joined = parallel_results
            .iter()
            .map(|result| result.text.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        let (_, expected_fragments) = golden_fragments
            .iter()
            .find(|(path, _)| path == image_path)
            .expect("missing page golden fragments");
        for fragment in expected_fragments {
            assert!(
                joined.contains(fragment),
                "parallel OCR output for {image_path} lost golden fragment {fragment:?}: {joined:?}"
            );
        }
    }
}

// ============================================================
// 方向分类模型测试
// ============================================================

#[test]
fn test_ori_model() {
    if !require_file(ORI_MODEL) || !require_file(TEST_IMAGE) {
        return;
    }

    let ori = OriModel::from_file(ORI_MODEL, None)
        .unwrap_or_else(|e| panic!("方向分类模型创建失败: {:?}", e));

    let image = load_test_image();
    let result = ori
        .classify(&image)
        .unwrap_or_else(|e| panic!("方向分类失败: {:?}", e));

    println!(
        "[ori] class_idx={}, angle={}°, confidence={:.4}",
        result.class_idx, result.angle, result.confidence
    );

    assert!(
        result.confidence >= 0.0 && result.confidence <= 1.0,
        "方向分类置信度应在 [0,1] 范围内"
    );
    // 正常英文文档应该是 0° 方向
    assert_eq!(result.angle, 0, "纯英文图片方向应为 0°");
}

// ============================================================
// 带方向模型的完整 pipeline 测试
// ============================================================

#[test]
fn test_pipeline_with_ori_model() {
    if !require_file(DET_V5)
        || !require_file(REC_V5)
        || !require_file(CHARSET_V5)
        || !require_file(ORI_MODEL)
        || !require_file(TEST_IMAGE)
    {
        return;
    }

    let engine = OcrEngine::new_with_ori(DET_V5, REC_V5, CHARSET_V5, ORI_MODEL, None)
        .expect("带方向模型的 OCR 引擎创建失败");

    let image = load_test_image();
    let results = engine.recognize(&image).expect("带方向模型识别失败");

    println!("[pipeline+ori] 识别到 {} 个文本区域", results.len());
    for r in &results {
        println!("  text={:?}  confidence={:.4}", r.text, r.confidence);
    }

    assert!(!results.is_empty(), "带方向模型应识别到文本");
}

// ============================================================
// 旋转图片方向检测测试（5.png = 1.png 顺时针旋转 90°）
// ============================================================

#[test]
fn test_ori_model_rotated_90() {
    if !require_file(ORI_MODEL) || !require_file(TEST_IMAGE_ROTATED_90) {
        return;
    }

    let ori = OriModel::from_file(ORI_MODEL, None)
        .unwrap_or_else(|e| panic!("方向分类模型创建失败: {:?}", e));

    let image = image::open(TEST_IMAGE_ROTATED_90).expect("无法打开测试图片 res/5.png");
    let result = ori
        .classify(&image)
        .unwrap_or_else(|e| panic!("方向分类失败: {:?}", e));

    println!(
        "[ori-rotated-90] class_idx={}, angle={}°, confidence={:.4}, scores={:?}",
        result.class_idx, result.angle, result.confidence, result.scores
    );

    assert!(
        result.confidence >= 0.0 && result.confidence <= 1.0,
        "方向分类置信度应在 [0,1] 范围内"
    );
    // 顺时针旋转 90° 的图片，方向模型应检测到 90°
    assert_eq!(
        result.angle, 90,
        "顺时针旋转 90° 的图片方向应为 90°，实际检测到 {}°",
        result.angle
    );
}

#[test]
fn test_pipeline_with_ori_model_rotated_90() {
    if !require_file(DET_V5)
        || !require_file(REC_V5)
        || !require_file(CHARSET_V5)
        || !require_file(ORI_MODEL)
        || !require_file(TEST_IMAGE_ROTATED_90)
    {
        return;
    }

    let engine = OcrEngine::new_with_ori(DET_V5, REC_V5, CHARSET_V5, ORI_MODEL, None)
        .expect("带方向模型的 OCR 引擎创建失败");

    // 使用旋转 90° 的图片进行识别，方向模型应自动纠正方向
    let image = image::open(TEST_IMAGE_ROTATED_90).expect("无法打开测试图片 res/5.png");
    let results = engine
        .recognize(&image)
        .expect("旋转图片带方向模型识别失败");

    println!(
        "[pipeline+ori+rotated90] 识别到 {} 个文本区域",
        results.len()
    );
    for r in &results {
        println!("  text={:?}  confidence={:.4}", r.text, r.confidence);
    }

    assert!(
        !results.is_empty(),
        "旋转 90° 的图片经方向校正后应识别到文本"
    );

    // 同时对比不带方向模型的结果
    let engine_no_ori =
        OcrEngine::new(DET_V5, REC_V5, CHARSET_V5, None).expect("无方向模型的 OCR 引擎创建失败");
    let results_no_ori = engine_no_ori
        .recognize(&image)
        .expect("旋转图片无方向模型识别失败");

    println!(
        "[pipeline+no_ori+rotated90] 无方向模型识别到 {} 个文本区域",
        results_no_ori.len()
    );
    for r in &results_no_ori {
        println!("  text={:?}  confidence={:.4}", r.text, r.confidence);
    }

    // 带方向模型的结果应该比不带方向模型的结果更好（识别出更多文本或更高置信度）
    println!(
        "\n[对比] 带方向模型: {} 个结果 vs 无方向模型: {} 个结果",
        results.len(),
        results_no_ori.len()
    );
}
