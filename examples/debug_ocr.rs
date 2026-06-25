//! OCR 调试示例
//!
//! 功能：
//! 1. 可视化文本检测框（绘制并保存）
//! 2. 输出详细的识别结果（文本、置信度、坐标）
//! 3. 适用于调试和验证 OCR 流程

use image::{GenericImageView, Rgb, RgbImage};
use imageproc::drawing::{draw_hollow_rect_mut, draw_line_segment_mut};
use imageproc::rect::Rect;
use ocr_rs::{OcrEngine, OcrEngineConfig};
use std::env;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 初始化日志
    env_logger::init();

    // 解析命令行参数
    let args: Vec<String> = env::args().collect();
    if args.len() < 5 {
        eprintln!("用法: debug_ocr <det_model> <rec_model> <keys> <image> [output]");
        eprintln!("\n示例:");
        eprintln!("  cargo run --example debug_ocr -- \\");
        eprintln!("    models/PP-OCRv5_mobile_det.mnn \\");
        eprintln!("    models/PP-OCRv5_mobile_rec.mnn \\");
        eprintln!("    models/ppocr_keys_v5.txt \\");
        eprintln!("    res/test.png \\");
        eprintln!("    output_debug.png");
        return Ok(());
    }

    let det_model = &args[1];
    let rec_model = &args[2];
    let keys_path = &args[3];
    let image_path = &args[4];
    let output_path = args
        .get(5)
        .map(|s| s.as_str())
        .unwrap_or("debug_ocr_result.png");

    println!("╔════════════════════════════════════════════╗");
    println!("║       OCR 调试工具 - Debug Tool          ║");
    println!("╚════════════════════════════════════════════╝\n");

    // 1. 加载模型
    println!("📦 加载模型...");
    println!("   检测模型: {}", det_model);
    println!("   识别模型: {}", rec_model);
    println!("   字符集:   {}", keys_path);

    let config = OcrEngineConfig::fast().with_min_result_confidence(0.7);
    let engine = OcrEngine::new(det_model, rec_model, keys_path, Some(config))?;
    println!("   ✅ 模型加载成功");

    // 2. 加载图像
    println!("🖼️  加载图像: {}", image_path);
    let image = image::open(image_path)?;
    let (width, height) = image.dimensions();
    println!("   尺寸: {}x{}\n", width, height);

    // 3. 执行 OCR 识别
    println!("🔍 执行 OCR 识别...");
    let results = engine.recognize(&image)?;
    println!("   ✅ 检测到 {} 个文本区域\n", results.len());

    // 4. 输出详细识别结果到命令行
    println!("╔════════════════════════════════════════════════════════════════════════╗");
    println!("║                        识别结果详情                                    ║");
    println!("╠════════════════════════════════════════════════════════════════════════╣");

    for (i, result) in results.iter().enumerate() {
        let bbox = &result.bbox;
        println!("📝 [{:2}] 文本: {}", i + 1, result.text);
        println!(
            "   置信度: {:.2}% | 位置: ({}, {}) | 尺寸: {}x{}",
            result.confidence * 100.0,
            bbox.rect.left(),
            bbox.rect.top(),
            bbox.rect.width(),
            bbox.rect.height()
        );

        // 如果有四个角点，也输出
        if let Some(points) = &bbox.points {
            println!(
                "   角点: [{:.0},{:.0}] [{:.0},{:.0}] [{:.0},{:.0}] [{:.0},{:.0}]",
                points[0].x,
                points[0].y,
                points[1].x,
                points[1].y,
                points[2].x,
                points[2].y,
                points[3].x,
                points[3].y
            );
        }
        println!();
    }

    println!("╚════════════════════════════════════════════════════════════════════════╝\n");

    // 5. 可视化：绘制边界框到图像
    println!("🎨 生成可视化结果...");
    let mut output_image = image.to_rgb8();

    // 预定义颜色方案（8种明亮的颜色）
    let colors = [
        Rgb([255u8, 0, 0]), // 红色
        Rgb([0, 255, 0]),   // 绿色
        Rgb([0, 0, 255]),   // 蓝色
        Rgb([255, 255, 0]), // 黄色
        Rgb([255, 0, 255]), // 品红
        Rgb([0, 255, 255]), // 青色
        Rgb([255, 128, 0]), // 橙色
        Rgb([128, 0, 255]), // 紫色
    ];

    for (i, result) in results.iter().enumerate() {
        let color = colors[i % colors.len()];
        let bbox = &result.bbox;

        if let Some(points) = &bbox.points {
            for idx in 0..4 {
                let start = points[idx];
                let end = points[(idx + 1) % 4];
                draw_line_segment_mut(&mut output_image, (start.x, start.y), (end.x, end.y), color);
                draw_line_segment_mut(
                    &mut output_image,
                    (start.x + 1.0, start.y + 1.0),
                    (end.x + 1.0, end.y + 1.0),
                    color,
                );
            }
        } else {
            // 绘制矩形边框（绘制2次让边框更明显）
            let rect = Rect::at(bbox.rect.left(), bbox.rect.top())
                .of_size(bbox.rect.width(), bbox.rect.height());

            draw_hollow_rect_mut(&mut output_image, rect, color);

            // 绘制加粗边框
            if bbox.rect.left() > 0 && bbox.rect.top() > 0 {
                let rect2 = Rect::at(bbox.rect.left() - 1, bbox.rect.top() - 1)
                    .of_size(bbox.rect.width() + 2, bbox.rect.height() + 2);
                draw_hollow_rect_mut(&mut output_image, rect2, color);
            }
        }

        // 可选：绘制索引标签（如果需要在图像上显示序号）
        draw_index_label(
            &mut output_image,
            i + 1,
            bbox.rect.left(),
            bbox.rect.top(),
            color,
        );
    }

    // 6. 保存可视化结果
    output_image.save(output_path)?;
    println!("   ✅ 可视化结果已保存到: {}\n", output_path);

    // 7. 统计信息
    println!("📊 统计信息:");
    if !results.is_empty() {
        let avg_confidence =
            results.iter().map(|r| r.confidence).sum::<f32>() / results.len() as f32;
        let max_confidence = results
            .iter()
            .map(|r| r.confidence)
            .fold(0.0f32, |a, b| a.max(b));
        let min_confidence = results
            .iter()
            .map(|r| r.confidence)
            .fold(1.0f32, |a, b| a.min(b));

        println!("   总文本区域数: {}", results.len());
        println!("   平均置信度:   {:.2}%", avg_confidence * 100.0);
        println!("   最高置信度:   {:.2}%", max_confidence * 100.0);
        println!("   最低置信度:   {:.2}%", min_confidence * 100.0);
    } else {
        println!("   未检测到任何文本");
    }

    println!("\n✨ 调试完成！");
    Ok(())
}

/// 在图像上绘制索引标签
fn draw_index_label(image: &mut RgbImage, _index: usize, x: i32, y: i32, color: Rgb<u8>) {
    // 计算标签位置（稍微偏移到框的左上角外侧）
    let label_x = (x - 20).max(0);
    let label_y = (y - 20).max(0);

    // 绘制标签背景（小方块）
    let label_size = 18;
    for dy in 0..label_size {
        for dx in 0..label_size {
            let px = label_x + dx;
            let py = label_y + dy;
            if px >= 0 && py >= 0 && (px as u32) < image.width() && (py as u32) < image.height() {
                image.put_pixel(px as u32, py as u32, color);
            }
        }
    }
}
