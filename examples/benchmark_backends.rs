//! GPU 后端性能测试
//!
//! GPU Backend Performance Benchmark
//!
//! 测试不同后端的推理速度,包括 CPU、Metal、OpenCL、Vulkan 等

use ocr_rs::{Backend, DetOptions, OcrEngine, OcrEngineConfig};
use std::error::Error;
use std::time::{Duration, Instant};

/// 执行多次推理并统计性能
fn benchmark_backend(
    backend: Backend,
    image_path: &str,
    det_model: &str,
    rec_model: &str,
    charset: &str,
    iterations: usize,
) -> Result<(Duration, Duration, Duration, Duration), Box<dyn Error>> {
    println!("\n{}", "=".repeat(60));
    println!("测试后端: {:?}", backend);
    println!("{}", "=".repeat(60));

    // 配置引擎
    let config = OcrEngineConfig::new()
        .with_backend(backend)
        .with_threads(4)
        .with_det_options(DetOptions::fast());

    // 创建引擎
    print!("创建引擎... ");
    let create_start = Instant::now();
    let engine = OcrEngine::new(det_model, rec_model, charset, Some(config))?;
    let create_time = create_start.elapsed();
    println!("完成 ({:?})", create_time);

    // 加载图像
    let image = image::open(image_path)?;
    println!("图像尺寸: {}x{} 像素", image.width(), image.height());

    // 预热 (首次推理通常较慢)
    print!("预热推理... ");
    let warmup_start = Instant::now();
    let _ = engine.recognize(&image)?;
    let warmup_time = warmup_start.elapsed();
    println!("完成 ({:?})", warmup_time);

    // 性能测试
    println!("\n执行 {} 次推理...", iterations);
    let mut durations = Vec::with_capacity(iterations);

    for i in 1..=iterations {
        let start = Instant::now();
        let _results = engine.recognize(&image)?;
        let duration = start.elapsed();
        durations.push(duration);

        print!("\r进度: {}/{} - 本次: {:?}", i, iterations, duration);
    }
    println!(); // 换行

    // 计算统计数据
    let total: Duration = durations.iter().sum();
    let avg = total / iterations as u32;
    let min = *durations.iter().min().unwrap();
    let max = *durations.iter().max().unwrap();

    // 输出结果
    println!("\n{}", "─".repeat(60));
    println!("性能统计:");
    println!("{}", "─".repeat(60));
    println!("  总耗时:   {:?}", total);
    println!("  平均耗时: {:?}", avg);
    println!("  最短耗时: {:?}", min);
    println!("  最长耗时: {:?}", max);
    println!("  平均吞吐: {:.2} FPS", 1000.0 / avg.as_millis() as f64);

    Ok((total, avg, min, max))
}

fn main() -> Result<(), Box<dyn Error>> {
    // 初始化日志
    env_logger::init();

    println!("\n{}", "#".repeat(60));
    println!("  OCR 后端性能基准测试");
    println!("{}", "#".repeat(60));

    // 固定配置
    let image_path = "/Users/chenzibo/git/rust-paddle-ocr/res/1.png";
    let det_model = "models/PP-OCRv5_mobile_det_fp16.mnn";
    let rec_model = "models/PP-OCRv5_mobile_rec_fp16.mnn";
    let charset = "models/ppocr_keys_v5.txt";
    let iterations = 10;

    println!("\n配置信息:");
    println!("  图像:     {}", image_path);
    println!("  检测模型: {}", det_model);
    println!("  识别模型: {}", rec_model);
    println!("  字符集:   {}", charset);
    println!("  测试次数: {}", iterations);

    // 要测试的后端列表
    let backends = vec![
        Backend::CPU,
        Backend::Metal,
        Backend::OpenCL,
        Backend::Vulkan,
    ];

    // 存储所有后端的测试结果
    let mut results = Vec::new();

    // 测试每个后端
    for backend in backends {
        match benchmark_backend(
            backend, image_path, det_model, rec_model, charset, iterations,
        ) {
            Ok((total, avg, min, max)) => {
                results.push((backend, total, avg, min, max));
            }
            Err(e) => {
                eprintln!("\n❌ 后端 {:?} 测试失败: {}", backend, e);
            }
        }
    }

    // 输出汇总对比
    if !results.is_empty() {
        println!("\n\n{}", "#".repeat(60));
        println!("  汇总对比");
        println!("{}", "#".repeat(60));
        println!(
            "\n{:<15} {:>12} {:>12} {:>12} {:>10}",
            "后端", "平均耗时", "最短耗时", "最长耗时", "吞吐(FPS)"
        );
        println!("{}", "─".repeat(60));

        for (backend, _total, avg, min, max) in &results {
            let fps = 1000.0 / avg.as_millis() as f64;
            println!(
                "{:<15} {:>12?} {:>12?} {:>12?} {:>10.2}",
                format!("{:?}", backend),
                avg,
                min,
                max,
                fps
            );
        }

        // 找出最快的后端
        if let Some((fastest_backend, _, fastest_avg, _, _)) =
            results.iter().min_by_key(|(_, _, avg, _, _)| avg)
        {
            println!(
                "\n🏆 最快后端: {:?} (平均 {:?})",
                fastest_backend, fastest_avg
            );
        }
    }

    println!("\n{}", "#".repeat(60));
    println!("  测试完成");
    println!("{}\n", "#".repeat(60));

    Ok(())
}
