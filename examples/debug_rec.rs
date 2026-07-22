//! 调试识别模型输出

use image::GenericImageView;
use ocr_rs::mnn::{InferenceConfig, InferenceEngine};
use ocr_rs::preprocess::{preprocess_for_rec, NormalizeParams};
use std::env;
use std::fs;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 4 {
        eprintln!("用法: debug_rec <模型路径> <字符集路径> <图像路径>");
        return;
    }

    let model_path = &args[1];
    let charset_path = &args[2];
    let image_path = &args[3];

    // 加载字符集
    println!("加载字符集: {}", charset_path);
    let charset_content = fs::read_to_string(charset_path).expect("读取字符集失败");
    let mut charset: Vec<char> = vec![' ']; // blank token
    for ch in charset_content.chars() {
        if ch != '\n' && ch != '\r' {
            charset.push(ch);
        }
    }
    charset.push(' '); // padding token
    println!("字符集大小: {}", charset.len());
    println!("前10个字符: {:?}", &charset[..10.min(charset.len())]);

    // 加载模型
    println!("\n加载模型: {}", model_path);
    let config = InferenceConfig::default();
    let engine = InferenceEngine::from_file(model_path, Some(config)).unwrap();

    println!("模型输入形状: {:?}", engine.input_shape());
    println!("模型输出形状: {:?}", engine.output_shape());
    println!("是否动态形状: {}", engine.has_dynamic_shape());

    // 加载图像
    println!("\n加载图像: {}", image_path);
    let image = image::open(image_path).expect("加载图像失败");
    let (w, h) = image.dimensions();
    println!("图像尺寸: {}x{}", w, h);

    // 预处理
    let target_height = 48u32;
    let params = NormalizeParams::paddle_rec();
    let input = preprocess_for_rec(&image, target_height, &params).expect("预处理失败");
    println!("输入张量形状: {:?}", input.shape());

    // 推理
    println!("\n执行推理...");
    let output = engine.run_dynamic(input.view().into_dyn()).unwrap();
    let output_shape = output.shape();
    println!("输出张量形状: {:?}", output_shape);

    // 分析输出
    let output_data: Vec<f32> = output.iter().cloned().collect();
    let min_val = output_data.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_val = output_data
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);
    println!("输出值范围: [{:.4}, {:.4}]", min_val, max_val);

    // 解析输出形状
    let (seq_len, num_classes) = if output_shape.len() == 3 {
        (output_shape[1], output_shape[2])
    } else if output_shape.len() == 2 {
        (output_shape[0], output_shape[1])
    } else {
        eprintln!("无效的输出形状!");
        return;
    };
    println!("序列长度: {}, 类别数: {}", seq_len, num_classes);
    println!("字符集大小: {} (应该接近类别数)", charset.len());

    // CTC 解码
    println!("\n=== CTC 解码 ===");
    let mut decoded_text = String::new();
    let mut prev_idx = 0usize;
    let mut char_details = Vec::new();

    for t in 0..seq_len {
        let start = t * num_classes;
        let end = start + num_classes;
        let probs = &output_data[start..end];

        // 找最大值
        let (max_idx, &max_val) = probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();

        // 计算 softmax 分数
        let max_logit = probs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let sum_exp: f32 = probs.iter().map(|&x| (x - max_logit).exp()).sum();
        let score = (max_val - max_logit).exp() / sum_exp;

        if t < 10 || (max_idx != 0 && max_idx != prev_idx) {
            let ch = if max_idx < charset.len() {
                charset[max_idx]
            } else {
                '?'
            };
            if t < 10 {
                println!(
                    "  t={:3}: idx={:5}, val={:8.4}, score={:.4}, char='{}'",
                    t, max_idx, max_val, score, ch
                );
            }
        }

        // CTC 解码规则
        if max_idx != 0 && max_idx != prev_idx && max_idx < charset.len() {
            let ch = charset[max_idx];
            decoded_text.push(ch);
            char_details.push((ch, score));
        }

        prev_idx = max_idx;
    }

    println!("\n=== 解码结果 ===");
    println!("文本: '{}'", decoded_text);
    println!("字符数: {}", decoded_text.len());
    if !char_details.is_empty() {
        println!("字符详情 (前20个):");
        for (i, (ch, score)) in char_details.iter().take(20).enumerate() {
            println!("  [{:2}] '{}' ({:.2}%)", i, ch, score * 100.0);
        }
    }
}
