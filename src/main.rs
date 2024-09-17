use onnxruntime::ndarray::prelude::*;
use onnxruntime::ndarray::{ArrayD, Axis, IxDyn}; // Import necessary types
use onnxruntime::GraphOptimizationLevel;
use onnxruntime::{environment::Environment, tensor::OrtOwnedTensor, LoggingLevel};
use std::error::Error;
use tokenizers::Encoding;
use tokenizers::Tokenizer;

fn cumulative_sum(arr: &ArrayD<i32>, axis: Axis) -> ArrayD<i32> {
    let mut result = arr.clone();

    for mut row in result.axis_iter_mut(axis) {
        for i in 1..row.len() {
            row[i] += row[i - 1];
        }
    }
    result
}

fn get_tokenizer(model_name_or_path: &str) -> Result<Tokenizer, Box<dyn Error + Send + Sync>> {
    let mut tokenizer = Tokenizer::from_pretrained(model_name_or_path, None)?;
    let pad_id = tokenizer.get_vocab(false)["<|endoftext|>"];
    tokenizer.with_padding(Some(tokenizers::PaddingParams {
        strategy: tokenizers::PaddingStrategy::BatchLongest,
        direction: tokenizers::PaddingDirection::Left,
        pad_to_multiple_of: None,
        pad_id: pad_id,
        pad_type_id: 0,
        pad_token: "<|endoftext|>".to_string(),
    }));

    Ok(tokenizer)
}

fn get_example_inputs(
    prompt_text: &str,
) -> (ArrayD<i32>, ArrayD<i32>, ArrayD<i32>, Vec<ArrayD<i32>>) {
    let num_attention_heads = 12;
    let num_layers = 12;
    let hidden_size = 768;

    let tokenizer = get_tokenizer("gpt2").expect("Failed to load tokenizer");

    let encoding: Encoding = tokenizer.encode(prompt_text, true).unwrap();

    let input_ids: Vec<u32> = encoding.get_ids().to_vec();
    let attention_mask: Vec<u32> = encoding.get_attention_mask().to_vec();

    let input_ids = ArrayD::from_shape_vec(IxDyn(&[1, input_ids.len()]), input_ids).unwrap();
    let attention_mask =
        ArrayD::from_shape_vec(IxDyn(&[1, attention_mask.len()]), attention_mask).unwrap();

    let mut position_ids = cumulative_sum(&attention_mask.mapv(|mask| mask as i32), Axis(1));
    position_ids.iter_mut().for_each(|x| *x = (*x).max(0));

    let mut empty_past: Vec<ArrayD<i32>> = Vec::new();
    let batch_size = input_ids.shape()[0];
    let hidden_size_per_head = hidden_size / num_attention_heads;

    let past_shape = IxDyn(&[2, batch_size, num_attention_heads, 0, hidden_size_per_head]);

    for _ in 0..num_layers {
        empty_past.push(ArrayD::zeros(past_shape.clone()));
    }

    (
        input_ids.mapv(|x| x as i32),
        attention_mask.mapv(|x| x as i32),
        position_ids,
        empty_past,
    )
}

fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    // Initialize the ONNX environment
    let environment = Environment::builder()
        .with_name("test")
        .with_log_level(LoggingLevel::Verbose)
        .build()?;

    // Load the ONNX model session
    let mut session = environment
        .new_session_builder()?
        .with_optimization_level(GraphOptimizationLevel::Basic)?
        .with_number_threads(1)?
        .with_model_from_file("models/gpt2_onnxv2/gpt2.onnx")?;

    let input_prompt = "Hello";
    let (input_ids, attention_mask, position_ids, empty_past) = get_example_inputs(input_prompt);

    let mut ort_inputs: Vec<ArrayD<i32>> = Vec::new();

    ort_inputs.push(input_ids);
    ort_inputs.push(attention_mask);
    ort_inputs.push(position_ids);

    for past_i in empty_past.iter() {
        ort_inputs.push(past_i.clone());
    }

    // Run inference
    let ort_outputs: Vec<OrtOwnedTensor<f32, IxDyn>> = session.run(ort_inputs)?;

    println!("Model output: {:?}", ort_outputs);

    Ok(())
}
