use anyhow::Result;
use ndarray_npy::read_npy;
use rust_tokenizers::tokenizer::{Gpt2Tokenizer, Tokenizer, TruncationStrategy};
use tract_ndarray::{s, ArrayD, ArrayViewD, IxDyn};
use tract_onnx::prelude::Datum;
use tract_onnx::prelude::*;
struct Config {
    target_len: usize,
    model_path: String,
    tokenizer_vocab_path: String,
    tokenizer_merges_path: String,
    tokenizer_special_tokens_path: String,
}

impl Config {
    fn new() -> Self {
        Self {
            target_len: 256,
            model_path: "models/gpt2_with_kv_logits.onnx".to_string(),
            tokenizer_vocab_path: "tokenizer/vocab.json".to_string(),
            tokenizer_merges_path: "tokenizer/merges.txt".to_string(),
            tokenizer_special_tokens_path: "tokenizer/special_tokens_map.json".to_string(),
        }
    }
}

struct TextGenerator {
    model: SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>,
    tokenizer: Gpt2Tokenizer,
    config: Config,
}

impl TextGenerator {
    fn new(config: Config) -> Result<Self> {
        let model = tract_onnx::onnx()
            .model_for_path(&config.model_path)?
            .into_optimized()?
            .into_runnable()?;

        let tokenizer = Gpt2Tokenizer::from_file_with_special_token_mapping(
            &config.tokenizer_vocab_path,
            &config.tokenizer_merges_path,
            false,
            &config.tokenizer_special_tokens_path,
        )?;

        Ok(Self {
            model,
            tokenizer,
            config,
        })
    }

    fn generate(&self, input: &str, num_tokens: usize) -> Result<String> {
        let (mut input_ids, mut attention_mask) = self.get_input_ids(input)?;
        let mut generated_tokens = input_ids.clone();
        let mut past_key_values_tensor = create_empty_past_key_values(24, 1, 12, 0, 64)?;

        for _ in 0..num_tokens {
            let input_ids_tensor = create_tensor(&input_ids)?;
            let attention_mask_tensor = create_tensor(&attention_mask)?;

            let inputs: TVec<TValue> = tvec!(
                input_ids_tensor.into(),
                attention_mask_tensor.into(),
                past_key_values_tensor.clone().into()
            );

            let outputs = self.model.run(inputs)?;

            let logits = outputs[0].to_array_view::<f32>()?.to_owned();
            past_key_values_tensor = outputs[1].clone().into_tensor();

            // Apply top-k and top-p filtering
            let k_filtered_logits = top_k_filtering(&logits, 50);
            let p_filtered_logits = top_p_filtering(&k_filtered_logits, 0.9);

            // Use argmax to get the next token ID
            let next_token_id = argmax(&p_filtered_logits);

            generated_tokens.push(next_token_id);
            let output = self.tokenizer.decode(&generated_tokens, false, true);
            println!("{}", output);
            input_ids = vec![next_token_id];
            attention_mask.push(1);
        }

        Ok(self.tokenizer.decode(&generated_tokens, true, true))
    }

    fn get_input_ids(&self, text: &str) -> Result<(Vec<i64>, Vec<i8>)> {
        let tokens = self.tokenizer.encode(
            text,
            None,
            self.config.target_len,
            &TruncationStrategy::DoNotTruncate,
            0,
        );

        let input_ids = tokens.token_ids.clone();
        let attention_mask: Vec<i8> = tokens.token_ids.iter().map(|_| 1).collect();

        Ok((input_ids, attention_mask))
    }
}

fn softmax(logits: &mut ArrayViewD<f32>) -> ArrayD<f32> {
    let max = logits.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp: ArrayD<f32> = logits.mapv(|x| (x - max).exp());
    let sum = exp.sum();
    exp / sum
}
fn top_k_filtering(logits: &ArrayD<f32>, top_k: usize) -> ArrayD<f32> {
    let shape = logits.shape();
    let vocab_size = shape[2];
    let mut filtered_logits = logits.clone(); // Create a separate copy to modify

    for i in 0..shape[0] {
        for j in 0..shape[1] {
            // Create a vector of (index, value) pairs for sorting
            let mut logit_values: Vec<(usize, f32)> =
                (0..vocab_size).map(|k| (k, logits[[i, j, k]])).collect();

            // Sort by logit values in descending order
            logit_values.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            // Collect indices that should be set to NEG_INFINITY beyond top_k
            let indices_to_neg_inf = logit_values.iter().skip(top_k).map(|&(idx, _)| idx);

            // Set the identified indices to NEG_INFINITY in the filtered logits
            for index in indices_to_neg_inf {
                filtered_logits[[i, j, index]] = f32::NEG_INFINITY;
            }
        }
    }

    filtered_logits
}

fn top_p_filtering(logits: &ArrayD<f32>, top_p: f32) -> ArrayD<f32> {
    let shape = logits.shape();
    let vocab_size = shape[2];
    let mut filtered_logits = logits.clone(); // Create a mutable copy to apply changes

    for i in 0..shape[0] {
        for j in 0..shape[1] {
            // Extract the logits slice and convert it to a vector
            let mut logit_values: Vec<(usize, f32)> = logits
                .slice(s![i, j, ..])
                .iter()
                .cloned()
                .enumerate()
                .collect();

            // Sort logit values in descending order
            logit_values.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            // Calculate cumulative probabilities
            let mut cumulative_probs = 0.0;
            for (k, &(_, logit)) in logit_values.iter().enumerate() {
                cumulative_probs += logit.exp();
                if cumulative_probs > top_p {
                    // Set logits to NEG_INFINITY beyond the cutoff
                    for &(index, _) in &logit_values[k..] {
                        filtered_logits[[i, j, index]] = f32::NEG_INFINITY;
                    }
                    break;
                }
            }
        }
    }

    filtered_logits
}

fn argmax(logits: &ArrayD<f32>) -> i64 {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(index, _)| index as i64)
        .unwrap()
}

fn main() -> Result<()> {
    let config = Config::new();
    let generator = TextGenerator::new(config)?;

    let input = "Machine learning is great for humanity. It helps";
    let generated_text = generator.generate(input, 100)?;

    println!("Final generated text: {}", generated_text);
    Ok(())
}

fn create_tensor<T: Clone + Datum>(data: &[T]) -> Result<Tensor> {
    let shape = [1, data.len()];
    let array = tract_ndarray::Array::from_shape_vec(shape, data.to_vec())?;
    Ok(array.into_tensor())
}

fn create_empty_past_key_values(
    num_layers: usize,
    batch_size: usize,
    num_heads: usize,
    seq_length: usize,
    head_dim: usize,
) -> Result<Tensor> {
    let shape = [num_layers, batch_size, num_heads, seq_length, head_dim];
    let array = tract_ndarray::Array::from_shape_vec(
        IxDyn(&shape),
        vec![0.0_f32; num_layers * batch_size * num_heads * seq_length * head_dim],
    )?;
    Ok(array.into_tensor())
}
