use anyhow::Result;
use ndarray_npy::read_npy;
use rust_tokenizers::tokenizer::{Gpt2Tokenizer, Tokenizer, TruncationStrategy};
use tract_ndarray::{ArrayD, ArrayViewD, IxDyn};
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

            let logits = outputs[0].to_array_view::<f32>()?;
            past_key_values_tensor = outputs[1].clone().into_tensor();
            println!("{:?}", past_key_values_tensor);
            // Use argmax to get the next token ID
            let next_token_id = logits
                .index_axis(ndarray::Axis(1), 0)
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(index, _)| index as i64)
                .unwrap();

            generated_tokens.push(next_token_id);
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

fn main() -> Result<()> {
    let config = Config::new();
    let generator = TextGenerator::new(config)?;

    let input = "what is blockchain";
    let generated_text = generator.generate(input, 20)?;

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
