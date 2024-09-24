use anyhow::Result;
use ndarray_npy::read_npy;
use rand::distributions::WeightedIndex;
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

// Define a struct to hold optional parameters with default values
#[derive(Clone)] // Add this line
struct GenerateConfig {
    temperature: f32,
    top_k: usize,
    top_p: f32,
}

impl GenerateConfig {
    // Method to create a new config with default values
    fn new() -> Self {
        Self {
            temperature: 0.5,
            top_k: 50,
            top_p: 0.9,
        }
    }

    // Builder methods to set individual parameters
    fn temperature(mut self, value: f32) -> Self {
        self.temperature = value;
        self
    }

    fn top_k(mut self, value: usize) -> Self {
        self.top_k = value;
        self
    }

    fn top_p(mut self, value: f32) -> Self {
        self.top_p = value;
        self
    }
}
#[derive(Debug, Clone)] // Add this line
enum SamplingMethod {
    Sampling,
    Argmax,
}

struct TextGenerator {
    model: SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>,
    tokenizer: Gpt2Tokenizer,
    config: Config,
    stop_tokens: Vec<String>,
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

        let stop_tokens = vec!["<|endoftext|>".to_string(), "Question".to_string()];

        Ok(Self {
            model,
            tokenizer,
            config,
            stop_tokens,
        })
    }

    fn generate(
        &self,
        input: &str,
        num_tokens: usize,
        config: GenerateConfig,
        use_top_k: bool,
        use_top_p: bool,
        sampling_method: SamplingMethod,
    ) -> Result<String> {
        let (mut input_ids, mut attention_mask) = self.get_input_ids(input)?;
        let mut generated_tokens = input_ids.clone();
        let mut past_key_values_tensor = create_empty_past_key_values(24, 1, 12, 0, 64)?;

        for i in 0..num_tokens {
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

            let mut filtered_logits = if use_top_k {
                top_k_filtering(&logits, config.top_k)
            } else {
                logits.clone() // No filtering
            };
            filtered_logits = if use_top_p {
                top_p_filtering(&filtered_logits, config.top_p)
            } else {
                filtered_logits // No filtering
            };

            let next_token_id = match sampling_method {
                SamplingMethod::Sampling => {
                    sample_from_logits(&filtered_logits, config.temperature)
                }
                SamplingMethod::Argmax => argmax(&filtered_logits),
            };
            // Debugging output

            generated_tokens.push(next_token_id);
            let output = self.tokenizer.decode(&generated_tokens, false, true);
            if self.stop_tokens.contains(&output) {
                println!("Stopping generation due to stop token.");
                break;
            }
            println!("Output: {}", output);
            input_ids = vec![next_token_id];
            attention_mask.push(1);
        }

        Ok(self.tokenizer.decode(&generated_tokens, false, true))
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

fn softmax(logits: &ArrayViewD<f32>) -> ArrayD<f32> {
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

fn sample_from_logits(logits: &ArrayD<f32>, temperature: f32) -> i64 {
    let adjusted_logits = logits.mapv(|x| x / temperature);
    let probabilities = softmax(&adjusted_logits.view());

    // Flatten the probabilities array to a 1D vector for sampling
    let mut prob_vec: Vec<f32> = probabilities.iter().cloned().collect();

    // Ensure all probabilities are valid (non-negative and not NaN)
    prob_vec.iter_mut().for_each(|p| {
        if !p.is_finite() || *p < 0.0 {
            *p = 0.0;
        }
    });

    // If all probabilities are zero, return a random index
    if prob_vec.iter().all(|&p| p == 0.0) {
        return rand::Rng::gen_range(&mut rand::thread_rng(), 0..prob_vec.len()) as i64;
    }

    // Create a weighted index distribution based on the probabilities
    let dist = WeightedIndex::new(&prob_vec).expect("Probabilities should be valid");

    // Randomly select an index based on the weighted probabilities
    let mut rng = rand::thread_rng();
    rand::prelude::Distribution::sample(&dist, &mut rng) as i64
}

fn main() -> Result<()> {
    let config = Config::new();
    let generator = TextGenerator::new(config)?;

    let input = "Read the question and give an honest answer. Question: What is Artificial Intelligence? Answer:";
    let generated_text = generator.generate(
        input,
        100,
        GenerateConfig::new(),
        true,
        true,
        SamplingMethod::Sampling,
    )?;

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
