use bytes::Bytes;
use half::f16;
use prost::Message;
use std::cell::RefCell;
use tokenizers::tokenizer::Tokenizer;
use tract_ndarray::{ArrayD, IxDyn};
use tract_onnx::prelude::*;

const TARGET_LEN: usize = 256;

// Thread-local storage for model and tokenizer
thread_local! {
    static MODEL: RefCell<Option<SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>>> = RefCell::new(None);
    static TOKENIZER: RefCell<Option<Tokenizer>> = RefCell::new(None);
}

pub fn setup_model(bytes: Bytes) -> TractResult<()> {
    // let bytes = bytes::Bytes::from_static(MODEL_BYTES);
    let proto: tract_onnx::pb::ModelProto = tract_onnx::pb::ModelProto::decode(bytes)?;
    let model = tract_onnx::onnx()
        .model_for_proto_model(&proto)?
        .into_optimized()?
        .into_runnable()?;
    MODEL.with_borrow_mut(|m| {
        *m = Some(model);
    });
    Ok(())
}

// Function to initialize and return the tokenizer for the current thread
pub fn setup_tokenizer(bytes: Bytes) -> Result<Tokenizer, ()> {
    TOKENIZER.with(|tokenizer| {
        let mut tokenizer_ref = tokenizer.borrow_mut();
        if tokenizer_ref.is_none() {
            let tokenizer_from_bytes = Tokenizer::from_bytes(&bytes.to_vec()).map_err(|_| ())?; // Handle the error without panic
            *tokenizer_ref = Some(tokenizer_from_bytes);
        }
        Ok(tokenizer_ref.as_ref().unwrap().clone()) // Clone and return the tokenizer
    })
}

// Function to get input IDs without padding
fn get_input_ids(tokenizer: &Tokenizer, text: &str) -> (Vec<i64>, Vec<i8>) {
    let encoding = tokenizer.encode(text, false).unwrap();
    let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
    let attention_mask: Vec<i8> = vec![1; input_ids.len()];
    (input_ids, attention_mask)
}

fn inference(input: &str) -> TractResult<()> {
    // Setup model and tokenizer
    MODEL.with(|model_cell| -> TractResult<()> {
        let model_opt = model_cell.borrow();
        let model = model_opt
            .as_ref()
            .ok_or(anyhow::anyhow!("Model not initialized"))?;

        TOKENIZER.with(|tokenizer_cell| -> TractResult<()> {
            let tokenizer_opt = tokenizer_cell.borrow();
            let tokenizer = tokenizer_opt
                .as_ref()
                .ok_or(anyhow::anyhow!("Tokenizer not initialized"))?;

            // Tokenize the input
            let (mut input_ids, mut attention_mask) = get_input_ids(&tokenizer, input);
            let mut generated_tokens: Vec<i64> = vec![];

            // Create empty past key values tensor
            let mut past_key_values_tensor = create_empty_past_key_values(32, 1, 8, 0, 64)?;

            // Generate tokens for a fixed number of steps or until stopping token
            for _ in 0..TARGET_LEN {
                // Convert input IDs and attention mask to tensors
                let input_ids_tensor = create_tensor_i64(&input_ids)?;
                let attention_mask_tensor = create_tensor_i8(&attention_mask)?;

                // Prepare inputs for the model
                let inputs: TVec<TValue> = tvec!(
                    input_ids_tensor.into(),
                    attention_mask_tensor.into(),
                    past_key_values_tensor.clone().into()
                );

                // Run the model
                let outputs = match model.run(inputs) {
                    Ok(o) => o,
                    Err(e) => {
                        eprintln!("Model run failed: {:?}", e);
                        return Err(e);
                    }
                };

                // Extract outputs: next token and updated past key values
                past_key_values_tensor = outputs[1].clone().into_tensor();
                let next_token_tensor = outputs[0].to_array_view::<i64>()?;
                let next_token = next_token_tensor[[0, 0]];

                // Stop if the token is the stopping token (50258 here)
                if next_token == 50258 {
                    break;
                }

                println!("Next token: {}", next_token);
                generated_tokens.push(next_token);

                // Prepare for the next iteration
                input_ids = vec![next_token];
                attention_mask.push(1);
            }

            // Convert generated tokens back to text
            let generated_tokens_u32: Vec<u32> =
                generated_tokens.iter().map(|&x| x as u32).collect();
            let final_output = tokenizer.decode(&generated_tokens_u32, false).unwrap();
            println!("{}", final_output);

            Ok(())
        })?;

        Ok(())
    })?;

    Ok(())
}

// Function to create tensors
fn create_tensor_i64(data: &[i64]) -> TractResult<Tensor> {
    let shape = [1, data.len()];
    let array = tract_ndarray::Array::from_shape_vec(shape, data.to_vec())
        .map_err(|_| anyhow::anyhow!("Failed to create tensor from shape and values"))?;
    Ok(array.into_tensor())
}

fn create_tensor_i8(data: &[i8]) -> TractResult<Tensor> {
    let shape = [1, data.len()];
    let array = ArrayD::from_shape_vec(IxDyn(&shape), data.to_vec())
        .map_err(|_| anyhow::anyhow!("Failed to create tensor from shape and values"))?;
    Ok(array.into_tensor())
}

// Function to create empty past key values tensor
fn create_empty_past_key_values(
    num_layers: usize,
    batch_size: usize,
    num_heads: usize,
    seq_length: usize,
    head_dim: usize,
) -> TractResult<Tensor> {
    let shape = [num_layers, batch_size, num_heads, seq_length, head_dim];
    let array = tract_ndarray::Array::from_shape_vec(
        IxDyn(&shape),
        vec![f16::ZERO; num_layers * batch_size * num_heads * seq_length * head_dim],
    )
    .map_err(|_| anyhow::anyhow!("Failed to create tensor from shape and values"))?;
    Ok(array.into_tensor())
}
