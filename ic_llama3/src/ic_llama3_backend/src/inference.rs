// use half::f16;
// use std::cell::RefCell;
// use std::time::Instant;
// use tokenizers::tokenizer::Tokenizer;
// use tract_ndarray::{ArrayD, IxDyn};
// use tract_onnx::prelude::*;

// const TARGET_LEN: usize = 256;

// // Thread-local storage for model and tokenizer
// thread_local! {
//     static MODEL: RefCell<Option<SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>>> = RefCell::new(None);
//     static TOKENIZER: RefCell<Option<Tokenizer>> = RefCell::new(None);
// }

// // Function to initialize and return the model for the current thread
// fn get_model() -> SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>> {
//     MODEL.with(|model| {
//         let mut model_ref = model.borrow_mut();
//         if model_ref.is_none() {
//             let new_model = tract_onnx::onnx()
//                 .model_for_path("models/fp16/llama3.2_1b.onnx")
//                 .unwrap()
//                 .into_optimized()
//                 .unwrap()
//                 .into_runnable()
//                 .unwrap();
//             *model_ref = Some(new_model);
//         }
//         model_ref.as_mut().unwrap().clone() // Clone the model and return it
//     })
// }

// // Function to initialize and return the tokenizer for the current thread
// fn get_tokenizer() -> Tokenizer {
//     TOKENIZER.with(|tokenizer| {
//         let mut tokenizer_ref = tokenizer.borrow_mut();
//         if tokenizer_ref.is_none() {
//             *tokenizer_ref = Some(Tokenizer::from_file("tokenizer/llama3/tokenizer.json").unwrap());
//         }
//         tokenizer_ref.as_ref().unwrap().clone() // Clone and return the tokenizer
//     })
// }

// // Function to get input IDs without padding
// fn get_input_ids(tokenizer: &Tokenizer, text: &str) -> (Vec<i64>, Vec<i8>) {
//     let encoding = tokenizer.encode(text, false).unwrap();
//     let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
//     let attention_mask: Vec<i8> = vec![1; input_ids.len()];
//     (input_ids, attention_mask)
// }

// fn main() -> TractResult<()> {
//     // Access the thread-local model and tokenizer
//     let mut model = get_model();
//     let tokenizer = get_tokenizer();

//     let mut past_key_values_tensor = create_empty_past_key_values(32, 1, 8, 0, 64)?;

//     let input = "<|begin_of_text|>Context: The Internet Computer is a public blockchain network enabled by new science from first principles. It is millions of times more powerful and can replace clouds and traditional IT.<|eot_id|>
//  <|begin_of_text|>What is internet computer? generate in one sentance <|eot_id|> <|begin_of_text|>Answer:";
//     let (mut input_ids, mut attention_mask) = get_input_ids(&tokenizer, input);
//     let mut generated_tokens: Vec<i64> = vec![];

//     for j in 0..5 {
//         let input_ids_tensor = create_tensor_i64(&input_ids)?;
//         let attention_mask_tensor = create_tensor_i8(&attention_mask)?;

//         let inputs: TVec<TValue> = tvec!(
//             input_ids_tensor.into(),
//             attention_mask_tensor.into(),
//             past_key_values_tensor.clone().into()
//         );

//         let outputs = match model.run(inputs) {
//             Ok(o) => o,
//             Err(e) => {
//                 println!("Model run failed: {:?}", e);
//                 return Err(e);
//             }
//         };

//         past_key_values_tensor = outputs[1].clone().into_tensor();
//         let next_token_tensor = outputs[0].to_array_view::<i64>()?;
//         let next_token = next_token_tensor[[0, 0]];
//         if next_token == 50258 {
//             break;
//         }
//         println!("Next token: {}", next_token);
//         generated_tokens.push(next_token);

//         input_ids = vec![next_token];
//         attention_mask.push(1);
//     }

//     let generated_tokens_u32: Vec<u32> = generated_tokens.iter().map(|&x| x as u32).collect();
//     let final_output = tokenizer.decode(&generated_tokens_u32, false).unwrap();
//     println!("{}", final_output);

//     Ok(())
// }

// // Function to create tensors
// fn create_tensor_i64(data: &[i64]) -> TractResult<Tensor> {
//     let shape = [1, data.len()];
//     let array = tract_ndarray::Array::from_shape_vec(shape, data.to_vec())
//         .map_err(|_| anyhow::anyhow!("Failed to create tensor from shape and values"))?;
//     Ok(array.into_tensor())
// }

// fn create_tensor_i8(data: &[i8]) -> TractResult<Tensor> {
//     let shape = [1, data.len()];
//     let array = ArrayD::from_shape_vec(IxDyn(&shape), data.to_vec())
//         .map_err(|_| anyhow::anyhow!("Failed to create tensor from shape and values"))?;
//     Ok(array.into_tensor())
// }

// // Function to create empty past key values tensor
// fn create_empty_past_key_values(
//     num_layers: usize,
//     batch_size: usize,
//     num_heads: usize,
//     seq_length: usize,
//     head_dim: usize,
// ) -> TractResult<Tensor> {
//     let shape = [num_layers, batch_size, num_heads, seq_length, head_dim];
//     let array = tract_ndarray::Array::from_shape_vec(
//         IxDyn(&shape),
//         vec![f16::ZERO; num_layers * batch_size * num_heads * seq_length * head_dim],
//     )
//     .map_err(|_| anyhow::anyhow!("Failed to create tensor from shape and values"))?;
//     Ok(array.into_tensor())
// }
