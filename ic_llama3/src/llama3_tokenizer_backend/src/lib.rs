use bytes::Bytes;
use std::cell::RefCell;
// use tokenizers::tokenizer::Tokenizer;
//
// thread_local! {
// static TOKENIZER: RefCell<Option<Tokenizer>> = RefCell::new(None);
// }
const TOKENIZER_FILE: &str = "tokenizer.json";
#[ic_cdk::query]
fn greet(name: String) -> String {
    format!("Hello, {}!", name)
}

// pub fn setup_tokenizer(bytes: Bytes) -> Result<(), ()> {
//     TOKENIZER.with(|tokenizer| {
//         let mut tokenizer_ref = tokenizer.borrow_mut();
//         if tokenizer_ref.is_none() {
//             let tokenizer_from_bytes = Tokenizer::from_bytes(&bytes.to_vec())
//                 .map_err(|_| ())
//                 .unwrap(); // Handle the error without panic
//             *tokenizer_ref = Some(tokenizer_from_bytes);
//         }
//         Ok(()) // Just return `Ok(())` after saving the tokenizer
//     })
// }
