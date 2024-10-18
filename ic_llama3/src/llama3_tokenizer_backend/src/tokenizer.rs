use bytes::Bytes;
use std::cell::RefCell;
use std::io::Write;
use tokenizers::tokenizer::Tokenizer;

thread_local! {
static TOKENIZER: RefCell<Option<Tokenizer>> = RefCell::new(None);
}

pub fn bytes(filename: &str) -> Bytes {
    std::fs::read(filename).unwrap().into()
}

pub fn append_bytes(filename: &str, bytes: Vec<u8>) {
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(filename)
        .unwrap();
    file.write_all(&bytes).unwrap();
}

pub fn decode_output(output_ids: Vec<u32>) -> Result<String, String> {
    TOKENIZER.with(|tokenizer_ref| {
        let tokenizer = tokenizer_ref.borrow();
        if let Some(tokenizer) = tokenizer.as_ref() {
            let generated_text = tokenizer.decode(&output_ids, true).unwrap();
            Ok(generated_text)
        } else {
            Err("Tokenizer not initialized".to_string())
        }
    })
}

pub fn encode_input(text: String) -> Result<Vec<i64>, String> {
    TOKENIZER.with(|tokenizer_ref| {
        let tokenizer = tokenizer_ref.borrow();
        if let Some(tokenizer) = tokenizer.as_ref() {
            let encoding = tokenizer
                .encode(text, false)
                .map_err(|e| format!("Encoding error: {:?}", e))?;
            let token_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
            Ok(token_ids)
        } else {
            Err("Tokenizer not initialized".to_string())
        }
    })
}

pub fn setup_tokenizer(bytes: Bytes) -> Result<(), ()> {
    let tokenizer_from_bytes = Tokenizer::from_bytes(&bytes.to_vec()).map_err(|_| ())?; // Handle the error and propagate it

    TOKENIZER.with_borrow_mut(|tokenizer_ref| {
        *tokenizer_ref = Some(tokenizer_from_bytes);
    });

    Ok(()) // Return Ok(()) after saving the tokenizer
}
