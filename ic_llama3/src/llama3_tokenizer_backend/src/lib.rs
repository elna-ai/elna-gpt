// extern crate ic_llama3_backend;
// use ic_llama3_backend::storage;
// use super::ic_llama3_backend;
use bytes::Bytes;
use ic_stable_structures::{
    memory_manager::{MemoryId, MemoryManager},
    DefaultMemoryImpl,
};
use std::cell::RefCell;
use std::io::Write;
use tokenizers::tokenizer::Tokenizer;

const WASI_MEMORY_ID: MemoryId = MemoryId::new(0);

thread_local! {
static MEMORY_MANAGER: RefCell<MemoryManager<DefaultMemoryImpl>> =
    RefCell::new(MemoryManager::init(DefaultMemoryImpl::default()));
static TOKENIZER: RefCell<Option<Tokenizer>> = RefCell::new(None);
}
const TOKENIZER_FILE: &str = "tokenizer.json";

#[ic_cdk::init]
fn init() {
    // Initialize ic_wasi_polyfill with a memory manager.
    let wasi_memory = MEMORY_MANAGER.with(|m| m.borrow().get(WASI_MEMORY_ID));
    ic_wasi_polyfill::init_with_memory(&[0u8; 32], &[], wasi_memory);
}

#[ic_cdk::post_upgrade]
fn post_upgrade() {
    // Reinitialize ic_wasi_polyfill after canister upgrade.
    let wasi_memory = MEMORY_MANAGER.with(|m| m.borrow().get(WASI_MEMORY_ID));
    ic_wasi_polyfill::init_with_memory(&[0u8; 32], &[], wasi_memory);
}

#[ic_cdk::update]
fn append_tokenizer_bytes(bytes: Vec<u8>) {
    append_bytes(TOKENIZER_FILE, bytes);
}

#[ic_cdk::update]
fn setup() -> Result<(), String> {
    // setup_tokenizer(storage::bytes(TOKENIZER_FILE))
    //     .map_err(|err| format!("Failed to setup model: {:?}", err))?;
    setup_tokenizer(bytes(TOKENIZER_FILE))
        .map_err(|err| format!("Failed to setup model: {:?}", err))
}

pub fn setup_tokenizer(bytes: Bytes) -> Result<(), ()> {
    let tokenizer_from_bytes = Tokenizer::from_bytes(&bytes.to_vec()).map_err(|_| ())?; // Handle the error and propagate it

    TOKENIZER.with_borrow_mut(|tokenizer_ref| {
        *tokenizer_ref = Some(tokenizer_from_bytes);
    });

    Ok(()) // Return Ok(()) after saving the tokenizer
}

pub fn get_token_ids(text: &str) -> Result<(Vec<i64>), String> {
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

ic_cdk::export_candid!();
