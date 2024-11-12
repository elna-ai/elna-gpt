mod onnx;
mod storage;
mod tokenizer;

use candid::Principal;
use ic_cdk::api::call::RejectionCode;
use ic_stable_structures::{
    memory_manager::{MemoryId, MemoryManager},
    DefaultMemoryImpl,
};
use onnx::setup_model;
// use sha2::{Digest, Sha256}; // Add for hashing
use std::cell::RefCell;
use tokenizer::{decode, encode};
const WASI_MEMORY_ID: MemoryId = MemoryId::new(0);
const MODEL_FILE: &str = "model.onnx";
// const MODEL_FILE2: &str = "9975db42-89df-11ef-b330-0242ac1c000c";

thread_local! {

    static TOKENIZER: RefCell<String> = RefCell::new(String::new());

    // The memory manager is used for simulating multiple memories.
    static MEMORY_MANAGER: RefCell<MemoryManager<DefaultMemoryImpl>> =
        RefCell::new(MemoryManager::init(DefaultMemoryImpl::default()));
}

#[target_feature(enable = "simd128")]
#[ic_cdk::update]
async fn model_inference(text: String) -> Result<String, (RejectionCode, std::string::String)> {
    let result = encode(text).await;
    match result {
        Ok(token_ids) => {
            let llm_result = onnx::run(token_ids).map_err(|err| err.to_string());
            match llm_result {
                Ok(token_ids) => {
                    let text = decode(token_ids).await;
                    text
                }
                Err(rejection) => Err((RejectionCode::CanisterReject, rejection)),
            }
        }

        Err(rejection) => Err(rejection),
    }
}

#[target_feature(enable = "simd128")]
#[ic_cdk::update]
fn test_model(token_ids: Vec<i64>) -> Vec<u32> {
    onnx::run(token_ids).unwrap()
}

#[ic_cdk::init]
fn init(canister_id: Principal) {
    // Initialize ic_wasi_polyfill with a memory manager.
    let wasi_memory = MEMORY_MANAGER.with(|m| m.borrow().get(WASI_MEMORY_ID));
    ic_wasi_polyfill::init_with_memory(&[0u8; 32], &[], wasi_memory);

    TOKENIZER.with(|o| *o.borrow_mut() = canister_id.to_string());
}

#[ic_cdk::post_upgrade]
fn post_upgrade(canister_id: Principal) {
    // Reinitialize ic_wasi_polyfill after canister upgrade.
    let wasi_memory = MEMORY_MANAGER.with(|m| m.borrow().get(WASI_MEMORY_ID));
    ic_wasi_polyfill::init_with_memory(&[0u8; 32], &[], wasi_memory);
    init(canister_id);
}

#[ic_cdk::update]
fn clear_model_bytes() {
    storage::clear_bytes(MODEL_FILE);
}
// Appends the given chunk to the model file.
// This is used for incremental chunk uploading of large files.
#[ic_cdk::update]
fn append_model_bytes(bytes: Vec<u8>) {
    storage::append_bytes(MODEL_FILE, bytes);
}

#[ic_cdk::update]
fn setup() -> Result<(), String> {
    setup_model().map_err(|err| format!("Failed to setup model: {}", err))
}
ic_cdk::export_candid!();
