mod onnx;
mod storage;
use ic_stable_structures::{
    memory_manager::{MemoryId, MemoryManager},
    DefaultMemoryImpl,
};
use onnx::{setup_model, setup_tokenizer};
use std::cell::RefCell;

const WASI_MEMORY_ID: MemoryId = MemoryId::new(0);
const MODEL_FILE: &str = "model.onnx";
const TOKENIZER_FILE: &str = "tokenizer.json";

thread_local! {
    // The memory manager is used for simulating multiple memories.
    static MEMORY_MANAGER: RefCell<MemoryManager<DefaultMemoryImpl>> =
        RefCell::new(MemoryManager::init(DefaultMemoryImpl::default()));
}

#[target_feature(enable = "simd128")]
#[ic_cdk::update]
fn inference_engine(text: String) -> String {
    let generated_text = onnx::inference(&text).unwrap();
    generated_text
}

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

/// Appends the given chunk to the model file.
/// This is used for incremental chunk uploading of large files.
#[ic_cdk::update]
fn append_model_bytes(bytes: Vec<u8>) {
    storage::append_bytes(MODEL_FILE, bytes);
}

#[ic_cdk::update]
fn setup() -> Result<(), String> {
    setup_tokenizer(storage::bytes(TOKENIZER_FILE))
        .map_err(|err| format!("Failed to setup model: {:?}", err))?;
    setup_model(storage::bytes(MODEL_FILE)).map_err(|err| format!("Failed to setup model: {}", err))
}
ic_cdk::export_candid!();
