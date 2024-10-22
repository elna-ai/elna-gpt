mod tokenizer;
use ic_stable_structures::{
    memory_manager::{MemoryId, MemoryManager},
    DefaultMemoryImpl,
};
use std::cell::RefCell;
use tokenizer::{append_bytes, bytes, clear_bytes, decode_output, encode_input, setup_tokenizer};

const WASI_MEMORY_ID: MemoryId = MemoryId::new(0);

thread_local! {
static MEMORY_MANAGER: RefCell<MemoryManager<DefaultMemoryImpl>> =
    RefCell::new(MemoryManager::init(DefaultMemoryImpl::default()));
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
fn clear_tokenizer_bytes() {
    clear_bytes(TOKENIZER_FILE);
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

#[ic_cdk::query]
fn get_token_ids(text: String) -> Result<Vec<i64>, String> {
    let token_ids = encode_input(text);
    token_ids
}

#[ic_cdk::query]
fn get_output_text(output_ids: Vec<u32>) -> Result<String, String> {
    let output_text = decode_output(output_ids);
    output_text
}

ic_cdk::export_candid!();
