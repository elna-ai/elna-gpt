use bytes::Bytes;
use ic_stable_structures::{
    memory_manager::{MemoryId, MemoryManager},
    DefaultMemoryImpl,
};
use std::cell::RefCell;
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
pub fn setup_tokenizer(bytes: Bytes) -> Result<(), ()> {
    TOKENIZER.with(|tokenizer| {
        let mut tokenizer_ref = tokenizer.borrow_mut();
        if tokenizer_ref.is_none() {
            let tokenizer_from_bytes = Tokenizer::from_bytes(&bytes.to_vec())
                .map_err(|_| ())
                .unwrap(); // Handle the error without panic
            *tokenizer_ref = Some(tokenizer_from_bytes);
        }
        Ok(()) // Just return `Ok(())` after saving the tokenizer
    })
}
ic_cdk::export_candid!();
