mod inference;
mod storage;

const MODEL_FILE: &str = "model.onnx";

#[ic_cdk::update]
fn append_model_bytes(bytes: Vec<u8>) {
    storage::append_bytes(MODEL_FILE, bytes);
}
#[ic_cdk::query]
fn greet(name: String) -> String {
    format!("Hello, {}!", name)
}

ic_cdk::export_candid!();
