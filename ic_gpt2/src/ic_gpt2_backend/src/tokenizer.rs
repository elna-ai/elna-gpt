#![allow(dead_code, unused_imports)]
use crate::TOKENIZER;
use candid::{self, CandidType, Decode, Deserialize, Encode, Principal};
use ic_cdk::api::call::{CallResult, RejectionCode};

#[derive(CandidType, Deserialize)]
pub enum Result_ {
    Ok(String),
    Err(String),
}

#[derive(CandidType, Deserialize, Debug)]
pub enum Result1 {
    Ok(Vec<i64>),
    Err(String),
}

pub struct Service(pub Principal);
impl Service {
    pub async fn get_output_text(&self, arg0: Vec<u32>) -> CallResult<(Result_,)> {
        ic_cdk::call(self.0, "get_output_text", (arg0,)).await
    }
    pub async fn get_token_ids(&self, arg0: String) -> CallResult<(Result1,)> {
        ic_cdk::call(self.0, "get_token_ids", (arg0,)).await
    }
}

pub async fn encode(text: String) -> Result<Vec<i64>, (RejectionCode, std::string::String)> {
    let canister_id = TOKENIZER.with(|owner| owner.borrow().clone());
    ic_cdk::println!("Token Canister ID:{:?}", canister_id);
    let tokenizer = Service(Principal::from_text(canister_id).unwrap());

    let result = tokenizer.get_token_ids(text).await;
    ic_cdk::println!("Tokens:{:?}", result);

    match result {
        Ok(result1) => match result1 {
            (Result1::Ok(token_ids),) => Ok(token_ids),

            (Result1::Err(err),) => Err((RejectionCode::CanisterError, err.to_string())),
        },
        Err(rejection) => Err((rejection.0, rejection.1)),
    }
}

pub async fn decode(token_ids: Vec<u32>) -> Result<String, (RejectionCode, std::string::String)> {
    let canister_id = TOKENIZER.with(|owner| owner.borrow().clone());
    let tokenizer = Service(Principal::from_text(canister_id).unwrap());

    let result = tokenizer.get_output_text(token_ids).await;

    match result {
        Ok(result1) => match result1 {
            (Result_::Ok(text),) => Ok(text),

            (Result_::Err(err),) => Err((RejectionCode::CanisterError, err.to_string())),
        },
        Err(rejection) => Err((rejection.0, rejection.1)),
    }
}
