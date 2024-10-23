#!/bin/bash
set -ex
MODULE=$1
export RUSTFLAGS=$RUSTFLAGS' -C target-feature=+simd128'
cargo build --release --target=wasm32-wasi
wasi2ic ./target/wasm32-wasi/release/$MODULE.wasm ./target/wasm32-wasi/release/$MODULE.wasm
wasm-opt -Os -o ./target/wasm32-wasi/release/$MODULE.wasm \
         ./target/wasm32-wasi/release/$MODULE.wasm

# candid-extractor "target/wasm32-wasi/release/ic_embedding_backend.wasm" > "src/ic-embedding-backend/ic-embedding-backend.did"