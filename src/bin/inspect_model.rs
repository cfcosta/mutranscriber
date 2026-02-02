use std::path::PathBuf;

use candle_core::{Device, safetensors};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        let home = std::env::var("HOME").unwrap();
        format!("{home}/.cache/huggingface/hub/models--Qwen--Qwen3-ASR-0.6B/snapshots/5eb144179a02acc5e5ba31e748d22b0cf3e303b0/model.safetensors")
    });

    println!("Loading: {}", path);
    let tensors = safetensors::load(PathBuf::from(&path), &Device::Cpu)?;

    let mut keys: Vec<_> = tensors.keys().collect();
    keys.sort();

    println!("\nConv layer shapes:");
    for key in keys.iter().filter(|k| k.contains("conv")) {
        let tensor = tensors.get(*key).unwrap();
        println!("  {}: {:?}", key, tensor.shape());
    }

    println!("\nFirst layer shapes:");
    for key in keys.iter().filter(|k| k.contains("layers.0.")) {
        let tensor = tensors.get(*key).unwrap();
        println!("  {}: {:?}", key, tensor.shape());
    }

    println!("\nOutput projection shapes:");
    for key in keys
        .iter()
        .filter(|k| k.contains("ln_post") || k.contains("proj1") || k.contains("proj2"))
    {
        let tensor = tensors.get(*key).unwrap();
        println!("  {}: {:?}", key, tensor.shape());
    }

    println!("\nTotal: {} tensors", keys.len());
    Ok(())
}
