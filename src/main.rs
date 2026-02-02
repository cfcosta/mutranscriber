//! CLI tool for transcribing audio files using Qwen3-ASR.
//!
//! Usage:
//!   mutranscriber <input-file> [options]
//!
//! Examples:
//!   mutranscriber recording.mp4
//!   mutranscriber audio.wav --model large --output transcript.txt
//!   mutranscriber recording.mp4 --cpu

use std::path::PathBuf;

use clap::Parser;
use mutranscriber::{ModelVariant, Transcriber, TranscriberConfig};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

/// Transcribe audio files using Qwen3-ASR
#[derive(Parser, Debug)]
#[command(name = "mutranscriber")]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Input audio or video file to transcribe
    #[arg(required = true)]
    input: PathBuf,

    /// Output file for the transcript (default: same as input with .txt extension)
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Model variant to use: "small" (0.6B) or "large" (1.7B)
    #[arg(short, long, default_value = "small")]
    model: String,

    /// Force CPU mode (disable GPU acceleration)
    #[arg(long)]
    cpu: bool,

    /// Print transcript to stdout instead of/in addition to file
    #[arg(short, long)]
    print: bool,

    /// Only print to stdout, don't save to file
    #[arg(long)]
    stdout_only: bool,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "mutranscriber=info".into()),
        )
        .with(tracing_subscriber::fmt::layer().with_target(false))
        .init();

    let args = Args::parse();

    // Validate input file
    if !args.input.exists() {
        eprintln!("Error: Input file not found: {}", args.input.display());
        std::process::exit(1);
    }

    // Parse model variant
    let variant = match args.model.to_lowercase().as_str() {
        "large" | "1.7b" => ModelVariant::Large,
        "small" | "0.6b" | _ => ModelVariant::Small,
    };

    // Build configuration
    let output_dir = args
        .output
        .as_ref()
        .and_then(|p| p.parent())
        .map(PathBuf::from);
    let config = TranscriberConfig {
        variant,
        use_gpu: !args.cpu,
        sample_rate: 16000,
        output_dir,
    };

    tracing::info!("Input: {}", args.input.display());
    tracing::info!("Model: {:?}", variant);
    tracing::info!(
        "Device: {}",
        if args.cpu {
            "CPU"
        } else {
            "GPU (if available)"
        }
    );

    // Create transcriber
    let transcriber = Transcriber::with_config(config);

    // Preload model
    tracing::info!("Loading model...");
    transcriber.preload().await?;

    // Transcribe
    tracing::info!("Transcribing...");

    if args.stdout_only {
        // Extract audio and transcribe directly
        let transcript_path = transcriber.transcribe(&args.input).await?;
        let text = tokio::fs::read_to_string(&transcript_path).await?;

        // Remove the temporary file
        tokio::fs::remove_file(&transcript_path).await.ok();

        // Print to stdout
        println!("{}", text);
    } else {
        // Standard file-based transcription
        let transcript_path = transcriber.transcribe(&args.input).await?;

        // Handle custom output path
        let final_path = if let Some(output) = &args.output {
            if output != &transcript_path {
                tokio::fs::rename(&transcript_path, output).await?;
                output.clone()
            } else {
                transcript_path
            }
        } else {
            transcript_path
        };

        tracing::info!("Transcript saved to: {}", final_path.display());

        // Optionally print to stdout
        if args.print {
            let text = tokio::fs::read_to_string(&final_path).await?;
            println!("\n--- Transcript ---\n{}\n--- End ---", text);
        }
    }

    Ok(())
}
