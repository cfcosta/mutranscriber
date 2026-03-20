use std::{path::PathBuf, time::Instant};

use clap::Parser;
use mutranscriber::{ModelVariant, Transcriber, TranscriberConfig, load_wav_pcm16_mono};

#[derive(Debug, Parser)]
#[command(name = "autoresearch_bench")]
struct Args {
    /// Input WAV fixture (16-bit PCM, 16kHz mono)
    #[arg(long, default_value = "tests/fixtures/test_audio.wav")]
    input: PathBuf,

    /// Repeat the fixture this many times to create a longer workload
    #[arg(long, default_value_t = 6)]
    repeat: usize,

    /// Warmup runs on the repeated workload before measuring
    #[arg(long, default_value_t = 1)]
    warmup_runs: usize,

    /// Number of measured runs
    #[arg(long, default_value_t = 5)]
    measure_runs: usize,

    /// Force CPU mode instead of CUDA/Metal
    #[arg(long)]
    cpu: bool,

    /// Model variant to benchmark
    #[arg(long, default_value = "small")]
    model: String,
}

fn normalize_text(text: &str) -> String {
    text.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .trim()
        .to_lowercase()
}

fn median_ms(times_ms: &mut [f64]) -> f64 {
    times_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mid = times_ms.len() / 2;
    if times_ms.len().is_multiple_of(2) {
        (times_ms[mid - 1] + times_ms[mid]) / 2.0
    } else {
        times_ms[mid]
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    anyhow::ensure!(args.repeat >= 1, "--repeat must be >= 1");
    anyhow::ensure!(args.measure_runs >= 1, "--measure-runs must be >= 1");

    let variant = match args.model.as_str() {
        "small" => ModelVariant::Small,
        "large" => ModelVariant::Large,
        other => anyhow::bail!("unsupported model variant: {other}"),
    };

    let short_audio = load_wav_pcm16_mono(&args.input).map_err(anyhow::Error::msg)?;
    let mut long_audio = Vec::with_capacity(short_audio.len() * args.repeat);
    for _ in 0..args.repeat {
        long_audio.extend_from_slice(&short_audio);
    }

    let transcriber = Transcriber::with_config(TranscriberConfig {
        variant,
        use_gpu: !args.cpu,
        sample_rate: 16000,
        output_dir: None,
    });

    transcriber.preload().await?;

    let short_transcript = transcriber.transcribe_audio(&short_audio).await?;
    let short_transcript_norm = normalize_text(&short_transcript);

    let mut warmup_transcript_norm = String::new();
    for _ in 0..args.warmup_runs {
        warmup_transcript_norm = normalize_text(&transcriber.transcribe_audio(&long_audio).await?);
    }

    let mut run_times_ms = Vec::with_capacity(args.measure_runs);
    let mut measured_transcript_norm = String::new();
    let mut measured_transcript = String::new();

    for run_idx in 0..args.measure_runs {
        let start = Instant::now();
        let transcript = transcriber.transcribe_audio(&long_audio).await?;
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        run_times_ms.push(elapsed_ms);
        measured_transcript_norm = normalize_text(&transcript);
        measured_transcript = transcript;
        println!("RUN {} {:.3}", run_idx + 1, elapsed_ms);
    }

    let median_ms = median_ms(&mut run_times_ms);
    let audio_seconds = long_audio.len() as f64 / 16000.0;
    let throughput_x = audio_seconds / (median_ms / 1000.0);

    println!("AUDIO_SECONDS {:.3}", audio_seconds);
    println!("MEDIAN_MS {:.3}", median_ms);
    println!("THROUGHPUT_X {:.6}", throughput_x);
    println!("SHORT_TRANSCRIPT {}", short_transcript.replace('\n', " "));
    println!("SHORT_TRANSCRIPT_NORM {}", short_transcript_norm);
    if !warmup_transcript_norm.is_empty() {
        println!("WARMUP_TRANSCRIPT_NORM {}", warmup_transcript_norm);
    }
    println!("LONG_TRANSCRIPT {}", measured_transcript.replace('\n', " "));
    println!("LONG_TRANSCRIPT_NORM {}", measured_transcript_norm);

    Ok(())
}
