# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

mutranscriber is a Rust-based audio transcription tool implementing the Qwen3-ASR model using Candle (Hugging Face's Rust ML framework). It provides a complete speech-to-text pipeline: audio input → mel spectrogram extraction → audio encoder (Conv2D + Transformer) → LLM decoder (Qwen3) → text output.

## Build Commands

```bash
cargo build                          # Development build
cargo build --release                # Release build
cargo build --features gstreamer     # With audio/video file support
cargo build --features cuda          # With CUDA GPU support
cargo build --features metal         # With Metal GPU support (macOS)
```

## Testing

```bash
cargo test --lib                     # Run unit tests only
cargo test --test integration_test   # Run integration tests
cargo test test_name                 # Run specific test
cargo test -- --nocapture            # With output printing
cargo test -- --ignored              # Run slow tests (requires model download)
```

Test fixtures are in `tests/fixtures/` (includes a 10-second test audio file).

## Linting & Formatting

```bash
cargo fmt                            # Format code
cargo fmt -- --check                 # Check formatting
cargo clippy                         # Run lints
```

The project uses 80-character line width. See `rustfmt.toml` for full config.

## Development Environment

The project uses Nix flakes. Enter the dev environment with:
```bash
nix develop    # Or use direnv (auto-activates)
```

Includes: bacon (test watcher), cargo-machete, cargo-nextest, treefmt, GStreamer libs.

## Architecture

### Core Pipeline

1. **Audio Input** (`transcriber.rs`): High-level async API using Tokio. Supports raw f32 samples at 16kHz or audio extraction from files via GStreamer.

2. **Mel Spectrogram** (`mel.rs`): Converts audio to 128-bin mel spectrogram using custom FFT implementation with Whisper-style normalization.

3. **Audio Encoder** (`audio_encoder.rs`): Qwen3-AuT transformer with Conv2D downsampling (8x) followed by transformer layers (18 for 0.6B, 24 for 1.7B model).

4. **Text Decoder** (`qwen3_decoder.rs`): Custom Qwen3 LLM supporting embedding injection for multimodal audio features. Uses RoPE, GQA, and KV cache for efficient generation.

### Model Variants

| Variant | Params | Audio Encoder | Text Decoder |
|---------|--------|---------------|--------------|
| Small   | 0.6B   | 896-dim, 18 layers | 1024-dim, 28 layers |
| Large   | 1.7B   | 1024-dim, 24 layers | 2048-dim, 28 layers |

Models auto-download from HuggingFace (`Qwen/Qwen3-ASR-0.6B`, `Qwen/Qwen3-ASR-1.7B`).

### Module Structure

```
lib.rs              # Public API
├── transcriber.rs  # High-level async transcription API
├── model.rs        # Full model composition
│   ├── audio_encoder.rs
│   ├── qwen3_decoder.rs
│   └── config.rs
└── mel.rs          # Mel spectrogram extraction

main.rs             # CLI binary
```

### Key Constants (mel.rs)

- Sample rate: 16kHz
- FFT size: 400
- Hop length: 160
- Mel bins: 128

## CLI Usage

```bash
mutranscriber <audio_file> [options]
  --model <small|large>   # Model variant (default: small)
  --gpu                   # Use GPU
  --output <file>         # Output file path
  --stdout                # Print to stdout
```

## Agent Instructions

This project uses **bd** (beads) for issue tracking. Run `bd onboard` to get started.

## Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --status in_progress  # Claim work
bd close <id>         # Complete work
bd sync               # Sync with git
```

## Landing the Plane

**When the user says "let's land the plane"**, you MUST complete ALL steps below. The plane is NOT landed until `jj git push` succeeds. NEVER stop before pushing. NEVER say "ready to push when you are!" - that is a FAILURE.

**MANDATORY WORKFLOW - COMPLETE ALL STEPS:**

1. **File beads issues for any remaining work** that needs follow-up
2. **Ensure all quality gates pass** (only if code changes were made):
   - Run `ruff check`, `ruff format`, `ty check`
   - File P0 issues if quality gates are broken
3. **Update beads issues** - close finished work, update status
4. **PUSH TO REMOTE - NON-NEGOTIABLE** - This step is MANDATORY. Execute ALL commands below:
   ```bash
   # Fetch and rebase to catch any remote changes
   jj git fetch
   jj rebase -d main@origin

   # If conflicts in .beads/issues.jsonl, resolve thoughtfully:
   #   - jj restore --from @- .beads/issues.jsonl (accept previous version)
   #   - bd import -i .beads/issues.jsonl (re-import)
   #   - Or manually resolve, then jj resolve --mark .beads/issues.jsonl

   # Sync the database (exports to JSONL, commits)
   bd sync

   # MANDATORY: Push everything to remote
   # DO NOT STOP BEFORE THIS COMMAND COMPLETES
   jj git push

   # MANDATORY: Verify push succeeded
   jj log -r 'main@origin'  # MUST show your latest changes
   ```

   **CRITICAL RULES:**
   - The plane has NOT landed until `jj git push` completes successfully
   - NEVER stop before `jj git push` - that leaves work stranded locally
   - NEVER say "ready to push when you are!" - YOU must push, not the user
   - If `jj git push` fails, resolve the issue and retry until it succeeds
   - The user is managing multiple agents - unpushed work breaks their coordination workflow

5. **Clean up jj state** - Abandon empty/obsolete changes:
   ```bash
   jj abandon 'empty()'              # Remove empty changes
   ```
6. **Verify clean state** - Ensure all changes are committed AND PUSHED, no untracked files remain
7. **Choose a follow-up issue for next session**
   - Provide a prompt for the user to give to you in the next session
   - Format: "Continue work on bd-X: [issue title]. [Brief context about what's been done and what's next]"

**REMEMBER: Landing the plane means EVERYTHING is pushed to remote. No exceptions. No "ready when you are". PUSH IT.**

**Example "land the plane" session:**

```bash
# 1. File remaining work
bd create "Add integration tests for sync" -t task -p 2 --json

# 2. Run quality gates (only if code changes were made)
ruff check src/
ruff format src/

# 3. Close finished issues
bd close bd-42 bd-43 --reason "Completed" --json

# 4. PUSH TO REMOTE - MANDATORY, NO STOPPING BEFORE THIS IS DONE
jj git fetch
jj rebase -d main@origin
# If conflicts in .beads/issues.jsonl, resolve thoughtfully:
#   - jj restore --from @- .beads/issues.jsonl (accept previous)
#   - bd import -i .beads/issues.jsonl (re-import)
#   - Or manually resolve, then jj resolve --mark
bd sync           # Export/import/commit
jj git push       # MANDATORY - THE PLANE IS STILL IN THE AIR UNTIL THIS SUCCEEDS
jj log -r 'main@origin'  # MUST verify changes are on remote

# 5. Clean up jj state
jj abandon 'empty()'

# 6. Verify everything is clean and pushed
jj status

# 7. Choose next work
bd ready --json
bd show bd-44 --json
```

**Then provide the user with:**

- Summary of what was completed this session
- What issues were filed for follow-up
- Status of quality gates (all passing / issues filed)
- Confirmation that ALL changes have been pushed to remote
- Recommended prompt for next session

**CRITICAL: Never end a "land the plane" session without successfully pushing. The user is coordinating multiple agents and unpushed work causes severe rebase conflicts.**
