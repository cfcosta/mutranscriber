use std::path::Path;

/// Load a PCM16 mono WAV file into normalized `f32` samples.
///
/// This parser walks RIFF chunks instead of assuming a fixed 44-byte header,
/// so it correctly handles files that include extra metadata chunks.
pub fn load_wav_pcm16_mono(path: &Path) -> Result<Vec<f32>, String> {
    let data =
        std::fs::read(path).map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;

    if data.len() < 12 {
        return Err(format!("WAV file too short: {}", path.display()));
    }
    if &data[0..4] != b"RIFF" || &data[8..12] != b"WAVE" {
        return Err(format!("Not a RIFF/WAVE file: {}", path.display()));
    }

    let mut offset = 12usize;
    let mut audio_format = None;
    let mut channels = None;
    let mut bits_per_sample = None;
    let mut sample_rate = None;
    let mut audio_data = None;

    while offset + 8 <= data.len() {
        let chunk_id = &data[offset..offset + 4];
        let chunk_size = u32::from_le_bytes([
            data[offset + 4],
            data[offset + 5],
            data[offset + 6],
            data[offset + 7],
        ]) as usize;
        offset += 8;

        let chunk_end = offset
            .checked_add(chunk_size)
            .ok_or_else(|| format!("Invalid chunk size in {}", path.display()))?;
        if chunk_end > data.len() {
            return Err(format!("Truncated WAV chunk in {}", path.display()));
        }

        match chunk_id {
            b"fmt " => {
                if chunk_size < 16 {
                    return Err(format!("fmt chunk too small in {}", path.display()));
                }
                audio_format = Some(u16::from_le_bytes([data[offset], data[offset + 1]]));
                channels = Some(u16::from_le_bytes([data[offset + 2], data[offset + 3]]));
                sample_rate = Some(u32::from_le_bytes([
                    data[offset + 4],
                    data[offset + 5],
                    data[offset + 6],
                    data[offset + 7],
                ]));
                bits_per_sample = Some(u16::from_le_bytes([data[offset + 14], data[offset + 15]]));
            }
            b"data" => {
                audio_data = Some(&data[offset..chunk_end]);
            }
            _ => {}
        }

        offset = chunk_end;
        if chunk_size % 2 == 1 && offset < data.len() {
            offset += 1;
        }
    }

    let audio_format =
        audio_format.ok_or_else(|| format!("Missing fmt chunk in {}", path.display()))?;
    let channels = channels.ok_or_else(|| format!("Missing channel count in {}", path.display()))?;
    let sample_rate =
        sample_rate.ok_or_else(|| format!("Missing sample rate in {}", path.display()))?;
    let bits_per_sample =
        bits_per_sample.ok_or_else(|| format!("Missing bit depth in {}", path.display()))?;
    let audio_data = audio_data.ok_or_else(|| format!("Missing data chunk in {}", path.display()))?;

    if audio_format != 1 {
        return Err(format!(
            "Unsupported WAV encoding {} in {} (expected PCM)",
            audio_format,
            path.display()
        ));
    }
    if channels != 1 {
        return Err(format!(
            "Unsupported channel count {} in {} (expected mono)",
            channels,
            path.display()
        ));
    }
    if sample_rate != 16_000 {
        return Err(format!(
            "Unsupported sample rate {} in {} (expected 16000 Hz)",
            sample_rate,
            path.display()
        ));
    }
    if bits_per_sample != 16 {
        return Err(format!(
            "Unsupported bit depth {} in {} (expected PCM16)",
            bits_per_sample,
            path.display()
        ));
    }

    Ok(audio_data
        .chunks_exact(2)
        .map(|chunk| {
            let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
            sample as f32 / 32768.0
        })
        .collect())
}
