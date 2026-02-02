//! Mel spectrogram computation for Qwen3-ASR.
//!
//! This module implements mel spectrogram extraction with 128 mel bins,
//! adapted from Whisper's implementation but configured for Qwen3-ASR.

use std::{f64::consts::PI, sync::Arc};

/// Mel spectrogram parameters for Qwen3-ASR.
pub const N_FFT: usize = 400;
pub const HOP_LENGTH: usize = 160;
pub const N_MELS: usize = 128;
pub const SAMPLE_RATE: usize = 16000;
pub const CHUNK_LENGTH: usize = 30; // seconds

/// Pre-computed mel filterbank for 128 bins.
pub struct MelFilters {
    filters: Vec<f32>,
    n_freqs: usize,
}

impl MelFilters {
    /// Create mel filterbank for specified parameters.
    pub fn new(n_mels: usize, n_fft: usize, sample_rate: usize) -> Self {
        let filters = create_mel_filterbank(n_mels, n_fft, sample_rate);
        let n_freqs = n_fft / 2 + 1;
        Self { filters, n_freqs }
    }

    /// Create default 128-bin mel filterbank.
    #[cfg(test)]
    pub fn default_128() -> Self {
        Self::new(N_MELS, N_FFT, SAMPLE_RATE)
    }

    /// Get filter value at (mel_bin, fft_bin).
    #[inline]
    pub fn get(&self, mel_bin: usize, fft_bin: usize) -> f32 {
        self.filters[mel_bin * self.n_freqs + fft_bin]
    }

    #[cfg(test)]
    pub fn n_mels(&self) -> usize {
        self.filters.len() / self.n_freqs
    }

    /// Get the raw filter data for a specific mel bin.
    /// Returns a slice of n_freqs values.
    pub fn get_filter(&self, mel_bin: usize) -> &[f32] {
        let start = mel_bin * self.n_freqs;
        &self.filters[start..start + self.n_freqs]
    }

    /// Get the number of frequency bins.
    pub fn n_freqs(&self) -> usize {
        self.n_freqs
    }

    /// Dump filterbank info for diagnostic comparison.
    pub fn dump_diagnostics(&self) -> String {
        let n_mels = self.filters.len() / self.n_freqs;
        let mut output = String::new();

        output.push_str(&format!(
            "Mel filterbank: {} mels x {} freqs\n",
            n_mels, self.n_freqs
        ));

        // For each mel bin, find the non-zero range and peak
        for m in 0..n_mels.min(10) {
            // First 10 bins for brevity
            let filter = self.get_filter(m);
            let mut first_nonzero = None;
            let mut last_nonzero = 0;
            let mut peak_idx = 0;
            let mut peak_val = 0.0f32;

            for (i, &v) in filter.iter().enumerate() {
                if v > 1e-6 {
                    if first_nonzero.is_none() {
                        first_nonzero = Some(i);
                    }
                    last_nonzero = i;
                    if v > peak_val {
                        peak_val = v;
                        peak_idx = i;
                    }
                }
            }

            if let Some(first) = first_nonzero {
                output.push_str(&format!(
                    "  Mel {}: bins {}..{}, peak at {} = {:.6}\n",
                    m, first, last_nonzero, peak_idx, peak_val
                ));
            }
        }

        output
    }
}

/// Convert frequency to mel scale.
fn hz_to_mel(freq: f64) -> f64 {
    2595.0 * (1.0 + freq / 700.0).log10()
}

/// Convert mel scale to frequency.
fn mel_to_hz(mel: f64) -> f64 {
    700.0 * (10.0_f64.powf(mel / 2595.0) - 1.0)
}

/// Create triangular mel filterbank matrix.
/// Uses the same approach as librosa and WhisperFeatureExtractor.
fn create_mel_filterbank(
    n_mels: usize,
    n_fft: usize,
    sample_rate: usize,
) -> Vec<f32> {
    let n_freqs = n_fft / 2 + 1;
    let mut filters = vec![0.0f32; n_mels * n_freqs];

    let f_min = 0.0;
    let f_max = sample_rate as f64 / 2.0;

    let mel_min = hz_to_mel(f_min);
    let mel_max = hz_to_mel(f_max);

    // Create n_mels + 2 equally spaced points in mel scale
    let mel_points: Vec<f64> = (0..=n_mels + 1)
        .map(|i| mel_min + (mel_max - mel_min) * i as f64 / (n_mels + 1) as f64)
        .collect();

    // Convert mel points back to Hz
    let hz_points: Vec<f64> =
        mel_points.iter().map(|&m| mel_to_hz(m)).collect();

    // Convert Hz to FFT bin indices using floating point for interpolation
    // This is the key fix: use f64 for bin positions to enable proper interpolation
    let bin_points: Vec<f64> = hz_points
        .iter()
        .map(|&f| (n_fft as f64) * f / sample_rate as f64)
        .collect();

    // Create triangular filters using floating point interpolation
    // This properly handles the case where mel centers are closer together than FFT bins
    for m in 0..n_mels {
        let f_left = bin_points[m];
        let f_center = bin_points[m + 1];
        let f_right = bin_points[m + 2];

        // Iterate over all FFT bins that could be affected by this filter
        let k_start = f_left.floor() as usize;
        let k_end = (f_right.ceil() as usize).min(n_freqs - 1);

        for k in k_start..=k_end {
            let k_f = k as f64;

            // Rising slope: from f_left to f_center
            if k_f >= f_left && k_f < f_center && f_center > f_left {
                let weight = (k_f - f_left) / (f_center - f_left);
                filters[m * n_freqs + k] = weight as f32;
            }
            // Falling slope: from f_center to f_right
            else if k_f >= f_center && k_f <= f_right && f_right > f_center {
                let weight = (f_right - k_f) / (f_right - f_center);
                filters[m * n_freqs + k] = weight as f32;
            }
        }
    }

    // Normalize filters (slaney normalization)
    for m in 0..n_mels {
        let enorm = 2.0
            / (mel_to_hz(mel_points[m + 2]) - mel_to_hz(mel_points[m])) as f32;
        for k in 0..n_freqs {
            filters[m * n_freqs + k] *= enorm;
        }
    }

    filters
}

/// Pre-computed Hann window for Fft.
pub struct HannWindow {
    window: Vec<f32>,
}

impl HannWindow {
    pub fn new(size: usize) -> Self {
        let window: Vec<f32> = (0..size)
            .map(|i| {
                0.5 * (1.0 - (2.0 * PI * i as f64 / size as f64).cos()) as f32
            })
            .collect();
        Self { window }
    }

    pub fn as_slice(&self) -> &[f32] {
        &self.window
    }
}

/// Fft computation using Cooley-Tukey algorithm.
pub struct Fft {
    size: usize,
    twiddles_re: Vec<f32>,
    twiddles_im: Vec<f32>,
}

impl Fft {
    pub fn new(size: usize) -> Self {
        assert!(size.is_power_of_two(), "Fft size must be power of 2");

        let mut twiddles_re = Vec::with_capacity(size / 2);
        let mut twiddles_im = Vec::with_capacity(size / 2);

        for i in 0..size / 2 {
            let angle = -2.0 * PI * i as f64 / size as f64;
            twiddles_re.push(angle.cos() as f32);
            twiddles_im.push(angle.sin() as f32);
        }

        Self {
            size,
            twiddles_re,
            twiddles_im,
        }
    }

    /// Compute Fft magnitude spectrum (only positive frequencies).
    /// Output is size/2 + 1 magnitudes.
    pub fn magnitude_spectrum(&self, input: &[f32], output: &mut [f32]) {
        assert!(input.len() >= self.size);
        assert!(output.len() > self.size / 2);

        // Allocate complex buffer
        let mut real = vec![0.0f32; self.size];
        let mut imag = vec![0.0f32; self.size];

        // Copy input to real part
        real[..self.size].copy_from_slice(&input[..self.size]);

        // In-place Fft
        self.fft_inplace(&mut real, &mut imag);

        // Compute magnitude squared for positive frequencies
        for i in 0..=self.size / 2 {
            output[i] = real[i] * real[i] + imag[i] * imag[i];
        }
    }

    fn fft_inplace(&self, real: &mut [f32], imag: &mut [f32]) {
        let n = self.size;

        // Bit-reversal permutation
        let mut j = 0;
        for i in 0..n {
            if i < j {
                real.swap(i, j);
                imag.swap(i, j);
            }
            let mut m = n >> 1;
            while m > 0 && j >= m {
                j -= m;
                m >>= 1;
            }
            j += m;
        }

        // Cooley-Tukey iterative Fft
        let mut len = 2;
        while len <= n {
            let half_len = len / 2;
            let step = n / len;

            for i in (0..n).step_by(len) {
                for k in 0..half_len {
                    let idx = k * step;
                    let tw_re = self.twiddles_re[idx];
                    let tw_im = self.twiddles_im[idx];

                    let u_re = real[i + k];
                    let u_im = imag[i + k];
                    let v_re = real[i + k + half_len];
                    let v_im = imag[i + k + half_len];

                    let t_re = tw_re * v_re - tw_im * v_im;
                    let t_im = tw_re * v_im + tw_im * v_re;

                    real[i + k] = u_re + t_re;
                    imag[i + k] = u_im + t_im;
                    real[i + k + half_len] = u_re - t_re;
                    imag[i + k + half_len] = u_im - t_im;
                }
            }
            len *= 2;
        }
    }
}

/// Mel spectrogram extractor for Qwen3-ASR.
pub struct MelSpectrogram {
    mel_filters: Arc<MelFilters>,
    hann_window: HannWindow,
    fft: Fft,
    n_fft: usize,
    hop_length: usize,
    n_mels: usize,
}

impl MelSpectrogram {
    /// Create mel spectrogram extractor with default Qwen3-ASR parameters.
    pub fn new() -> Self {
        Self::with_params(N_MELS, N_FFT, HOP_LENGTH, SAMPLE_RATE)
    }

    /// Create mel spectrogram extractor with custom parameters.
    pub fn with_params(
        n_mels: usize,
        n_fft: usize,
        hop_length: usize,
        sample_rate: usize,
    ) -> Self {
        // Round up n_fft to next power of 2 for Fft
        let fft_size = n_fft.next_power_of_two();

        Self {
            mel_filters: Arc::new(MelFilters::new(n_mels, n_fft, sample_rate)),
            hann_window: HannWindow::new(n_fft),
            fft: Fft::new(fft_size),
            n_fft,
            hop_length,
            n_mels,
        }
    }

    /// Compute log mel spectrogram from audio samples.
    ///
    /// Input: f32 audio samples at 16kHz
    /// Output: (n_frames, n_mels) mel spectrogram
    pub fn compute(&self, audio: &[f32]) -> Vec<f32> {
        let n_samples = audio.len();
        let n_frames =
            (n_samples.saturating_sub(self.n_fft)) / self.hop_length + 1;

        if n_frames == 0 {
            return vec![0.0; self.n_mels];
        }

        let mut mel_spec = vec![0.0f32; n_frames * self.n_mels];
        let fft_size = self.n_fft.next_power_of_two();
        let n_freqs = self.n_fft / 2 + 1;

        // Temporary buffers
        let mut windowed = vec![0.0f32; fft_size];
        let mut magnitudes = vec![0.0f32; fft_size / 2 + 1];

        for frame in 0..n_frames {
            let start = frame * self.hop_length;
            let end = (start + self.n_fft).min(n_samples);

            // Apply Hann window
            windowed.fill(0.0);
            for (i, &w) in self.hann_window.as_slice().iter().enumerate() {
                if start + i < end {
                    windowed[i] = audio[start + i] * w;
                }
            }

            // Compute Fft magnitude spectrum
            self.fft.magnitude_spectrum(&windowed, &mut magnitudes);

            // Apply mel filterbank
            for m in 0..self.n_mels {
                let sum: f32 = magnitudes[..n_freqs]
                    .iter()
                    .enumerate()
                    .map(|(k, &mag)| mag * self.mel_filters.get(m, k))
                    .sum();
                // Log mel spectrogram with floor (using log10 like Whisper)
                mel_spec[frame * self.n_mels + m] = (sum.max(1e-10)).log10();
            }
        }

        // Whisper-style normalization:
        // 1. Clamp to (max - 8.0) to limit dynamic range
        // 2. Normalize to roughly [-1, 1] range
        let max_val =
            mel_spec.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        for v in &mut mel_spec {
            // Clamp minimum to max - 8.0 (80dB dynamic range)
            *v = v.max(max_val - 8.0);
            // Shift and scale: (x + 4.0) / 4.0 maps to roughly [-1, 1]
            *v = (*v + 4.0) / 4.0;
        }

        mel_spec
    }

    /// Compute mel spectrogram and return as 2D structure.
    /// Returns (mel_spec, n_frames, n_mels).
    pub fn compute_2d(&self, audio: &[f32]) -> (Vec<f32>, usize, usize) {
        let n_samples = audio.len();
        let n_frames =
            (n_samples.saturating_sub(self.n_fft)) / self.hop_length + 1;
        let mel_spec = self.compute(audio);
        (mel_spec, n_frames.max(1), self.n_mels)
    }
}

impl Default for MelSpectrogram {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mel_filters_creation() {
        let filters = MelFilters::default_128();
        assert_eq!(filters.n_mels(), 128);
    }

    #[test]
    fn test_hann_window() {
        let window = HannWindow::new(400);
        assert_eq!(window.as_slice().len(), 400);
        // First and last values should be close to 0
        assert!(window.as_slice()[0].abs() < 0.01);
        assert!(window.as_slice()[399].abs() < 0.01);
        // Middle value should be close to 1
        assert!((window.as_slice()[200] - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_fft_size() {
        let fft = Fft::new(512);
        assert_eq!(fft.size, 512);
    }

    #[test]
    fn test_mel_spectrogram_basic() {
        let mel_spec = MelSpectrogram::new();

        // Create 1 second of silence
        let audio = vec![0.0f32; 16000];
        let result = mel_spec.compute(&audio);

        // Should produce some frames
        assert!(!result.is_empty());
    }

    #[test]
    fn test_mel_spectrogram_sine_wave() {
        let mel_spec = MelSpectrogram::new();

        // Create 1 second of 440Hz sine wave
        let audio: Vec<f32> = (0..16000)
            .map(|i| {
                (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16000.0).sin()
            })
            .collect();

        let result = mel_spec.compute(&audio);
        assert!(!result.is_empty());
    }

    #[test]
    fn test_mel_filterbank_diagnostics() {
        let filters = MelFilters::default_128();
        let diag = filters.dump_diagnostics();
        eprintln!("{}", diag);

        // Verify basic properties
        assert_eq!(filters.n_freqs(), N_FFT / 2 + 1); // 201 frequency bins

        // Debug: print first few filters in detail
        for m in 0..5 {
            let filter = filters.get_filter(m);
            let nonzero: Vec<(usize, f32)> = filter
                .iter()
                .enumerate()
                .filter(|(_, &v)| v > 1e-10)
                .map(|(i, &v)| (i, v))
                .collect();
            eprintln!("Filter {}: {} non-zero bins: {:?}", m, nonzero.len(), nonzero);
        }

        // Also check filter 64 (middle)
        let filter64 = filters.get_filter(64);
        let nonzero64: Vec<(usize, f32)> = filter64
            .iter()
            .enumerate()
            .filter(|(_, &v)| v > 1e-10)
            .map(|(i, &v)| (i, v))
            .collect();
        eprintln!("Filter 64: {} non-zero bins", nonzero64.len());

        // Filters should have overlapping triangular shapes
        // This is a known issue with 128 mels and n_fft=400: low freq filters are sparse
    }
}
