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
#[allow(dead_code)]
pub struct MelFilters {
    filters: Vec<f32>,
    n_mels: usize,
    n_fft: usize,
}

impl MelFilters {
    /// Create mel filterbank for specified parameters.
    pub fn new(n_mels: usize, n_fft: usize, sample_rate: usize) -> Self {
        let filters = create_mel_filterbank(n_mels, n_fft, sample_rate);
        Self {
            filters,
            n_mels,
            n_fft,
        }
    }

    /// Create default 128-bin mel filterbank.
    #[allow(dead_code)]
    pub fn default_128() -> Self {
        Self::new(N_MELS, N_FFT, SAMPLE_RATE)
    }

    /// Get filter value at (mel_bin, fft_bin).
    #[inline]
    pub fn get(&self, mel_bin: usize, fft_bin: usize) -> f32 {
        self.filters[mel_bin * (self.n_fft / 2 + 1) + fft_bin]
    }

    /// Get the full filterbank as a slice.
    #[allow(dead_code)]
    pub fn as_slice(&self) -> &[f32] {
        &self.filters
    }

    pub fn n_mels(&self) -> usize {
        self.n_mels
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
fn create_mel_filterbank(n_mels: usize, n_fft: usize, sample_rate: usize) -> Vec<f32> {
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
    let hz_points: Vec<f64> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();

    // Convert Hz to FFT bin indices
    let bin_points: Vec<usize> = hz_points
        .iter()
        .map(|&f| ((n_fft as f64 + 1.0) * f / sample_rate as f64).floor() as usize)
        .collect();

    // Create triangular filters
    for m in 0..n_mels {
        let f_left = bin_points[m];
        let f_center = bin_points[m + 1];
        let f_right = bin_points[m + 2];

        // Rising slope
        for k in f_left..f_center {
            if k < n_freqs && f_center > f_left {
                filters[m * n_freqs + k] = (k - f_left) as f32 / (f_center - f_left) as f32;
            }
        }

        // Falling slope
        for k in f_center..=f_right {
            if k < n_freqs && f_right > f_center {
                filters[m * n_freqs + k] = (f_right - k) as f32 / (f_right - f_center) as f32;
            }
        }
    }

    // Normalize filters (slaney normalization)
    for m in 0..n_mels {
        let enorm = 2.0 / (mel_to_hz(mel_points[m + 2]) - mel_to_hz(mel_points[m])) as f32;
        for k in 0..n_freqs {
            filters[m * n_freqs + k] *= enorm;
        }
    }

    filters
}

/// Pre-computed Hann window for FFT.
pub struct HannWindow {
    window: Vec<f32>,
}

impl HannWindow {
    pub fn new(size: usize) -> Self {
        let window: Vec<f32> = (0..size)
            .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f64 / size as f64).cos()) as f32)
            .collect();
        Self { window }
    }

    pub fn as_slice(&self) -> &[f32] {
        &self.window
    }
}

/// FFT computation using Cooley-Tukey algorithm.
pub struct FFT {
    size: usize,
    twiddles_re: Vec<f32>,
    twiddles_im: Vec<f32>,
}

impl FFT {
    pub fn new(size: usize) -> Self {
        assert!(size.is_power_of_two(), "FFT size must be power of 2");

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

    /// Compute FFT magnitude spectrum (only positive frequencies).
    /// Output is size/2 + 1 magnitudes.
    pub fn magnitude_spectrum(&self, input: &[f32], output: &mut [f32]) {
        assert!(input.len() >= self.size);
        assert!(output.len() >= self.size / 2 + 1);

        // Allocate complex buffer
        let mut real = vec![0.0f32; self.size];
        let mut imag = vec![0.0f32; self.size];

        // Copy input to real part
        real[..self.size].copy_from_slice(&input[..self.size]);

        // In-place FFT
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

        // Cooley-Tukey iterative FFT
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
    fft: FFT,
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
    pub fn with_params(n_mels: usize, n_fft: usize, hop_length: usize, sample_rate: usize) -> Self {
        // Round up n_fft to next power of 2 for FFT
        let fft_size = n_fft.next_power_of_two();

        Self {
            mel_filters: Arc::new(MelFilters::new(n_mels, n_fft, sample_rate)),
            hann_window: HannWindow::new(n_fft),
            fft: FFT::new(fft_size),
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
        let n_frames = (n_samples.saturating_sub(self.n_fft)) / self.hop_length + 1;

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

            // Compute FFT magnitude spectrum
            self.fft.magnitude_spectrum(&windowed, &mut magnitudes);

            // Apply mel filterbank
            for m in 0..self.n_mels {
                let mut sum = 0.0f32;
                for k in 0..n_freqs {
                    sum += magnitudes[k] * self.mel_filters.get(m, k);
                }
                // Log mel spectrogram with floor
                mel_spec[frame * self.n_mels + m] = (sum.max(1e-10)).log10();
            }
        }

        // Normalize to [0, 1] range
        let max_val = mel_spec.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let min_val = mel_spec.iter().copied().fold(f32::INFINITY, f32::min);
        let range = (max_val - min_val).max(1e-6);

        for v in &mut mel_spec {
            *v = (*v - min_val) / range;
        }

        mel_spec
    }

    /// Compute mel spectrogram and return as 2D structure.
    /// Returns (mel_spec, n_frames, n_mels).
    pub fn compute_2d(&self, audio: &[f32]) -> (Vec<f32>, usize, usize) {
        let n_samples = audio.len();
        let n_frames = (n_samples.saturating_sub(self.n_fft)) / self.hop_length + 1;
        let mel_spec = self.compute(audio);
        (mel_spec, n_frames.max(1), self.n_mels)
    }

    /// Get the number of mel bins.
    #[allow(dead_code)]
    pub fn n_mels(&self) -> usize {
        self.n_mels
    }

    /// Get the hop length.
    #[allow(dead_code)]
    pub fn hop_length(&self) -> usize {
        self.hop_length
    }

    /// Get the FFT size.
    #[allow(dead_code)]
    pub fn n_fft(&self) -> usize {
        self.n_fft
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
        let fft = FFT::new(512);
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
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16000.0).sin())
            .collect();

        let result = mel_spec.compute(&audio);
        assert!(!result.is_empty());
    }
}
