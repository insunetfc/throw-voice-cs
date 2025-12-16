
# dsp_spectral_gate.py
# ---------------------------------------------------
# Lightweight spectral denoiser tailored for telephony TTS.
# - Works best when run at 24 kHz BEFORE downsampling to 8 kHz.
# - Uses STFT, minima-tracking noise estimate, smoothed Wiener mask.
# - Avoids "musical noise" with time/freq smoothing + noise floor.
#
# Usage:
#   from dsp_spectral_gate import spectral_denoise
#   y = spectral_denoise(x, sr=24000, noise_ms=300, reduce_db=12)
#
import numpy as np

def _stft(x, n_fft=1024, hop=240, win=None):
    if win is None:
        win = np.hanning(n_fft).astype(np.float32)
    n_frames = 1 + (len(x) - n_fft) // hop if len(x) >= n_fft else 1
    frames = np.zeros((n_frames, n_fft), dtype=np.float32)
    for i in range(n_frames):
        s = i * hop
        e = min(s + n_fft, len(x))
        chunk = np.zeros(n_fft, dtype=np.float32)
        chunk[:e - s] = x[s:e]
        frames[i] = chunk * win
    spec = np.fft.rfft(frames, n=n_fft, axis=1)
    return spec, win, hop

def _istft(spec, win, hop, n_fft=1024, length=None):
    n_frames = spec.shape[0]
    sig_len = (n_frames - 1) * hop + n_fft
    if length is None:
        length = sig_len
    y = np.zeros(sig_len, dtype=np.float32)
    win_norm = np.zeros(sig_len, dtype=np.float32) + 1e-8
    frames = np.fft.irfft(spec, n=n_fft, axis=1).astype(np.float32)
    for i in range(n_frames):
        s = i * hop
        e = s + n_fft
        y[s:e] += frames[i] * win
        win_norm[s:e] += win * win
    y /= win_norm
    return y[:length]

def spectral_denoise(
    x: np.ndarray,
    sr: int = 24000,
    n_fft: int = 1024,
    hop: int = 240,           # 10 ms @ 24k
    noise_ms: int = 300,      # noise profile from first 300 ms
    reduce_db: float = 12.0,  # target noise reduction
    floor_db: float = -30.0,  # don't kill noise completely (avoid musical noise)
    time_smooth: int = 4,     # frames for temporal smoothing of mask
    freq_smooth: int = 3      # bins for frequency smoothing of mask
) -> np.ndarray:
    if x.size == 0:
        return x.astype(np.float32)
    x = x.astype(np.float32)
    spec, win, hop_sz = _stft(x, n_fft=n_fft, hop=hop)
    mag = np.abs(spec)
    phase = np.angle(spec)

    # Estimate noise magnitude from first noise_ms
    n_frames_noise = max(1, int((noise_ms/1000.0) * sr / hop_sz))
    noise_mag = np.median(mag[:n_frames_noise], axis=0, keepdims=True)

    # Minima tracking (simple)
    min_mag = np.copy(noise_mag)
    alpha = 0.95
    for t in range(1, mag.shape[0]):
        min_mag = np.minimum(alpha * min_mag + (1-alpha) * mag[t:t+1], mag[t:t+1])

    # Build Wiener-like mask
    noise_floor = 10 ** (floor_db / 20.0)
    att = 10 ** (-reduce_db / 20.0)
    # Avoid div by zero
    denom = (min_mag + 1e-6)
    gain = np.clip(1 - (min_mag / np.maximum(mag, denom)) * (1 - att), att, 1.0)
    gain = np.maximum(gain, noise_floor)

    # Smooth in time
    if time_smooth > 1:
        k = np.ones((time_smooth, 1), dtype=np.float32) / time_smooth
        gain = np.convolve(gain.flatten(), k.flatten(), mode="same").reshape(gain.shape)

    # Smooth in frequency
    if freq_smooth > 1:
        from scipy.ndimage import uniform_filter1d
        gain = uniform_filter1d(gain, size=freq_smooth, axis=1, mode="nearest")

    # Apply mask
    denoised_mag = mag * gain
    spec_out = denoised_mag * np.exp(1j * phase)
    y = _istft(spec_out, win, hop_sz, n_fft=n_fft, length=len(x))

    # Light tail fade to avoid clicks
    if len(y) > sr // 50:
        n = sr // 200
        y[-n:] *= np.linspace(1.0, 0.98, n, dtype=np.float32)
    return y.astype(np.float32)
