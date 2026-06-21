import torch
from einops import rearrange

def to_spectrogram(x, target_frames, target_bins, return_complex=False):
    
    def _next_pow2(n: int) -> int:
        p = 1
        while p < n:
            p <<= 1
        return p

    if x.dim() == 3:
        waveform = x.mean(dim=1)  # (batch, time)
    else:
        waveform = x  # (batch, time)

    B, T = waveform.shape

    if target_frames <= 1:
        raise ValueError("target_frames must be at least 1")
    if target_bins < 2:
        raise ValueError("target_bins must be >= 2")
        
    # bins = n_fft // 2 + 1  =>  n_fft = 2 * (bins - 1)
    n_fft = _next_pow2(2 * (target_bins - 1))
    n_fft = max(2, n_fft)
    
    hop_length = max(1, (T - n_fft) // (target_frames - 1))

    spec = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        return_complex=return_complex,
        center=True,
    )
    
    mag = rearrange(spec.abs(), "batch freq frames complex -> batch frames freq complex")
    mag = mag[..., :target_bins, 0]
    
    return mag