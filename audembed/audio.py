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


def generate_audio(freqs):

    def _generate_phase(freqs):
        """
        Given a (timesteps, voices)-tensor, output a (time, voices)-tensor
        of phases where time = timesteps * fs
        """
        delta = 2 * torch.pi * freqs / float(44100)
        initial_phase = torch.rand(freqs.shape[1]) * 2 * torch.pi
        phase = initial_phase.unsqueeze(0) + torch.cumsum(delta, axis=0)  # start from random phase
        phase = torch.fmod(phase, 2 * torch.pi) # wrap phase to prevent overflow
        return phase
    
    def _mix(audio):
        GAIN = 0.4
        """ 
        Sum the voice channels of a (channels, time, voices)-tensor, 
        normalize to int16 
        """
        mix = torch.sum(audio, dim=2)
        mix /= torch.max(torch.abs(mix))
        return (mix * 32767 * GAIN).to(torch.int16)
    
    phase = _generate_phase(freqs)
    audio = torch.sin(phase)
    stereo = audio.unsqueeze(0).repeat(2, 1, 1)
    stereo = _mix(stereo)
    # stereo = stereo.permute(1, 0)

    return stereo

if __name__ == "__main__":
    import sounddevice as sd
    import soundfile as sf
    audio = generate_audio(torch.tensor([440, 660]).unsqueeze(0).repeat(44100, 1))
    sd.play(audio.numpy(), samplerate=44100)
    sd.wait()