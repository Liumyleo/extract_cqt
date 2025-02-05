import os
import subprocess
import numpy as np
import torch.nn.functional as F
import torch

def load_wave(audio_path, sr=16000, pos=[0, -1]):
    start_pos, end_pos = pos
    audio_data = open(audio_path, 'rb').read()
    if end_pos == -1:
        command = ['ffmpeg', '-i', '-', '-ss', '{}'.format(start_pos), '-ac', '1', '-ar', '{}'.format(sr), '-f', 's16le', '-']
    else:
        command = ['ffmpeg', '-i', '-', '-ss', '{}'.format(start_pos), '-to', '{}'.format(end_pos), '-ac', '1', '-ar', '{}'.format(sr), '-f', 's16le', '-']
    p = subprocess.run(
        command,
        input=audio_data,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if p.returncode != 0:
        raise RuntimeError(f'Failed to load audio: {p.returncode}')
    return np.frombuffer(p.stdout, np.int16).flatten().astype(np.float32) / 32768.0

def mel_filters(device, n_mels: int = 80) -> torch.Tensor:
    """
    load the mel filterbank matrix for projecting STFT into a Mel spectrogram.
    Allows decoupling librosa dependency; saved using:

        np.savez_compressed(
            "mel_filters.npz",
            mel_80=librosa.filters.mel(sr=16000, n_fft=400, n_mels=80),
        )
    """
    assert n_mels == 80, f"Unsupported n_mels: {n_mels}"
    with np.load(
        os.path.join(os.path.dirname(__file__), "mel_filters.npz")
    ) as f:
        return torch.from_numpy(f[f"mel_{n_mels}"]).to(device)

def log_mel_spectrogram(
    audio: np.ndarray,
    n_fft: int = 400,
    hop_length: int = 160,
    n_mels: int = 80,
    padding: int = 0,
):
    if not isinstance(audio, torch.Tensor):
        audio = torch.from_numpy(audio)

    if padding > 0:
        audio = F.pad(audio, (0, padding))
    window = torch.hann_window(n_fft, device=audio.device)
    stft = torch.stft(audio, n_fft, hop_length, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2

    filters = mel_filters(audio.device, n_mels)
    mel_spec = filters @ magnitudes

    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec



def main(pathIn):
    audio = load_wave(pathIn)
    logmel = log_mel_spectrogram(audio)
    logmel = logmel.detach().cpu().numpy()
    return logmel


if __name__=='__main__':
    pathIn = "./testcase/84_42.wav"
    logmel = main(pathIn)
    print(f"{logmel.shape=}")
