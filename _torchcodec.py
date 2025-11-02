"""
Fallback TorchCodec integration for TorchAudio.
Implements load/save equivalents using torchaudio or soundfile — no torchcodec required.
Should replace ./torchaudio/_torchcodec.py in case of any torchcodec issues
"""

import os
from typing import BinaryIO, Optional, Tuple, Union

import torch

try:
    import torchaudio
except ImportError as e:
    raise ImportError("torchaudio is required for this replacement module") from e

try:
    import soundfile as sf
except ImportError:
    sf = None


def load_with_torchcodec(
    uri: Union[BinaryIO, str, os.PathLike],
    frame_offset: int = 0,
    num_frames: int = -1,
    normalize: bool = True,
    channels_first: bool = True,
    format: Optional[str] = None,
    buffer_size: int = 4096,
    backend: Optional[str] = None,
) -> Tuple[torch.Tensor, int]:
    """
    Load audio data using torchaudio or soundfile (torchcodec-free).
    Returns (waveform, sample_rate).
    """
    # Try torchaudio first
    try:
        waveform, sr = torchaudio.load(
            uri,
            frame_offset=frame_offset,
            num_frames=num_frames,
            normalize=normalize,
            channels_first=channels_first,
            format=format,
        )
        return waveform, sr
    except Exception as e:
        if sf is None:
            raise RuntimeError(f"Failed to load audio with torchaudio: {e}") from e

    # Fallback to soundfile if torchaudio fails
    data, sr = sf.read(uri, always_2d=True)
    tensor = torch.from_numpy(data.T if channels_first else data).float()

    if normalize:
        tensor = torch.clamp(tensor, -1.0, 1.0)

    # Handle frame offsets
    if frame_offset > 0:
        tensor = tensor[:, frame_offset:]
    if num_frames > 0:
        tensor = tensor[:, :num_frames]

    return tensor, sr


def save_with_torchcodec(
    uri: Union[str, os.PathLike],
    src: torch.Tensor,
    sample_rate: int,
    channels_first: bool = True,
    format: Optional[str] = None,
    encoding: Optional[str] = None,
    bits_per_sample: Optional[int] = None,
    buffer_size: int = 4096,
    backend: Optional[str] = None,
    compression: Optional[Union[float, int]] = None,
) -> None:
    """
    Save audio data to file using torchaudio or soundfile (torchcodec-free).
    """
    # Ensure tensor is float32
    src = src.float()
    if channels_first:
        src = src

    # Try torchaudio first
    try:
        torchaudio.save(
            uri,
            src,
            sample_rate,
            channels_first=channels_first,
            format=format,
            encoding=encoding,
            bits_per_sample=bits_per_sample,
        )
        return
    except Exception as e:
        if sf is None:
            raise RuntimeError(f"Failed to save audio with torchaudio: {e}") from e

    # Fallback to soundfile
    data = src.T.numpy() if channels_first else src.numpy()
    sf.write(uri, data, sample_rate)
