import numpy as np
import torch
import torch.nn.functional as F
from torch_radon import cuda_backend
from torch_radon.filtering import FourierFilters


def mu2hu(volume: torch.Tensor, mu_water: float) -> torch.Tensor:
    return (volume - mu_water)/mu_water * 1000


def hu2mu(volume: torch.Tensor, mu_water: float) -> torch.Tensor:
    return (volume * mu_water)/1000 + mu_water


def filter_sinogram_3d(sinogram: torch.Tensor, filter_name="ramp"):
    fft_cache = cuda_backend.FFTCache(8)
    fourier_filters = FourierFilters()
    sinogram = sinogram.permute(0, 1, 3, 2, 4)
    sino_shape = sinogram.shape
    sinogram = sinogram.reshape(np.prod(sino_shape[:-2]), sino_shape[-2], sino_shape[-1])
    size = sinogram.size(-1)
    n_angles = sinogram.size(-2)

    # Pad sinogram to improve accuracy
    padded_size = max(64, int(2 ** np.ceil(np.log2(2 * size))))
    pad = padded_size - size
    padded_sinogram = F.pad(sinogram.float(), (0, pad, 0, 0))

    sino_fft = cuda_backend.rfft(padded_sinogram, fft_cache) / np.sqrt(padded_size)

    # get filter and apply
    f = fourier_filters.get(padded_size, filter_name, sinogram.device)
    filtered_sino_fft = sino_fft * f

    # Inverse fft
    filtered_sinogram = cuda_backend.irfft(filtered_sino_fft, fft_cache) / np.sqrt(padded_size)
    filtered_sinogram = filtered_sinogram[:, :, :-pad] * (np.pi / (2 * n_angles))

    return filtered_sinogram.to(dtype=sinogram.dtype).reshape(sino_shape).permute(0, 1, 3, 2, 4)
