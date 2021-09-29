from typing import Callable, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch_radon import cuda_backend, ConeBeam
from torch_radon.filtering import FourierFilters
from torch_radon.volumes import Volume3D

from utility.ict_system import ArtisQSystem, DetectorBinning


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


def create_projections(
        volume_path: str,
        load_fn: Callable,
        center: Tuple[float, float, float] = (0, 0, 0)) \
        -> Tuple[np.ndarray, Tuple[float, float], Tuple[float, float, float]]:
    ct_system = ArtisQSystem(DetectorBinning.BINNING4x4)
    angles = np.linspace(0, 2*np.pi, 360, endpoint=False, dtype=np.float32)
    src_dist = ct_system.carm_span*4/6
    det_dist = ct_system.carm_span*2/6
    # src_det_dist = src_dist + det_dist
    det_spacing_v = ct_system.pixel_dims[1]
    radon = ConeBeam(
        det_count_u=ct_system.nb_pixels[0],
        angles=angles,
        src_dist=src_dist,
        det_dist=det_dist,
        det_count_v=ct_system.nb_pixels[1],
        det_spacing_u=ct_system.pixel_dims[0],
        det_spacing_v=det_spacing_v,
        pitch=0.0,
        base_z=0.0,
    )

    # create head projections
    volume, voxel_size = load_fn(volume_path)
    volume = volume.transpose()
    # print(volume.shape, voxel_size)
    volume_t = torch.from_numpy(volume).float().cuda()[None, None, ...]
    volume_t = hu2mu(volume_t, 0.02)
    volume_t[volume_t < 0] = 0
    radon.volume = Volume3D(
            depth=volume.shape[0],
            height=volume.shape[1],
            width=volume.shape[2],
            voxel_size=voxel_size,
            center=center)
    projections_t = radon.forward(volume_t).nan_to_num()

    return projections_t[0, 0].cpu().numpy().transpose(), \
        ct_system.pixel_dims, voxel_size, volume.shape
