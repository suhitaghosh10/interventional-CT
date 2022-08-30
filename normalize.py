from typing import Union

import numpy as np
import torch


def hu2mu(volume: Union[np.ndarray, torch.Tensor], mu_water: float = 0.02) -> Union[np.ndarray, torch.Tensor]:
    return (volume * mu_water)/1000 + mu_water


def hu_to_normalized(volume: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    return hu2mu(volume)/hu2mu(1720.43359375)


if __name__ == '__main__':
    print(hu2mu(1538775200.0))
