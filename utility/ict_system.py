from enum import IntEnum


class DetectorBinning(IntEnum):
    BINNING1x1 = 1
    BINNING2x2 = 2
    BINNING4x4 = 4


class ArtisQSystem:
    def __init__(self, detector_binning: DetectorBinning):
        self.nb_pixels = (2480//detector_binning, 1920//detector_binning)
        self.pixel_dims = (0.154*detector_binning, 0.154*detector_binning)
        self.carm_span = 1200.0  # mm
