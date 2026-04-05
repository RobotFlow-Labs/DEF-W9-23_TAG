"""TAG: Thermal Anti-Ghosting — Universal Computational Thermal Imaging.

Physics-based SLOT algorithm for hyperspectral thermal TeX decomposition.
Decomposes thermal radiance into Temperature, Emissivity, and teXture (view factor).

Paper: "Universal computational thermal imaging overcoming the ghosting effect"
arXiv: 2604.01542
"""

__version__ = "0.1.0"

from tag.losses import SLOTObjective
from tag.model import HADARDecomposer, SLOTDecomposer, ThermalForwardModel
from tag.utils import cubic_bspline_basis, planck_radiance, second_order_diff_operator

__all__ = [
    "SLOTDecomposer",
    "ThermalForwardModel",
    "HADARDecomposer",
    "SLOTObjective",
    "planck_radiance",
    "cubic_bspline_basis",
    "second_order_diff_operator",
]

# Lazy import CUDA components (only when GPU available)
def get_cuda_decomposer():
    """Get CUDA-accelerated SLOT decomposer."""
    from tag.cuda.slot_cuda import SLOTDecomposerCUDA
    return SLOTDecomposerCUDA
