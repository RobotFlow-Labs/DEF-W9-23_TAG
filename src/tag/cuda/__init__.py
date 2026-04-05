"""CUDA-accelerated kernels for TAG thermal imaging pipeline.

Provides fused GPU implementations of:
- Planck radiance computation (batched)
- B-spline basis evaluation (vectorized, no recursion)
- Fused SLOT forward model (Planck + rendering equation)
- Batched Voronoi region assignment for synthetic data generation
"""

from tag.cuda.kernels import (
    bspline_basis_cuda,
    fused_slot_forward_cuda,
    planck_radiance_cuda,
    voronoi_assign_cuda,
)

__all__ = [
    "planck_radiance_cuda",
    "bspline_basis_cuda",
    "fused_slot_forward_cuda",
    "voronoi_assign_cuda",
]
