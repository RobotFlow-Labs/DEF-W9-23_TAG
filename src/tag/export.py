"""Export pipeline for TAG: ONNX + TensorRT FP16/FP32.

TAG exports the batched forward model (Planck + rendering equation)
as a traceable module. The SLOT optimization loop itself uses L-BFGS
which is not directly exportable, but the forward model + B-spline
evaluation can be exported for fast inference on edge devices.

Export chain: PyTorch → safetensors → ONNX → TRT FP16 → TRT FP32
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor

from tag.cuda.kernels import bspline_basis_cuda
from tag.utils import (
    C1,
    C2,
    default_wavenumber_grid,
    planck_radiance,
    second_order_diff_operator,
)

PROJECT = "project_tag"
ARTIFACTS = "/mnt/artifacts-datai"


class TAGForwardModule(nn.Module):
    """Exportable TAG forward model for inference.

    Given (T, beta, V, s_sky, s_ground), computes:
    1. Emissivity from B-spline coefficients: e = beta @ Phi^T
    2. Planck radiance: B(T)
    3. Ambient texture: X = V*S_sky + (1-V)*S_ground
    4. Rendering equation: S = e*B + (1-e)*X
    5. Texture map: mean(e)

    This is the forward model that runs during SLOT iterations.
    Exporting it enables fast batch inference on edge/embedded.
    """

    def __init__(self, n_knots: int = 20, wavenumber_grid: Tensor | None = None):
        super().__init__()
        if wavenumber_grid is None:
            wavenumber_grid = default_wavenumber_grid()

        phi = bspline_basis_cuda(wavenumber_grid, n_knots)

        self.register_buffer("wavenumber_grid", wavenumber_grid)
        self.register_buffer("phi", phi)
        self.register_buffer("c1", torch.tensor(C1))
        self.register_buffer("c2", torch.tensor(C2))

    def forward(
        self,
        temperature: Tensor,
        beta: Tensor,
        view_factor: Tensor,
        s_sky: Tensor,
        s_ground: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Forward pass for export.

        Args:
            temperature: [N] temperatures in K.
            beta: [N, K] B-spline coefficients.
            view_factor: [N] view factors.
            s_sky: [C] sky reference.
            s_ground: [C] ground reference.

        Returns:
            (s_model, emissivity, texture): [N,C], [N,C], [N].
        """
        # Emissivity
        emissivity = beta @ self.phi.T  # [N, C]

        # Planck radiance
        v = self.wavenumber_grid
        t = temperature.unsqueeze(-1)
        exponent = torch.clamp(self.c2 * v / t, max=500.0)
        b_planck = self.c1 * v.pow(3) / (torch.exp(exponent) - 1.0)

        # Ambient
        vf = view_factor.unsqueeze(-1)
        x_ambient = vf * s_sky + (1.0 - vf) * s_ground

        # Rendering
        s_model = emissivity * b_planck + (1.0 - emissivity) * x_ambient

        # Texture
        texture = emissivity.mean(dim=-1)

        return s_model, emissivity, texture


class TAGDecomposeModule(nn.Module):
    """Exportable single-step SLOT iteration for inference.

    Performs one forward pass + objective computation.
    Used for TensorRT inference in iterative optimization loops
    on edge devices.
    """

    def __init__(self, n_knots: int = 20, reg_lambda: float = 1.0,
                 constraint_mu: float = 10.0, wavenumber_grid: Tensor | None = None):
        super().__init__()
        if wavenumber_grid is None:
            wavenumber_grid = default_wavenumber_grid()

        phi = bspline_basis_cuda(wavenumber_grid, n_knots)
        d_beta = second_order_diff_operator(phi.shape[1])

        self.register_buffer("wavenumber_grid", wavenumber_grid)
        self.register_buffer("phi", phi)
        self.register_buffer("d_beta", d_beta)
        self.register_buffer("c1", torch.tensor(C1))
        self.register_buffer("c2", torch.tensor(C2))
        self.reg_lambda = reg_lambda
        self.constraint_mu = constraint_mu

    def forward(
        self,
        s_obs: Tensor,
        temperature: Tensor,
        beta: Tensor,
        view_factor: Tensor,
        s_sky: Tensor,
        s_ground: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Single SLOT iteration step.

        Returns:
            (objective, s_model, emissivity): scalar loss, [N,C], [N,C].
        """
        emissivity = beta @ self.phi.T

        v = self.wavenumber_grid
        t = temperature.unsqueeze(-1)
        exponent = torch.clamp(self.c2 * v / t, max=500.0)
        b_planck = self.c1 * v.pow(3) / (torch.exp(exponent) - 1.0)

        vf = view_factor.unsqueeze(-1)
        x_ambient = vf * s_sky + (1.0 - vf) * s_ground
        s_model = emissivity * b_planck + (1.0 - emissivity) * x_ambient

        # Objective
        residual = ((s_obs - s_model) ** 2).sum(dim=-1).mean()
        diff = beta @ self.d_beta.T
        smoothness = (self.reg_lambda / 2.0) * (diff ** 2).sum(dim=-1).mean()
        bound_penalty = self.constraint_mu * (
            torch.relu(-emissivity + 0.01).sum(dim=-1).mean()
            + torch.relu(emissivity - 0.99).sum(dim=-1).mean()
        )
        objective = residual + smoothness + bound_penalty

        return objective, s_model, emissivity


def export_safetensors(model: nn.Module, save_path: Path) -> Path:
    """Export model state to safetensors format."""
    from safetensors.torch import save_file

    state = {k: v.contiguous() for k, v in model.state_dict().items()}
    save_file(state, str(save_path))
    print(f"  [SAFE] Saved: {save_path} ({save_path.stat().st_size / 1024:.1f} KB)")
    return save_path


def export_onnx(model: nn.Module, save_path: Path, n_pixels: int = 1024,
                device: torch.device = torch.device("cpu")) -> Path:
    """Export model to ONNX format."""
    model = model.to(device).eval()
    wg = model.wavenumber_grid
    c = len(wg)
    k = model.phi.shape[1]

    # Create dummy inputs
    if isinstance(model, TAGForwardModule):
        dummy = (
            torch.full((n_pixels,), 300.0, device=device),
            torch.full((n_pixels, k), 0.04, device=device),
            torch.full((n_pixels,), 0.5, device=device),
            planck_radiance(wg, torch.tensor(240.0, device=device)),
            planck_radiance(wg, torch.tensor(290.0, device=device)),
        )
        input_names = ["temperature", "beta", "view_factor", "s_sky", "s_ground"]
        output_names = ["s_model", "emissivity", "texture"]
        dynamic_axes = {
            "temperature": {0: "n_pixels"},
            "beta": {0: "n_pixels"},
            "view_factor": {0: "n_pixels"},
            "s_model": {0: "n_pixels"},
            "emissivity": {0: "n_pixels"},
            "texture": {0: "n_pixels"},
        }
    else:
        dummy = (
            torch.randn(n_pixels, c, device=device).abs() * 1e-5,
            torch.full((n_pixels,), 300.0, device=device),
            torch.full((n_pixels, k), 0.04, device=device),
            torch.full((n_pixels,), 0.5, device=device),
            planck_radiance(wg, torch.tensor(240.0, device=device)),
            planck_radiance(wg, torch.tensor(290.0, device=device)),
        )
        input_names = ["s_obs", "temperature", "beta", "view_factor", "s_sky", "s_ground"]
        output_names = ["objective", "s_model", "emissivity"]
        dynamic_axes = {
            "s_obs": {0: "n_pixels"},
            "temperature": {0: "n_pixels"},
            "beta": {0: "n_pixels"},
            "view_factor": {0: "n_pixels"},
            "s_model": {0: "n_pixels"},
            "emissivity": {0: "n_pixels"},
        }

    torch.onnx.export(
        model,
        dummy,
        str(save_path),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=17,
        do_constant_folding=True,
    )

    # Verify
    import onnx
    onnx_model = onnx.load(str(save_path))
    onnx.checker.check_model(onnx_model)

    print(f"  [ONNX] Saved: {save_path} ({save_path.stat().st_size / 1024:.1f} KB)")
    return save_path


def export_tensorrt(onnx_path: Path, save_path: Path, precision: str = "fp16") -> Path | None:
    """Export ONNX to TensorRT engine.

    Args:
        onnx_path: path to ONNX model.
        save_path: output TRT engine path.
        precision: 'fp16' or 'fp32'.

    Returns:
        Path to TRT engine or None if TRT not available.
    """
    # Try shared TRT toolkit first
    trt_toolkit = Path("/mnt/forge-data/shared_infra/trt_toolkit/export_to_trt.py")
    if trt_toolkit.exists():
        import subprocess
        cmd = [
            "python", str(trt_toolkit),
            "--onnx", str(onnx_path),
            "--output", str(save_path),
            "--precision", precision,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"  [TRT-{precision.upper()}] Saved: {save_path}")
            return save_path
        print(f"  [TRT-{precision.upper()}] Toolkit failed: {result.stderr[:200]}")

    # Fallback: try trtexec
    try:
        import subprocess
        precision_flag = "--fp16" if precision == "fp16" else ""
        cmd = f"trtexec --onnx={onnx_path} --saveEngine={save_path} {precision_flag} --workspace=4096"
        result = subprocess.run(cmd.split(), capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print(f"  [TRT-{precision.upper()}] Saved: {save_path}")
            return save_path
        print(f"  [TRT-{precision.upper()}] trtexec failed: {result.stderr[:200]}")
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        print(f"  [TRT-{precision.upper()}] Not available: {e}")

    # Fallback: try tensorrt Python API
    try:
        import tensorrt as trt

        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, logger)

        with open(str(onnx_path), "rb") as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    print(f"    Parse error: {parser.get_error(i)}")
                return None

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)  # 4GB

        if precision == "fp16" and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)

        engine = builder.build_serialized_network(network, config)
        if engine is None:
            print(f"  [TRT-{precision.upper()}] Build failed")
            return None

        with open(str(save_path), "wb") as f:
            f.write(engine)

        print(f"  [TRT-{precision.upper()}] Saved: {save_path} ({save_path.stat().st_size / 1024:.1f} KB)")
        return save_path
    except ImportError:
        print(f"  [TRT-{precision.upper()}] tensorrt not installed — skipping")
        return None


def run_export(config: dict, device_str: str = "cuda:0"):
    """Run full export pipeline: pth → safetensors → ONNX → TRT."""
    export_dir = Path(f"{ARTIFACTS}/exports/{PROJECT}")
    export_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    slot_cfg = config.get("slot", {})
    n_knots = slot_cfg.get("n_knots", 20)
    reg_lambda = slot_cfg.get("reg_lambda", 1.0)
    constraint_mu = slot_cfg.get("constraint_mu", 10.0)

    print(f"\n{'=' * 60}")
    print(f"[EXPORT] TAG Export Pipeline")
    print(f"[CONFIG] n_knots={n_knots}, device={device}")
    print(f"[OUTPUT] {export_dir}")
    print(f"{'=' * 60}")

    wg = default_wavenumber_grid(device)

    # --- 1. Forward Model ---
    print("\n[1/5] Forward Model (Planck + Rendering)")
    forward_model = TAGForwardModule(n_knots=n_knots, wavenumber_grid=wg).to(device)

    # PyTorch checkpoint
    pth_path = export_dir / f"tag_forward_{run_id}.pth"
    torch.save({
        "model_state": forward_model.state_dict(),
        "config": config,
        "n_knots": n_knots,
    }, pth_path)
    print(f"  [PTH] Saved: {pth_path}")

    # Safetensors
    safe_path = export_dir / f"tag_forward_{run_id}.safetensors"
    export_safetensors(forward_model, safe_path)

    # ONNX
    onnx_fwd_path = export_dir / f"tag_forward_{run_id}.onnx"
    export_onnx(forward_model, onnx_fwd_path, device=device)

    # TRT FP16
    trt_fp16_path = export_dir / f"tag_forward_{run_id}_fp16.engine"
    export_tensorrt(onnx_fwd_path, trt_fp16_path, precision="fp16")

    # TRT FP32
    trt_fp32_path = export_dir / f"tag_forward_{run_id}_fp32.engine"
    export_tensorrt(onnx_fwd_path, trt_fp32_path, precision="fp32")

    # --- 2. Decompose Module (single SLOT step) ---
    print("\n[2/5] Decompose Module (single SLOT iteration)")
    decompose_model = TAGDecomposeModule(
        n_knots=n_knots, reg_lambda=reg_lambda, constraint_mu=constraint_mu,
        wavenumber_grid=wg,
    ).to(device)

    pth_dec_path = export_dir / f"tag_decompose_{run_id}.pth"
    torch.save({
        "model_state": decompose_model.state_dict(),
        "config": config,
    }, pth_dec_path)
    print(f"  [PTH] Saved: {pth_dec_path}")

    safe_dec_path = export_dir / f"tag_decompose_{run_id}.safetensors"
    export_safetensors(decompose_model, safe_dec_path)

    onnx_dec_path = export_dir / f"tag_decompose_{run_id}.onnx"
    export_onnx(decompose_model, onnx_dec_path, device=device)

    trt_dec_fp16_path = export_dir / f"tag_decompose_{run_id}_fp16.engine"
    export_tensorrt(onnx_dec_path, trt_dec_fp16_path, precision="fp16")

    trt_dec_fp32_path = export_dir / f"tag_decompose_{run_id}_fp32.engine"
    export_tensorrt(onnx_dec_path, trt_dec_fp32_path, precision="fp32")

    # --- Summary ---
    print(f"\n{'=' * 60}")
    print("[EXPORT COMPLETE]")
    exports = list(export_dir.glob(f"*{run_id}*"))
    for e in sorted(exports):
        size = e.stat().st_size
        print(f"  {e.name}: {size / 1024:.1f} KB")
    print(f"{'=' * 60}")

    # Save manifest
    manifest = {
        "run_id": run_id,
        "config": config,
        "exports": [str(e) for e in sorted(exports)],
    }
    manifest_path = export_dir / f"manifest_{run_id}.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="TAG export pipeline")
    parser.add_argument("--config", type=str, default="configs/paper.toml")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    try:
        import tomllib
    except ImportError:
        import tomli as tomllib

    with open(args.config, "rb") as f:
        config = tomllib.load(f)

    run_export(config, device_str=f"cuda:{args.gpu}")


if __name__ == "__main__":
    main()
