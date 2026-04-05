"""TAG serving module: FastAPI endpoints and ROS2 node for thermal decomposition.

Provides:
  - POST /decompose — submit hyperspectral thermal cube, get T/e/V maps
  - POST /texture — submit cube, get high-fidelity texture image
  - GET /health, /ready, /info — standard ANIMA endpoints
"""

from __future__ import annotations

import os
import time

import numpy as np
import torch

from tag.model import SLOTDecomposer
from tag.utils import default_wavenumber_grid

MODULE_NAME = os.getenv("ANIMA_MODULE_NAME", "project_tag")
MODULE_VERSION = os.getenv("ANIMA_MODULE_VERSION", "0.1.0")
SERVE_PORT = int(os.getenv("ANIMA_SERVE_PORT", "8080"))
DEVICE = os.getenv("ANIMA_DEVICE", "auto")


class TAGServer:
    """TAG inference server wrapping SLOTDecomposer."""

    def __init__(self):
        self.decomposer: SLOTDecomposer | None = None
        self.device: torch.device = torch.device("cpu")
        self.ready = False
        self.start_time = time.time()

    def setup(self):
        """Initialize the SLOT decomposer."""
        if DEVICE == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(DEVICE)

        wg = default_wavenumber_grid(self.device)
        self.decomposer = SLOTDecomposer(
            n_knots=20,
            reg_lambda=1.0,
            max_iter=100,
            wavenumber_grid=wg,
        ).to(self.device)

        self.ready = True
        print(f"[TAG] Server ready on {self.device}")

    def health(self) -> dict:
        """Health check endpoint."""
        gpu_vram_mb = 0
        if torch.cuda.is_available():
            gpu_vram_mb = torch.cuda.memory_allocated() / 1024 / 1024

        return {
            "status": "healthy" if self.ready else "starting",
            "module": MODULE_NAME,
            "uptime_s": time.time() - self.start_time,
            "gpu_vram_mb": gpu_vram_mb,
        }

    def ready_check(self) -> dict:
        """Readiness check endpoint."""
        return {
            "ready": self.ready,
            "module": MODULE_NAME,
            "version": MODULE_VERSION,
            "weights_loaded": self.decomposer is not None,
        }

    def info(self) -> dict:
        """Module info endpoint."""
        return {
            "module": MODULE_NAME,
            "version": MODULE_VERSION,
            "description": "TAG: Thermal Anti-Ghosting via SLOT decomposition",
            "device": str(self.device),
            "endpoints": ["/decompose", "/texture", "/health", "/ready", "/info"],
        }

    def decompose(
        self,
        s_obs: np.ndarray,
        s_sky: np.ndarray,
        s_ground: np.ndarray,
    ) -> dict:
        """Run TeX decomposition on input hyperspectral data.

        Args:
            s_obs: [H, W, C] observed spectral radiance.
            s_sky: [C] sky reference.
            s_ground: [C] ground reference.

        Returns:
            Dict with temperature, emissivity, view_factor, texture maps.
        """
        if not self.ready:
            raise RuntimeError("Server not ready")

        s_obs_t = torch.from_numpy(s_obs).float().to(self.device)
        s_sky_t = torch.from_numpy(s_sky).float().to(self.device)
        s_ground_t = torch.from_numpy(s_ground).float().to(self.device)

        result = self.decomposer.decompose(s_obs_t, s_sky_t, s_ground_t)

        return {
            "temperature": result.temperature.cpu().numpy().tolist(),
            "emissivity": result.emissivity.cpu().numpy().tolist(),
            "view_factor": result.view_factor.cpu().numpy().tolist(),
            "texture": result.texture.cpu().numpy().tolist(),
            "objective": result.objective,
            "n_iterations": result.n_iterations,
        }


def create_app():
    """Create FastAPI application."""
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import JSONResponse
    except ImportError:
        print("[TAG] FastAPI not available. Install with: pip install fastapi uvicorn")
        return None

    app = FastAPI(
        title="TAG — Thermal Anti-Ghosting",
        version=MODULE_VERSION,
        description="SLOT-based hyperspectral thermal TeX decomposition",
    )

    server = TAGServer()

    @app.on_event("startup")
    async def startup():
        server.setup()

    @app.get("/health")
    async def health():
        return server.health()

    @app.get("/ready")
    async def ready():
        info = server.ready_check()
        if not info["ready"]:
            raise HTTPException(status_code=503, detail="Not ready")
        return info

    @app.get("/info")
    async def info():
        return server.info()

    @app.post("/decompose")
    async def decompose(data: dict):
        """Decompose hyperspectral thermal data.

        Expected JSON body:
        {
            "s_obs": [[...], ...],    # [H, W, C] nested list
            "s_sky": [...],           # [C] list
            "s_ground": [...]         # [C] list
        }
        """
        try:
            s_obs = np.array(data["s_obs"], dtype=np.float32)
            s_sky = np.array(data["s_sky"], dtype=np.float32)
            s_ground = np.array(data["s_ground"], dtype=np.float32)
            result = server.decompose(s_obs, s_sky, s_ground)
            return JSONResponse(content=result)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.post("/texture")
    async def texture(data: dict):
        """Extract high-fidelity texture from thermal data.

        Same input as /decompose, returns only texture map.
        """
        try:
            s_obs = np.array(data["s_obs"], dtype=np.float32)
            s_sky = np.array(data["s_sky"], dtype=np.float32)
            s_ground = np.array(data["s_ground"], dtype=np.float32)
            result = server.decompose(s_obs, s_sky, s_ground)
            return JSONResponse(content={"texture": result["texture"]})
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    return app


def main():
    """Run the TAG serving module."""
    app = create_app()
    if app is not None:
        import uvicorn

        uvicorn.run(app, host="0.0.0.0", port=SERVE_PORT)
    else:
        print("[TAG] Cannot start server without FastAPI. Exiting.")


if __name__ == "__main__":
    main()
