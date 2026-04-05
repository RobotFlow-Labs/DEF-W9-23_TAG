# PRD-07: Integration

> Status: TODO
> Module: TAG (Thermal Anti-Ghosting)
> Depends on: PRD-06

## Objective

Docker serving, ROS2 node, FastAPI endpoint for TAG thermal decomposition service.

## Deliverables

### 1. Dockerfile.serve
- 3-layer build from anima-serve:jazzy base
- Install TAG package + dependencies
- Runtime: load SLOT model, serve decomposition API

### 2. docker-compose.serve.yml
- Profiles: serve (full), ros2 (ROS2 only), api (FastAPI only), test
- GPU passthrough for Planck/matrix acceleration

### 3. ROS2 Node (`src/tag/serve.py`)

```python
class TAGNode(AnimaNode):
    def setup_inference(self):
        self.decomposer = SLOTDecomposer(config)

    def process(self, input_data):
        # input_data: hyperspectral thermal cube
        return self.decomposer.decompose(input_data)
```

### 4. API Endpoints
- `POST /decompose` — submit hyperspectral cube, get T/e/V maps
- `POST /texture` — submit cube, get high-fidelity texture image
- `GET /health`, `GET /ready`, `GET /info`

### 5. .env.serve
```
ANIMA_MODULE_NAME=project_tag
ANIMA_MODULE_VERSION=0.1.0
ANIMA_HF_REPO=ilessio-aiflowlab/project_tag
ANIMA_SERVE_PORT=8080
```

## Acceptance Criteria
- [ ] Docker image builds successfully
- [ ] Health endpoint returns 200
- [ ] /decompose endpoint processes synthetic input and returns valid T/e/V
- [ ] ROS2 node publishes decomposition results
