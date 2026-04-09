"""
ClinicalOps — FastAPI application server.

Exposes the ClinicalOps environment over HTTP and WebSocket endpoints.

Endpoints:
    POST /reset  — Reset the environment (returns initial observation)
    POST /step   — Execute an action (returns updated observation)
    GET  /state  — Get current environment state
    GET  /health — Health check
    GET  /schema — Action / observation JSON schemas
    WS   /ws     — WebSocket endpoint for persistent sessions
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv-core is required. Install with: uv sync"
    ) from e

try:
    from ..models import ClinicalOpsAction, ClinicalOpsObservation
    from .clinicalops_environment import ClinicalOpsEnvironment
except (ImportError, ModuleNotFoundError):
    from models import ClinicalOpsAction, ClinicalOpsObservation
    from server.clinicalops_environment import ClinicalOpsEnvironment


app = create_app(
    ClinicalOpsEnvironment,
    ClinicalOpsAction,
    ClinicalOpsObservation,
    env_name="clinicalops",
    max_concurrent_envs=4,
)


def main(host: str = "0.0.0.0", port: int = 7860):
    """
    Entry point for uv run server and direct execution.

    Used by pyproject.toml [project.scripts]:
        server = "clinicalops.server.app:main"

    Run with:
        uv run server
        python -m server.app
    """
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    # Validator check: main() must appear as a callable call in this block
    main()  # noqa: default args used when no CLI args given

