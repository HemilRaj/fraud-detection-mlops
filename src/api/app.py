"""FastAPI application factory and lifecycle management."""

import os
from contextlib import asynccontextmanager
import joblib
import yaml
from fastapi import FastAPI
from src.api.routes import router


def load_config(config_path: str = "configs/service_config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifecycle manager: pre-loads the model artifact into memory before serving requests.
    """
    config = load_config()
    app.state.config = config
    model_path = config["model"]["model_path"]

    if os.path.exists(model_path):
        print(f"🚀 Preloading model artifact into memory from {model_path}...")
        app.state.model = joblib.load(model_path)
        print("✅ Model loaded and warm for inference.")
    else:
        print(f"⚠️ Warning: Model artifact not found at {model_path}. Run training first.")
        app.state.model = None

    yield
    print("🛑 Shutting down serving service...")


def create_app() -> FastAPI:
    config = load_config()
    app = FastAPI(
        title=config["service"]["title"],
        version=config["service"]["version"],
        description=config["service"]["description"],
        lifespan=lifespan,
    )
    app.include_router(router, prefix="/api/v1")
    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000, reload=True)