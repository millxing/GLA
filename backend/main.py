import os
from importlib.metadata import PackageNotFoundError, version as pkg_version

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

REQUIRED_SKLEARN_VERSION = "1.8.0"


def _assert_runtime_dependencies() -> None:
    try:
        sklearn_version = pkg_version("scikit-learn")
    except PackageNotFoundError as exc:
        raise RuntimeError("scikit-learn is required and must be installed at version 1.8.0") from exc
    if sklearn_version != REQUIRED_SKLEARN_VERSION:
        raise RuntimeError(
            f"scikit-learn=={REQUIRED_SKLEARN_VERSION} is required; found {sklearn_version}. "
            "Install/activate the correct environment before starting the API."
        )


_assert_runtime_dependencies()

load_dotenv()

from routers.api import router as api_router

app = FastAPI(
    title="NBA Game Log Analytics API",
    description="API for NBA game log analytics with Four Factors analysis",
    version="1.0.0",
)

allowed_origins = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", "http://localhost:5173,http://localhost:3000").split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

app.include_router(api_router)

@app.get("/")
async def root():
    return {
        "message": "NBA Game Log Analytics API",
        "version": "1.0.0",
        "docs": "/docs",
    }

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
