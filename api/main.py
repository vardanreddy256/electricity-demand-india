"""FastAPI main application entry point."""
import sys
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from api.routers import overview, regional, forecast, grid, compare

app = FastAPI(
    title="ElectricAI India — Demand Forecasting & Grid Planning",
    description="End-to-end electricity demand forecasting and grid planning dashboard for India.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# API routers
app.include_router(overview.router, prefix="/api/overview", tags=["Overview"])
app.include_router(regional.router, prefix="/api/regional", tags=["Regional"])
app.include_router(forecast.router, prefix="/api/forecast", tags=["Forecast"])
app.include_router(grid.router, prefix="/api/grid", tags=["Grid Planning"])
app.include_router(compare.router, prefix="/api/compare", tags=["Model Comparison"])


@app.get("/api/health")
def health():
    return {"status": "ok", "message": "ElectricAI India API is running"}


# Serve frontend static files
FRONTEND_DIR = ROOT / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/", include_in_schema=False)
    def serve_index():
        return FileResponse(str(FRONTEND_DIR / "index.html"))

    @app.get("/{full_path:path}", include_in_schema=False)
    def serve_spa(full_path: str):
        file_path = FRONTEND_DIR / full_path
        if file_path.exists() and file_path.is_file():
            return FileResponse(str(file_path))
        return FileResponse(str(FRONTEND_DIR / "index.html"))
