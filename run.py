"""One-command startup: checks models, starts FastAPI server."""
import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).parent
MODELS_DIR = ROOT / "saved_models"
METRICS_FILE = MODELS_DIR / "metrics.json"


def main():
    print("=" * 60)
    print("  ⚡ ElectricAI India — Starting Up")
    print("=" * 60)

    # Check if models are trained
    if not METRICS_FILE.exists():
        print("\n⚠️  Models not found. Running training pipeline first...")
        print("   (This may take 5–10 minutes for the first run)\n")
        result = subprocess.run([sys.executable, "scripts/run_pipeline.py"], cwd=str(ROOT))
        if result.returncode != 0:
            print("\n❌ Training failed. Check the output above.")
            sys.exit(1)
    else:
        print("\n✅ Pre-trained models found — skipping training.")

    print("\n🚀 Starting FastAPI server...")
    print("   Dashboard → http://localhost:8000")
    print("   API Docs  → http://localhost:8000/docs")
    print("\n   Press Ctrl+C to stop\n")

    subprocess.run([
        sys.executable, "-m", "uvicorn",
        "api.main:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--reload",
    ], cwd=str(ROOT))


if __name__ == "__main__":
    main()
