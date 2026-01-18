from __future__ import annotations

from pathlib import Path
import subprocess
import sys


BASE_DIR = Path(__file__).resolve().parent

# This must match where the big HealthCard script lives in your repo:
ENGINE_SCRIPT = (
    BASE_DIR
    / "Scripts"
    / "OUT_HEALTH_CARD"
    / "health_card_master_v1_9g_REPORTCARD_FOLDERS_GIS_SYNTH.py"
)

def run_healthcard(lat: float, lon: float, out_dir: Path) -> Path:
    """
    Runs the HealthCard engine script and returns the path to the generated HealthCard.html.

    Requirements:
      - ENGINE_SCRIPT exists
      - The engine supports: --lat --lon --out_dir
      - The engine writes HealthCard.html into out_dir
    """
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not ENGINE_SCRIPT.exists():
        raise FileNotFoundError(f"Engine script not found at: {ENGINE_SCRIPT}")

    cmd = [
        sys.executable,              # uses same python as the web server
        str(ENGINE_SCRIPT),
        "--lat", str(lat),
        "--lon", str(lon),
        "--out_dir", str(out_dir),
    ]

    # IMPORTANT: run with cwd at the engine folder so its relative paths work
    subprocess.run(cmd, check=True, cwd=str(ENGINE_SCRIPT.parent))

    html = out_dir / "HealthCard.html"
    if not html.exists():
        raise FileNotFoundError(f"Expected HealthCard.html not found in: {out_dir}")

    return html

