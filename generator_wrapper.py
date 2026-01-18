from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import shutil
import os


BASE_DIR = Path(__file__).resolve().parent  # .../app

# Engine script path (must exist in the GitHub repo at deploy time)
ENGINE_SCRIPT = (
    BASE_DIR
    / "Scripts"
    / "OUT_HEALTH_CARD"
    / "health_card_master_v1_9g_REPORTCARD_FOLDERS_GIS_SYNTH.py"
)

ENGINE_DIR = ENGINE_SCRIPT.parent  # .../app/Scripts/OUT_HEALTH_CARD


def _debug_tree(p: Path, depth: int = 3) -> str:
    """Small directory tree to help diagnose Render path issues."""
    if not p.exists():
        return f"[missing] {p}"
    lines = [f"[tree] {p}"]
    base_parts = len(p.parts)

    for child in sorted(p.rglob("*")):
        rel_depth = len(child.parts) - base_parts
        if rel_depth > depth:
            continue
        mark = "/" if child.is_dir() else ""
        lines.append("  " * rel_depth + f"- {child.name}{mark}")
    return "\n".join(lines)


def run_healthcard(lat: float, lon: float, out_dir: Path) -> Path:
    """
    Run the HealthCard engine and return a Path to HealthCard.html
    that is inside `out_dir` (so FastAPI can serve it under /jobs).

    This version matches your engine behavior:
      - Engine is run with --lat/--lon only
      - Engine writes output under ENGINE_DIR/OUT_HEALTH_CARD or ENGINE_DIR/HealthCard_* (depending on your script)
      - We locate the newest HealthCard_* folder and copy it into out_dir
    """
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Verify engine exists (Render is Linux: path is case-sensitive)
    if not ENGINE_SCRIPT.exists():
        # Print trees to diagnose what Render actually has
        msg = (
            f"Engine script not found at: {ENGINE_SCRIPT}\n\n"
            f"{_debug_tree(BASE_DIR, depth=4)}\n\n"
            f"{_debug_tree(BASE_DIR / 'Scripts', depth=4)}\n"
        )
        raise FileNotFoundError(msg)

    # 2) Run engine (NO --out_dir unless your engine truly supports it)
    cmd = [
        sys.executable,
        str(ENGINE_SCRIPT),
        "--lat", str(lat),
        "--lon", str(lon),
    ]

    subprocess.run(cmd, check=True, cwd=str(ENGINE_DIR))

    # 3) Find newest HealthCard_* folder produced by the engine
    #    Your earlier local logic used OUT_HEALTH_CARD/HealthCard_*
    out_root = ENGINE_DIR / "OUT_HEALTH_CARD"
    candidates = []

    if out_root.exists():
        candidates.extend(out_root.glob("HealthCard_*"))
    # sometimes scripts write directly under ENGINE_DIR
    candidates.extend(ENGINE_DIR.glob("HealthCard_*"))

    candidates = [c for c in candidates if c.is_dir()]
    if not candidates:
        raise FileNotFoundError(
            "No HealthCard_* output folder found after engine run.\n\n"
            f"Looked in:\n- {out_root}\n- {ENGINE_DIR}\n\n"
            f"{_debug_tree(ENGINE_DIR, depth=3)}"
        )

    newest = max(candidates, key=lambda p: p.stat().st_mtime)

    # 4) Copy into the job folder (so it’s served by /jobs/<job_id>/...)
    dest = out_dir / newest.name
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(newest, dest)

    html = dest / "HealthCard.html"
    if not html.exists():
        raise FileNotFoundError(
            f"Expected HealthCard.html missing at: {html}\n\n"
            f"{_debug_tree(dest, depth=3)}"
        )

    return html
