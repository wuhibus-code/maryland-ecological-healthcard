from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import uuid
import shutil
import subprocess
import os
import sys
import traceback

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent
JOBS_DIR = BASE_DIR / "jobs"
JOBS_DIR.mkdir(exist_ok=True)

app.mount("/jobs", StaticFiles(directory=str(JOBS_DIR)), name="jobs")

# ✅ Point to the real engine script in your repo
ENGINE_SCRIPT = (
    BASE_DIR / "Scripts" / "OUT_HEALTH_CARD" / "health_card_master_v1_9g_REPORTCARD_FOLDERS_GIS_SYNTH.py"
)

# ✅ Point to the folder where the engine writes HealthCard_* outputs
OUT_ROOT = BASE_DIR / "Scripts" / "OUT_HEALTH_CARD" / "OUT_HEALTH_CARD"
# If in your repo it's different, change ONLY this line.

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <h1>Maryland Ecological HealthCard</h1>
    <form method="post" action="/generate">
      Latitude: <input name="lat"><br><br>
      Longitude: <input name="lon"><br><br>
      <button type="submit">Generate</button>
    </form>
    """

@app.post("/generate", response_class=HTMLResponse)
def generate(lat: float = Form(...), lon: float = Form(...)):
    try:
        job_id = str(uuid.uuid4())[:8]
        job_dir = JOBS_DIR / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        # --- sanity checks ---
        if not ENGINE_SCRIPT.exists():
            raise FileNotFoundError(f"Engine script not found: {ENGINE_SCRIPT}")

        if not OUT_ROOT.exists():
            raise FileNotFoundError(f"OUT_HEALTH_CARD folder not found: {OUT_ROOT}")

        # --- run the engine ---
        subprocess.run(
            [
                sys.executable,  # ✅ correct python (Render + local)
                str(ENGINE_SCRIPT),
                "--lat", str(lat),
                "--lon", str(lon),
            ],
            cwd=str(ENGINE_SCRIPT.parent),  # ✅ run in engine folder so relative paths work
            check=True,
        )

        # --- find newest HealthCard_* folder produced ---
        hc_folders = sorted(OUT_ROOT.glob("HealthCard_*"), key=os.path.getmtime)
        if not hc_folders:
            raise FileNotFoundError(f"No HealthCard_* folders found in: {OUT_ROOT}")

        hc = hc_folders[-1]

        dest = job_dir / hc.name
        shutil.copytree(hc, dest, dirs_exist_ok=True)

        # final html must exist
        html_path = dest / "HealthCard.html"
        if not html_path.exists():
            raise FileNotFoundError(f"HealthCard.html not found at: {html_path}")

        return f"""
        <h3>Generated</h3>
        <p><b>Lat:</b> {lat} | <b>Lon:</b> {lon}</p>
        <iframe src="/jobs/{job_id}/{hc.name}/HealthCard.html"
                style="width:100%;height:900px"></iframe>
        """

    except Exception:
        err = traceback.format_exc()
        return HTMLResponse(f"<h2>Internal Server Error</h2><pre>{err}</pre>", status_code=500)
