from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import uuid
import shutil
import subprocess
import os

app = FastAPI()

BASE_DIR = Path(__file__).parent
JOBS_DIR = BASE_DIR / "jobs"
JOBS_DIR.mkdir(exist_ok=True)

app.mount("/jobs", StaticFiles(directory=JOBS_DIR), name="jobs")

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
    job_id = str(uuid.uuid4())[:8]
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True)

    # THIS is where your real generator is called
    subprocess.run(
        [
            "python",
            "health_card_master_v1_9g_REPORTCARD_FOLDERS_GIS_SYNTH.py",
            "--lat", str(lat),
            "--lon", str(lon)
        ],
        cwd=str(BASE_DIR),
        check=True
    )

    # Copy the newly generated HealthCard folder
    out_root = BASE_DIR / "OUT_HEALTH_CARD"
    hc_folders = sorted(out_root.glob("HealthCard_*"), key=os.path.getmtime)
    hc = hc_folders[-1]

    shutil.copytree(hc, job_dir / hc.name)

    return f"""
    <h3>Generated</h3>
    <iframe src="/jobs/{job_id}/{hc.name}/HealthCard.html"
            style="width:100%;height:900px"></iframe>
    """
