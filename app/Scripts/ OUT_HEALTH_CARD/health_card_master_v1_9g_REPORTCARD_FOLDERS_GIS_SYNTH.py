#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Health Card Master Orchestrator v1.9g (FULL) — v1.9e FLOW + StreamStats fix + Ranking clarity
PLUS: Mean BIBI references on the HealthCard (site / DNR12 / MDE8) + improved Region stdout tail display.

PATCH (requested by Wuhib):
1) If Mean BIBI (DNR12/MDE8) is NA from MBSS_Merged_All.csv but Region stdout prints it,
   parse those "Mean BIBI in DNR12..." / "Mean BIBI in MDE8..." lines and fill NA.
2) Hydraulics invocation robustness: try argparse style (--lat/--lon) then fallback to stdin.
3) StreamStats top pills: tolerant key matching so common items show even if labels differ.

Run:
  python health_card_master_v1_9g_REPORTCARD_FOLDERS_GIS_SYNTH.py
then enter lat/lon when prompted.
"""

import os
import sys
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd

import argparse

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lat", type=float, required=True)
    p.add_argument("--lon", type=float, required=True)
    return p.parse_args()



# ============================
# USER CONFIG (GIS EXPORT)
# ============================
DNR12_FOLDER = Path(r"C:\Users\wbayou\Documents\Documents - Copy\STream Restoration Dissertation\FINAL\Maryland_Watersheds_-_12_Digit_Watersheds")
MDE8_FOLDER  = Path(r"C:\Users\wbayou\Documents\Documents - Copy\STream Restoration Dissertation\FINAL\Maryland_Watersheds_-_8_Digit_Watersheds (1)")

DNR12_SHP_HINT = None  # optional explicit shapefile path
MDE8_SHP_HINT  = None

ENABLE_GIS_EXPORT = True   # set False to skip ArcPy export


# ----------------------------
# Helpers / plumbing
# ----------------------------
@dataclass
class RunResult:
    name: str
    ok: bool
    returncode: int
    stdout_path: Path
    stderr_path: Path
    stdout_text: str
    stderr_text: str


def safe_slug(s: str) -> str:
    s = (s or "").strip().replace(" ", "_")
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    return s[:180]


def read_text(p: Path, max_chars: int = 300_000) -> str:
    try:
        t = p.read_text(encoding="utf-8", errors="replace")
        return t[:max_chars]
    except Exception:
        return ""


def run_script(
    script_path: Path,
    name: str,
    workdir: Path,
    logs_dir: Path,
    stdin_text: str = "",
    extra_args: Optional[List[str]] = None,
    timeout_sec: int = 1200,
) -> RunResult:
    """
    CRITICAL: Forces UTF-8 for child scripts to prevent Windows 'charmap' crashes
              when stdout is piped (e.g., Region printing '≈').
    """
    py = sys.executable
    args = [py, "-X", "utf8", str(script_path)]
    if extra_args:
        args += extra_args

    stdout_path = logs_dir / f"{name}_stdout.txt"
    stderr_path = logs_dir / f"{name}_stderr.txt"

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    try:
        proc = subprocess.run(
            args,
            input=stdin_text,
            cwd=str(workdir),
            env=env,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_sec,
        )
        stdout_path.write_text(proc.stdout or "", encoding="utf-8", errors="replace")
        stderr_path.write_text(proc.stderr or "", encoding="utf-8", errors="replace")
        ok = (proc.returncode == 0)

        return RunResult(
            name=name, ok=ok, returncode=proc.returncode,
            stdout_path=stdout_path, stderr_path=stderr_path,
            stdout_text=read_text(stdout_path), stderr_text=read_text(stderr_path),
        )
    except subprocess.TimeoutExpired as e:
        stdout_path.write_text((e.stdout or ""), encoding="utf-8", errors="replace")
        stderr_path.write_text("TIMEOUT\n" + (e.stderr or ""), encoding="utf-8", errors="replace")
        return RunResult(
            name=name, ok=False, returncode=124,
            stdout_path=stdout_path, stderr_path=stderr_path,
            stdout_text=read_text(stdout_path), stderr_text=read_text(stderr_path),
        )
    except Exception as e:
        stderr_path.write_text(f"EXCEPTION: {e}\n", encoding="utf-8", errors="replace")
        return RunResult(
            name=name, ok=False, returncode=1,
            stdout_path=stdout_path, stderr_path=stderr_path,
            stdout_text=read_text(stdout_path), stderr_text=read_text(stderr_path),
        )


def prompt_float(msg: str) -> float:
    while True:
        s = input(msg).strip()
        try:
            return float(s)
        except Exception:
            print("Please enter a valid number.")


def find_newest_in_roots(roots: List[Path], filename: str, since_ts: float) -> Optional[Path]:
    newest = None
    newest_mtime = -1.0
    for r in roots:
        if not r or not Path(r).exists():
            continue
        for p in Path(r).rglob(filename):
            try:
                mt = p.stat().st_mtime
            except Exception:
                continue
            if mt >= since_ts and mt > newest_mtime:
                newest = p
                newest_mtime = mt
    return newest


def fnum(x) -> float:
    try:
        if pd.isna(x):
            return np.nan
        return float(x)
    except Exception:
        return np.nan


# ----------------------------
# Script discovery (NEW)
# ----------------------------
def _pick_newest_matching(workdir: Path, patterns: List[str]) -> Optional[Path]:
    cand: List[Path] = []
    for pat in patterns:
        cand.extend(list(workdir.glob(pat)))
    cand = [p for p in cand if p.exists() and p.is_file()]
    if not cand:
        return None
    cand.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cand[0]


def discover_region_script(workdir: Path, default_name: str) -> Optional[Path]:
    p = workdir / default_name
    if p.exists():
        return p
    return _pick_newest_matching(
        workdir,
        patterns=[
            "region_analysis_master*.py",
            "region*_analysis*.py",
            "region*.py",
        ],
    )


def discover_hydrology_script(workdir: Path, default_name: str) -> Optional[Path]:
    p = workdir / default_name
    if p.exists():
        return p
    return _pick_newest_matching(
        workdir,
        patterns=[
            "md_coordinate_hydro_engine*.py",
            "md_coordinate_hydro*_PLUS*.py",
            "md_coordinate_hydro*.py",
        ],
    )


def discover_hydraulics_script(workdir: Path, default_name: str) -> Optional[Path]:
    p = workdir / default_name
    if p.exists():
        return p
    return _pick_newest_matching(
        workdir,
        patterns=[
            "mbss_hydraulics_from_discharge*.py",
            "mbss_hydraulics*.py",
        ],
    )


# ----------------------------
# Parsing (stdout ‘SITEYR’ / watershed block)
# ----------------------------
def extract_siteyr_from_stdout(stdout_text: str) -> Optional[str]:
    m = re.search(r"SITEYR\s*\(nearest\)\s*:\s*([A-Z0-9_-]+)", stdout_text)
    if m:
        return m.group(1).strip()
    m = re.search(r"\bSITEYR\b\s*:\s*([A-Z0-9_-]+)", stdout_text)
    return m.group(1).strip() if m else None


def extract_watershed_block(stdout_text: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for key in ["DNR12DIG (HUC-12)", "DNR12DIG", "MDE8", "MDE8 Name", "Stream name", "Province/Physio"]:
        patt = re.escape(key) + r"\s*:\s*(.*)"
        m = re.search(patt, stdout_text)
        if m:
            out[key.replace(" (HUC-12)", "")] = m.group(1).strip()
    return out


def parse_streamstats_slope_ftft(hydrology_stdout: str) -> float:
    m = re.search(r"\bSLOPE\b\s*:\s*([-+]?\d+(\.\d+)?([eE][-+]?\d+)?)\s*ft/ft", hydrology_stdout)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return np.nan
    m = re.search(r"\bSLOPE\b\s*:\s*([-+]?\d+(\.\d+)?([eE][-+]?\d+)?)", hydrology_stdout)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return np.nan
    return np.nan


def parse_predicted_bibi_from_region(stdout_text: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    patterns = [
        ("pred_combined", r"Combined\s+BIBI\s+prediction\s*\(local-priority\)\s*:\s*([0-9.]+)"),
        ("pred_baseline", r"Baseline\s+\(priority\)\s*:\s*([0-9.]+)"),
        ("pred_state", r"Statewide\s+\w+\s+predicted\s+BIBI\s*:\s*([0-9.]+)"),
        ("pred_prov", r"Province\s+\w+\s+predicted\s+BIBI\s*:\s*([0-9.]+)"),
        ("pred_mde8", r"MDE8\s+\w+\s+predicted\s+BIBI\s*:\s*([0-9.]+)"),
    ]
    for k, pat in patterns:
        m = re.search(pat, stdout_text)
        if m:
            try:
                out[k] = float(m.group(1))
            except Exception:
                pass
    if "pred_baseline" not in out and "pred_combined" in out:
        out["pred_baseline"] = out["pred_combined"]
    return out


# ----------------------------
# NEW: Parse Mean BIBI lines from Region stdout (fills NA when MBSS lookup fails)
# ----------------------------
def parse_mean_bibi_from_region_stdout(region_stdout: str) -> Dict[str, float]:
    """
    Expected lines (example):
      Mean  BIBI  in DNR12 21405020196: 2.000  (n=1)
      Mean  BIBI  in MDE8 2140502: 2.966  (n=67)
    """
    out = {"mean_bibi_dnr12_from_region": np.nan, "mean_bibi_mde8_from_region": np.nan}
    if not region_stdout:
        return out

    m1 = re.search(r"Mean\s*BIBI\s*in\s*DNR12\s*[^:]+:\s*([0-9]+(?:\.[0-9]+)?)", region_stdout, flags=re.IGNORECASE)
    if m1:
        try:
            out["mean_bibi_dnr12_from_region"] = float(m1.group(1))
        except Exception:
            pass

    m2 = re.search(r"Mean\s*BIBI\s*in\s*MDE8\s*[^:]+:\s*([0-9]+(?:\.[0-9]+)?)", region_stdout, flags=re.IGNORECASE)
    if m2:
        try:
            out["mean_bibi_mde8_from_region"] = float(m2.group(1))
        except Exception:
            pass

    return out


def discover_region_output_folder(region_stdout: str, workdir: Path) -> Optional[Path]:
    """
    Detect region output folder in a robust way:
      - If stdout prints “FI plots saved in: ...”
      - Else scan for most recently modified folder that contains a 'pdp' directory
    """
    m = re.search(r"FI plots saved in:\s*(.+)", region_stdout)
    if m:
        p = Path(m.group(1).strip().strip('"'))
        if p.exists():
            return p

    candidates = []
    for p in workdir.rglob("pdp"):
        if p.is_dir():
            candidates.append(p.parent)
    if not candidates:
        return None
    candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return candidates[0]


def read_first_row_csv(p: Path) -> Dict[str, Any]:
    try:
        df = pd.read_csv(p)
        if df.empty:
            return {}
        return df.iloc[0].to_dict()
    except Exception:
        return {}


# ----------------------------
# MBSS Observed + Mean-BIBI reference computation
# ----------------------------
def _find_col_by_candidates(df: pd.DataFrame, candidates: List[str], contains_ok: bool = True) -> Optional[str]:
    cols = list(df.columns)
    cmap = {str(c).strip().lower(): c for c in cols}
    for want in candidates:
        wl = want.strip().lower()
        if wl in cmap:
            return cmap[wl]

    if not contains_ok:
        return None

    scored: List[Tuple[int, str]] = []
    for c in cols:
        cl = str(c).lower()
        hits = 0
        for want in candidates:
            if want.strip().lower() in cl:
                hits += 1
        if hits > 0:
            bonus = 0
            if "mbss" in cl:
                bonus += 2
            if "siteyr" in cl:
                bonus += 3
            if "dnr12" in cl or "huc12" in cl:
                bonus += 2
            if "mde8" in cl or "huc8" in cl:
                bonus += 2
            scored.append((hits * 10 + bonus - len(cl) // 80, c))
    if not scored:
        return None
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]


def _site_base_from_siteyr(siteyr: str) -> str:
    if not siteyr:
        return ""
    m = re.match(r"^(.+)-(\d{4})$", siteyr.strip())
    if m:
        return m.group(1)
    parts = siteyr.strip().split("-")
    if len(parts) >= 2 and parts[-1].isdigit() and len(parts[-1]) == 4:
        return "-".join(parts[:-1])
    return siteyr.strip()


def load_mbss_observed(merged_csv: Path, siteyr: str) -> Dict[str, Any]:
    if not merged_csv.exists():
        return {}

    df = pd.read_csv(merged_csv, low_memory=False)

    site_col = _find_col_by_candidates(df, ["SITEYR"], contains_ok=True)
    if site_col is None:
        return {}

    sub = df.loc[df[site_col].astype(str).str.strip().eq(siteyr)]
    if sub.empty:
        return {}

    row = sub.iloc[0]

    bibi_candidates = ["MBSS-BIBI95_23__bibi_05", "BIBI_05", "bibi_05", "BIBI", "bibi"]
    fib_candidates = ["FIBI_05", "fibi_05", "FIBI", "fibi"]

    out: Dict[str, Any] = {"SITEYR": siteyr}

    def pick_first(cols: List[str]) -> Tuple[Optional[str], Any]:
        for c in cols:
            if c in sub.columns:
                return c, row.get(c)
        for c in sub.columns:
            cl = str(c).lower()
            for want in cols:
                if want.lower() in cl:
                    return c, row.get(c)
        return None, np.nan

    bcol, bval = pick_first(bibi_candidates)
    fcol, fval = pick_first(fib_candidates)

    out["Observed_BIBI"] = bval
    out["Observed_FIBI"] = fval
    out["_bibi_col"] = bcol or ""
    out["_fibi_col"] = fcol or ""
    return out


def compute_mean_bibi_references_from_merged(
    merged_csv: Path,
    siteyr: str,
    dnr12_code: str,
    mde8_code: str,
) -> Dict[str, float]:
    out = {"mean_bibi_site": np.nan, "mean_bibi_dnr12": np.nan, "mean_bibi_mde8": np.nan}
    if (not merged_csv.exists()) or (not siteyr):
        return out

    try:
        df = pd.read_csv(merged_csv, low_memory=False)
    except Exception:
        return out

    siteyr_col = _find_col_by_candidates(df, ["SITEYR"], contains_ok=True)
    if not siteyr_col:
        return out

    bibi_col = _find_col_by_candidates(df, ["MBSS-BIBI95_23__bibi_05", "bibi_05", "bibi"], contains_ok=True)
    if not bibi_col:
        return out

    dnr12_col = _find_col_by_candidates(df, ["DNR12DIG", "HUC12", "HUC_12", "DNR12"], contains_ok=True)
    mde8_col  = _find_col_by_candidates(df, ["MDE8", "HUC8", "HUC_8", "BASIN8"], contains_ok=True)

    df_b = pd.to_numeric(df[bibi_col], errors="coerce")

    base = _site_base_from_siteyr(siteyr)
    if base:
        mask_site = df[siteyr_col].astype(str).str.strip().str.startswith(base + "-")
        vals = df_b[mask_site].dropna()
        if vals.size > 0:
            out["mean_bibi_site"] = float(vals.mean())

    if dnr12_col and dnr12_code and dnr12_code != "UNKNOWN_DNR12":
        mask_d = df[dnr12_col].astype(str).str.strip().eq(str(dnr12_code).strip())
        vals = df_b[mask_d].dropna()
        if vals.size > 0:
            out["mean_bibi_dnr12"] = float(vals.mean())

    if mde8_col and mde8_code and mde8_code != "UNKNOWN_MDE8":
        mask_m = df[mde8_col].astype(str).str.strip().eq(str(mde8_code).strip())
        vals = df_b[mask_m].dropna()
        if vals.size > 0:
            out["mean_bibi_mde8"] = float(vals.mean())

    return out


# ----------------------------
# StreamStats parsing
# ----------------------------
def parse_streamstats_from_stdout(hydrology_stdout: str) -> List[Tuple[str, str]]:
    if not hydrology_stdout:
        return []

    lines = hydrology_stdout.splitlines()
    out: List[Tuple[str, str]] = []

    def add(k: str, v: str) -> None:
        k = (k or "").strip()
        v = (v or "").strip()
        if k and v:
            out.append((k, v))

    start = None
    for i, ln in enumerate(lines):
        if "Basin characteristics" in ln and "selected" in ln.lower():
            start = i + 1
            break

    if start is not None:
        for ln in lines[start:]:
            if not ln.strip():
                break
            m = re.match(r"^\s*([A-Za-z0-9_()%/\- .]+?)\s*:\s*(.+?)\s*$", ln)
            if m:
                add(m.group(1), m.group(2))

    start2 = None
    for i, ln in enumerate(lines):
        if "Nearest MBSS site context" in ln:
            start2 = i + 1
            break
    if start2 is not None:
        for ln in lines[start2:]:
            if not ln.strip():
                break
            m = re.match(r"^\s*([A-Za-z0-9_()%/\- .]+?)\s*:\s*(.+?)\s*$", ln)
            if m:
                add(m.group(1), m.group(2))

    for ln in lines:
        m = re.search(r"\b(Q(?:2|5|10|25|50|100|200|500))\b\s*[:=]\s*(.+)$", ln, flags=re.IGNORECASE)
        if m:
            add(m.group(1).upper(), m.group(2).strip())

    seen = set()
    dedup: List[Tuple[str, str]] = []
    for k, v in out:
        key = (k.strip().lower(), v.strip())
        if key in seen:
            continue
        seen.add(key)
        dedup.append((k, v))
    return dedup


def streamstats_warning_note(hydrology_stdout: str) -> str:
    t = hydrology_stdout or ""
    if "Snapped pour point not clearly found" in t:
        return ("Note: StreamStats snap/pour-point was not clearly found. If StreamStats outputs show NA or 0, "
                "it often means the pour point/blue flowline was not properly selected/snapped (StreamStats UI behavior). "
                "Re-run and ensure the flowline/pour point is correctly set.")
    if re.search(r"Slope\s*:\s*NA", t) or re.search(r"Slope\s*:\s*0(\.0+)?\b", t):
        return ("Note: StreamStats slope/basin values appear missing or zero. If StreamStats outputs show NA or 0, "
                "it often means the pour point/blue flowline was not properly selected/snapped (StreamStats UI behavior). "
                "Re-run and ensure the flowline/pour point is correctly set.")
    return ""


# ----------------------------
# Hydraulics recompute (with slope fix)
# ----------------------------
def resolve_slope_m_m(hyd_row: Dict[str, Any], hydrology_stdout: str) -> Tuple[float, str]:
    s1 = fnum(hyd_row.get("Slope_m_m"))
    if np.isfinite(s1) and s1 > 0:
        return s1, "Slope source: hydraulics CSV Slope_m_m"

    sg = fnum(hyd_row.get("ST_GRAD_pct"))
    if np.isfinite(sg) and sg > 0:
        return sg / 100.0, "Slope source: hydraulics CSV ST_GRAD_pct/100"

    ftft = parse_streamstats_slope_ftft(hydrology_stdout)
    if np.isfinite(ftft) and ftft > 0:
        return ftft, "Slope source: hydrology stdout StreamStats SLOPE (ft/ft)"

    return np.nan, "Slope source: not found"


def recompute_tau_omega(hyd_row: Dict[str, Any], slope_m_m: float) -> Dict[str, Any]:
    rho = 1000.0
    g = 9.80665

    Q_m3s = fnum(hyd_row.get("Q_C_m3s"))
    W = fnum(hyd_row.get("AVGWID_m"))
    R = fnum(hyd_row.get("HydraulicRadius_m_rect"))

    if not np.isfinite(R):
        A = fnum(hyd_row.get("Area_m2_rect"))
        P = fnum(hyd_row.get("WettedPerimeter_m_rect"))
        if np.isfinite(A) and np.isfinite(P) and P > 0:
            R = A / P

    if not np.isfinite(R):
        D = fnum(hyd_row.get("AVGTHAL_m"))
        if np.isfinite(W) and np.isfinite(D) and (W + 2 * D) > 0:
            A = W * D
            P = W + 2 * D
            R = A / P

    out = dict(hyd_row)

    if not (np.isfinite(Q_m3s) and Q_m3s > 0):
        out["ShearStress_Pa_calc"] = np.nan
        out["StreamPowerPerLength_W_per_m_calc"] = np.nan
        out["UnitStreamPower_W_per_m2_calc"] = np.nan
        out["Hydraulics_note"] = "Missing Q (Q_C_m3s) for tau/Omega."
        return out

    if not (np.isfinite(slope_m_m) and slope_m_m > 0):
        out["ShearStress_Pa_calc"] = np.nan
        out["StreamPowerPerLength_W_per_m_calc"] = np.nan
        out["UnitStreamPower_W_per_m2_calc"] = np.nan
        out["Hydraulics_note"] = "Missing slope for tau/Omega."
        return out

    if not (np.isfinite(R) and R > 0):
        out["ShearStress_Pa_calc"] = np.nan
        out["StreamPowerPerLength_W_per_m_calc"] = np.nan
        out["UnitStreamPower_W_per_m2_calc"] = np.nan
        out["Hydraulics_note"] = "Missing hydraulic radius R for tau/Omega."
        return out

    tau = rho * g * R * slope_m_m
    Omega = rho * g * Q_m3s * slope_m_m
    omega = Omega / W if (np.isfinite(W) and W > 0) else np.nan

    out["ShearStress_Pa_calc"] = tau
    out["StreamPowerPerLength_W_per_m_calc"] = Omega
    out["UnitStreamPower_W_per_m2_calc"] = omega
    out["Hydraulics_note"] = "tau/Omega recomputed using visit discharge (Q_C_m3s) and resolved slope."
    return out


# ----------------------------
# Threshold concerns + management  (UNCHANGED)
# ----------------------------
def infer_direction(stressor: str) -> str:
    s = (stressor or "").upper()
    lower = ["IMPSURF", "IMP", "COND", "CL_", "SO4", "TN", "TP", "NO3", "TURB", "EMBED", "TEMP", "TSS"]
    higher = ["FOREST", "RIP", "SHAD", "BANKSTAB", "NUMROOT", "WOOD", "HAB", "RIFF", "POOL"]
    if any(k in s for k in higher):
        return "HigherBetter"
    if any(k in s for k in lower):
        return "LowerBetter"
    return "Unknown"


def choose_threshold(local_thr: float, prov_thr: float, state_thr: float) -> float:
    if np.isfinite(local_thr):
        return local_thr
    if np.isfinite(prov_thr):
        return prov_thr
    if np.isfinite(state_thr):
        return state_thr
    return np.nan


def flag_threshold_exceed(row: pd.Series) -> Optional[str]:
    stressor = str(row.get("Stressor", ""))
    site = fnum(row.get("Site_value"))
    local_thr = fnum(row.get("Local_thr"))
    prov_thr = fnum(row.get("Prov_thr"))
    state_thr = fnum(row.get("State_thr"))

    if not np.isfinite(site):
        return None

    direction = infer_direction(stressor)
    thr = choose_threshold(local_thr, prov_thr, state_thr)
    if not np.isfinite(thr):
        return None

    if direction == "LowerBetter":
        if site > thr:
            return f"{stressor}: site {site:.3g} worse than threshold {thr:.3g} (lower-is-better)"
    elif direction == "HigherBetter":
        if site < thr:
            return f"{stressor}: site {site:.3g} worse than threshold {thr:.3g} (higher-is-better)"
    return None


def management_suggestions(key_concerns: List[str], hab_interp: Optional[pd.DataFrame]) -> List[str]:
    sug: List[str] = []

    for c in key_concerns[:6]:
        cu = c.upper()
        if "IMPSURF" in cu:
            sug.append("Imperviousness pressure: prioritize LID/MS4 retrofits (disconnect runoff, storage, infiltration) upstream of the reach; target peak-rate and volume reduction.")
        if "COND" in cu or "CL_" in cu or "SO4" in cu:
            sug.append("Ionic signal: reduce road-salt and groundwater salinization sources (winter maintenance optimization, brining controls, salt storage, hotspot BMPs).")
        if "NO3" in cu or "TN" in cu or "TP" in cu:
            sug.append("Nutrient signal: strengthen riparian uptake + source control (fertilizer management, septic/WW inputs, agricultural BMPs where relevant).")
        if "TURB" in cu or "EMBED" in cu:
            sug.append("Sediment/substrate: control upstream erosion sources first; then consider targeted riffle/substrate enhancement and bank stabilization where needed.")
        if "TEMP" in cu:
            sug.append("Thermal stress: increase riparian shading, reduce warm runoff inputs, and consider hyporheic connectivity where feasible.")
        if "BANKSTAB" in cu:
            sug.append("Bank stability: combine toe protection + vegetation + floodplain reconnection where feasible; treat flashy inflows as the root driver.")

    if hab_interp is not None and not hab_interp.empty:
        def pct(metric: str) -> float:
            sub = hab_interp.loc[hab_interp["HabitatMetric"].astype(str).eq(metric)]
            if sub.empty:
                return np.nan
            return fnum(sub.iloc[0].get("Percentile"))

        sh = pct("SHADING")
        if np.isfinite(sh) and sh < 25:
            sug.append("Habitat: low SHADING percentile → riparian canopy planting/protection is a high-leverage action.")
        bs = pct("BANKSTAB")
        if np.isfinite(bs) and bs < 25:
            sug.append("Habitat: low BANKSTAB percentile → prioritize bank stabilization using bioengineering and root reinforcement.")

    if not sug:
        sug.append("No strong red flags were detected from available thresholds/habitat; prioritize watershed stressor control + targeted habitat complexity improvements and monitor response.")

    dedup = []
    seen = set()
    for s in sug:
        if s not in seen:
            dedup.append(s)
            seen.add(s)
    return dedup


# ----------------------------
# Chemistry window helpers (UNCHANGED)
# ----------------------------
def percentile_of_value(series: pd.Series, value: float) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna().to_numpy()
    if s.size == 0 or not np.isfinite(value):
        return np.nan
    s.sort()
    k = np.searchsorted(s, value, side="right")
    return 100.0 * (k / s.size)


def find_siteyr_col(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        if str(c).strip().upper() == "SITEYR":
            return c
    for c in df.columns:
        cu = str(c).strip().upper()
        if cu in {"SITE_YR", "SITEYEAR", "SITE_YR_ID"}:
            return c
    for c in df.columns:
        if "SITEYR" in str(c).upper():
            return c
    return None


def pick_col_regex(cols: List[str], patterns: List[str], avoid_patterns: List[str]) -> Optional[str]:
    compiled = [re.compile(p, flags=re.IGNORECASE) for p in patterns]
    avoidc = [re.compile(p, flags=re.IGNORECASE) for p in avoid_patterns]
    scored = []
    for c in cols:
        if any(a.search(c) for a in avoidc):
            continue
        hits = sum(1 for rg in compiled if rg.search(c))
        if hits <= 0:
            continue
        score = hits * 10
        cu = c.upper()
        if "WATERCHEMISTRY" in cu:
            score += 5
        if "MBSS-WATERCHEMISTRY" in cu or "MBSS_WATERCHEMISTRY" in cu:
            score += 6
        if any(k in cu for k in ["_FLD", "_FIELD", "_LAB"]):
            score += 2
        scored.append((score, c))
    if not scored:
        return None
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]


def pick_chem_column(cols: List[str], kind: str) -> Optional[str]:
    avoid_common = [r"\bPHOS\b", r"\bPHOSPH", r"\bTP\b", r"\bTN\b", r"\bTURB\b", r"\bGRAPH\b"]

    if kind == "ph":
        return pick_col_regex(
            cols,
            patterns=[r"(^|[_\s])PH($|[_\s])", r"\bpH\b", r"PH[_-]?(FLD|FIELD|LAB)\b"],
            avoid_patterns=avoid_common + [r"\bTPH\b", r"\bALP\b", r"PHOS"]
        )
    if kind == "do":
        return pick_col_regex(
            cols,
            patterns=[r"(^|[_\s])DO($|[_\s])", r"DISSOLVED[_\s-]*OXY", r"(^|[_\s])OXYGEN($|[_\s])", r"OXY[_\s-]*MG"],
            avoid_patterns=[r"\bDOC\b", r"\bDOSE\b"]
        )
    if kind == "tn":
        return pick_col_regex(cols, patterns=[r"(^|[_\s])TN($|[_\s])", r"TOTAL[_\s-]*N(ITROGEN)?\b"], avoid_patterns=[r"\bTNA\b", r"\bTND\b"])
    if kind == "tp":
        return pick_col_regex(cols, patterns=[r"(^|[_\s])TP($|[_\s])", r"TOTAL[_\s-]*P(HOSPHORUS)?\b"], avoid_patterns=[r"\bTMP\b"])
    if kind == "no3":
        return pick_col_regex(cols, patterns=[r"\bNO3\b", r"NITRATE\b", r"NO[_\s-]*3"], avoid_patterns=[])
    if kind == "cond":
        return pick_col_regex(cols, patterns=[r"\bCOND\b", r"CONDUCTIV", r"SPECIFIC[_\s-]*COND", r"CONDUCTANCE\b"], avoid_patterns=[r"CONDUCTANCEPRED"])
    if kind == "turb":
        return pick_col_regex(cols, patterns=[r"\bTURB\b", r"TURBID"], avoid_patterns=[])
    return None


def classify_ph(ph: float) -> str:
    if not np.isfinite(ph):
        return "Missing"
    if ph < 6.5:
        return "Risky (acidic; <6.5)"
    if ph > 8.5:
        return "Risky (basic; >8.5)"
    return "OK (6.5–8.5)"


def classify_do(do_mgL: float) -> str:
    if not np.isfinite(do_mgL):
        return "Missing"
    if do_mgL < 5.0:
        return "Risky (<5 mg/L)"
    if do_mgL < 6.0:
        return "Borderline (5–6 mg/L)"
    return "OK (≥6 mg/L)"


def classify_generic_lower_is_better(x: float, thr: float, name: str) -> str:
    if not np.isfinite(x):
        return "Missing"
    if np.isfinite(thr):
        if x <= thr:
            return f"OK (≤ target {thr:.3g})"
        return f"High (>{thr:.3g})"
    return "Observed"


def build_threshold_lookup(thr_df: Optional[pd.DataFrame]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    if thr_df is None or thr_df.empty:
        return out

    df = thr_df.copy()
    df.columns = [c.strip() for c in df.columns]

    need = {"Stressor", "State_thr", "Prov_thr", "Local_thr", "Site_value"}
    if not need.issubset(set(df.columns)):
        m = {}
        for c in df.columns:
            cl = c.lower()
            if "stressor" in cl or cl == "feature":
                m[c] = "Stressor"
            elif "state" in cl:
                m[c] = "State_thr"
            elif "prov" in cl:
                m[c] = "Prov_thr"
            elif "local" in cl:
                m[c] = "Local_thr"
            elif "site" in cl or "value" in cl:
                m[c] = "Site_value"
        df = df.rename(columns=m)

    if not need.issubset(set(df.columns)):
        return out

    for _, r in df.iterrows():
        st = str(r.get("Stressor", "")).strip()
        if not st:
            continue
        out[st] = {
            "state": fnum(r.get("State_thr")),
            "prov":  fnum(r.get("Prov_thr")),
            "local": fnum(r.get("Local_thr")),
            "site":  fnum(r.get("Site_value")),
        }
    return out


def lookup_best_threshold_for_stressor(thr_map: Dict[str, Dict[str, float]], stressor_key_contains: str) -> Tuple[Optional[str], float]:
    if not thr_map:
        return None, np.nan
    key = stressor_key_contains.upper()
    matches = [k for k in thr_map.keys() if key in k.upper()]
    if not matches:
        return None, np.nan
    matches.sort(key=lambda s: (("WATERCHEMISTRY" not in s.upper()), len(s)))
    m = matches[0]
    d = thr_map[m]
    thr = choose_threshold(d.get("local", np.nan), d.get("prov", np.nan), d.get("state", np.nan))
    return m, thr


def load_mbss_chemistry_window_from_df(df: pd.DataFrame, siteyr: str, thr_df: Optional[pd.DataFrame]) -> Dict[str, Any]:
    site_col = find_siteyr_col(df)
    if site_col is None:
        return {}

    sub = df.loc[df[site_col].astype(str).str.strip().eq(siteyr)]
    if sub.empty:
        return {}

    row = sub.iloc[0]
    cols = list(df.columns)

    ph_col   = pick_chem_column(cols, "ph")
    do_col   = pick_chem_column(cols, "do")
    tn_col   = pick_chem_column(cols, "tn")
    tp_col   = pick_chem_column(cols, "tp")
    no3_col  = pick_chem_column(cols, "no3")
    cond_col = pick_chem_column(cols, "cond")
    turb_col = pick_chem_column(cols, "turb")

    thr_map = build_threshold_lookup(thr_df)
    _, thr_tn   = lookup_best_threshold_for_stressor(thr_map, "TN")
    _, thr_tp   = lookup_best_threshold_for_stressor(thr_map, "TP")
    _, thr_no3  = lookup_best_threshold_for_stressor(thr_map, "NO3")
    _, thr_cond = lookup_best_threshold_for_stressor(thr_map, "COND")
    _, thr_turb = lookup_best_threshold_for_stressor(thr_map, "TURB")

    def getnum(c: Optional[str], vmin: float = -np.inf, vmax: float = np.inf) -> float:
        if not c or c not in df.columns:
            return np.nan
        x = fnum(row.get(c))
        if np.isfinite(x) and (x < vmin or x > vmax):
            return np.nan
        return x

    ph_val   = getnum(ph_col,   0.0, 14.0)
    do_val   = getnum(do_col,   0.0, 30.0)
    tn_val   = getnum(tn_col,   0.0, 1e6)
    tp_val   = getnum(tp_col,   0.0, 1e6)
    no3_val  = getnum(no3_col,  0.0, 1e6)
    cond_val = getnum(cond_col, 0.0, 1e9)
    turb_val = getnum(turb_col, 0.0, 1e9)

    def pct(col: Optional[str], val: float) -> float:
        if not col or col not in df.columns:
            return np.nan
        return percentile_of_value(df[col], val)

    out: Dict[str, Any] = {"SITEYR": siteyr}
    out.update({
        "pH_col": ph_col or "", "DO_col": do_col or "",
        "TN_col": tn_col or "", "TP_col": tp_col or "", "NO3_col": no3_col or "",
        "COND_col": cond_col or "", "TURB_col": turb_col or "",
        "pH": ph_val, "DO_mgL": do_val,
        "TN": tn_val, "TP": tp_val, "NO3": no3_val,
        "COND": cond_val, "TURB": turb_val,
        "pH_statewide_pct": pct(ph_col, ph_val),
        "DO_statewide_pct": pct(do_col, do_val),
        "TN_statewide_pct": pct(tn_col, tn_val),
        "TP_statewide_pct": pct(tp_col, tp_val),
        "NO3_statewide_pct": pct(no3_col, no3_val),
        "COND_statewide_pct": pct(cond_col, cond_val),
        "TURB_statewide_pct": pct(turb_col, turb_val),
        "pH_status": classify_ph(ph_val),
        "DO_status": classify_do(do_val),
        "TN_status": classify_generic_lower_is_better(tn_val, thr_tn, "TN"),
        "TP_status": classify_generic_lower_is_better(tp_val, thr_tp, "TP"),
        "NO3_status": classify_generic_lower_is_better(no3_val, thr_no3, "NO3"),
        "COND_status": classify_generic_lower_is_better(cond_val, thr_cond, "COND"),
        "TURB_status": classify_generic_lower_is_better(turb_val, thr_turb, "TURB"),
        "TN_target": thr_tn, "TP_target": thr_tp, "NO3_target": thr_no3,
        "COND_target": thr_cond, "TURB_target": thr_turb,
    })
    return out


def load_mbss_chemistry_window(
    merged_csv: Path,
    siteyr: str,
    thr_df: Optional[pd.DataFrame],
    waterchem_csv: Optional[Path] = None,
    master_full_csv: Optional[Path] = None,
) -> Dict[str, Any]:
    # Reads only if files exist. NO generation is triggered here.
    if waterchem_csv and waterchem_csv.exists():
        try:
            dfc = pd.read_csv(waterchem_csv, low_memory=False)
            out = load_mbss_chemistry_window_from_df(dfc, siteyr, thr_df)
            if out:
                out["_chem_source"] = str(waterchem_csv)
                return out
        except Exception:
            pass

    if master_full_csv and master_full_csv.exists():
        try:
            dfm = pd.read_csv(master_full_csv, low_memory=False)
            out = load_mbss_chemistry_window_from_df(dfm, siteyr, thr_df)
            if out:
                out["_chem_source"] = str(master_full_csv)
                return out
        except Exception:
            pass

    if merged_csv.exists():
        try:
            df = pd.read_csv(merged_csv, low_memory=False)
            out = load_mbss_chemistry_window_from_df(df, siteyr, thr_df)
            if out:
                out["_chem_source"] = str(merged_csv)
                return out
        except Exception:
            pass

    return {}


# ----------------------------
# Report Card Grades (A–F)  (UNCHANGED)
# ----------------------------
def clamp01(x: float) -> float:
    if not np.isfinite(x):
        return np.nan
    return max(0.0, min(1.0, x))


def grade_from_score_0_5(x: float) -> str:
    if not np.isfinite(x):
        return "NA"
    if x >= 4.0:
        return "A"
    if x >= 3.0:
        return "B"
    if x >= 2.0:
        return "C"
    if x >= 1.0:
        return "D"
    return "F"


def summarize_grade_reason(current_bibi: float, exceed_rate: float, uplift_delta: float) -> str:
    parts = []
    if np.isfinite(current_bibi):
        parts.append(f"BIBI={current_bibi:.2f}")
    if np.isfinite(exceed_rate):
        parts.append(f"pressure={exceed_rate*100:.0f}%")
    if np.isfinite(uplift_delta):
        parts.append(f"Δpot={uplift_delta:+.2f}")
    return ", ".join(parts) if parts else ""


def compute_threshold_exceedance_rate(thr_df: Optional[pd.DataFrame]) -> float:
    if thr_df is None or thr_df.empty:
        return np.nan

    df = thr_df.copy()
    df.columns = [c.strip() for c in df.columns]

    need = {"Stressor", "State_thr", "Prov_thr", "Local_thr", "Site_value"}
    if not need.issubset(set(df.columns)):
        m = {}
        for c in df.columns:
            cl = c.lower()
            if "stressor" in cl or cl == "feature":
                m[c] = "Stressor"
            elif "state" in cl:
                m[c] = "State_thr"
            elif "prov" in cl:
                m[c] = "Prov_thr"
            elif "local" in cl:
                m[c] = "Local_thr"
            elif "site" in cl or "value" in cl:
                m[c] = "Site_value"
        df = df.rename(columns=m)

    if not need.issubset(set(df.columns)):
        return np.nan

    n_ok = 0
    n_worse = 0
    for _, r in df.iterrows():
        stressor = str(r.get("Stressor", ""))
        site = fnum(r.get("Site_value"))
        thr = choose_threshold(fnum(r.get("Local_thr")), fnum(r.get("Prov_thr")), fnum(r.get("State_thr")))
        if not (np.isfinite(site) and np.isfinite(thr)):
            continue
        direction = infer_direction(stressor)
        n_ok += 1
        if direction == "LowerBetter" and site > thr:
            n_worse += 1
        elif direction == "HigherBetter" and site < thr:
            n_worse += 1

    if n_ok == 0:
        return np.nan
    return n_worse / n_ok


def compute_current_score_0_5(observed_bibi: float, exceed_rate: float) -> float:
    if not np.isfinite(observed_bibi):
        return np.nan
    pen = 0.75 * clamp01(exceed_rate) if np.isfinite(exceed_rate) else 0.0
    return max(0.0, min(5.0, observed_bibi - pen))


def compute_potential_score_0_5(best_pred_bibi: float, exceed_rate: float) -> float:
    if not np.isfinite(best_pred_bibi):
        return np.nan
    pen = 0.40 * clamp01(exceed_rate) if np.isfinite(exceed_rate) else 0.0
    return max(0.0, min(5.0, best_pred_bibi - pen))


def build_report_card(
    obs_bibi: float,
    pred_base: float,
    pred_best: float,
    thr_df: Optional[pd.DataFrame],
    mean_bibi_dnr12: float,
    mean_bibi_mde8: float,
) -> Dict[str, Any]:
    exceed_rate = compute_threshold_exceedance_rate(thr_df)
    uplift_delta = (pred_best - pred_base) if (np.isfinite(pred_best) and np.isfinite(pred_base)) else np.nan

    dnr12_current_bibi = mean_bibi_dnr12 if np.isfinite(mean_bibi_dnr12) else obs_bibi
    mde8_current_bibi  = mean_bibi_mde8  if np.isfinite(mean_bibi_mde8)  else obs_bibi

    dnr12_current_score = compute_current_score_0_5(dnr12_current_bibi, exceed_rate)
    mde8_current_score  = compute_current_score_0_5(mde8_current_bibi, exceed_rate)

    dnr12_pot_score = compute_potential_score_0_5(pred_best, exceed_rate)
    mde8_pot_score  = compute_potential_score_0_5(pred_best, exceed_rate)

    out = {
        "exceed_rate": exceed_rate,
        "uplift_delta": uplift_delta,
        "dnr12_current_bibi": dnr12_current_bibi,
        "mde8_current_bibi": mde8_current_bibi,
        "dnr12_current_score": dnr12_current_score,
        "mde8_current_score": mde8_current_score,
        "dnr12_potential_score": dnr12_pot_score,
        "mde8_potential_score": mde8_pot_score,
        "dnr12_current_grade": grade_from_score_0_5(dnr12_current_score),
        "mde8_current_grade": grade_from_score_0_5(mde8_current_score),
        "dnr12_potential_grade": grade_from_score_0_5(dnr12_pot_score),
        "mde8_potential_grade": grade_from_score_0_5(mde8_pot_score),
    }
    out["dnr12_reason"] = summarize_grade_reason(dnr12_current_bibi, exceed_rate, uplift_delta)
    out["mde8_reason"]  = summarize_grade_reason(mde8_current_bibi, exceed_rate, uplift_delta)
    return out


# ----------------------------
# HTML utilities
# ----------------------------
def esc_html(s: str) -> str:
    return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def df_to_html_table(df: Optional[pd.DataFrame], max_rows: int = 14) -> str:
    if df is None or df.empty:
        return "<div class='muted'>No data found.</div>"
    d = df.head(max_rows).copy()
    return d.to_html(index=False, escape=True, classes="tbl")



def build_ranking_process_html(
    siteyr: str,
    dnr12_code: str,
    mde8_code: str,
    mde8_name: str,
    obs_bibi: float,
    mean_site_bibi: float,
    mean_dnr12_bibi: float,
    mean_mde8_bibi: float,
    pred_base: float,
    pred_best: float,
    exceed_rate: float,
    uplift_delta: float,
    key_concerns: List[str],
    thr_df: Optional[pd.DataFrame],
    scen_df: Optional[pd.DataFrame],
) -> str:
    """Create a self-contained Ranking_Process.html so the HealthCard hyperlink always works.

    The goal is clarity:
      - what the 'rank/grade' represents (composite, not BIBI alone)
      - the exact formulas used
      - what inputs were available/missing
      - the stressors driving pressure and the top scenario bundles
    """
    def fmt(x: Any, nd: int = 3) -> str:
        v = fnum(x)
        return "NA" if not np.isfinite(v) else f"{v:.{nd}f}"

    press = fnum(exceed_rate)
    uplift = fnum(uplift_delta)

    current_bibi_dnr12 = mean_dnr12_bibi if np.isfinite(mean_dnr12_bibi) else obs_bibi
    current_bibi_mde8  = mean_mde8_bibi  if np.isfinite(mean_mde8_bibi)  else obs_bibi

    current_score_dnr12 = compute_current_score_0_5(current_bibi_dnr12, press)
    current_score_mde8  = compute_current_score_0_5(current_bibi_mde8,  press)
    pot_score_dnr12     = compute_potential_score_0_5(pred_best, press)
    pot_score_mde8      = compute_potential_score_0_5(pred_best, press)

    grade_dnr12_c = grade_from_score_0_5(current_score_dnr12)
    grade_mde8_c  = grade_from_score_0_5(current_score_mde8)
    grade_dnr12_p = grade_from_score_0_5(pot_score_dnr12)
    grade_mde8_p  = grade_from_score_0_5(pot_score_mde8)

    # Small helper tables
    inputs_tbl = pd.DataFrame([
        ["Observed BIBI (site-year)", fmt(obs_bibi, 3)],
        ["Mean BIBI (site across years)", fmt(mean_site_bibi, 3)],
        ["Mean BIBI (DNR12)", fmt(mean_dnr12_bibi, 3)],
        ["Mean BIBI (MDE8)", fmt(mean_mde8_bibi, 3)],
        ["Baseline predicted BIBI", fmt(pred_base, 3)],
        ["Best-scenario predicted BIBI", fmt(pred_best, 3)],
        ["Threshold pressure (share exceedances)", ("NA" if not np.isfinite(press) else f"{press*100:.1f}%")],
        ["Uplift potential (best - baseline)", fmt(uplift, 3)],
    ], columns=["Input", "Value"])

    score_tbl = pd.DataFrame([
        ["DNR12 Current", fmt(current_score_dnr12, 2), grade_dnr12_c],
        ["DNR12 Potential", fmt(pot_score_dnr12, 2), grade_dnr12_p],
        ["MDE8 Current", fmt(current_score_mde8, 2), grade_mde8_c],
        ["MDE8 Potential", fmt(pot_score_mde8, 2), grade_mde8_p],
    ], columns=["Category", "Score (0–5)", "Grade"])

    # Drivers table: use key_concerns (already filtered exceedances) first; fallback to thr_df scan
    driver_lines = list(key_concerns or [])
    if (not driver_lines) and (thr_df is not None) and (not thr_df.empty):
        try:
            tmp = thr_df.copy()
            tmp.columns = [c.strip() for c in tmp.columns]
            needed = {"Stressor", "State_thr", "Prov_thr", "Local_thr", "Site_value"}
            if needed.issubset(set(tmp.columns)):
                for _, r in tmp.iterrows():
                    msg = flag_threshold_exceed(r)
                    if msg:
                        driver_lines.append(msg)
            driver_lines = driver_lines[:20]
        except Exception:
            pass

    driver_html = "<div class='muted'>No threshold exceedance drivers available (thresholds missing or site values NA).</div>"
    if driver_lines:
        driver_html = "<ol>" + "".join([f"<li>{esc_html(s)}</li>" for s in driver_lines[:20]]) + "</ol>"

    # Scenario preview
    scen_html = "<div class='muted'>No scenario bundle table found.</div>"
    if scen_df is not None and not scen_df.empty:
        try:
            scen_html = df_to_html_table(scen_df, max_rows=14)
        except Exception:
            pass

    # Threshold preview (top rows)
    thr_html = "<div class='muted'>No threshold comparison table found.</div>"
    if thr_df is not None and not thr_df.empty:
        try:
            thr_html = df_to_html_table(thr_df, max_rows=18)
        except Exception:
            pass

    html = f"""<!doctype html>
<html>
<head>
  <meta charset='utf-8'/>
  <meta name='viewport' content='width=device-width, initial-scale=1'/>
  <title>Ranking Process — {esc_html(siteyr)}</title>
  <style>
    body {{ font-family: Arial, Helvetica, sans-serif; margin: 0; background:#0f1220; color:#eef2ff; }}
    a {{ color:#9ad1ff; }}
    .wrap {{ max-width: 1100px; margin: 0 auto; padding: 18px; }}
    h1 {{ margin:0 0 6px 0; font-size: 22px; }}
    h2 {{ margin:18px 0 10px 0; font-size: 18px; }}
    .muted {{ color:#a6b0d4; font-size: 12.5px; line-height: 1.35; }}
    .card {{ background:#121634; border-radius:16px; padding: 14px; box-shadow: 0 6px 18px rgba(0,0,0,0.22); margin-top: 14px; }}
    .tbl {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
    .tbl th, .tbl td {{ border-bottom: 1px solid rgba(255,255,255,0.08); padding: 6px 8px; text-align:left; vertical-align: top; }}
    .tbl th {{ color:#cbd5ff; font-weight: 700; }}
    code {{ background:#0b1020; padding:2px 6px; border-radius:8px; border:1px solid rgba(255,255,255,0.10); }}
    .formula {{ background:#0b1020; padding: 10px 12px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.10); }}
  </style>
</head>
<body>
  <div class='wrap'>
    <h1>Ranking Process</h1>
    <div class='muted'>
      SITEYR: <b>{esc_html(siteyr)}</b> |
      DNR12: <b>{esc_html(dnr12_code)}</b> |
      MDE8: <b>{esc_html(mde8_code)}</b> ({esc_html(mde8_name)})
    </div>

    <div class='card'>
      <h2>What the “rank/grade” is (and is not)</h2>
      <div class='muted'>
        The HealthCard “rank/grade” is a <b>composite</b> indicator meant to summarize overall biological condition and stressor pressure:
        it uses the current BIBI level (mean for DNR12/MDE8 when available), penalizes it by modeled threshold pressure,
        and reports a second “potential” grade based on best-scenario predicted BIBI under bundled stressor reduction.
        <br/><br/>
        It is <b>not</b> just BIBI alone, and it is <b>not</b> a statewide numeric percentile rank.
      </div>
    </div>

    <div class='card'>
      <h2>Exact formulas used</h2>
      <div class='formula'>
        <div><b>Pressure</b> = share of stressors where site value is worse than the selected breakpoint/threshold (local → province → statewide fallback).</div>
        <div style='margin-top:8px;'><b>CurrentScore</b> = <code>clamp(CurrentBIBI − 0.75 × Pressure, 0, 5)</code></div>
        <div><b>PotentialScore</b> = <code>clamp(BestPredBIBI − 0.40 × Pressure, 0, 5)</code></div>
        <div style='margin-top:8px;'>Grade bands: <code>A ≥ 4</code>, <code>B ≥ 3</code>, <code>C ≥ 2</code>, <code>D ≥ 1</code>, else <code>F</code>.</div>
      </div>
      <div class='muted' style='margin-top:10px;'>
        CurrentBIBI is taken as Mean BIBI in DNR12/MDE8 when available; otherwise it falls back to the observed site-year BIBI.
      </div>
    </div>

    <div class='card'>
      <h2>Inputs used for this run</h2>
      {df_to_html_table(inputs_tbl, max_rows=20)}
    </div>

    <div class='card'>
      <h2>Computed scores and grades</h2>
      {df_to_html_table(score_tbl, max_rows=10)}
    </div>

    <div class='card'>
      <h2>Pressure drivers (threshold exceedances)</h2>
      {driver_html}
    </div>

    <div class='card'>
      <h2>Threshold comparison preview</h2>
      {thr_html}
    </div>

    <div class='card'>
      <h2>Scenario bundles ranked (preview)</h2>
      {scen_html}
    </div>

    <div class='muted' style='margin-top:16px;'>
      Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    </div>
  </div>
</body>
</html>"""
    return html
def thumb_grid(img_paths: List[Path], rel_to: Path, cols: int = 4, height_px: int = 78) -> str:
    if not img_paths:
        return "<div class='muted'>No plots found.</div>"
    tiles = []
    for p in img_paths:
        try:
            rel = p.relative_to(rel_to).as_posix()
        except Exception:
            rel = p.as_posix()
        title = esc_html(p.name)
        tiles.append(
            f"<a class='thumb' href='{rel}' target='_blank' title='{title}'>"
            f"<img style='height:{height_px}px;' src='{rel}'/>"
            f"</a>"
        )
    return f"<div class='thumb-grid' style='grid-template-columns: repeat({cols}, 1fr);'>" + "".join(tiles) + "</div>"


def render_streamstats_top(
    ss_items: List[Tuple[str, str]],
    mean_site_bibi: float,
    mean_dnr12_bibi: float,
    mean_mde8_bibi: float,
) -> str:
    pills: List[Tuple[str, str]] = []
    if ss_items:
        ss_map = {str(k).strip().lower(): str(v).strip() for k, v in ss_items}

        def pick(label: str, contains: List[str]) -> Optional[Tuple[str, str]]:
            l = label.strip().lower()
            if l in ss_map:
                return (label, ss_map[l])
            for k, v in ss_map.items():
                if any(c in k for c in contains):
                    return (label, v)
            return None

        priority = [
            ("Drainage Area", ["drainage area", "drainage"]),
            ("Impervious", ["impervious"]),
            ("Forest", ["forest"]),
            ("Precipitation", ["precip"]),
            ("Slope", ["slope"]),
            ("DNR12DIG", ["dnr12", "huc12"]),
            ("MDE8 unit", ["mde8", "huc8", "basin8"]),
            ("Province/Physio", ["province", "physio"]),
        ]
        for lab, keys in priority:
            got = pick(lab, keys)
            if got:
                pills.append(got)

        used = {k.lower() for k, _ in pills}
        for k, v in ss_items:
            if len(pills) >= 12:
                break
            if str(k).strip().lower() not in used:
                pills.append((k, v))
                used.add(str(k).strip().lower())

    else:
        pills.append(("StreamStats", "not parsed"))

    if np.isfinite(mean_site_bibi):
        pills.append(("Mean BIBI (site)", f"{mean_site_bibi:.3f}"))
    if np.isfinite(mean_dnr12_bibi):
        pills.append(("Mean BIBI (DNR12)", f"{mean_dnr12_bibi:.3f}"))
    if np.isfinite(mean_mde8_bibi):
        pills.append(("Mean BIBI (MDE8)", f"{mean_mde8_bibi:.3f}"))

    html = "".join([f"<span class='pill'><b>{esc_html(k)}:</b> {esc_html(v)}</span>" for k, v in pills])
    return f"<div class='topstats'>{html}</div>"


def _tail_lines(text: str, n: int = 140) -> str:
    if not text:
        return ""
    lines = text.splitlines()
    tail = lines[-n:] if len(lines) > n else lines
    return "\n".join(tail)


def _extract_region_key_lines(text: str, max_lines: int = 24) -> str:
    if not text:
        return ""
    pats = [
        r"\bBEST\b",
        r"\bBaseline\b",
        r"Combined\s+BIBI\s+prediction",
        r"scenario",
        r"threshold",
        r"\bMean\s+BIBI\b",
        r"\bERROR\b",
        r"\bFATAL\b",
        r"\bWARNING\b",
        r"\bEXCEPTION\b",
        r"\bTraceback\b",
    ]
    rx = re.compile("|".join(pats), flags=re.IGNORECASE)
    picks = [ln for ln in text.splitlines() if rx.search(ln)]
    seen = set()
    ded = []
    for ln in picks:
        s = ln.strip()
        if not s:
            continue
        if s in seen:
            continue
        seen.add(s)
        ded.append(s)
        if len(ded) >= max_lines:
            break
    return "\n".join(ded)


def _na(x: Any, nd: int = 3) -> str:
    v = fnum(x)
    if not np.isfinite(v):
        return "NA"
    return f"{v:.{nd}f}"


def _hydraulic_impact_note(tau_pa: float, omega_wm2: float) -> str:
    """
    Heuristic: does NOT alter score. Helps interpret hydraulics “impact direction” on habitat/BIBI.
    """
    t = fnum(tau_pa)
    w = fnum(omega_wm2)
    if not np.isfinite(t) and not np.isfinite(w):
        return "Hydraulics impact: NA (missing τ/ω)."

    msgs = []
    if np.isfinite(w):
        if w >= 60:
            msgs.append("Very high unit stream power → likely bed/bank instability risk; can suppress BIBI if persistent.")
        elif w >= 30:
            msgs.append("High unit stream power → erosive tendency; bio response depends on stability/habitat complexity.")
        elif w <= 5:
            msgs.append("Low unit stream power → depositional tendency; embeddedness/fine sediment may suppress BIBI.")
        else:
            msgs.append("Moderate unit stream power → hydraulics not extreme; watershed stressors may dominate.")
    elif np.isfinite(t):
        if t >= 60:
            msgs.append("Very high shear stress → bed/bank mobilization risk; can suppress BIBI if habitat destabilizes.")
        elif t >= 30:
            msgs.append("High shear stress → erosive tendency; check bank stability + substrate.")
        elif t <= 5:
            msgs.append("Low shear stress → deposition/embeddedness risk; check fines and riffle quality.")
        else:
            msgs.append("Moderate shear stress → hydraulics not extreme; check other stressors.")

    return "Hydraulics impact: " + " ".join(msgs)


# ----------------------------
# OPTIONAL GIS EXPORT (ArcPy) (UNCHANGED)
# ----------------------------
def _find_first_shp(folder: Path, hint: Optional[str] = None) -> Optional[Path]:
    if hint:
        p = Path(hint)
        if p.exists() and p.suffix.lower() == ".shp":
            return p
    if not folder.exists():
        return None
    shps = sorted(folder.rglob("*.shp"), key=lambda p: p.stat().st_mtime, reverse=True)
    return shps[0] if shps else None


def _pick_field_name(fields: List[str], candidates: List[str]) -> Optional[str]:
    fset = {f.upper(): f for f in fields}
    for c in candidates:
        if c.upper() in fset:
            return fset[c.upper()]
    for f in fields:
        fu = f.upper()
        for c in candidates:
            if c.upper() in fu:
                return f
    return None


def export_watersheds_arcpy(
    out_gis_dir: Path,
    dnr12_code: str,
    mde8_code: str,
    dnr12_folder: Path,
    mde8_folder: Path,
    dnr12_hint: Optional[str] = None,
    mde8_hint: Optional[str] = None,
) -> List[str]:
    notes = []
    try:
        import arcpy  # type: ignore
    except Exception:
        notes.append("GIS export skipped: ArcPy not available in this Python environment.")
        return notes

    out_gis_dir.mkdir(parents=True, exist_ok=True)
    arcpy.env.overwriteOutput = True

    dnr12_shp = _find_first_shp(dnr12_folder, dnr12_hint)
    mde8_shp  = _find_first_shp(mde8_folder,  mde8_hint)

    if not dnr12_shp or not dnr12_shp.exists():
        notes.append("GIS export: could not locate 12-digit watershed shapefile in provided folder.")
        return notes
    if not mde8_shp or not mde8_shp.exists():
        notes.append("GIS export: could not locate 8-digit watershed shapefile in provided folder.")
        return notes

    dnr12_lyr = "dnr12_lyr_tmp"
    mde8_lyr  = "mde8_lyr_tmp"
    arcpy.MakeFeatureLayer_management(str(dnr12_shp), dnr12_lyr)
    arcpy.MakeFeatureLayer_management(str(mde8_shp),  mde8_lyr)

    dnr12_fields = [f.name for f in arcpy.ListFields(dnr12_lyr)]
    mde8_fields  = [f.name for f in arcpy.ListFields(mde8_lyr)]

    dnr12_field = _pick_field_name(dnr12_fields, ["DNR12DIG", "HUC12", "HUC_12", "HUC12DIG", "HUC12_CODE"])
    mde8_field  = _pick_field_name(mde8_fields,  ["MDE8", "MDE_8", "BASIN8", "BASIN", "HUC8", "HUC_8", "HUC8_CODE"])

    if not dnr12_field:
        notes.append(f"GIS export: could not identify DNR12/HUC12 field in 12-digit layer. Fields={dnr12_fields[:20]}...")
        return notes
    if not mde8_field:
        notes.append(f"GIS export: could not identify MDE8/HUC8 field in 8-digit layer. Fields={mde8_fields[:20]}...")
        return notes

    dnr12_where = f"{arcpy.AddFieldDelimiters(dnr12_lyr, dnr12_field)} = '{dnr12_code}'"
    mde8_where  = f"{arcpy.AddFieldDelimiters(mde8_lyr,  mde8_field)}  = '{mde8_code}'"

    arcpy.SelectLayerByAttribute_management(dnr12_lyr, "NEW_SELECTION", dnr12_where)
    arcpy.SelectLayerByAttribute_management(mde8_lyr,  "NEW_SELECTION", mde8_where)

    dnr12_out_folder = out_gis_dir / f"DNR12_{dnr12_code}"
    mde8_out_folder  = out_gis_dir / f"MDE8_{mde8_code}"
    dnr12_out_folder.mkdir(parents=True, exist_ok=True)
    mde8_out_folder.mkdir(parents=True, exist_ok=True)

    dnr12_out_shp = dnr12_out_folder / f"DNR12_{dnr12_code}.shp"
    mde8_out_shp  = mde8_out_folder  / f"MDE8_{mde8_code}.shp"
    arcpy.CopyFeatures_management(dnr12_lyr, str(dnr12_out_shp))
    arcpy.CopyFeatures_management(mde8_lyr,  str(mde8_out_shp))
    notes.append(f"GIS export: wrote {dnr12_out_shp}")
    notes.append(f"GIS export: wrote {mde8_out_shp}")

    try:
        dnr12_kmz = out_gis_dir / f"DNR12_{dnr12_code}.kmz"
        mde8_kmz  = out_gis_dir / f"MDE8_{mde8_code}.kmz"
        arcpy.LayerToKML_conversion(dnr12_lyr, str(dnr12_kmz))
        arcpy.LayerToKML_conversion(mde8_lyr,  str(mde8_kmz))
        notes.append(f"GIS export: wrote {dnr12_kmz}")
        notes.append(f"GIS export: wrote {mde8_kmz}")
    except Exception as e:
        notes.append(f"GIS export: KML/KMZ export failed (LayerToKML). {e}")

    try:
        arcpy.Delete_management(dnr12_lyr)
        arcpy.Delete_management(mde8_lyr)
    except Exception:
        pass

    return notes


# =====================================================================
# POST HEALTH CARD SYNTHESIS (kept)
# =====================================================================
def _fmt(x: Any, nd: int = 2) -> str:
    v = fnum(x)
    if not np.isfinite(v):
        return "NA"
    return f"{v:.{nd}f}"


def _pct(x: Any, nd: int = 1) -> str:
    v = fnum(x)
    if not np.isfinite(v):
        return "NA"
    return f"{v:.{nd}f}"


def _safe(s: Any) -> str:
    return str(s) if s is not None else ""


def _top_habitat_low_points(hab_interp: Optional[pd.DataFrame], k: int = 4) -> List[str]:
    if hab_interp is None or hab_interp.empty:
        return []
    df = hab_interp.copy()
    if "Percentile" not in df.columns or "HabitatMetric" not in df.columns:
        return []
    df["p"] = pd.to_numeric(df["Percentile"], errors="coerce")
    df = df.dropna(subset=["p"])
    if df.empty:
        return []
    df = df.sort_values("p", ascending=True).head(k)
    out = []
    for _, r in df.iterrows():
        out.append(f"{_safe(r.get('HabitatMetric'))} (pct={_safe(r.get('Percentile'))}, value={_safe(r.get('Value'))})")
    return out


def build_post_synthesis_text(
    siteyr: str,
    wblock: Dict[str, str],
    report_card: Dict[str, Any],
    obs_bibi: float,
    pred_base: float,
    pred_best: float,
    pred_best_delta: float,
    resid: float,
    key_concerns: List[str],
    suggestions: List[str],
    chem: Dict[str, Any],
    hab_interp: Optional[pd.DataFrame],
    hyd_row2: Dict[str, Any],
    ss_items: List[Tuple[str, str]],
    ss_note: str,
    mean_site_bibi: float,
    mean_dnr12_bibi: float,
    mean_mde8_bibi: float,
) -> str:
    dnr12_c = _safe(report_card.get("dnr12_current_grade", "NA"))
    dnr12_p = _safe(report_card.get("dnr12_potential_grade", "NA"))
    mde8_c  = _safe(report_card.get("mde8_current_grade", "NA"))
    mde8_p  = _safe(report_card.get("mde8_potential_grade", "NA"))
    press   = fnum(report_card.get("exceed_rate", np.nan))
    uplift  = fnum(report_card.get("uplift_delta", np.nan))

    dnr12_code = wblock.get("DNR12DIG", "")
    mde8_code  = wblock.get("MDE8", "")
    mde8_name  = wblock.get("MDE8 Name", "")
    prov       = wblock.get("Province/Physio", "")
    streamnm   = wblock.get("Stream name", "")

    tau = fnum(hyd_row2.get("ShearStress_Pa_calc", np.nan))
    omega = fnum(hyd_row2.get("UnitStreamPower_W_per_m2_calc", np.nan))
    slope = fnum(hyd_row2.get("Slope_m_m_resolved", np.nan))
    q = fnum(hyd_row2.get("Q_C_m3s", np.nan))
    wid = fnum(hyd_row2.get("AVGWID_m", np.nan))

    hab_lows = _top_habitat_low_points(hab_interp, k=5)
    thr_drivers = key_concerns[:8] if key_concerns else []

    chem_lines = []
    if chem:
        chem_lines.append(f"pH={_fmt(chem.get('pH'),2)} ({_safe(chem.get('pH_status'))}, pct={_pct(chem.get('pH_statewide_pct'))})")
        chem_lines.append(f"DO={_fmt(chem.get('DO_mgL'),2)} mg/L ({_safe(chem.get('DO_status'))}, pct={_pct(chem.get('DO_statewide_pct'))})")
        chem_lines.append(f"COND={_fmt(chem.get('COND'),1)} ({_safe(chem.get('COND_status'))}, pct={_pct(chem.get('COND_statewide_pct'))})")
        chem_lines.append(f"TN={_fmt(chem.get('TN'),3)} ({_safe(chem.get('TN_status'))}, pct={_pct(chem.get('TN_statewide_pct'))})")
        chem_lines.append(f"TP={_fmt(chem.get('TP'),3)} ({_safe(chem.get('TP_status'))}, pct={_pct(chem.get('TP_statewide_pct'))})")
        chem_lines.append(f"NO3={_fmt(chem.get('NO3'),3)} ({_safe(chem.get('NO3_status'))}, pct={_pct(chem.get('NO3_statewide_pct'))})")
        chem_lines.append(f"Turb={_fmt(chem.get('TURB'),2)} ({_safe(chem.get('TURB_status'))}, pct={_pct(chem.get('TURB_statewide_pct'))})")

    resid_note = ""
    if np.isfinite(resid):
        if resid > 0.5:
            resid_note = "Observed BIBI is notably higher than the model’s baseline expectation (site performing better than predicted for its stressor context)."
        elif resid < -0.5:
            resid_note = "Observed BIBI is notably lower than the model’s baseline expectation (site under-performing relative to its stressor context)."
        else:
            resid_note = "Observed BIBI is close to the model’s baseline expectation."

    press_note = ""
    if np.isfinite(press):
        if press >= 0.6:
            press_note = "High threshold pressure: many modeled stressors are on the wrong side of their breakpoints; watershed constraints likely dominate channel actions."
        elif press >= 0.3:
            press_note = "Moderate threshold pressure: several stressors exceed breakpoints; combined watershed + reach actions are needed."
        else:
            press_note = "Lower threshold pressure: fewer stressors exceed breakpoints; reach-scale actions have a better chance to translate into biological response."

    pot_note = ""
    if np.isfinite(uplift):
        if uplift >= 0.5:
            pot_note = "Scenario modeling indicates meaningful uplift is plausible if the top limiting stressors are reduced simultaneously (bundled actions)."
        elif uplift >= 0.2:
            pot_note = "Scenario modeling indicates modest uplift is plausible with targeted stressor reduction."
        else:
            pot_note = "Scenario modeling indicates limited uplift; without watershed-scale change, biological gains may remain small."

    ss_map = {k.upper(): v for k, v in ss_items} if ss_items else {}
    area = ss_map.get("DRAINAGE AREA", "")
    imp = ss_map.get("IMPERVIOUS", "")
    pcp = ss_map.get("PRECIPITATION", "")

    L = []
    L.append("POST HEALTH CARD SYNTHESIS (DETAILED)")
    L.append("=" * 88)
    L.append("1) What this card says (executive interpretation)")
    L.append(f"- Site: {siteyr}")
    L.append(f"- Context: DNR12={dnr12_code} | MDE8={mde8_code} ({mde8_name}) | Province={prov} | Stream={streamnm}")
    L.append(f"- Grades: DNR12 Current={dnr12_c}, Potential={dnr12_p} | MDE8 Current={mde8_c}, Potential={mde8_p}")

    ref_bits = []
    if np.isfinite(mean_site_bibi):
        ref_bits.append(f"mean(site)={mean_site_bibi:.3f}")
    if np.isfinite(mean_dnr12_bibi):
        ref_bits.append(f"mean(DNR12)={mean_dnr12_bibi:.3f}")
    if np.isfinite(mean_mde8_bibi):
        ref_bits.append(f"mean(MDE8)={mean_mde8_bibi:.3f}")
    if ref_bits:
        L.append(f"- Reference means (MBSS/Region): " + " | ".join(ref_bits))

    if np.isfinite(press):
        L.append(f"- Threshold pressure: {press*100:.1f}% → {press_note}")
    if np.isfinite(uplift):
        L.append(f"- Uplift potential (best - baseline): {uplift:+.3f} → {pot_note}")
    if np.isfinite(obs_bibi):
        L.append(f"- Observed BIBI: {obs_bibi:.3f}")
    if np.isfinite(pred_base):
        L.append(f"- Baseline predicted BIBI: {pred_base:.3f}")
    if np.isfinite(pred_best):
        L.append(f"- Best-scenario predicted BIBI: {pred_best:.3f} (Δ {pred_best_delta:+.3f})")
    if np.isfinite(resid):
        L.append(f"- Residual (Obs - Pred baseline): {resid:+.3f} → {resid_note}")
    L.append("")

    L.append("2) Watershed learning")
    if area or imp or pcp:
        L.append("- StreamStats snapshot (if available):")
        if area: L.append(f"  • Drainage area: {area}")
        if imp:  L.append(f"  • Imperviousness: {imp}")
        if pcp:  L.append(f"  • Mean annual precipitation: {pcp}")
    if ss_note:
        L.append(f"- StreamStats caution: {ss_note}")
    L.append("")

    L.append("3) Primary limiting factors detected")
    if key_concerns:
        L.append("- Modeled threshold exceedances (top drivers):")
        for s in key_concerns[:8]:
            L.append(f"  • {s}")
    else:
        L.append("- No threshold exceedances were detected (or thresholds/site values were missing).")
    if hab_lows:
        L.append("- Habitat weak points (lowest percentiles):")
        for s in hab_lows:
            L.append(f"  • {s}")
    if chem_lines:
        L.append("- Water chemistry flags (value, status, statewide percentile):")
        for s in chem_lines:
            L.append(f"  • {s}")
    L.append("")

    L.append("4) Hydraulics interpretation")
    if np.isfinite(q) or np.isfinite(tau) or np.isfinite(omega) or np.isfinite(slope):
        L.append(f"- Resolved slope: {_fmt(slope,4)} (m/m)")
        L.append(f"- Visit discharge: {_fmt(q,4)} (m³/s)")
        L.append(f"- Mean width: {_fmt(wid,2)} (m)")
        L.append(f"- Shear stress (τ): {_fmt(tau,2)} Pa")
        L.append(f"- Unit stream power (ω): {_fmt(omega,2)} W/m²")
        L.append(f"- {_hydraulic_impact_note(tau, omega)}")
    else:
        L.append("- Hydraulics metrics were not available (missing Q/slope/hydraulic radius).")
    L.append("")

    L.append("5) Practitioner actions (ranked)")
    if suggestions:
        for s in suggestions[:10]:
            L.append(f"  • {s}")
    else:
        L.append("  • No suggestions available.")
    L.append("")

    return "\n".join(L)


def build_post_synthesis_html(text_block: str) -> str:
    return "<div class='card full'><h2>Post HealthCard Synthesis (detailed)</h2>" \
           "<div class='muted'>Auto-generated narrative interpretation from the card values, thresholds, habitat, chemistry, and hydraulics.</div>" \
           f"<pre class='scroll' style='max-height:520px; white-space:pre-wrap; background:#0b1020; color:#e7eefc;'>{esc_html(text_block)}</pre>" \
           "</div>"


# ----------------------------
# HealthCard HTML
# ----------------------------
def build_healthcard_html(
    title: str,
    lat: float,
    lon: float,
    siteyr: str,
    wblock: Dict[str, str],
    ss_items: List[Tuple[str, str]],
    ss_note: str,
    obs_bibi: float,
    obs_fibi: float,
    mean_site_bibi: float,
    mean_dnr12_bibi: float,
    mean_mde8_bibi: float,
    preds: Dict[str, float],
    pred_base: float,
    pred_best: float,
    pred_best_delta: float,
    resid: float,
    report_card: Dict[str, Any],
    thr_df: Optional[pd.DataFrame],
    scen_df: Optional[pd.DataFrame],
    hyd_row2: Dict[str, Any],
    slope_note: str,
    hab_interp: Optional[pd.DataFrame],
    chem: Dict[str, Any],
    suggestions: List[str],
    key_concerns: List[str],
    fi_imgs: List[Path],
    pdp_imgs: List[Path],
    root_dir: Path,
    ranking_html: str,
    region_stdout: str,
    region_stderr: str,
    synth_html_block: str,
    gis_notes: List[str],
) -> str:
    dnr12_code = wblock.get("DNR12DIG", "UNKNOWN_DNR12")
    mde8_code  = wblock.get("MDE8", "UNKNOWN_MDE8")
    mde8_name  = wblock.get("MDE8 Name", "UNKNOWN_MDE8_NAME")
    prov       = wblock.get("Province/Physio", "UNKNOWN_PROVINCE")
    streamnm   = wblock.get("Stream name", "UNKNOWN_STREAM")

    ss_top = render_streamstats_top(ss_items, mean_site_bibi, mean_dnr12_bibi, mean_mde8_bibi)

    # BIBI table
    pred_mde8 = preds.get("pred_mde8", np.nan)
    pred_prov = preds.get("pred_prov", np.nan)
    pred_state = preds.get("pred_state", np.nan)

    bibi_tbl = pd.DataFrame([
        ["Observed (site-year)", _na(obs_bibi, 3)],
        ["Mean BIBI (site across years)", _na(mean_site_bibi, 3)],
        ["Mean BIBI (DNR12)", _na(mean_dnr12_bibi, 3)],
        ["Mean BIBI (MDE8)", _na(mean_mde8_bibi, 3)],
        ["Predicted BIBI (Baseline; site priority)", _na(pred_base, 3)],
        ["Predicted BIBI (Best scenario)", _na(pred_best, 3)],
        ["Best uplift (Δ best - baseline)", _na(pred_best_delta, 3)],
        ["Residual (Obs - baseline)", _na(resid, 3)],
        ["Predicted BIBI (MDE8 model)", _na(pred_mde8, 3)],
        ["Predicted BIBI (Province model)", _na(pred_prov, 3)],
        ["Predicted BIBI (Statewide model)", _na(pred_state, 3)],
    ], columns=["Metric", "Value"])
    bibi_tbl_html = df_to_html_table(bibi_tbl, max_rows=25)

    # Report card block
    press = fnum(report_card.get("exceed_rate", np.nan))
    uplift = fnum(report_card.get("uplift_delta", np.nan))
    report_html = f"""
    <div class='card'>
      <h2>Rank / Report Card</h2>
      <div class='muted'>
        This “rank” is a composite health grade based on: (i) current BIBI level (mean in DNR12/MDE8 if available),
        (ii) modeled threshold pressure (share of stressors beyond breakpoints), and (iii) modeled uplift potential (best–baseline).
        It is not just “BIBI alone”.
      </div>
      <div class='grid2'>
        <div class='box'>
          <h3>DNR12</h3>
          <div><b>Current:</b> <span class='grade'>{esc_html(str(report_card.get("dnr12_current_grade","NA")))}</span>
               <span class='muted'>(score={_na(report_card.get("dnr12_current_score", np.nan), 2)})</span></div>
          <div><b>Potential:</b> <span class='grade'>{esc_html(str(report_card.get("dnr12_potential_grade","NA")))}</span>
               <span class='muted'>(score={_na(report_card.get("dnr12_potential_score", np.nan), 2)})</span></div>
          <div class='muted'>Reason: {esc_html(str(report_card.get("dnr12_reason","")))}</div>
        </div>
        <div class='box'>
          <h3>MDE8</h3>
          <div><b>Current:</b> <span class='grade'>{esc_html(str(report_card.get("mde8_current_grade","NA")))}</span>
               <span class='muted'>(score={_na(report_card.get("mde8_current_score", np.nan), 2)})</span></div>
          <div><b>Potential:</b> <span class='grade'>{esc_html(str(report_card.get("mde8_potential_grade","NA")))}</span>
               <span class='muted'>(score={_na(report_card.get("mde8_potential_score", np.nan), 2)})</span></div>
          <div class='muted'>Reason: {esc_html(str(report_card.get("mde8_reason","")))}</div>
        </div>
      </div>
      <div class='muted' style='margin-top:10px;'>
        Threshold pressure: {("NA" if not np.isfinite(press) else f"{press*100:.1f}%")} |
        Uplift potential (best - baseline): {("NA" if not np.isfinite(uplift) else f"{uplift:+.3f}")}
      </div>
      <div style='margin-top:12px;'>
        <a class='btn' href='Ranking_Process.html' target='_self'>Open Ranking Process (details)</a>
      </div>
    </div>
    """

    # Hydraulics block
    tau = fnum(hyd_row2.get("ShearStress_Pa_calc", np.nan))
    omega_wm2 = fnum(hyd_row2.get("UnitStreamPower_W_per_m2_calc", np.nan))
    q = fnum(hyd_row2.get("Q_C_m3s", np.nan))
    slope = fnum(hyd_row2.get("Slope_m_m_resolved", np.nan))
    wid = fnum(hyd_row2.get("AVGWID_m", np.nan))
    hyd_note = _hydraulic_impact_note(tau, omega_wm2)

    hydraulics_html = f"""
    <div class='card'>
      <h2>Hydraulics (visit discharge)</h2>
      <div class='grid2'>
        <div class='box'>
          <div><b>Resolved slope:</b> {_na(slope, 4)} <span class='muted'>(m/m)</span></div>
          <div class='muted'>{esc_html(slope_note)}</div>
          <div><b>Visit Q:</b> {_na(q, 4)} <span class='muted'>(m³/s)</span></div>
          <div><b>Mean width:</b> {_na(wid, 2)} <span class='muted'>(m)</span></div>
        </div>
        <div class='box'>
          <div><b>Shear stress (τ):</b> {_na(tau, 2)} <span class='muted'>(Pa)</span></div>
          <div><b>Unit stream power (ω):</b> {_na(omega_wm2, 2)} <span class='muted'>(W/m²)</span></div>
          <div class='muted' style='margin-top:6px;'>{esc_html(hyd_note)}</div>
        </div>
      </div>
      <div class='muted' style='margin-top:10px;'>{esc_html(str(hyd_row2.get("Hydraulics_note","")))}</div>
    </div>
    """

    # Thresholds and scenarios tables
    thr_html = "<div class='card'><h2>Threshold comparison (top rows)</h2>" + df_to_html_table(thr_df, 18) + "</div>"
    scen_html = "<div class='card'><h2>Scenario bundles ranked (top rows)</h2>" + df_to_html_table(scen_df, 12) + "</div>"

    # Key concerns list
    if key_concerns:
        kc = "<ul>" + "".join([f"<li>{esc_html(s)}</li>" for s in key_concerns[:12]]) + "</ul>"
    else:
        kc = "<div class='muted'>No threshold exceedance flags (or missing thresholds/site values).</div>"
    concerns_html = f"<div class='card'><h2>Key concerns (threshold exceedances)</h2>{kc}</div>"

    # Suggestions list
    sug_html = "<ol>" + "".join([f"<li>{esc_html(s)}</li>" for s in suggestions[:10]]) + "</ol>" if suggestions else "<div class='muted'>No suggestions.</div>"
    actions_html = f"<div class='card'><h2>Practitioner actions (ranked)</h2>{sug_html}</div>"

    # Habitat interpretation (if any)
    hab_html = "<div class='card'><h2>Habitat interpretation</h2>" + df_to_html_table(hab_interp, 20) + "</div>"

    # Chemistry block (if any)
    if chem:
        chem_tbl = pd.DataFrame([
            ["pH", _na(chem.get("pH", np.nan), 2), str(chem.get("pH_status","")), _na(chem.get("pH_statewide_pct", np.nan), 1)],
            ["DO (mg/L)", _na(chem.get("DO_mgL", np.nan), 2), str(chem.get("DO_status","")), _na(chem.get("DO_statewide_pct", np.nan), 1)],
            ["COND", _na(chem.get("COND", np.nan), 1), str(chem.get("COND_status","")), _na(chem.get("COND_statewide_pct", np.nan), 1)],
            ["TN", _na(chem.get("TN", np.nan), 3), str(chem.get("TN_status","")), _na(chem.get("TN_statewide_pct", np.nan), 1)],
            ["TP", _na(chem.get("TP", np.nan), 3), str(chem.get("TP_status","")), _na(chem.get("TP_statewide_pct", np.nan), 1)],
            ["NO3", _na(chem.get("NO3", np.nan), 3), str(chem.get("NO3_status","")), _na(chem.get("NO3_statewide_pct", np.nan), 1)],
            ["Turb", _na(chem.get("TURB", np.nan), 2), str(chem.get("TURB_status","")), _na(chem.get("TURB_statewide_pct", np.nan), 1)],
        ], columns=["Metric", "Value", "Status", "Statewide pct"])
        chem_html = "<div class='card'><h2>Water chemistry window</h2>" + df_to_html_table(chem_tbl, 25) + \
                    f"<div class='muted'>Source: {esc_html(str(chem.get('_chem_source','')))}</div></div>"
    else:
        chem_html = "<div class='card'><h2>Water chemistry window</h2><div class='muted'>No chemistry values found for this SITEYR in available CSVs.</div></div>"

    # Plots
    fi_html = "<div class='card'><h2>Feature importance</h2>" + thumb_grid(fi_imgs, root_dir, cols=4, height_px=86) + "</div>"
    pdp_html = "<div class='card'><h2>Partial dependence (PDP)</h2>" + thumb_grid(pdp_imgs, root_dir, cols=4, height_px=86) + "</div>"

    # Region stdout tail display (improved)
    key_lines = _extract_region_key_lines(region_stdout, 28)
    tail = _tail_lines(region_stdout, 160)
    stderr_tail = _tail_lines(region_stderr, 80)

    region_tail_html = f"""
    <div class='card full'>
      <h2>Region stdout (high-signal lines + tail)</h2>
      <div class='muted'>Useful when you want to confirm the model outputs, mean-BIBI lines, scenarios, and any warnings.</div>
      <details open>
        <summary><b>Key lines (filtered)</b></summary>
        <pre class='scroll'>{esc_html(key_lines) if key_lines else "No key lines extracted."}</pre>
      </details>
      <details>
        <summary><b>Last ~160 lines (stdout tail)</b></summary>
        <pre class='scroll'>{esc_html(tail)}</pre>
      </details>
      <details>
        <summary><b>stderr tail (if any)</b></summary>
        <pre class='scroll'>{esc_html(stderr_tail) if stderr_tail else "No stderr."}</pre>
      </details>
    </div>
    """

    # GIS notes
    if gis_notes:
        gis_html = "<div class='card'><h2>GIS export</h2><ul>" + "".join([f"<li>{esc_html(n)}</li>" for n in gis_notes]) + "</ul></div>"
    else:
        gis_html = "<div class='card'><h2>GIS export</h2><div class='muted'>No GIS exports (skipped or unavailable).</div></div>"

    # Header summary
    header = f"""
    <div class='header'>
      <div>
        <h1>{esc_html(title)}</h1>
        <div class='muted'>
          Coordinate: ({lat:.6f}, {lon:.6f}) |
          SITEYR: <b>{esc_html(siteyr)}</b> |
          DNR12: <b>{esc_html(dnr12_code)}</b> |
          MDE8: <b>{esc_html(mde8_code)}</b> ({esc_html(mde8_name)}) |
          Province: <b>{esc_html(prov)}</b> |
          Stream: <b>{esc_html(streamnm)}</b>
        </div>
      </div>
    </div>
    """

    # StreamStats note block
    ss_note_html = f"<div class='note'>{esc_html(ss_note)}</div>" if ss_note else ""

    # Main HTML template
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>{esc_html(title)}</title>
  <style>
    body {{ font-family: Arial, Helvetica, sans-serif; margin: 0; background:#0f1220; color:#eef2ff; }}
    a {{ color:#9ad1ff; }}
    .wrap {{ max-width: 1180px; margin: 0 auto; padding: 18px; }}
    .header {{ background:#121634; padding:18px; border-radius:16px; box-shadow: 0 6px 20px rgba(0,0,0,0.25); }}
    h1 {{ margin:0 0 6px 0; font-size: 22px; }}
    h2 {{ margin:0 0 10px 0; font-size: 18px; }}
    h3 {{ margin:0 0 8px 0; font-size: 15px; }}
    .muted {{ color:#a6b0d4; font-size: 12.5px; line-height: 1.35; }}
    .grid {{ display:grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-top: 14px; }}
    .grid3 {{ display:grid; grid-template-columns: 1fr 1fr 1fr; gap: 14px; margin-top: 14px; }}
    .grid2 {{ display:grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
    .card {{ background:#121634; border-radius:16px; padding: 14px; box-shadow: 0 6px 18px rgba(0,0,0,0.22); }}
    .card.full {{ grid-column: 1 / -1; }}
    .box {{ background:#0f1430; border: 1px solid rgba(255,255,255,0.06); border-radius:14px; padding: 12px; }}
    .topstats {{ margin-top: 12px; display:flex; flex-wrap:wrap; gap: 8px; }}
    .pill {{ background:#0f1430; border:1px solid rgba(255,255,255,0.08); padding: 6px 10px; border-radius:999px; font-size: 12px; }}
    .tbl {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
    .tbl th, .tbl td {{ border-bottom: 1px solid rgba(255,255,255,0.08); padding: 6px 8px; text-align:left; vertical-align: top; }}
    .tbl th {{ color:#cbd5ff; font-weight: 700; }}
    .note {{ margin-top: 10px; background:#2b1d10; border:1px solid rgba(255,193,7,0.35); color:#ffe5b4; padding: 10px 12px; border-radius: 12px; font-size: 12.5px; }}
    .thumb-grid {{ display:grid; gap: 8px; }}
    .thumb img {{ border-radius: 10px; border: 1px solid rgba(255,255,255,0.08); }}
    .scroll {{ max-height: 340px; overflow:auto; background:#0b1020; color:#e7eefc; padding: 10px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.08); }}
    .grade {{ display:inline-block; padding: 3px 10px; border-radius: 999px; background:#0b1020; border:1px solid rgba(255,255,255,0.12); font-weight: 800; }}
    .btn {{ display:inline-block; padding: 8px 12px; border-radius: 10px; background:#0b1020; border:1px solid rgba(255,255,255,0.14); text-decoration:none; }}
    details summary {{ cursor:pointer; color:#cbd5ff; }}
    @media (max-width: 980px) {{
      .grid {{ grid-template-columns: 1fr; }}
      .grid3 {{ grid-template-columns: 1fr; }}
      .grid2 {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    {header}
    {ss_top}
    {ss_note_html}

    <div class="grid">
      <div class="card">
        <h2>BIBI / Predictions</h2>
        {bibi_tbl_html}
      </div>
      {report_html}
    </div>

    <div class="grid">
      {hydraulics_html}
      {chem_html}
    </div>

    <div class="grid">
      {concerns_html}
      {actions_html}
    </div>

    <div class="grid">
      {hab_html}
      {thr_html}
    </div>

    <div class="grid">
      {scen_html}
      {gis_html}
    </div>

    <div class="grid">
      {fi_html}
      {pdp_html}
    </div>

    <div class="grid">
      {synth_html_block}
    </div>

    <div class="grid">
      {region_tail_html}
    </div>

    <div class="muted" style="margin: 14px 4px;">
      Generated: {esc_html(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))}
    </div>
  </div>
</body>
</html>
"""
    return html


# ----------------------------
# Main
# ----------------------------
def main():
    workdir = Path(__file__).resolve().parent

    # Defaults (your original v1.9e names)
    hydrology_default  = "md_coordinate_hydro_engine_v5_2_PLUS_region_context.py"
    region_default     = "region_analysis_master_all_v2_11_PATCH_UTF8_FIXED.py"
    hydraulics_default = "mbss_hydraulics_from_dischargeC_v1_6_INTERACTIVE.py"

    hydrology_script  = discover_hydrology_script(workdir, hydrology_default)
    region_script     = discover_region_script(workdir, region_default)
    hydraulics_script = discover_hydraulics_script(workdir, hydraulics_default)

    merged_csv = workdir / "MBSS_Merged_All.csv"
    waterchem_csv = workdir / "MBSS-WaterChemistry95_23.csv"
    master_full_csv = workdir / "MBSS_Master_Full.csv"

    print("\n================ HEALTH CARD MASTER v1.9g (FULL) ================\n")
    print(f"Base directory: {workdir}\n")

    # Preflight checks
    missing: List[str] = []
    if hydrology_script is None or not hydrology_script.exists():
        missing.append(str(workdir / hydrology_default))
    if region_script is None or not region_script.exists():
        missing.append(str(workdir / region_default))
    if hydraulics_script is None or not hydraulics_script.exists():
        missing.append(str(workdir / hydraulics_default))
    if not merged_csv.exists():
        missing.append(str(merged_csv))

    missing = [m for m in missing if m and m.strip() and m.strip().lower() != "(not found)"]

    if missing:
        print("[FATAL] Missing required files:")
        for m in missing:
            print("  -", m)
        print("\nFix the missing items and re-run.\n")
        return

    print("[OK] Using scripts:")
    print("  Hydrology  :", hydrology_script.name)
    print("  Region     :", region_script.name)
    print("  Hydraulics :", hydraulics_script.name)
    print("")

    lat = prompt_float("Latitude (decimal degrees): ")
    lon = prompt_float("Longitude (decimal degrees): ")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_root = workdir / "OUT_HEALTH_CARD"
    base_root.mkdir(parents=True, exist_ok=True)

    tmp_root = base_root / "_TMP"
    tmp_root.mkdir(parents=True, exist_ok=True)
    run_folder_tmp = tmp_root / safe_slug(f"HealthCard_{lat}_{lon}_{ts}")

    logs_dir = run_folder_tmp / "logs"
    figs_dir = run_folder_tmp / "figs"
    csv_dir  = run_folder_tmp / "csv"
    gis_dir  = run_folder_tmp / "gis"
    for d in [logs_dir, figs_dir, csv_dir, gis_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 1) Hydraulics/snapping (gets SITEYR + watershed context)
    # PATCH: try argparse (--lat/--lon) first; fallback to stdin if script is purely interactive.
    comp3 = run_script(
        hydraulics_script,
        name="hydraulics",
        workdir=workdir,
        logs_dir=logs_dir,
        stdin_text="",
        extra_args=["--lat", str(lat), "--lon", str(lon)],
        timeout_sec=600,
    )
    if (not comp3.ok) or (extract_siteyr_from_stdout(comp3.stdout_text or "") is None):
        comp3 = run_script(
            hydraulics_script,
            name="hydraulics_retry_stdin",
            workdir=workdir,
            logs_dir=logs_dir,
            stdin_text=f"{lat}\n{lon}\n",
            extra_args=None,
            timeout_sec=600,
        )

    siteyr = extract_siteyr_from_stdout(comp3.stdout_text) or "UNKNOWN_SITEYR"
    wblock = extract_watershed_block(comp3.stdout_text)

    dnr12_code = (wblock.get("DNR12DIG") or "UNKNOWN_DNR12").strip()
    mde8_code  = (wblock.get("MDE8") or "UNKNOWN_MDE8").strip()
    mde8_name  = (wblock.get("MDE8 Name") or "UNKNOWN_MDE8_NAME").strip()

    final_name = safe_slug(f"HealthCard_{mde8_name}__MDE8_{mde8_code}__DNR12_{dnr12_code}__{siteyr}__{ts}")
    run_folder = base_root / final_name

    # 2) Hydrology (StreamStats + Q2/Q10/Q100 + slope in stdout)
    comp1 = run_script(
        hydrology_script,
        name="hydrology",
        workdir=workdir,
        logs_dir=logs_dir,
        stdin_text=f"{lat}\n{lon}\n",
        extra_args=None,
        timeout_sec=900,
    )

    ss_items = parse_streamstats_from_stdout(comp1.stdout_text or "")
    ss_note = streamstats_warning_note(comp1.stdout_text or "")

    # 3) Region (models + FI + thresholds + PDP + scenarios)
    comp2_start = datetime.now().timestamp()
    comp2 = run_script(
        region_script,
        name="region",
        workdir=workdir,
        logs_dir=logs_dir,
        stdin_text=f"{lat}\n{lon}\n",
        extra_args=None,
        timeout_sec=1800,
    )

    if not comp2.ok:
        print("\n[REGION FAILED] Region returned nonzero exit code.")
        print("Check logs/region_stderr.txt for the reason.\n")
    else:
        print("\n[REGION OK] Region completed. Collecting artifacts...\n")

    region_out_folder = discover_region_output_folder(comp2.stdout_text or "", workdir)

    search_roots = [workdir]
    if region_out_folder and region_out_folder.exists():
        search_roots.insert(0, region_out_folder)

    thr_csv = find_newest_in_roots(search_roots, "threshold_comparison_bibi.csv", since_ts=comp2_start)
    scen_csv = find_newest_in_roots(search_roots, "scenario_bundles_ranked.csv", since_ts=comp2_start)

    if (thr_csv is None) and (scen_csv is None):
        print("[WARNING] No threshold/scenario CSVs were found after Region run.")
        print("          Open logs/region_stdout.txt and logs/region_stderr.txt.\n")

    thr_df = pd.read_csv(thr_csv) if (thr_csv and thr_csv.exists()) else None
    scen_df = pd.read_csv(scen_csv) if (scen_csv and scen_csv.exists()) else None

    preds = parse_predicted_bibi_from_region(comp2.stdout_text or "")

    obs = load_mbss_observed(merged_csv, siteyr) if siteyr != "UNKNOWN_SITEYR" else {}
    obs_bibi = fnum(obs.get("Observed_BIBI", np.nan))
    obs_fibi = fnum(obs.get("Observed_FIBI", np.nan))

    # Hydraulics outputs
    hyd_csv = workdir / "OUT_MBSS_HYDRAULICS" / f"MBSS_Hydraulics_C_{siteyr}.csv"
    hab_interp_csv = workdir / "OUT_MBSS_HYDRAULICS" / f"MBSS_Habitat_Interpretation_{siteyr}.csv"
    hyd_row = read_first_row_csv(hyd_csv) if hyd_csv.exists() else {}
    hab_interp = pd.read_csv(hab_interp_csv) if hab_interp_csv.exists() else None

    slope_m_m, slope_note = resolve_slope_m_m(hyd_row, comp1.stdout_text or "")
    hyd_row["Slope_m_m_resolved"] = slope_m_m
    hyd_row["Slope_note"] = slope_note
    hyd_row2 = recompute_tau_omega(hyd_row, slope_m_m)

    # Key concerns from thresholds
    key_concerns: List[str] = []
    if thr_df is not None and not thr_df.empty:
        tmp = thr_df.copy()
        tmp.columns = [c.strip() for c in tmp.columns]
        needed = {"Stressor", "State_thr", "Prov_thr", "Local_thr", "Site_value"}
        if not needed.issubset(set(tmp.columns)):
            m = {}
            for c in tmp.columns:
                cl = c.lower()
                if "stressor" in cl or "feature" in cl:
                    m[c] = "Stressor"
                elif "state" in cl:
                    m[c] = "State_thr"
                elif "prov" in cl:
                    m[c] = "Prov_thr"
                elif "local" in cl:
                    m[c] = "Local_thr"
                elif "site" in cl or "value" in cl:
                    m[c] = "Site_value"
            tmp = tmp.rename(columns=m)

        if needed.issubset(set(tmp.columns)):
            for _, r in tmp.iterrows():
                msg = flag_threshold_exceed(r)
                if msg:
                    key_concerns.append(msg)
        thr_df = tmp

    suggestions = management_suggestions(key_concerns, hab_interp)

    # Best scenario values
    pred_base = preds.get("pred_baseline", np.nan)
    pred_best = np.nan
    pred_best_delta = np.nan

    m_best = re.search(r"BEST\s*\(combined bundles\).*?:\s*([0-9.]+)\s*\(Δ\s*([+-]?[0-9.]+)\)", comp2.stdout_text or "")
    if m_best:
        try:
            pred_best = float(m_best.group(1))
            pred_best_delta = float(m_best.group(2))
        except Exception:
            pass

    if not np.isfinite(pred_best) and scen_df is not None and not scen_df.empty:
        r0 = scen_df.iloc[0].to_dict()
        for k in r0.keys():
            kl = str(k).lower()
            if ("bibi" in kl) and ("pred" in kl or "priority" in kl or "best" in kl):
                pred_best = fnum(r0.get(k))
                if np.isfinite(pred_best):
                    break
        if np.isfinite(pred_base) and np.isfinite(pred_best):
            pred_best_delta = pred_best - pred_base

    resid = (obs_bibi - pred_base) if (np.isfinite(obs_bibi) and np.isfinite(pred_base)) else np.nan

    # Mean references from MBSS_Merged_All.csv
    means = compute_mean_bibi_references_from_merged(
        merged_csv=merged_csv,
        siteyr=siteyr,
        dnr12_code=dnr12_code,
        mde8_code=mde8_code,
    )
    mean_bibi_site = fnum(means.get("mean_bibi_site", np.nan))
    mean_bibi_dnr12 = fnum(means.get("mean_bibi_dnr12", np.nan))
    mean_bibi_mde8  = fnum(means.get("mean_bibi_mde8", np.nan))

    # PATCH: fill NA from Region stdout means
    region_means = parse_mean_bibi_from_region_stdout(comp2.stdout_text or "")
    if not np.isfinite(mean_bibi_dnr12):
        v = fnum(region_means.get("mean_bibi_dnr12_from_region", np.nan))
        if np.isfinite(v):
            mean_bibi_dnr12 = v
    if not np.isfinite(mean_bibi_mde8):
        v = fnum(region_means.get("mean_bibi_mde8_from_region", np.nan))
        if np.isfinite(v):
            mean_bibi_mde8 = v

    report_card = build_report_card(
        obs_bibi=obs_bibi,
        pred_base=pred_base,
        pred_best=pred_best,
        thr_df=thr_df,
        mean_bibi_dnr12=mean_bibi_dnr12,
        mean_bibi_mde8=mean_bibi_mde8,
    )

    # Chemistry
    chem: Dict[str, Any] = {}
    if siteyr != "UNKNOWN_SITEYR":
        chem = load_mbss_chemistry_window(
            merged_csv=merged_csv,
            siteyr=siteyr,
            thr_df=thr_df,
            waterchem_csv=waterchem_csv if waterchem_csv.exists() else None,
            master_full_csv=master_full_csv if master_full_csv.exists() else None,
        )

    # Copy CSVs into /csv (tight)
    def copy_to_csv_dir(p: Optional[Path]) -> None:
        if p and p.exists():
            try:
                shutil.copy2(p, csv_dir / p.name)
            except Exception:
                pass

    for p in [hyd_csv, hab_interp_csv, thr_csv, scen_csv]:
        copy_to_csv_dir(p)

    # REGION ARTIFACT COLLECTION
    if region_out_folder and region_out_folder.exists():
        for p in region_out_folder.rglob("*.csv"):
            copy_to_csv_dir(p)
        for p in region_out_folder.rglob("*.png"):
            try:
                shutil.copy2(p, figs_dir / p.name)
            except Exception:
                pass
        pdp_dir = region_out_folder / "pdp"
        if pdp_dir.exists():
            for p in pdp_dir.glob("*.png"):
                try:
                    shutil.copy2(p, figs_dir / p.name)
                except Exception:
                    pass

    fi_local = sorted([p for p in figs_dir.glob("fi_*.png")], key=lambda p: p.name)
    pdp_local = sorted([p for p in figs_dir.glob("pdp_*.png")], key=lambda p: p.name)

    # OPTIONAL GIS export
    gis_notes: List[str] = []
    if ENABLE_GIS_EXPORT and dnr12_code != "UNKNOWN_DNR12" and mde8_code != "UNKNOWN_MDE8":
        gis_notes = export_watersheds_arcpy(
            out_gis_dir=gis_dir,
            dnr12_code=dnr12_code,
            mde8_code=mde8_code,
            dnr12_folder=DNR12_FOLDER,
            mde8_folder=MDE8_FOLDER,
            dnr12_hint=DNR12_SHP_HINT,
            mde8_hint=MDE8_SHP_HINT,
        )


    # Ranking process HTML (always generate a local file so the hyperlink works)
    ranking_html = build_ranking_process_html(
        siteyr=siteyr,
        dnr12_code=dnr12_code,
        mde8_code=mde8_code,
        mde8_name=mde8_name,
        obs_bibi=obs_bibi,
        mean_site_bibi=mean_bibi_site,
        mean_dnr12_bibi=mean_bibi_dnr12,
        mean_mde8_bibi=mean_bibi_mde8,
        pred_base=pred_base,
        pred_best=pred_best,
        exceed_rate=report_card.get("exceed_rate", np.nan),
        uplift_delta=report_card.get("uplift_delta", np.nan),
        key_concerns=key_concerns,
        thr_df=thr_df,
        scen_df=scen_df,
    )
    (run_folder_tmp / "Ranking_Process.html").write_text(ranking_html, encoding="utf-8", errors="replace")

    # Build synthesis
    synth_text = build_post_synthesis_text(
        siteyr=siteyr,
        wblock=wblock,
        report_card=report_card,
        obs_bibi=obs_bibi,
        pred_base=pred_base,
        pred_best=pred_best,
        pred_best_delta=pred_best_delta,
        resid=resid,
        key_concerns=key_concerns,
        suggestions=suggestions,
        chem=chem,
        hab_interp=hab_interp,
        hyd_row2=hyd_row2,
        ss_items=ss_items,
        ss_note=ss_note,
        mean_site_bibi=mean_bibi_site,
        mean_dnr12_bibi=mean_bibi_dnr12,
        mean_mde8_bibi=mean_bibi_mde8,
    )
    synth_html_block = build_post_synthesis_html(synth_text)

    # TEXT report
    lines: List[str] = []
    lines.append("HEALTH CARD REPORT")
    lines.append("=" * 88)
    lines.append(f"Coordinate: ({lat}, {lon})")
    lines.append(f"Nearest SITEYR: {siteyr}")
    lines.append(f"DNR12DIG: {dnr12_code} | MDE8: {mde8_code} | MDE8 Name: {mde8_name}")
    lines.append("")

    lines.append("Mean BIBI references (from MBSS_Merged_All.csv; fallback from Region stdout if needed):")
    lines.append(f"  Mean BIBI (site across years): {mean_bibi_site}")
    lines.append(f"  Mean BIBI (DNR12): {mean_bibi_dnr12}")
    lines.append(f"  Mean BIBI (MDE8): {mean_bibi_mde8}")
    lines.append("")

    lines.append("Predicted BIBI (multi-scale, parsed from Region stdout when available):")
    lines.append(f"  Baseline (site priority): {pred_base}")
    lines.append(f"  Best scenario predicted: {pred_best}  (Δ={pred_best_delta})")
    lines.append(f"  MDE8 predicted: {preds.get('pred_mde8', np.nan)}")
    lines.append(f"  Province predicted: {preds.get('pred_prov', np.nan)}")
    lines.append(f"  Statewide predicted: {preds.get('pred_state', np.nan)}")
    lines.append("")
    lines.append(f"Residual (Obs - baseline): {resid}")
    lines.append("")

    lines.append("Component status:")
    for comp in [comp3, comp1, comp2]:
        lines.append(f"  {comp.name}: {'OK' if comp.ok else 'FAILED'} (rc={comp.returncode}) log={comp.stdout_path.name}")
    lines.append("")

    if ss_items:
        lines.append("StreamStats basin summary (parsed from hydrology engine):")
        for k, v in ss_items:
            lines.append(f"  {k}: {v}")
        lines.append("")
    else:
        lines.append("StreamStats basin summary: NA (not parsed from hydrology output)")
        lines.append("")
    if ss_note:
        lines.append("StreamStats caution:")
        lines.append(f"  {ss_note}")
        lines.append("")

    lines.append("Observed (MBSS):")
    lines.append(f"  Observed BIBI: {obs_bibi}")
    lines.append(f"  Observed FIBI: {obs_fibi}")
    lines.append("")

    lines.append("Hydraulics (recomputed where possible):")
    lines.append(f"  Resolved slope (m/m): {hyd_row2.get('Slope_m_m_resolved', np.nan)}")
    lines.append(f"  Slope note: {hyd_row2.get('Slope_note','')}")
    lines.append(f"  Q_C_m3s: {hyd_row2.get('Q_C_m3s', np.nan)}")
    lines.append(f"  ShearStress_Pa_calc: {hyd_row2.get('ShearStress_Pa_calc', np.nan)}")
    lines.append(f"  UnitStreamPower_W_per_m2_calc: {hyd_row2.get('UnitStreamPower_W_per_m2_calc', np.nan)}")
    lines.append(f"  Interpretation: {_hydraulic_impact_note(fnum(hyd_row2.get('ShearStress_Pa_calc',np.nan)), fnum(hyd_row2.get('UnitStreamPower_W_per_m2_calc',np.nan)))}")
    lines.append("")

    if not comp2.ok:
        lines.append("REGION ERROR (tail of region_stderr):")
        lines.append((comp2.stderr_text or "")[-2500:])
        lines.append("")

    lines.append("REPORT CARD (A–F)  [See Ranking_Process.html for details]")
    lines.append("-" * 88)
    lines.append(f"DNR12 Current   : {report_card.get('dnr12_current_grade','NA')}  (score={report_card.get('dnr12_current_score',np.nan)})  [{report_card.get('dnr12_reason','')}]")
    lines.append(f"DNR12 Potential : {report_card.get('dnr12_potential_grade','NA')}  (score={report_card.get('dnr12_potential_score',np.nan)})")
    lines.append(f"MDE8  Current   : {report_card.get('mde8_current_grade','NA')}  (score={report_card.get('mde8_current_score',np.nan)})  [{report_card.get('mde8_reason','')}]")
    lines.append(f"MDE8  Potential : {report_card.get('mde8_potential_grade','NA')}  (score={report_card.get('mde8_potential_score',np.nan)})")
    lines.append("")
    lines.append(f"Threshold pressure (share exceedances): {report_card.get('exceed_rate', np.nan)}")
    lines.append(f"Uplift potential (best - baseline): {report_card.get('uplift_delta', np.nan)}")
    lines.append("")

    if key_concerns:
        lines.append("Key concerns (threshold exceedances):")
        for s in key_concerns[:12]:
            lines.append(f"  - {s}")
        lines.append("")
    if suggestions:
        lines.append("Practitioner actions (ranked):")
        for s in suggestions[:10]:
            lines.append(f"  - {s}")
        lines.append("")

    if gis_notes:
        lines.append("GIS export notes:")
        for n in gis_notes:
            lines.append(f"  - {n}")
        lines.append("")

    lines.append("Post synthesis (same text embedded in HealthCard):")
    lines.append("-" * 88)
    lines.append(synth_text)
    lines.append("")

    # Write report files into tmp folder first
    (run_folder_tmp / "HealthCard.txt").write_text("\n".join(lines), encoding="utf-8", errors="replace")

    # Build HealthCard HTML
    title = f"MBSS HealthCard — {siteyr}"
    health_html = build_healthcard_html(
        title=title,
        lat=lat,
        lon=lon,
        siteyr=siteyr,
        wblock=wblock,
        ss_items=ss_items,
        ss_note=ss_note,
        obs_bibi=obs_bibi,
        obs_fibi=obs_fibi,
        mean_site_bibi=mean_bibi_site,
        mean_dnr12_bibi=mean_bibi_dnr12,
        mean_mde8_bibi=mean_bibi_mde8,
        preds=preds,
        pred_base=pred_base,
        pred_best=pred_best,
        pred_best_delta=pred_best_delta,
        resid=resid,
        report_card=report_card,
        thr_df=thr_df,
        scen_df=scen_df,
        hyd_row2=hyd_row2,
        slope_note=slope_note,
        hab_interp=hab_interp,
        chem=chem,
        suggestions=suggestions,
        key_concerns=key_concerns,
        fi_imgs=fi_local,
        pdp_imgs=pdp_local,
        root_dir=run_folder_tmp,
        ranking_html=ranking_html,
        region_stdout=comp2.stdout_text or "",
        region_stderr=comp2.stderr_text or "",
        synth_html_block=synth_html_block,
        gis_notes=gis_notes,
    )
    (run_folder_tmp / "HealthCard.html").write_text(health_html, encoding="utf-8", errors="replace")

    # Ensure final run folder exists, then move tmp run outputs into it
    if run_folder.exists():
        # avoid overwrite collisions
        run_folder = base_root / safe_slug(final_name + "__DUP_" + datetime.now().strftime("%H%M%S"))
    run_folder.mkdir(parents=True, exist_ok=True)

    # Move entire tmp folder contents to run_folder (robust)
    for item in run_folder_tmp.iterdir():
        dest = run_folder / item.name
        try:
            if dest.exists():
                if dest.is_dir():
                    shutil.rmtree(dest, ignore_errors=True)
                else:
                    dest.unlink(missing_ok=True)  # py3.8+ on Windows ok in 3.11+
            if item.is_dir():
                shutil.copytree(item, dest)
            else:
                shutil.copy2(item, dest)
        except Exception:
            # last resort: ignore
            pass

    # Cleanup tmp folder
    try:
        shutil.rmtree(run_folder_tmp, ignore_errors=True)
    except Exception:
        pass

    # Final console summary (short + actionable)
    print("\n==================== RUN COMPLETE ====================")
    print(f"Output folder : {run_folder}")
    print(f"HealthCard    : {run_folder / 'HealthCard.html'}")
    print(f"Text report   : {run_folder / 'HealthCard.txt'}")
    print(f"Ranking       : {run_folder / 'Ranking_Process.html'}")
    print(f"Logs          : {run_folder / 'logs'}")
    print(f"CSV artifacts : {run_folder / 'csv'}")
    print(f"Plots         : {run_folder / 'figs'}")
    if ENABLE_GIS_EXPORT:
        print(f"GIS           : {run_folder / 'gis'}")
    print("------------------------------------------------------")
    print(f"SITEYR: {siteyr}")
    print(f"DNR12:  {dnr12_code} | MDE8: {mde8_code} ({mde8_name})")
    print(f"Observed BIBI: {obs_bibi} | Baseline Pred: {pred_base} | Best Pred: {pred_best} (Δ {pred_best_delta})")
    print(f"Grades: DNR12 {report_card.get('dnr12_current_grade','NA')}→{report_card.get('dnr12_potential_grade','NA')} | "
          f"MDE8 {report_card.get('mde8_current_grade','NA')}→{report_card.get('mde8_potential_grade','NA')}")
    if ss_note:
        print(f"StreamStats note: {ss_note}")
    if not comp2.ok:
        print("NOTE: Region failed — open logs/region_stderr.txt.")
    print("======================================================\n")
if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
