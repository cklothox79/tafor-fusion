# app.py
"""
TAFOR Fusion Pro — Operational v2.4 (WARR / Sedati Gede)
- Full Streamlit app with styled download buttons and ZIP export
- Adds: daily logging to ./logs/, auto-alerts for PoP/Wind/High RH+CC
ADM4: 35.15.17.2011 (Sedati Gede)
"""

import os
import io
import json
import logging
import zipfile
from datetime import datetime, timedelta
import math

import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# Basic config
# -----------------------
st.set_page_config(page_title="TAFOR Fusion Pro — Operational v2.4 (WARR)", layout="centered")
st.title("🛫 TAFOR Fusion Pro — Operational (WARR / Sedati Gede)")
st.caption("Location: Sedati Gede (ADM4=35.15.17.2011). Fusi BMKG + Open-Meteo + METAR realtime")

# create folders
os.makedirs("output", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# logging basic
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# constants
LAT, LON = -7.379, 112.787
ADM4 = "35.15.17.2011"
REFRESH_TTL = 600  # cache TTL seconds
DEFAULT_WEIGHTS = {"bmkg": 0.45, "ecmwf": 0.25, "icon": 0.15, "gfs": 0.15}

# -----------------------
# UI Inputs
# -----------------------
col1, col2, col3 = st.columns(3)
with col1:
    issue_date = st.date_input("📅 Issue date (UTC)", datetime.utcnow().date())
with col2:
    jam_penting = [0, 3, 6, 9, 12, 15, 18, 21]
    default_hour = min(jam_penting, key=lambda j: abs(j - datetime.utcnow().hour))
    issue_time = st.selectbox("🕓 Issue time (UTC)", jam_penting, index=jam_penting.index(default_hour))
with col3:
    validity = st.number_input("🕐 Validity (hours)", min_value=6, max_value=36, value=24, step=6)

st.markdown("### ⚙️ Ensemble weights (BMKG priority)")
wcols = st.columns(4)
bmkg_w = wcols[0].number_input("BMKG", 0.0, 1.0, value=DEFAULT_WEIGHTS["bmkg"], step=0.05)
ecmwf_w = wcols[1].number_input("ECMWF", 0.0, 1.0, value=DEFAULT_WEIGHTS["ecmwf"], step=0.05)
icon_w = wcols[2].number_input("ICON", 0.0, 1.0, value=DEFAULT_WEIGHTS["icon"], step=0.05)
gfs_w = wcols[3].number_input("GFS", 0.0, 1.0, value=DEFAULT_WEIGHTS["gfs"], step=0.05)
sumw = bmkg_w + ecmwf_w + icon_w + gfs_w or 1.0
weights = {"bmkg": bmkg_w / sumw, "ecmwf": ecmwf_w / sumw, "icon": icon_w / sumw, "gfs": gfs_w / sumw}
st.caption(f"Normalized weights: {weights}")

st.divider()

# -----------------------
# Helpers
# -----------------------
def wind_to_uv(speed, deg):
    if speed is None or deg is None or (isinstance(speed, float) and math.isnan(speed)) or (isinstance(deg, float) and math.isnan(deg)):
        return np.nan, np.nan
    theta = math.radians((270.0 - deg) % 360.0)
    return speed * math.cos(theta), speed * math.sin(theta)

def uv_to_wind(u, v):
    try:
        spd = math.sqrt(u * u + v * v)
        theta = math.degrees(math.atan2(v, u))
        deg = (270.0 - theta) % 360.0
        return spd, deg
    except Exception:
        return np.nan, np.nan

def safe_to_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan

def safe_int(x, default=0):
    try:
        return int(round(float(x)))
    except Exception:
        return default

def weighted_mean(vals, ws):
    if not vals or not ws:
        return np.nan
    arr = np.array([np.nan if v is None else v for v in vals], dtype=float)
    w = np.array(ws[:len(arr)], dtype=float)
    if len(w) == 0:
        return np.nan
    mask = ~np.isnan(arr)
    if not mask.any():
        return np.nan
    w_mask = w[mask]
    if w_mask.sum() == 0:
        return float(np.nanmean(arr[mask]))
    return float((arr[mask] * w_mask).sum() / w_mask.sum())

def fmt_bytes(n):
    try:
        n = float(n)
    except Exception:
        return "0 B"
    for unit in ["B","KB","MB","GB","TB"]:
        if n < 1024.0:
            return f"{n:3.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} PB"

# -----------------------
# Fetchers (cached)
# -----------------------
@st.cache_data(ttl=REFRESH_TTL)
def fetch_bmkg(adm4=ADM4, local_fallback="JSON_BMKG.txt"):
    url = "https://cuaca.bmkg.go.id/api/df/v1/forecast/adm"
    params = {"adm1": "35", "adm2": "35.15", "adm3": "35.15.17", "adm4": adm4}
    try:
        r = requests.get(url, params=params, timeout=15, verify=False)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        logging.warning("BMKG API failed: %s", e)
        if os.path.exists(local_fallback):
            with open(local_fallback, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            return {"status": "Unavailable"}
    try:
        cuaca = data["data"][0]["cuaca"][0][0]
        return {"status": "OK", "raw": data, "cuaca": cuaca}
    except Exception:
        try:
            c = data.get("data", [{}])[0].get("cuaca")
            if isinstance(c, list):
                flattened = []
                for item in c:
                    if isinstance(item, list):
                        for sub in item:
                            flattened.append(sub)
                    else:
                        flattened.append(item)
                return {"status": "OK", "raw": data, "cuaca": flattened}
        except Exception:
            pass
    return {"status": "Unavailable", "raw": data}

@st.cache_data(ttl=REFRESH_TTL)
def fetch_openmeteo(model):
    base = f"https://api.open-meteo.com/v1/{model}"
    params = {
        "latitude": LAT,
        "longitude": LON,
        "hourly": "temperature_2m,relative_humidity_2m,cloud_cover,windspeed_10m,winddirection_10m,visibility",
        "forecast_days": 2,
        "timezone": "UTC"
    }
    try:
        r = requests.get(base, params=params, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logging.warning("Open-Meteo %s failed: %s", model, e)
        return None

@st.cache_data(ttl=REFRESH_TTL)
def fetch_metar_ogimet(station="WARR"):
    try:
        og = requests.get(f"https://ogimet.com/display_metars2.php?lang=en&icao={station}", timeout=10)
        if og.ok:
            text = og.text
            lines = [ln.strip() for ln in text.splitlines() if station in ln]
            if lines:
                last = lines[-1]
                import re
                last = re.sub("<[^<]+?>", "", last)
                idx = last.find(station)
                if idx >= 0:
                    return " ".join(last[idx:].split())
    except Exception as e:
        logging.warning("OGIMET failed: %s", e)
    try:
        r = requests.get(f"https://tgftp.nws.noaa.gov/data/observations/metar/stations/{station}.TXT", timeout=10)
        if r.ok:
            lines = r.text.strip().splitlines()
            return lines[-1].strip()
    except Exception as e:
        logging.warning("NOAA METAR fallback failed: %s", e)
    return None

# -----------------------
# Parsers & converters
# -----------------------
def bmkg_cuaca_to_df(cuaca):
    records = []
    if isinstance(cuaca, dict):
        records = [cuaca]
    elif isinstance(cuaca, list):
        for item in cuaca:
            if isinstance(item, dict):
                records.append(item)
            elif isinstance(item, list):
                for sub in item:
                    if isinstance(sub, dict):
                        records.append(sub)
    else:
        return pd.DataFrame()

    times, tvals, rhvals, tccvals, wsvals, wdvals, visvals = [], [], [], [], [], [], []
    for rec in records:
        if not isinstance(rec, dict):
            continue
        dt = rec.get("datetime") or rec.get("time") or rec.get("jamCuaca") or rec.get("date") or rec.get("valid_time")
        if isinstance(dt, str):
            try:
                t0 = pd.to_datetime(dt.replace("Z", "+00:00"), utc=True)
            except Exception:
                try:
                    t0 = pd.to_datetime(dt)
                except Exception:
                    t0 = None
        elif isinstance(dt, (int, float)):
            try:
                t0 = pd.to_datetime(dt, unit="s", utc=True)
            except Exception:
                t0 = None
        else:
            t0 = None
        if t0 is None:
            continue
        # store naive UTC
        try:
            times.append(t0.tz_convert("UTC").tz_localize(None))
        except Exception:
            try:
                times.append(t0.tz_localize(None))
            except Exception:
                times.append(pd.to_datetime(t0))
        tvals.append(safe_to_float(rec.get("t") or rec.get("temp") or rec.get("temperature")))
        rhvals.append(safe_to_float(rec.get("hu") or rec.get("rh") or rec.get("humidity")))
        tccvals.append(safe_to_float(rec.get("tcc") or rec.get("cloud") or rec.get("cloud_cover")))
        wsvals.append(safe_to_float(rec.get("ws") or rec.get("wind_speed")))
        wdvals.append(safe_to_float(rec.get("wd_deg") or rec.get("wind_dir") or rec.get("wind_direction")))
        visvals.append(rec.get("vs_text") or rec.get("visibility") or np.nan)

    if not times:
        return pd.DataFrame()

    df = pd.DataFrame({
        "time": times,
        "T_BMKG": tvals,
        "RH_BMKG": rhvals,
        "CC_BMKG": tccvals,
        "WS_BMKG": wsvals,
        "WD_BMKG": wdvals,
        "VIS_BMKG": visvals
    })
    df = df.sort_values("time").reset_index(drop=True)
    return df

def openmeteo_json_to_df(j, tag):
    if not j or "hourly" not in j:
        return None
    h = j["hourly"]
    df = pd.DataFrame({"time": pd.to_datetime(h["time"])})
    df[f"T_{tag}"] = h.get("temperature_2m")
    df[f"RH_{tag}"] = h.get("relative_humidity_2m")
    df[f"CC_{tag}"] = h.get("cloud_cover")
    df[f"WS_{tag}"] = h.get("windspeed_10m")
    df[f"WD_{tag}"] = h.get("winddirection_10m")
    df[f"VIS_{tag}"] = h.get("visibility", [np.nan] * len(df))
    return df

# -----------------------
# Align & Fuse
# -----------------------
def align_hourly(dfs):
    normalized = []
    for d in dfs:
        if d is None:
            continue
        if "time" in d.columns:
            d["time"] = pd.to_datetime(d["time"], errors="coerce")
            d = d.dropna(subset=["time"])
            try:
                d["time"] = d["time"].dt.tz_convert("UTC").dt.tz_localize(None)
            except Exception:
                try:
                    d["time"] = d["time"].dt.tz_localize(None)
                except Exception:
                    pass
            normalized.append(d)
    if not normalized:
        return None
    base = normalized[0][["time"]].copy()
    for d in normalized[1:]:
        base = pd.merge(base, d, on="time", how="outer")
    base = base.sort_values("time").reset_index(drop=True)
    return base

def fuse_ensemble(df_merged, weights, hours=24):
    rows = []
    now = pd.to_datetime(datetime.utcnow()).floor("H")
    df_merged = df_merged.sort_values("time").reset_index(drop=True)
    df_merged = df_merged[df_merged["time"] >= now].head(hours)

    for _, r in df_merged.iterrows():
        T_vals, RH_vals, CC_vals, VIS_vals = [], [], [], []
        u_vals, v_vals = [], []
        w_list = []
        if weights.get("bmkg", 0) > 0:
            t = r.get("T_BMKG"); rh = r.get("RH_BMKG"); cc = r.get("CC_BMKG")
            ws = r.get("WS_BMKG"); wd = r.get("WD_BMKG"); vis = r.get("VIS_BMKG")
            if not pd.isna(t): T_vals.append(t)
            if not pd.isna(rh): RH_vals.append(rh)
            if not pd.isna(cc): CC_vals.append(cc)
            try:
                VIS_vals.append(float(vis))
            except Exception:
                pass
            if not pd.isna(ws) and not pd.isna(wd):
                u, v = wind_to_uv(ws, wd); u_vals.append(u); v_vals.append(v)
            w_list.append(weights["bmkg"])
        for model in ["ecmwf", "icon", "gfs"]:
            tag = model.upper()
            wt = weights.get(model, 0)
            if wt <= 0:
                continue
            t = r.get(f"T_{tag}"); rh = r.get(f"RH_{tag}"); cc = r.get(f"CC_{tag}")
            ws = r.get(f"WS_{tag}"); wd = r.get(f"WD_{tag}"); vis = r.get(f"VIS_{tag}")
            if not pd.isna(t): T_vals.append(t)
            if not pd.isna(rh): RH_vals.append(rh)
            if not pd.isna(cc): CC_vals.append(cc)
            try:
                VIS_vals.append(float(vis))
            except Exception:
                pass
            if not pd.isna(ws) and not pd.isna(wd):
                u, v = wind_to_uv(ws, wd); u_vals.append(u); v_vals.append(v)
            w_list.append(wt)
        if not w_list:
            continue
        T_f = weighted_mean(T_vals, w_list)
        RH_f = weighted_mean(RH_vals, w_list)
        CC_f = weighted_mean(CC_vals, w_list)
        VIS_f = weighted_mean(VIS_vals, w_list)
        U_f = weighted_mean(u_vals, w_list) if u_vals else np.nan
        V_f = weighted_mean(v_vals, w_list) if v_vals else np.nan
        WS_f, WD_f = uv_to_wind(U_f, V_f)
        rows.append({
            "time": r["time"], "T": T_f, "RH": RH_f, "CC": CC_f, "VIS": VIS_f, "WS": WS_f, "WD": WD_f
        })
    return pd.DataFrame(rows)

# -----------------------
# Probabilities
# -----------------------
def compute_probabilities(df_merged, models_list=["GFS", "ECMWF", "ICON", "BMKG"]):
    probs = []
    for _, r in df_merged.iterrows():
        votes = 0
        nm = 0
        temps = []
        for src in models_list:
            nm += 1
            t = r.get(f"T_{src}") if src != "BMKG" else r.get("T_BMKG")
            rh = r.get(f"RH_{src}") if src != "BMKG" else r.get("RH_BMKG")
            cc = r.get(f"CC_{src}") if src != "BMKG" else r.get("CC_BMKG")
            if t is not None:
                temps.append(safe_to_float(t))
            try:
                if (safe_to_float(cc) >= 80) and (safe_to_float(rh) >= 85):
                    votes += 1
            except Exception:
                pass
        prob = votes / nm if nm > 0 else 0.0
        spread = float(np.nanstd([x for x in temps if not pd.isna(x)])) if temps else np.nan
        probs.append({"time": r["time"], "PoP_precip": prob, "T_spread": spread})
    return pd.DataFrame(probs)

# -----------------------
# TAF builder
# -----------------------
def tcc_to_cloud_label(cc):
    if pd.isna(cc):
        return "FEW020"
    try:
        c = float(cc)
    except Exception:
        return "FEW020"
    if c < 25: return "FEW020"
    elif c < 50: return "SCT025"
    elif c < 85: return "BKN030"
    else: return "OVC030"

def build_taf_from_fused(df_fused, df_merged_for_flags, metar, issue_dt, validity):
    taf_lines = []
    header = f"TAF WARR {issue_dt:%d%H%MZ} {issue_dt:%d%H}/{(issue_dt + timedelta(hours=validity)):%d%H}"
    taf_lines.append(header)
    if df_fused is None or df_fused.empty:
        taf_lines += ["00000KT 9999 FEW020", "NOSIG", "RMK AUTO FUSION BASED ON MODEL ONLY"]
        return taf_lines, []

    first = df_fused.iloc[0]
    wd = safe_int(first.WD, 90)
    ws = safe_int(first.WS, 5)
    vis = safe_int(first.VIS, 9999)
    cloud = tcc_to_cloud_label(first.CC)
    taf_lines.append(f"{wd:03d}{ws:02d}KT {vis:04d} {cloud}")

    WIND_CHANGE_DEG = 60
    WIND_SPEED_KT = 10
    CLOUD_CHANGE_PCT = 25

    becmg, tempo = [], []
    signif_times = []

    for i in range(1, len(df_fused)):
        prev = df_fused.iloc[i - 1]
        curr = df_fused.iloc[i]
        tstart = prev["time"].strftime("%d%H")
        tend = curr["time"].strftime("%d%H")
        wd_diff = abs((curr.WD or 0) - (prev.WD or 0))
        ws_diff = abs((curr.WS or 0) - (prev.WS or 0))
        cc_diff = abs((curr.CC or 0) - (prev.CC or 0))

        sig_wind = wd_diff >= WIND_CHANGE_DEG or ws_diff >= WIND_SPEED_KT
        sig_cloud = cc_diff >= CLOUD_CHANGE_PCT

        if sig_wind or sig_cloud:
            becmg.append(f"BECMG {tstart}/{tend} {safe_int(curr.WD):03d}{safe_int(curr.WS):02d}KT {safe_int(curr.VIS or 9999):04d} {tcc_to_cloud_label(curr.CC)}")
            signif_times.append(curr["time"])

        precip_flag = (curr.CC and curr.CC >= 80 and curr.RH and curr.RH >= 85)
        if precip_flag:
            tempo.append(f"TEMPO {tstart}/{tend} 4000 -RA SCT020CB")
            signif_times.append(curr["time"])

    if becmg: taf_lines += becmg
    if tempo: taf_lines += tempo
    if not becmg and not tempo:
        taf_lines.append("NOSIG")

    source_marker = "METAR+MODEL FUSION" if metar else "MODEL FUSION"
    taf_lines.append(f"RMK AUTO FUSION BASED ON {source_marker}")
    return taf_lines, sorted(list(set(signif_times)))

# -----------------------
# Export
# -----------------------
def export_results(df_fused, df_probs, taf_lines, issue_dt):
    stamp = issue_dt.strftime("%Y%m%d_%H%M")
    out_json = {
        "issued_at": issue_dt.isoformat(),
        "taf_lines": taf_lines,
        "fused": df_fused.to_dict(orient="records"),
        "probabilities": df_probs.to_dict(orient="records")
    }
    fname_json = f"output/fused_{stamp}.json"
    fname_csv = f"output/fused_{stamp}.csv"
    df_fused.to_csv(fname_csv, index=False)
    with open(fname_json, "w", encoding="utf-8") as f:
        json.dump(out_json, f, ensure_ascii=False, indent=2, default=str)
    return fname_json, fname_csv

# -----------------------
# MAIN ACTION
# -----------------------
if st.button("🚀 Generate Operational TAFOR (Fusion)"):
    issue_dt = datetime.combine(issue_date, datetime.utcnow().replace(hour=issue_time, minute=0, second=0).time())
    st.info("📡 Fetching BMKG / Open-Meteo / METAR ... (please wait)")

    bmkg = fetch_bmkg()
    gfs_json = fetch_openmeteo("gfs")
    ecmwf_json = fetch_openmeteo("ecmwf")
    icon_json = fetch_openmeteo("icon")
    metar = fetch_metar_ogimet("WARR")

    st.success("✅ Data fetched (or fallback used). Processing fusion...")

    df_gfs = openmeteo_json_to_df(gfs_json, "GFS")
    df_ecmwf = openmeteo_json_to_df(ecmwf_json, "ECMWF")
    df_icon = openmeteo_json_to_df(icon_json, "ICON")
    df_bmkg = bmkg_cuaca_to_df(bmkg["cuaca"]) if bmkg.get("status") == "OK" else None

    df_merged = align_hourly([df_gfs, df_ecmwf, df_icon, df_bmkg])
    if df_merged is None:
        st.error("No model data available to fuse.")
        st.stop()

    df_fused = fuse_ensemble(df_merged, weights, hours=validity)
    if df_fused is None or df_fused.empty:
        st.error("Fusion failed / empty result.")
        st.stop()

    df_probs = compute_probabilities(df_merged)

    taf_lines, signif_times = build_taf_from_fused(df_fused, df_merged, metar, issue_dt, validity)
    taf_html = "<br>".join(taf_lines)

    json_file, csv_file = export_results(df_fused, df_probs, taf_lines, issue_dt)

    # -----------------------
    # DISPLAY
    # -----------------------
    st.subheader("📊 Source summary")
    st.write({
        "BMKG ADM4 (Sedati Gede 35.15.17.2011)": "OK" if bmkg.get("status") == "OK" else "Unavailable",
        "GFS": "OK" if gfs_json else "Unavailable",
        "ECMWF": "OK" if ecmwf_json else "Unavailable",
        "ICON": "OK" if icon_json else "Unavailable",
        "METAR (OGIMET/NOAA)": "OK" if metar else "Unavailable"
    })

    st.markdown("### 📡 METAR (Realtime OGIMET/NOAA)")
    st.code(metar or "Not available")

    st.markdown("### 📝 Generated TAFOR (Operational)")
    st.markdown(f"<pre>{taf_html}</pre>", unsafe_allow_html=True)
    valid_to = issue_dt + timedelta(hours=validity)
    st.caption(f"Issued at {issue_dt:%d%H%MZ}, Valid {issue_dt:%d/%H}–{valid_to:%d/%H} UTC")

    # Plot
    st.markdown("### 📈 Fused 24h (T/RH/Cloud/WS) & Significant changes")
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(df_fused["time"], df_fused["T"], label="T (°C)")
    ax.plot(df_fused["time"], df_fused["RH"], label="RH (%)")
    ax.plot(df_fused["time"], df_fused["CC"], label="Cloud (%)")
    ax.plot(df_fused["time"], df_fused["WS"], label="Wind (kt)")
    for t in signif_times:
        ax.axvline(t, color="orange", linestyle="--", alpha=0.6)
    ax.legend(); ax.grid(True, linestyle="--", alpha=0.4)
    plt.xticks(rotation=35)
    st.pyplot(fig)

    st.markdown("### 🔢 Probabilistic Metrics (sample)")
    st.dataframe(df_probs.head(24))

    # -----------------------
    # DOWNLOAD SECTION (Styled)
    # -----------------------
    st.markdown("### 💾 Exported Files")
    st.caption("File hasil fusi otomatis tersimpan di folder `output/`. Pilih tombol untuk mengunduh (CSV / JSON / ZIP).")

    # Read file content bytes
    with open(json_file, "r", encoding="utf-8") as f:
        json_data = f.read()
    with open(csv_file, "r", encoding="utf-8") as f:
        csv_data = f.read()

    # prepare ZIP in-memory
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(os.path.basename(csv_file), csv_data)
        zf.writestr(os.path.basename(json_file), json_data)
    zip_buffer.seek(0)
    zip_bytes = zip_buffer.read()

    # file size info
    csv_size = os.path.getsize(csv_file) if os.path.exists(csv_file) else 0
    json_size = os.path.getsize(json_file) if os.path.exists(json_file) else 0
    zip_size = len(zip_bytes)

    st.markdown(
        """
        <style>
        div[data-testid="stDownloadButton"] > button {
            border-radius: 10px;
            height: 3em;
            padding: 0 1.2em;
            font-weight: 600;
            color: white;
            transition: all 0.12s ease-in-out;
            box-shadow: 0 2px 6px rgba(0,0,0,0.18);
        }
        /* first button (CSV) */
        div[data-testid="stDownloadButton"]:nth-child(1) > button {
            background: linear-gradient(90deg, #0066cc, #0099ff);
        }
        /* second button (JSON) */
        div[data-testid="stDownloadButton"]:nth-child(2) > button {
            background: linear-gradient(90deg, #28a745, #6ddf7a);
        }
        /* third button (ZIP) */
        div[data-testid="stDownloadButton"]:nth-child(3) > button {
            background: linear-gradient(90deg, #6f42c1, #a78bfa);
        }
        div[data-testid="stDownloadButton"] > button:hover {
            transform: scale(1.02);
            box-shadow: 0 6px 14px rgba(0,0,0,0.2);
        }
        </style>
        """, unsafe_allow_html=True
    )

    col_csv, col_json, col_zip = st.columns([1,1,1])
    with col_csv:
        st.download_button(
            label=f"⬇️ CSV ({fmt_bytes(csv_size)})",
            data=csv_data,
            file_name=os.path.basename(csv_file),
            mime="text/csv"
        )
    with col_json:
        st.download_button(
            label=f"📦 JSON ({fmt_bytes(json_size)})",
            data=json_data,
            file_name=os.path.basename(json_file),
            mime="application/json"
        )
    with col_zip:
        st.download_button(
            label=f"🗜️ ZIP (CSV+JSON) ({fmt_bytes(zip_size)})",
            data=zip_bytes,
            file_name=f"fused_{issue_dt.strftime('%Y%m%d_%H%M')}.zip",
            mime="application/zip"
        )

    st.success("✅ File berhasil diekspor dan siap diunduh.")

    # -----------------------
    # LOGGING + ALERTS (NEW)
    # -----------------------
    # prepare log entry
    log_file = f"logs/{issue_dt:%Y%m%d}_tafor_log.csv"
    taf_text = " | ".join(taf_lines)
    pop_max = round((df_probs["PoP_precip"].max() if not df_probs.empty else 0.0) * 100, 1)
    wind_max = round(df_fused["WS"].max() if not df_fused.empty else 0.0, 1)
    rh_max = round(df_fused["RH"].max() if not df_fused.empty else 0.0, 1)
    cc_max = round(df_fused["CC"].max() if not df_fused.empty else 0.0, 1)

    alerts = []
    if pop_max >= 70:
        alerts.append(f"⚠️ High PoP ({pop_max}%) — possible heavy RA/TS")
    if wind_max >= 20:
        alerts.append(f"💨 High wind: {wind_max} kt")
    if (rh_max >= 90) and (cc_max >= 85):
        alerts.append("🌫️ High RH & cloud cover — possible low visibility / convective cloud")

    log_df = pd.DataFrame([{
        "timestamp": datetime.utcnow().isoformat(),
        "issue_time": f"{issue_dt:%d%H%MZ}",
        "validity": validity,
        "metar": metar or "",
        "taf_text": taf_text,
        "pop_max_pct": pop_max,
        "wind_max_kt": wind_max,
        "rh_max_pct": rh_max,
        "cc_max_pct": cc_max,
        "alerts": "; ".join(alerts)
    }])

    if os.path.exists(log_file):
        log_df.to_csv(log_file, mode="a", header=False, index=False)
    else:
        log_df.to_csv(log_file, index=False)

    # show alerts
    if alerts:
        for a in alerts:
            st.warning(a)
    else:
        st.info("✅ No significant alerts detected — conditions stable.")

    with st.expander("🔍 Debug: raw BMKG JSON"):
        st.write(bmkg.get("raw"))

    st.success("✅ Operational TAFOR (fusion) created, exported, and logged. PLEASE VALIDATE before operational release.")
