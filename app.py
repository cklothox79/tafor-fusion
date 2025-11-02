# app_pro_fusion.py
"""
Auto TAFOR Fusion Pro (BMKG + GFS + ECMWF + ICON)
- BMKG ADM4 Sedati Gede (35.15.17.2001)
- Open-Meteo GFS / ECMWF / ICON (global models)
- Output: fused 24h forecast + TAF-like (ICAO & Perka-aware)
"""

import streamlit as st
from datetime import datetime, timedelta
import requests, json, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import radians, cos, sin, atan2, degrees

# === Streamlit setup ===
st.set_page_config(page_title="🛫 TAFOR Fusion Pro — Sedati Gede", layout="centered")
st.title("🛫 TAFOR Fusion Pro — Sedati Gede (WARR vicinity)")
st.caption("Fusi: BMKG (ADM4) + Open-Meteo models (GFS, ECMWF, ICON). Output: TAF-like (ICAO + Perka-aware)")
st.divider()

LAT, LON = -7.379, 112.787
REFRESH_TTL = 900
DEFAULT_WEIGHTS = {"bmkg": 0.45, "ecmwf": 0.25, "icon": 0.15, "gfs": 0.15}

# === Input UI ===
col1, col2, col3 = st.columns(3)
with col1:
    issue_date = st.date_input("📅 Issue date (UTC)", datetime.utcnow().date())
with col2:
    jam_penting = [0, 3, 6, 9, 12, 15, 18, 21]
    jam_sekarang = datetime.utcnow().hour
    default_jam = min(jam_penting, key=lambda j: abs(j - jam_sekarang))
    issue_time = st.selectbox("🕓 Issue time (UTC)", jam_penting, index=jam_penting.index(default_jam))
with col3:
    validity = st.number_input("🕐 Validity (hours)", min_value=6, max_value=36, value=24, step=6)

# === Bobot model ===
st.markdown("**⚙️ Ensemble weights (BMKG prioritas)** — ubah jika perlu")
cols = st.columns(4)
weights = {}
weights["bmkg"] = cols[0].number_input("BMKG", 0.0, 1.0, 0.45, step=0.05)
weights["ecmwf"] = cols[1].number_input("ECMWF", 0.0, 1.0, 0.25, step=0.05)
weights["icon"] = cols[2].number_input("ICON", 0.0, 1.0, 0.15, step=0.05)
weights["gfs"] = cols[3].number_input("GFS", 0.0, 1.0, 0.15, step=0.05)

sumw = sum(weights.values()) or 1
norm_weights = {k: v / sumw for k, v in weights.items()}
st.caption(f"Normalized weights: {', '.join([f'{k}={v:.2f}' for k,v in norm_weights.items()])}")

# === Wind helpers ===
def wind_to_uv(speed, deg):
    if pd.isna(speed) or pd.isna(deg): return None, None
    theta = radians((270 - deg) % 360)
    return speed * cos(theta), speed * sin(theta)

def uv_to_wind(u, v):
    spd = np.sqrt(u**2 + v**2)
    theta = degrees(atan2(v, u))
    deg = (270 - theta) % 360
    return spd, deg

def weighted_mean(vals, ws):
    arr = np.array([np.nan if v is None else v for v in vals], float)
    w = np.array(ws, float)
    mask = ~np.isnan(arr)
    return float((arr[mask] * w[mask]).sum() / w[mask].sum()) if mask.sum() else np.nan

# === Fetch BMKG ===
@st.cache_data(ttl=REFRESH_TTL)
def fetch_bmkg(adm4="35.15.17.2001", local_fallback="JSON_BMKG.txt"):
    url = "https://cuaca.bmkg.go.id/api/df/v1/forecast/adm"
    params = {"adm1": "35", "adm2": "35.15", "adm3": "35.15.17", "adm4": adm4}
    try:
        r = requests.get(url, params=params, timeout=15, verify=False)
        r.raise_for_status()
        data = r.json()
    except Exception:
        if os.path.exists(local_fallback):
            with open(local_fallback, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            return {"status": "Unavailable"}
    try:
        cuaca = data["data"][0]["cuaca"][0][0]
        return {"status": "OK", "raw": data, "cuaca": cuaca}
    except Exception:
        return {"status": "Unavailable", "raw": data}

# === BMKG Parser (versi fleksibel) ===
def bmkg_cuaca_to_df(cuaca_list):
    times, t, hu, tcc, ws, wd, vs = [], [], [], [], [], [], []
    for c in cuaca_list:
        # cari field waktu yang valid
        dt_val = (
            c.get("datetime") or c.get("date") or c.get("time")
            or c.get("valid_time") or c.get("jamCuaca")
        )
        if isinstance(dt_val, (int, float)):
            try:
                dt_val = datetime.utcfromtimestamp(dt_val)
            except Exception:
                dt_val = None
        elif isinstance(dt_val, str):
            dt_val = dt_val.replace("Z", "+00:00").replace("T", " ")
            try:
                dt_val = pd.to_datetime(dt_val, utc=True)
            except Exception:
                dt_val = None
        else:
            dt_val = None
        if dt_val is None:
            continue

        # ambil nilai umum
        times.append(dt_val)
        t.append(c.get("t") or c.get("temp") or c.get("temperature"))
        hu.append(c.get("hu") or c.get("rh") or c.get("humidity"))
        tcc.append(c.get("tcc") or c.get("cloud") or c.get("cloud_cover"))
        ws.append(c.get("ws") or c.get("wind_speed"))
        wd.append(c.get("wd_deg") or c.get("wind_dir") or c.get("wind_direction"))
        vs.append(c.get("vs_text") or c.get("visibility") or "")

    if not times:
        return pd.DataFrame(columns=["time", "T_BMKG", "RH_BMKG", "CC_BMKG", "WS_BMKG", "WD_BMKG", "VIS_BMKG"])

    df = pd.DataFrame({
        "time": times, "T_BMKG": t, "RH_BMKG": hu, "CC_BMKG": tcc,
        "WS_BMKG": ws, "WD_BMKG": wd, "VIS_BMKG": vs
    })
    return df.sort_values("time").reset_index(drop=True)

# === Fetch OpenMeteo models ===
def fetch_openmeteo_model(model):
    base = f"https://api.open-meteo.com/v1/{model}"
    params = {
        "latitude": LAT, "longitude": LON,
        "hourly": "temperature_2m,relative_humidity_2m,cloud_cover,windspeed_10m,winddirection_10m,visibility",
        "forecast_days": 2, "timezone": "UTC"
    }
    try:
        r = requests.get(base, params=params, timeout=15)
        return r.json()
    except Exception:
        return None

def om_to_df(j, tag):
    if not j or "hourly" not in j: return None
    h = j["hourly"]
    df = pd.DataFrame({
        "time": pd.to_datetime(h["time"]),
        f"T_{tag}": h["temperature_2m"],
        f"RH_{tag}": h["relative_humidity_2m"],
        f"CC_{tag}": h["cloud_cover"],
        f"WS_{tag}": h["windspeed_10m"],
        f"WD_{tag}": h["winddirection_10m"],
        f"VIS_{tag}": h.get("visibility", [None]*len(h["time"]))
    })
    return df

# === Fusion logic ===
def build_fused_hourly(bmkg_obj, om_gfs, om_ecmwf, om_icon, hours=24, weights_norm=None):
    dfs = [d for d in [om_to_df(om_gfs, "GFS"), om_to_df(om_ecmwf, "ECMWF"), om_to_df(om_icon, "ICON")] if d is not None]
    if not dfs:
        return None
    df = dfs[0][["time"]]
    for d in dfs: df = pd.merge(df, d, on="time", how="outer")
    df = df.sort_values("time").reset_index(drop=True)

    if bmkg_obj and bmkg_obj.get("status") == "OK":
        df_b = bmkg_cuaca_to_df(bmkg_obj["cuaca"])
        df = pd.merge(df, df_b, on="time", how="outer")

    rows = []
    for _, r in df.iterrows():
        T_vals, RH_vals, CC_vals, VIS_vals, U_vals, V_vals, w = [], [], [], [], [], [], []
        for key in ["bmkg", "ecmwf", "icon", "gfs"]:
            wt = weights_norm.get(key, 0)
            tag = key.upper()
            t = r.get(f"T_{tag}") if tag != "BMKG" else r.get("T_BMKG")
            rh = r.get(f"RH_{tag}") if tag != "BMKG" else r.get("RH_BMKG")
            cc = r.get(f"CC_{tag}") if tag != "BMKG" else r.get("CC_BMKG")
            vis = r.get(f"VIS_{tag}") if tag != "BMKG" else r.get("VIS_BMKG")
            ws = r.get(f"WS_{tag}") if tag != "BMKG" else r.get("WS_BMKG")
            wd = r.get(f"WD_{tag}") if tag != "BMKG" else r.get("WD_BMKG")
            if t is not None: T_vals.append(t)
            if rh is not None: RH_vals.append(rh)
            if cc is not None: CC_vals.append(cc)
            if vis is not None:
                try: VIS_vals.append(float(vis))
                except: VIS_vals.append(None)
            if ws and wd:
                u, v = wind_to_uv(ws, wd)
                if u and v:
                    U_vals.append(u); V_vals.append(v)
            w.append(wt)
        if not w: continue
        T_f = weighted_mean(T_vals, w)
        RH_f = weighted_mean(RH_vals, w)
        CC_f = weighted_mean(CC_vals, w)
        VIS_f = weighted_mean(VIS_vals, w)
        U_f = weighted_mean(U_vals, w)
        V_f = weighted_mean(V_vals, w)
        WS_f, WD_f = uv_to_wind(U_f, V_f)
        rows.append({"time": r["time"], "T": T_f, "RH": RH_f, "CC": CC_f, "VIS": VIS_f, "WS": WS_f, "WD": WD_f})

    df_fused = pd.DataFrame(rows).dropna(subset=["time"])
    now = pd.to_datetime(datetime.utcnow()).floor("H")
    return df_fused[df_fused["time"] >= now].head(hours)

# === TAFOR builder ===
def tcc_to_cloud(cc):
    if cc is None: return "FEW020"
    cc = float(cc)
    if cc < 25: return "FEW020"
    elif cc < 50: return "SCT025"
    elif cc < 85: return "BKN030"
    else: return "OVC030"

def build_tafor(df, issue_dt, validity=24):
    if df is None or df.empty:
        return ["TAF WARR " + issue_dt.strftime("%d%H%MZ") + f" {issue_dt:%d%H}/{(issue_dt+timedelta(hours=validity)):%d%H}",
                "9999 FEW020", "NOSIG"]
    header = f"TAF WARR {issue_dt:%d%H%MZ} {issue_dt:%d%H}/{(issue_dt+timedelta(hours=validity)):%d%H}"
    first = df.iloc[0]
    wind = f"{int(first.WD):03d}{int(round(first.WS or 5)):02d}KT"
    vis = str(int(first.VIS or 9999))
    cloud = tcc_to_cloud(first.CC)
    taf = [header, f"{wind} {vis} {cloud}"]
    df["precip"] = (df["CC"] > 80) & (df["RH"] > 85)
    for i, row in df.iterrows():
        if row.precip:
            taf.append(f"TEMPO {row.time:%d%H}/{(row.time+timedelta(hours=2)):%d%H} 4000 -RA SCT020CB")
            break
    if len(taf) == 2: taf.append("NOSIG")
    return taf

# === Main Action ===
if st.button("🚀 Generate TAFOR (Fusion)"):
    issue_dt = datetime.combine(issue_date, datetime.utcnow().replace(hour=issue_time, minute=0, second=0).time())
    valid_to = issue_dt + timedelta(hours=validity)
    st.info("🔎 Mengambil data model...")

    bmkg_obj = fetch_bmkg()
    om_gfs, om_ecmwf, om_icon = fetch_openmeteo_model("gfs"), fetch_openmeteo_model("ecmwf"), fetch_openmeteo_model("icon")

    st.success("✅ Data diterima, fusi sedang diproses...")
    df_fused = build_fused_hourly(bmkg_obj, om_gfs, om_ecmwf, om_icon, validity, norm_weights)
    taf = build_tafor(df_fused, issue_dt, validity)
    taf_html = "<br>".join(taf)

    st.subheader("📊 Ringkasan Sumber")
    st.write(f"| Sumber | Status |\n|:--|:--|\n| BMKG ADM4 | {'OK' if bmkg_obj and bmkg_obj.get('status')=='OK' else 'Unavailable'} |\n| GFS | {'OK' if om_gfs else 'Unavailable'} |\n| ECMWF | {'OK' if om_ecmwf else 'Unavailable'} |\n| ICON | {'OK' if om_icon else 'Unavailable'} |")

    st.markdown("### 📝 Hasil TAFOR (Auto Fusion)")
    st.markdown(f"<div style='padding:15px;border:2px solid #555;border-radius:10px;background:#f9f9f9'><p style='font-family:monospace;font-weight:700'>{taf_html}</p></div>", unsafe_allow_html=True)

    if df_fused is not None and not df_fused.empty:
        fig, ax = plt.subplots(figsize=(9,4))
        ax.plot(df_fused["time"], df_fused["T"], label="T (°C)")
        ax.plot(df_fused["time"], df_fused["RH"], label="RH (%)")
        ax.plot(df_fused["time"], df_fused["CC"], label="Cloud (%)")
        ax.plot(df_fused["time"], df_fused["WS"], label="Wind Speed (kt)")
        ax.legend(); plt.xticks(rotation=35)
        st.pyplot(fig)

    with st.expander("🔍 Debug: JSON BMKG Mentah"):
        st.write(bmkg_obj)
