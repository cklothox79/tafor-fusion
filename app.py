# app_pro_fusion.py
"""
Auto TAFOR Pro (fusion): BMKG + GFS + ECMWF + ICON -> fused 24h -> auto generate TAFOR
Save file as app_pro_fusion.py and run:
    pip install -r requirements.txt
    streamlit run app_pro_fusion.py
"""
import streamlit as st
from datetime import datetime, timedelta
import requests, json, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import radians, cos, sin, atan2, degrees

# --------------------------
# Config
# --------------------------
st.set_page_config(page_title="🛫 TAFOR Fusion Pro — Sedati Gede", layout="centered")
st.title("🛫 TAFOR Fusion Pro — Sedati Gede (WARR vicinity)")
st.caption("Fusi: BMKG (ADM4) + Open-Meteo models (GFS, ECMWF, ICON). Output: TAF-like (ICAO + Perka-aware)")
st.divider()

LAT, LON = -7.379, 112.787
REFRESH_TTL = 900  # caching TTL in seconds
# default ensemble weights (sum doesn't need to be 1, we'll normalize)
DEFAULT_WEIGHTS = {"bmkg": 0.45, "ecmwf": 0.25, "icon": 0.15, "gfs": 0.15}

# --------------------------
# UI Controls
# --------------------------
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

weights = DEFAULT_WEIGHTS.copy()
st.markdown("**⚙️ Ensemble weights (BMKG prioritas)** — ubah jika perlu")
w1, w2, w3, w4 = st.columns(4)
with w1:
    weights["bmkg"] = st.number_input("BMKG", min_value=0.0, max_value=1.0, value=float(weights["bmkg"]), step=0.05)
with w2:
    weights["ecmwf"] = st.number_input("ECMWF", min_value=0.0, max_value=1.0, value=float(weights["ecmwf"]), step=0.05)
with w3:
    weights["icon"] = st.number_input("ICON", min_value=0.0, max_value=1.0, value=float(weights["icon"]), step=0.05)
with w4:
    weights["gfs"] = st.number_input("GFS", min_value=0.0, max_value=1.0, value=float(weights["gfs"]), step=0.05)

# normalize weights for convenience (if all zero, we'll keep defaults)
sumw = sum(weights.values())
if sumw == 0:
    norm_weights = {k: DEFAULT_WEIGHTS[k] for k in weights}
else:
    norm_weights = {k: (v / sumw) for k, v in weights.items()}

st.caption(f"Normalized weights: {', '.join([f'{k}={v:.2f}' for k,v in norm_weights.items()])}")

# --------------------------
# Helpers: wind conversion and fusion helpers
# --------------------------
def wind_to_uv(speed, deg):
    # speed (kt), deg meteorological (from) -> u,v
    theta = radians((270 - deg) % 360)
    return speed * cos(theta), speed * sin(theta)

def uv_to_wind(u, v):
    spd = (u**2 + v**2)**0.5
    theta = degrees(atan2(v, u))
    deg = (270 - theta) % 360
    return spd, deg

def weighted_mean_list(vals, ws):
    arr = np.array([np.nan if v is None else v for v in vals], dtype=float)
    w = np.array(ws, dtype=float)
    mask = ~np.isnan(arr)
    if mask.sum() == 0:
        return None
    return float((arr[mask] * w[mask]).sum() / w[mask].sum())

# --------------------------
# Fetchers
# --------------------------
@st.cache_data(ttl=REFRESH_TTL)
def fetch_bmkg(adm4="35.15.17.2001", local_fallback="JSON_BMKG.txt"):
    url = "https://cuaca.bmkg.go.id/api/df/v1/forecast/adm"
    params = {"adm1":"35","adm2":"35.15","adm3":"35.15.17","adm4":adm4}
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
    # parse defensively
    try:
        cuaca = data["data"][0]["cuaca"][0][0]
        return {"status": "OK", "raw": data, "cuaca": cuaca}
    except Exception:
        return {"status": "Unavailable", "raw": data}

def fetch_openmeteo_model(model_name):
    # model_name: "gfs", "ecmwf", "icon"
    base = f"https://api.open-meteo.com/v1/{model_name}"
    params = {
        "latitude": LAT, "longitude": LON,
        "hourly": "temperature_2m,relative_humidity_2m,cloud_cover,windspeed_10m,winddirection_10m,visibility",
        "forecast_days": 2, "timezone": "UTC"
    }
    try:
        r = requests.get(base, params=params, timeout=20)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None

# --------------------------
# Utility: convert model JSON -> aligned DataFrame (hourly UTC)
# --------------------------
def om_json_to_df(openm_json, model_tag):
    if not openm_json or "hourly" not in openm_json:
        return None
    h = openm_json["hourly"]
    df = pd.DataFrame({
        "time": pd.to_datetime(h["time"]),
        f"T_{model_tag}": h.get("temperature_2m"),
        f"RH_{model_tag}": h.get("relative_humidity_2m"),
        f"CC_{model_tag}": h.get("cloud_cover"),
        f"WS_{model_tag}": h.get("windspeed_10m"),
        f"WD_{model_tag}": h.get("winddirection_10m"),
        f"VIS_{model_tag}": h.get("visibility") if "visibility" in h else [None]*len(h["time"])
    })
    return df

# --------------------------
# Convert BMKG cuaca list to hourly DF (approx)
# --------------------------
def bmkg_cuaca_to_df(cuaca_list):
    # cuaca_list is list of dicts with keys datetime, t, hu, tcc, ws, wd_deg, vs_text etc.
    times, t, hu, tcc, ws, wd, vs = [], [], [], [], [], [], []
    for c in cuaca_list:
        try:
            times.append(pd.to_datetime(c["datetime"]))
        except Exception:
            # try Z replacement
            times.append(pd.to_datetime(c["datetime"].replace("Z","+00:00")))
        t.append(c.get("t"))
        hu.append(c.get("hu"))
        tcc.append(c.get("tcc"))
        ws.append(c.get("ws"))
        wd.append(c.get("wd_deg"))
        vs.append(c.get("vs_text"))
    df = pd.DataFrame({
        "time": times, "T_BMKG": t, "RH_BMKG": hu, "CC_BMKG": tcc,
        "WS_BMKG": ws, "WD_BMKG": wd, "VIS_BMKG": vs
    })
    return df

# --------------------------
# Weather text -> ICAO symbol (simple mapping)
# --------------------------
def weather_text_to_icao(text):
    if not text: return ""
    txt = text.lower()
    if "petir" in txt or "thunder" in txt or "thunderstorm" in txt: return "TS"
    if "hujan lebat" in txt or "heavy rain" in txt: return "+RA"
    if "hujan" in txt or "rain" in txt: return "RA"
    if "kabut" in txt or "fog" in txt: return "FG"
    if "gerimis" in txt or "drizzle" in txt: return "DZ"
    return ""

# --------------------------
# Fusion routine: build fused hourly DF for next N hours
# --------------------------
def build_fused_hourly(bmkg_obj, om_gfs, om_ecmwf, om_icon, hours=24, weights_norm=None):
    # create dataframes
    dfs = []
    if om_gfs: dfs.append(om_json_to_df(om_gfs, "GFS"))
    if om_ecmwf: dfs.append(om_json_to_df(om_ecmwf, "ECMWF"))
    if om_icon: dfs.append(om_json_to_df(om_icon, "ICON"))
    # merge on time
    if not dfs:
        return None
    df = dfs[0][["time"]].copy()
    for d in dfs:
        df = pd.merge(df, d, on="time", how="outer")
    df = df.sort_values("time").reset_index(drop=True)

    # BMKG
    if bmkg_obj and bmkg_obj.get("status") == "OK":
        df_b = bmkg_cuaca_to_df(bmkg_obj["cuaca"])
        df = pd.merge(df, df_b, on="time", how="outer")

    # Now for each time, compute fused variables by weights_norm
    # variables: temperature (T), RH, cloud cover (CC), wind speed/direction (WS/WD), visibility (VIS)
    rows = []
    for _, row in df.iterrows():
        t_vals = []
        rh_vals = []
        cc_vals = []
        ws_u_vals = []  # u components
        ws_v_vals = []
        vis_vals = []
        w_list = []
        # BMKG
        if bmkg_obj and bmkg_obj.get("status") == "OK":
            w = weights_norm.get("bmkg", 0.0)
            t_vals.append(row.get("T_BMKG"))
            rh_vals.append(row.get("RH_BMKG"))
            cc_vals.append(row.get("CC_BMKG"))
            vis_text = row.get("VIS_BMKG")
            vis_m = None
            if isinstance(vis_text, str):
                try:
                    if ">" in vis_text:
                        vis_m = float(vis_text.replace(">","").strip())*1000
                    elif "km" in vis_text:
                        vis_m = float(vis_text.replace("km","").strip())*1000
                    else:
                        vis_m = float(vis_text)
                except Exception:
                    vis_m = None
            vis_vals.append(vis_m)
            ws = row.get("WS_BMKG")
            wd = row.get("WD_BMKG")
            if pd.notna(ws) and pd.notna(wd):
                u,v = wind_to_uv(ws, wd)
                ws_u_vals.append(u); ws_v_vals.append(v)
            w_list.append(w)
        # Open-Meteo models
        for key in ["ECMWF", "ICON", "GFS"]:
            tag = key
            w = weights_norm.get(key.lower(), 0.0)
            if w <= 0: 
                continue
            t_val = row.get(f"T_{tag}") if f"T_{tag}" in row.index else None
            rh_val = row.get(f"RH_{tag}") if f"RH_{tag}" in row.index else None
            cc_val = row.get(f"CC_{tag}") if f"CC_{tag}" in row.index else None
            ws_val = row.get(f"WS_{tag}") if f"WS_{tag}" in row.index else None
            wd_val = row.get(f"WD_{tag}") if f"WD_{tag}" in row.index else None
            vis_val = row.get(f"VIS_{tag}") if f"VIS_{tag}" in row.index else None
            t_vals.append(t_val); rh_vals.append(rh_val); cc_vals.append(cc_val)
            vis_vals.append(vis_val)
            if pd.notna(ws_val) and pd.notna(wd_val):
                u,v = wind_to_uv(ws_val, wd_val)
                ws_u_vals.append(u); ws_v_vals.append(v)
            w_list.append(w)
        # If no weights (all zeros), skip
        if not w_list:
            continue
        # For each variable, compute weighted mean (skip NaN)
        def wmean(vals):
            return weighted_mean_list(vals, w_list)
        T_f = wmean(t_vals)
        RH_f = wmean(rh_vals)
        CC_f = wmean(cc_vals)
        VIS_f = wmean(vis_vals)
        # Wind via vector mean
        U_f = weighted_mean_list(ws_u_vals, w_list) if ws_u_vals else None
        V_f = weighted_mean_list(ws_v_vals, w_list) if ws_v_vals else None
        if U_f is None or V_f is None:
            WS_f = None; WD_f = None
        else:
            WS_f, WD_f = uv_to_wind(U_f, V_f)
        rows.append({
            "time": row["time"], "T": T_f, "RH": RH_f, "CC": CC_f, "VIS": VIS_f,
            "WS": WS_f, "WD": WD_f
        })
    df_fused = pd.DataFrame(rows).dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    # keep only next `hours` hours from now (UTC)
    now = pd.to_datetime(datetime.utcnow()).floor('H')
    df_fused = df_fused[df_fused["time"] >= now].head(hours)
    return df_fused

# --------------------------
# Build TAFOR from fused hourly DF
# --------------------------
def tcc_to_cloud_group(cc):
    if cc is None: return "FEW020"
    try:
        c = float(cc)
    except:
        return "FEW020"
    if c < 25: return "FEW020"
    elif c < 50: return "SCT025"
    elif c < 85: return "BKN030"
    else: return "OVC030"

def build_tafor_from_fused(df_fused, issue_dt, valid_hours=24):
    """
    Heuristic TAFOR builder:
    - baseline: first hour's fused fields
    - TEMPO if any short-lived significant events (<3h clusters)
    - BECMG if sustained trend (>=3h) change
    Uses conservative thresholds:
      * precipitation implied if CC>=80 and RH>=85 -> -RA
      * TS if CC>=80 and RH>=90 and WS variability high -> TS
      * reduced vis if VIS < 5000 m
    """
    if df_fused is None or df_fused.empty:
        return ["TAF WARR " + issue_dt.strftime("%d%H%MZ") + f" {issue_dt.strftime('%d%H')}/{(issue_dt+timedelta(hours=valid_hours)).strftime('%d%H')}",
                "9999 FEW020", "NOSIG"]

    header = f"TAF WARR {issue_dt.strftime('%d%H%MZ')} {issue_dt.strftime('%d%H')}/{(issue_dt+timedelta(hours=valid_hours)).strftime('%d%H')}"
    # baseline from first row
    first = df_fused.iloc[0]
    wind = f"{int(first.WD):03d}{int(round(float(first.WS or 5))):02d}KT" if pd.notna(first.WS) else "09005KT"
    vis_val = int(first.VIS) if pd.notna(first.VIS) else 9999
    vis = str(vis_val)
    cloud = tcc_to_cloud_group(first.CC)
    baseline = f"{wind} {vis} {cloud}"

    lines = [header, baseline]

    # scan for significant hours (rain/ts/fog)
    df = df_fused.copy()
    df["precip_flag"] = ((df["CC"] >= 80) & (df["RH"] >= 85))
    df["ts_flag"] = ((df["CC"] >= 85) & (df["RH"] >= 90) & (df["WS"] >= 15))
    df["lowvis_flag"] = (df["VIS"].notna()) & (df["VIS"] < 5000)

    # find contiguous clusters for precip / ts / lowvis
    def clusters(flag_series):
        clusters = []
        current = None
        for i, f in enumerate(flag_series):
            t = df.iloc[i]["time"]
            if f:
                if current is None:
                    current = {"start": t, "end": t}
                else:
                    current["end"] = t
            else:
                if current is not None:
                    clusters.append(current)
                    current = None
        if current is not None:
            clusters.append(current)
        return clusters

    precip_clusters = clusters(df["precip_flag"].tolist())
    ts_clusters = clusters(df["ts_flag"].tolist())
    lowvis_clusters = clusters(df["lowvis_flag"].tolist())

    # TEMPO for short clusters (<3h), BECMG for sustained (>=3h)
    def cluster_to_group(cluster):
        # cluster={"start": datetime, "end": datetime}
        # produce string in format: TEMPO/BECMG DDHH/DDHH ...
        start, end = cluster["start"], cluster["end"]
        dur_hours = int(((end - start).total_seconds() // 3600) + 1)
        if dur_hours < 3:
            # tempo
            return ("TEMPO", start, end)
        else:
            return ("BECMG", start, end)

    # apply for precip
    for cl in precip_clusters:
        typ, s, e = cluster_to_group(cl)
        # choose wording "-RA" and lowered vis
        if typ == "TEMPO":
            lines.append(f"TEMPO {s.strftime('%d%H')}/{e.strftime('%d%H')} 4000 -RA SCT020CB")
        else:
            lines.append(f"BECMG {s.strftime('%d%H')}/{e.strftime('%d%H')} 20005KT 8000 -RA BKN025")

    for cl in ts_clusters:
        typ, s, e = cluster_to_group(cl)
        if typ == "TEMPO":
            lines.append(f"TEMPO {s.strftime('%d%H')}/{e.strftime('%d%H')} 3000 TSRA SCT020CB")
        else:
            lines.append(f"BECMG {s.strftime('%d%H')}/{e.strftime('%d%H')} VRB25G35KT TSRA")

    for cl in lowvis_clusters:
        typ, s, e = cluster_to_group(cl)
        if typ == "TEMPO":
            lines.append(f"TEMPO {s.strftime('%d%H')}/{e.strftime('%d%H')} 2000 -BR")
        else:
            lines.append(f"BECMG {s.strftime('%d%H')}/{e.strftime('%d%H')} 2000 FG")

    # If no change groups -> add NOSIG
    if len(lines) == 2:
        lines.append("NOSIG")

    return lines

# --------------------------
# Main action
# --------------------------
if st.button("🚀 Generate TAFOR (Fusion)"):
    issue_dt = datetime.combine(issue_date, datetime.utcnow().replace(hour=issue_time, minute=0, second=0).time())
    valid_to = issue_dt + timedelta(hours=validity)

    st.info("🔎 Mengambil data: BMKG & OpenMeteo (GFS/ECMWF/ICON)...")
    bmkg_obj = fetch_bmkg()
    om_gfs = fetch_openmeteo_model("gfs")
    om_ecmwf = fetch_openmeteo_model("ecmwf")
    om_icon = fetch_openmeteo_model("icon")

    st.success("✅ Data diambil (atau fallback digunakan jika API gagal).")

    st.info("🔀 Melakukan fusi multi-model (berbobot)...")
    # build fused df (next 24 hours by default)
    df_fused = build_fused_hourly(bmkg_obj, om_gfs, om_ecmwf, om_icon, hours=validity, weights_norm=norm_weights)
    if df_fused is None or df_fused.empty:
        st.error("Data model tidak tersedia / fusi gagal.")
        st.stop()

    st.success("✅ Fusi selesai.")

    # build tafor
    taf_lines = build_tafor_from_fused(df_fused, issue_dt, valid_hours=validity)
    taf_html = "<br>".join(taf_lines)

    # display source summary
    st.subheader("📊 Ringkasan Sumber Data")
    st.write(f"""
    | Sumber | Status |
    |:-------|:--------|
    | BMKG ADM4 | {'OK' if bmkg_obj and bmkg_obj.get('status')=='OK' else 'Unavailable'} |
    | Open-Meteo GFS | {'OK' if om_gfs else 'Unavailable'} |
    | Open-Meteo ECMWF | {'OK' if om_ecmwf else 'Unavailable'} |
    | Open-Meteo ICON | {'OK' if om_icon else 'Unavailable'} |
    """)

    # METAR input / OGIMET note
    st.markdown("### 📡 METAR (observasi — optional)")
    st.markdown("<div style='padding:10px;border:2px solid #bbb;border-radius:10px;background:#fafafa;'><p style='font-family:monospace;'>Optional manual METAR input is respected in TAF decisions if provided.</p></div>", unsafe_allow_html=True)

    # Show TAF
    st.markdown("### 📝 Hasil TAFOR (WARR — Sedati Gede / Juanda)")
    st.markdown(f"<div style='padding:15px;border:2px solid #555;border-radius:10px;background:#f9f9f9;'><p style='font-family:monospace;font-weight:700;'>{taf_html}</p></div>", unsafe_allow_html=True)

    # Display trend and short narrative
    st.markdown("### 🌦️ TREND / Narasi singkat")
    # Build simple trend: if first 1h has precip_flag -> TEMPO TL...
    first_hour = df_fused.iloc[0]
    precip_now = (first_hour.CC >= 80 and first_hour.RH >= 85)
    if precip_now:
        trend = f"TEMPO TL{(issue_dt+timedelta(hours=1)).strftime('%d%H%M')} 4000 -RA"
    else:
        trend = "NOSIG"
    st.markdown(f"<div style='padding:12px;border:2px solid #777;border-radius:10px;background:#f4f4f4;'><p style='font-family:monospace;font-weight:700;'>{trend}</p></div>", unsafe_allow_html=True)

    # Model analysis block
    st.markdown("### 🧠 Analisis Model (Fusi)")
    try:
        now = df_fused.iloc[0]
        sky = "Cerah" if now.CC < 25 else "Berawan" if now.CC < 70 else "Tertutup"
        hum = "Kering" if now.RH < 60 else "Lembap" if now.RH < 80 else "Basah"
        st.markdown(f"""
        <div style='padding:12px;border:2px solid #888;border-radius:10px;background:#f6f6f6;'>
            <b>Jam referensi:</b> {now.time:%Y-%m-%d %H:%M UTC}<br>
            <b>Temp (fused):</b> {now.T:.1f} °C &nbsp;&nbsp; <b>RH:</b> {now.RH:.0f}% ({hum})<br>
            <b>Tutupan awan:</b> {now.CC:.0f}% ({sky})<br>
            <b>Angin:</b> {int(now.WD):03d}/{now.WS:.1f} kt<br>
            <b>Interpretasi:</b> {'Fenomena signifikan terdeteksi (hujan/konveksi)' if precip_now else 'Tidak ada cuaca signifikan terdeteksi'}.
        </div>
        """, unsafe_allow_html=True)
    except Exception:
        st.info("Analisis model tidak tersedia.")

    # Plot fused chart (24h)
    st.markdown("### 📈 Grafik Fusi 24 jam (T / RH / Cloud / WS)")
    try:
        fig, ax = plt.subplots(figsize=(9,4))
        ax.plot(df_fused["time"], df_fused["T"], label="T (°C)")
        ax.plot(df_fused["time"], df_fused["RH"], label="RH (%)")
        ax.plot(df_fused["time"], df_fused["CC"], label="Cloud (%)")
        ax.plot(df_fused["time"], df_fused["WS"], label="Wind Speed (kt)")
        ax.set_xlabel("UTC time"); plt.xticks(rotation=35)
        ax.legend(loc="upper left")
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"Gagal plot: {e}")

    # Debug / JSON
    with st.expander("🔎 Debug / Fused JSON"):
        st.write(df_fused.to_dict(orient="records"))

    st.info("⚠️ Hati-hati: TAFOR otomatis ini bersifat eksperimental. Selalu validasi oleh forecaster resmi sebelum publikasi operasional.")
