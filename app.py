# app_pro_fusion_v2_2_sedatigede.py
"""
TAFOR Fusion Pro v2.2 — Optimized for Sedati Gede (WARR vicinity)
Data sources:
 - BMKG ADM4 (35.15.17.2001)
 - Open-Meteo (GFS, ECMWF, ICON)
 - METAR realtime (OGIMET/NOAA)
Output:
 - ICAO/Perka-compliant TAFOR (auto)
 - Trend & fusion analysis
"""

import streamlit as st
from datetime import datetime, timedelta
import requests, json, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import radians, cos, sin, atan2, degrees

# === Streamlit setup ===
st.set_page_config(page_title="🛫 TAFOR Fusion Pro v2.2 — WARR", layout="centered")
st.title("🛫 TAFOR Fusion Pro v2.2 — Sedati Gede (WARR vicinity)")
st.caption("Optimized fusion for Juanda area: BMKG ADM4 + Open-Meteo (GFS, ECMWF, ICON) + METAR realtime")
st.divider()

# === Constants ===
LAT, LON = -7.379, 112.787
ADM4 = "35.15.17.2001"
REFRESH_TTL = 900
DEFAULT_WEIGHTS = {"bmkg": 0.45, "ecmwf": 0.25, "icon": 0.15, "gfs": 0.15}

# === UI inputs ===
col1, col2, col3 = st.columns(3)
with col1:
    issue_date = st.date_input("📅 Issue date (UTC)", datetime.utcnow().date())
with col2:
    jam_penting = [0, 3, 6, 9, 12, 15, 18, 21]
    jam_sekarang = datetime.utcnow().hour
    default_jam = min(jam_penting, key=lambda j: abs(j - jam_sekarang))
    issue_time = st.selectbox("🕓 Issue time (UTC)", jam_penting, index=jam_penting.index(default_jam))
with col3:
    validity = st.number_input("🕐 Validity (hours)", 6, 36, 24, 6)

cols = st.columns(4)
weights = {}
weights["bmkg"] = cols[0].number_input("BMKG", 0.0, 1.0, 0.45, step=0.05)
weights["ecmwf"] = cols[1].number_input("ECMWF", 0.0, 1.0, 0.25, step=0.05)
weights["icon"] = cols[2].number_input("ICON", 0.0, 1.0, 0.15, step=0.05)
weights["gfs"] = cols[3].number_input("GFS", 0.0, 1.0, 0.15, step=0.05)
norm_weights = {k: v / (sum(weights.values()) or 1) for k, v in weights.items()}
st.caption(f"Normalized weights: {', '.join([f'{k}={v:.2f}' for k,v in norm_weights.items()])}")

# === Helper functions ===
def wind_to_uv(speed, deg):
    if pd.isna(speed) or pd.isna(deg): return None, None
    th = radians((270 - deg) % 360)
    return speed * cos(th), speed * sin(th)

def uv_to_wind(u, v):
    spd = np.sqrt(u**2 + v**2)
    th = degrees(atan2(v, u))
    deg = (270 - th) % 360
    return spd, deg

def weighted_mean(vals, ws):
    """Hitung rata-rata berbobot dengan panjang aman"""
    arr = np.array([np.nan if v is None else v for v in vals], float)
    w = np.array(ws, float)

    # samakan panjang antara arr dan w
    n = min(len(arr), len(w))
    if n == 0:
        return np.nan
    arr = arr[:n]
    w = w[:n]

    mask = ~np.isnan(arr)
    if not mask.any():
        return np.nan
    return float(np.nansum(arr[mask] * w[mask]) / np.nansum(w[mask]))

# === Fetch BMKG optimized ===
@st.cache_data(ttl=REFRESH_TTL)
def fetch_bmkg_sedatigede():
    url = "https://cuaca.bmkg.go.id/api/df/v1/forecast/adm"
    params = {"adm1":"35","adm2":"35.15","adm3":"35.15.17","adm4":ADM4}
    try:
        r = requests.get(url, params=params, timeout=15, verify=False)
        data = r.json()
    except Exception:
        return {"status":"Unavailable"}
    try:
        # JSON struktur: data[0]["cuaca"][0][0]
        cuaca = data["data"][0]["cuaca"][0][0]
        return {"status":"OK","raw":data,"cuaca":cuaca}
    except Exception:
        return {"status":"Unavailable","raw":data}

# === Parser khusus Sedati Gede ===
def bmkg_to_df_sedatigede(cuaca):
    """
    Struktur BMKG terbaru (Nov 2025) untuk ADM4:
    cuaca = [[{datetime,t,hu,tcc,ws,wd_deg,vs_text,weather}]...]
    """
    records = []
    # handle nested list
    if isinstance(cuaca, list):
        for sub in cuaca:
            if isinstance(sub, dict):
                records.append(sub)
            elif isinstance(sub, list):
                for s in sub:
                    if isinstance(s, dict):
                        records.append(s)
    elif isinstance(cuaca, dict):
        records = [cuaca]

    if not records:
        return pd.DataFrame()

    times, t, rh, cc, ws, wd, vs, wx = [], [], [], [], [], [], [], []
    for c in records:
        dt = c.get("datetime")
        if not dt: continue
        try:
            dt = pd.to_datetime(dt.replace("Z","+00:00"))
        except: continue
        times.append(dt)
        t.append(c.get("t"))
        rh.append(c.get("hu"))
        cc.append(c.get("tcc"))
        ws.append(c.get("ws"))
        wd.append(c.get("wd_deg"))
        vs.append(c.get("vs_text"))
        wx.append(c.get("weather"))

    return pd.DataFrame({
        "time":times,"T_BMKG":t,"RH_BMKG":rh,"CC_BMKG":cc,
        "WS_BMKG":ws,"WD_BMKG":wd,"VIS_BMKG":vs,"WX_BMKG":wx
    }).sort_values("time").reset_index(drop=True)

# === Open-Meteo fetch ===
def fetch_openmeteo(model):
    url=f"https://api.open-meteo.com/v1/{model}"
    params={"latitude":LAT,"longitude":LON,
            "hourly":"temperature_2m,relative_humidity_2m,cloud_cover,windspeed_10m,winddirection_10m,visibility",
            "forecast_days":2,"timezone":"UTC"}
    try:
        return requests.get(url,params=params,timeout=15).json()
    except: return None

def om_to_df(j,tag):
    if not j or "hourly" not in j: return None
    h=j["hourly"]
    return pd.DataFrame({
        "time":pd.to_datetime(h["time"]),
        f"T_{tag}":h["temperature_2m"],
        f"RH_{tag}":h["relative_humidity_2m"],
        f"CC_{tag}":h["cloud_cover"],
        f"WS_{tag}":h["windspeed_10m"],
        f"WD_{tag}":h["winddirection_10m"],
        f"VIS_{tag}":h.get("visibility",[None]*len(h["time"]))
    })

# === METAR realtime ===
@st.cache_data(ttl=REFRESH_TTL)
def fetch_metar_ogimet(station="WARR"):
    try:
        r=requests.get(f"https://tgftp.nws.noaa.gov/data/observations/metar/stations/{station}.TXT",timeout=10)
        return r.text.strip().split("\n")[-1]
    except: return None

# === Fusion ===
def build_fused(bmkg, gfs, ecmwf, icon, w, hours=24):
    """Gabungkan data dari BMKG dan tiga model Open-Meteo dengan bobot ensemble"""

    # helper: paksa time menjadi datetime dan buang null
    def normalize_time(df):
        if df is not None and "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"], errors="coerce")
            df = df.dropna(subset=["time"])
        return df

    # ambil model-model openmeteo
    dfs = [normalize_time(d) for d in [om_to_df(gfs, "GFS"),
                                       om_to_df(ecmwf, "ECMWF"),
                                       om_to_df(icon, "ICON")] if d is not None]

    if not dfs:
        return None

    # merge model-model openmeteo
    df = dfs[0][["time"]]
    for d in dfs[1:]:
        if d is not None:
            df = pd.merge(df, d, on="time", how="outer")

    # tambahkan BMKG (jika ada)
    if bmkg and bmkg.get("status") == "OK":
        df_b = bmkg_to_df_sedatigede(bmkg["cuaca"])
        df_b = normalize_time(df_b)

    if df_b is not None and not df_b.empty:
       # pastikan semua waktu sama-sama naive (tanpa zona waktu)
       df_b["time"] = pd.to_datetime(df_b["time"], utc=True).dt.tz_convert(None)
       df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_convert(None)
       try:
        df = pd.merge(df, df_b, on="time", how="outer")
       except Exception as e:
        st.warning(f"⚠️ BMKG merge skipped: {e}")
    else:
        st.info("ℹ️ BMKG data unavailable — using Open-Meteo only")

    # bersihkan dan urutkan
    df = normalize_time(df)
    if df.empty:
        return None

    df = df.sort_values("time").reset_index(drop=True)

    # proses fusi berbobot
    out = []
    for _, r in df.iterrows():
        vals = {"T": [], "RH": [], "CC": [], "VIS": [], "U": [], "V": [], "w": []}
        for key in ["bmkg", "ecmwf", "icon", "gfs"]:
            wt = w[key]
            tag = key.upper()
            t = r.get(f"T_{tag}") if tag != "BMKG" else r.get("T_BMKG")
            rh = r.get(f"RH_{tag}") if tag != "BMKG" else r.get("RH_BMKG")
            cc = r.get(f"CC_{tag}") if tag != "BMKG" else r.get("CC_BMKG")
            vis = r.get(f"VIS_{tag}") if tag != "BMKG" else r.get("VIS_BMKG")
            ws = r.get(f"WS_{tag}") if tag != "BMKG" else r.get("WS_BMKG")
            wd = r.get(f"WD_{tag}") if tag != "BMKG" else r.get("WD_BMKG")

            if t is not None: vals["T"].append(t)
            if rh is not None: vals["RH"].append(rh)
            if cc is not None: vals["CC"].append(cc)
            if vis:
                try: vals["VIS"].append(float(vis))
                except: pass
            if ws and wd:
                u, v = wind_to_uv(ws, wd)
                vals["U"].append(u); vals["V"].append(v)
            vals["w"].append(wt)

        if not vals["w"]: 
            continue

        ws_ = vals["w"]
        T = weighted_mean(vals["T"], ws_)
        RH = weighted_mean(vals["RH"], ws_)
        CC = weighted_mean(vals["CC"], ws_)
        VIS = weighted_mean(vals["VIS"], ws_)
        U = weighted_mean(vals["U"], ws_)
        V = weighted_mean(vals["V"], ws_)
        WS, WD = uv_to_wind(U, V)

        out.append({
            "time": r["time"],
            "T": T, "RH": RH, "CC": CC, "VIS": VIS, "WS": WS, "WD": WD
        })

    df_out = pd.DataFrame(out)
    if df_out.empty:
        return None

    now = pd.to_datetime(datetime.utcnow()).floor("H")
    return df_out[df_out["time"] >= now].head(hours)


# === TAFOR builder ===
def tcc_to_cloud(cc):
    if cc is None: return "FEW020"
    cc=float(cc)
    if cc<25: return "FEW020"
    elif cc<50: return "SCT025"
    elif cc<85: return "BKN030"
    else: return "OVC030"

def build_taf(df, metar, issue_dt, validity):
    """Bangun TAF resmi (ICAO + Perka BMKG) berdasarkan hasil fusi"""
    taf_lines = []

    # === HEADER ===
    taf_header = f"TAF WARR {issue_dt:%d%H%MZ} {issue_dt:%d%H}/{(issue_dt + timedelta(hours=validity)):%d%H}"
    taf_lines.append(taf_header)

    # === Jika tidak ada data fusi, fallback ===
    if df is None or df.empty:
        taf_lines += ["00000KT 9999 FEW020", "NOSIG", "RMK AUTO FUSION BASED ON MODEL ONLY"]
        return taf_lines

    # === Baris dasar ===
    base = df.iloc[0]
    wd = int(round(base.WD or 0))
    ws = int(round(base.WS or 5))
    vis = base.VIS
    try:
        vis = int(float(vis))
        if vis <= 0:
            vis = 9999
    except Exception:
        vis = 9999
    cc_code = tcc_to_cloud(base.CC)
    taf_lines.append(f"{wd:03d}{ws:02d}KT {vis:04d} {cc_code}")

    # === Analisis tren dari data fusi ===
    df["precip"] = (df["CC"] > 80) & (df["RH"] > 85)
    df["wind_change"] = abs(df["WD"] - wd) > 45

    # === Deteksi periode signifikan dengan ambang ICAO ===
    WIND_CHANGE_THRESHOLD_DEG = 60
    WIND_SPEED_THRESHOLD_KT = 10
    CLOUD_CHANGE_THRESHOLD = 25  # persen

    becmg_periods, tempo_periods = [], []

    for i in range(1, len(df)):
        prev = df.iloc[i - 1]
        curr = df.iloc[i]

        tstart = prev["time"].strftime("%d%H")
        tend = curr["time"].strftime("%d%H")
        tcode = tcc_to_cloud(curr.CC)
        tvis = int(float(curr.VIS)) if not pd.isna(curr.VIS) else 9999

        # deteksi perubahan signifikan arah/kecepatan angin
        wd_diff = abs((curr.WD or 0) - (prev.WD or 0))
        ws_diff = abs((curr.WS or 0) - (prev.WS or 0))
        cc_diff = abs((curr.CC or 0) - (prev.CC or 0))

        significant_wind = wd_diff >= WIND_CHANGE_THRESHOLD_DEG or ws_diff >= WIND_SPEED_THRESHOLD_KT
        significant_cloud = cc_diff >= CLOUD_CHANGE_THRESHOLD

        if significant_wind or significant_cloud:
            becmg_periods.append(
                f"BECMG {tstart}/{tend} {int(curr.WD):03d}{int(curr.WS):02d}KT {tvis:04d} {tcode}"
            )

        # deteksi potensi fenomena sementara (hujan, badai)
        if (curr["CC"] > 80) and (curr["RH"] > 85):
            tempo_periods.append(
                f"TEMPO {tstart}/{tend} 4000 -RA SCT020CB"
            )

    # === Tambahkan hasil periodik (jika ada) ===
    if becmg_periods:
        taf_lines += becmg_periods
    if tempo_periods:
        taf_lines += tempo_periods

    # === Jika stabil total, tambahkan NOSIG ===
    if not becmg_periods and not tempo_periods:
        taf_lines.append("NOSIG")

    # === Remark akhir ===
    source_text = "METAR+MODEL FUSION" if metar else "MODEL FUSION"
    taf_lines.append(f"RMK AUTO FUSION BASED ON {source_text}")

    return taf_lines




# === Run ===
if st.button("🚀 Generate TAFOR (Optimized Fusion)"):
    issue_dt=datetime.combine(issue_date,datetime.utcnow().replace(hour=issue_time,minute=0,second=0).time())
    st.info("📡 Fetching BMKG + OpenMeteo + METAR...")
    bmkg=fetch_bmkg_sedatigede()
    gfs,ecmwf,icon=fetch_openmeteo("gfs"),fetch_openmeteo("ecmwf"),fetch_openmeteo("icon")
    metar=fetch_metar_ogimet("WARR")

    st.success("✅ Data ready. Processing fusion...")
    df_fused=build_fused(bmkg,gfs,ecmwf,icon,norm_weights,validity)
    taf=build_taf(df_fused,metar,issue_dt,validity)
    taf_html="<br>".join(taf)

    st.subheader("📊 Ringkasan Sumber Data")
    st.write(f"""
    | Sumber | Status |
    |:--|:--|
    | BMKG ADM4 | {'OK' if bmkg.get('status')=='OK' else 'Unavailable'} |
    | GFS | {'OK' if gfs else 'Unavailable'} |
    | ECMWF | {'OK' if ecmwf else 'Unavailable'} |
    | ICON | {'OK' if icon else 'Unavailable'} |
    | METAR | {'✅' if metar else '❌'} |
    """)

    st.markdown("### 📡 METAR (Realtime OGIMET/NOAA)")
    st.markdown(f"<div style='padding:12px;border:2px solid #bbb;border-radius:10px;background:#fafafa;'><p style='font-family:monospace;font-weight:700'>{metar or 'Tidak tersedia'}</p></div>",unsafe_allow_html=True)

    st.markdown("### 📝 Hasil TAFOR (Optimized Fusion)")
    st.markdown(f"<div style='padding:15px;border:2px solid #555;border-radius:10px;background:#f9f9f9;'><p style='font-family:monospace;font-weight:700'>{taf_html}</p></div>",unsafe_allow_html=True)

    if df_fused is not None and not df_fused.empty:
        st.markdown("### 📈 Grafik Fusi 24 jam (T/RH/Awan/Angin)")
        fig,ax=plt.subplots(figsize=(9,4))
        ax.plot(df_fused["time"],df_fused["T"],label="T (°C)")
        ax.plot(df_fused["time"],df_fused["RH"],label="RH (%)")
        ax.plot(df_fused["time"],df_fused["CC"],label="Cloud (%)")
        ax.plot(df_fused["time"],df_fused["WS"],label="Wind (kt)")
        ax.legend();plt.xticks(rotation=35)
        st.pyplot(fig)

    with st.expander("🔍 Debug JSON BMKG"):
        st.write(bmkg)
