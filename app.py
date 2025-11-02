# app_pro_fusion_v2_1.py
"""
TAFOR Fusion Pro v2.1 (BMKG + GFS + ECMWF + ICON + METAR OGIMET)
Fusi multi-model numerik untuk lokasi Sedati Gede (Juanda – WARR)
Output: TAF-like (ICAO + Perka-aware)
"""

import streamlit as st
from datetime import datetime, timedelta
import requests, json, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import radians, cos, sin, atan2, degrees
from xml.etree import ElementTree as ET

# === Streamlit setup ===
st.set_page_config(page_title="🛫 TAFOR Fusion Pro v2.1 — WARR", layout="centered")
st.title("🛫 TAFOR Fusion Pro v2.1 — Sedati Gede (WARR vicinity)")
st.caption("Fusi: BMKG (ADM4) + Open-Meteo (GFS, ECMWF, ICON) + METAR realtime (OGIMET/NOAA)")
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
    validity = st.number_input("🕐 Validity (hours)", 6, 36, 24, 6)

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

# === Helpers ===
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
    arr = np.array([np.nan if v is None else v for v in vals], float)
    w = np.array(ws, float)
    mask = ~np.isnan(arr)
    return float((arr[mask]*w[mask]).sum()/w[mask].sum()) if mask.sum() else np.nan

# === BMKG fetcher ===
@st.cache_data(ttl=REFRESH_TTL)
def fetch_bmkg():
    url = "https://cuaca.bmkg.go.id/api/df/v1/forecast/adm"
    params = {"adm1":"35","adm2":"35.15","adm3":"35.15.17","adm4":"35.15.17.2001"}
    try:
        r = requests.get(url, params=params, timeout=15, verify=False)
        data = r.json()
    except Exception:
        return {"status":"Unavailable"}
    try:
        cuaca = data["data"][0]["cuaca"][0][0]
        return {"status":"OK","raw":data,"cuaca":cuaca}
    except Exception:
        return {"status":"Unavailable","raw":data}

# === BMKG parser ===
def bmkg_to_df(cuaca):
    times, t, rh, cc, ws, wd, vs = [], [], [], [], [], [], []
    for c in cuaca:
        dt = c.get("datetime") or c.get("time") or c.get("jamCuaca") or c.get("date")
        if isinstance(dt, str):
            try: dt = pd.to_datetime(dt.replace("Z","+00:00"))
            except: continue
        else:
            continue
        times.append(dt)
        t.append(c.get("t")); rh.append(c.get("hu")); cc.append(c.get("tcc"))
        ws.append(c.get("ws")); wd.append(c.get("wd_deg")); vs.append(c.get("vs_text"))
    if not times: return pd.DataFrame()
    return pd.DataFrame({"time":times,"T_BMKG":t,"RH_BMKG":rh,"CC_BMKG":cc,
                         "WS_BMKG":ws,"WD_BMKG":wd,"VIS_BMKG":vs}).sort_values("time")

# === OpenMeteo fetcher ===
def fetch_openmeteo(model):
    url = f"https://api.open-meteo.com/v1/{model}"
    params = {"latitude":LAT,"longitude":LON,
              "hourly":"temperature_2m,relative_humidity_2m,cloud_cover,windspeed_10m,winddirection_10m,visibility",
              "forecast_days":2,"timezone":"UTC"}
    try:
        r = requests.get(url, params=params, timeout=15)
        return r.json()
    except: return None

def om_to_df(j, tag):
    if not j or "hourly" not in j: return None
    h = j["hourly"]
    return pd.DataFrame({
        "time":pd.to_datetime(h["time"]),
        f"T_{tag}":h["temperature_2m"],
        f"RH_{tag}":h["relative_humidity_2m"],
        f"CC_{tag}":h["cloud_cover"],
        f"WS_{tag}":h["windspeed_10m"],
        f"WD_{tag}":h["winddirection_10m"],
        f"VIS_{tag}":h.get("visibility",[None]*len(h["time"]))
    })

# === METAR fetch (OGIMET XML / NOAA fallback) ===
@st.cache_data(ttl=REFRESH_TTL)
def fetch_metar_ogimet(station="WARR"):
    try:
        url = f"https://ogimet.com/display_metars2.php?lang=en&icao={station}"
        r = requests.get(url, timeout=10)
        if not r.ok: return None
        text = r.text
        # try parse latest METAR line
        lines = [ln.strip() for ln in text.split("\n") if station in ln]
        if not lines: return None
        last = lines[-1].split(">")[-1].strip()
        return last
    except Exception:
        # NOAA fallback
        try:
            r = requests.get(f"https://tgftp.nws.noaa.gov/data/observations/metar/stations/{station}.TXT",timeout=10)
            return r.text.strip().split("\n")[-1]
        except:
            return None

# === Fusion core ===
def build_fused(bmkg, gfs, ecmwf, icon, weights, hours=24):
    dfs = [d for d in [om_to_df(gfs,"GFS"),om_to_df(ecmwf,"ECMWF"),om_to_df(icon,"ICON")] if d is not None]
    if not dfs: return None
    df = dfs[0][["time"]]
    for d in dfs: df = pd.merge(df, d, on="time", how="outer")
    if bmkg and bmkg.get("status")=="OK":
        df_b = bmkg_to_df(bmkg["cuaca"])
        df = pd.merge(df, df_b, on="time", how="outer")
    df = df.sort_values("time").reset_index(drop=True)
    out=[]
    for _,r in df.iterrows():
        vals={"T":[],"RH":[],"CC":[],"VIS":[],"U":[],"V":[],"w":[]}
        for key in ["bmkg","ecmwf","icon","gfs"]:
            wt=weights[key]; tag=key.upper()
            t=r.get(f"T_{tag}") if tag!="BMKG" else r.get("T_BMKG")
            rh=r.get(f"RH_{tag}") if tag!="BMKG" else r.get("RH_BMKG")
            cc=r.get(f"CC_{tag}") if tag!="BMKG" else r.get("CC_BMKG")
            vis=r.get(f"VIS_{tag}") if tag!="BMKG" else r.get("VIS_BMKG")
            ws=r.get(f"WS_{tag}") if tag!="BMKG" else r.get("WS_BMKG")
            wd=r.get(f"WD_{tag}") if tag!="BMKG" else r.get("WD_BMKG")
            if t is not None: vals["T"].append(t)
            if rh is not None: vals["RH"].append(rh)
            if cc is not None: vals["CC"].append(cc)
            if vis is not None:
                try: vals["VIS"].append(float(vis))
                except: pass
            if ws and wd:
                u,v=wind_to_uv(ws,wd); vals["U"].append(u); vals["V"].append(v)
            vals["w"].append(wt)
        if not vals["w"]: continue
        w=vals["w"]
        T=weighted_mean(vals["T"],w)
        RH=weighted_mean(vals["RH"],w)
        CC=weighted_mean(vals["CC"],w)
        VIS=weighted_mean(vals["VIS"],w)
        U=weighted_mean(vals["U"],w); V=weighted_mean(vals["V"],w)
        WS,WD=uv_to_wind(U,V)
        out.append({"time":r["time"],"T":T,"RH":RH,"CC":CC,"VIS":VIS,"WS":WS,"WD":WD})
    df_out=pd.DataFrame(out)
    now=pd.to_datetime(datetime.utcnow()).floor("H")
    return df_out[df_out["time"]>=now].head(hours)

# === TAFOR builder ===
def tcc_to_cloud(cc):
    if cc is None: return "FEW020"
    cc=float(cc)
    if cc<25: return "FEW020"
    elif cc<50: return "SCT025"
    elif cc<85: return "BKN030"
    else: return "OVC030"

def build_taf(df, metar, issue_dt, validity):
    header=f"TAF WARR {issue_dt:%d%H%MZ} {issue_dt:%d%H}/{(issue_dt+timedelta(hours=validity)):%d%H}"
    if df is None or df.empty:
        return [header,"9999 FEW020","NOSIG"]
    base=df.iloc[0]
    wind=f"{int(base.WD):03d}{int(round(base.WS or 5)):02d}KT"
    vis=str(int(base.VIS or 9999))
    cloud=tcc_to_cloud(base.CC)
    taf=[header,f"{wind} {vis} {cloud}"]
    df["precip"]=(df["CC"]>80)&(df["RH"]>85)
    if any(df["precip"]): taf.append("TEMPO 4000 -RA SCT020CB")
    else: taf.append("NOSIG")
    taf.append(f"RMK BASED ON { 'METAR INPUT' if metar else 'MODEL FUSION' }")
    return taf

# === Main run ===
if st.button("🚀 Generate TAFOR Fusion Pro"):
    issue_dt=datetime.combine(issue_date, datetime.utcnow().replace(hour=issue_time,minute=0,second=0).time())
    st.info("📡 Mengambil data BMKG, OpenMeteo & METAR...")
    bmkg=fetch_bmkg()
    gfs,ecmwf,icon=fetch_openmeteo("gfs"),fetch_openmeteo("ecmwf"),fetch_openmeteo("icon")
    metar=fetch_metar_ogimet("WARR")

    st.success("✅ Data diterima, memproses fusi numerik...")
    df_fused=build_fused(bmkg,gfs,ecmwf,icon,norm_weights,validity)
    taf=build_taf(df_fused,metar,issue_dt,validity)
    taf_html="<br>".join(taf)

    st.subheader("📊 Ringkasan Sumber Data")
    st.write(f"""
    | Sumber | Status |
    |:--|:--|
    | BMKG ADM4 | {'OK' if bmkg.get('status')=='OK' else 'Unavailable'} |
    | Open-Meteo GFS | {'OK' if gfs else 'Unavailable'} |
    | Open-Meteo ECMWF | {'OK' if ecmwf else 'Unavailable'} |
    | Open-Meteo ICON | {'OK' if icon else 'Unavailable'} |
    | METAR OGIMET | {'✅ Tersedia' if metar else '❌ Tidak ada'} |
    """)

    st.markdown("### 📡 METAR (Realtime OGIMET/NOAA)")
    st.markdown(f"<div style='padding:12px;border:2px solid #bbb;border-radius:10px;background:#fafafa;'><p style='font-family:monospace;font-weight:700'>{metar or 'Tidak tersedia'}</p></div>",unsafe_allow_html=True)

    st.markdown("### 📝 Hasil TAFOR (Auto Fusion)")
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
