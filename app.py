# =========================================================
# TAFOR Fusion Pro — FINAL Operational
# Auto TAF + Auto Narasi Resmi BMKG
# Location : Juanda (WARR)
# =========================================================

import os, io, json, math, zipfile, logging
from datetime import datetime, timedelta

import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
st.set_page_config(page_title="TAFOR Fusion Pro (WARR)", layout="centered")
st.title("🛫 TAFOR Fusion Pro — Operational (WARR / Juanda)")
st.caption("Auto TAFOR + Narasi Resmi | Fusion BMKG – ECMWF – ICON – GFS")

LAT, LON = -7.379, 112.787
ADM4 = "35.15.17.2011"
VALIDITY_DEFAULT = 24

os.makedirs("output", exist_ok=True)
os.makedirs("logs", exist_ok=True)

logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------
# UI INPUT
# ---------------------------------------------------------
col1, col2, col3 = st.columns(3)
with col1:
    issue_date = st.date_input("📅 Issue date (UTC)", datetime.utcnow().date())
with col2:
    issue_hour = st.selectbox("🕓 Issue hour (UTC)", [0,3,6,9,12,15,18,21])
with col3:
    validity = st.number_input("⏱ Validity (hours)", 6, 36, VALIDITY_DEFAULT, 6)

# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------
def wind_uv(ws, wd):
    theta = math.radians((270 - wd) % 360)
    return ws * math.cos(theta), ws * math.sin(theta)

def uv_wind(u, v):
    spd = math.sqrt(u*u + v*v)
    deg = (270 - math.degrees(math.atan2(v, u))) % 360
    return spd, deg

def tcc_to_cloud(cc):
    if cc < 25: return "FEW020"
    if cc < 50: return "SCT025"
    if cc < 85: return "BKN030"
    return "OVC030"

# ---------------------------------------------------------
# FETCH DATA
# ---------------------------------------------------------
@st.cache_data(ttl=600)
def fetch_openmeteo(model):
    url = f"https://api.open-meteo.com/v1/{model}"
    p = {
        "latitude": LAT, "longitude": LON,
        "hourly": "temperature_2m,relative_humidity_2m,cloud_cover,windspeed_10m,winddirection_10m,visibility",
        "forecast_days": 2, "timezone": "UTC"
    }
    r = requests.get(url, params=p, timeout=15)
    r.raise_for_status()
    return r.json()

def openmeteo_df(js, tag):
    h = js["hourly"]
    return pd.DataFrame({
        "time": pd.to_datetime(h["time"]),
        f"T_{tag}": h["temperature_2m"],
        f"RH_{tag}": h["relative_humidity_2m"],
        f"CC_{tag}": h["cloud_cover"],
        f"WS_{tag}": h["windspeed_10m"],
        f"WD_{tag}": h["winddirection_10m"],
        f"VIS_{tag}": h["visibility"]
    })

# ---------------------------------------------------------
# FUSION
# ---------------------------------------------------------
def fuse(df, hours):
    rows = []
    for _, r in df.head(hours).iterrows():
        ws = np.nanmean([r.WS_GFS, r.WS_ECMWF, r.WS_ICON])
        wd = np.nanmean([r.WD_GFS, r.WD_ECMWF, r.WD_ICON])
        rows.append({
            "time": r.time,
            "T": np.nanmean([r.T_GFS, r.T_ECMWF, r.T_ICON]),
            "RH": np.nanmean([r.RH_GFS, r.RH_ECMWF, r.RH_ICON]),
            "CC": np.nanmean([r.CC_GFS, r.CC_ECMWF, r.CC_ICON]),
            "VIS": np.nanmean([r.VIS_GFS, r.VIS_ECMWF, r.VIS_ICON]),
            "WS": ws,
            "WD": wd
        })
    return pd.DataFrame(rows)

# ---------------------------------------------------------
# BUILD TAF
# ---------------------------------------------------------
def build_taf(df, issue_dt, validity):
    taf = []
    taf.append(f"TAF WARR {issue_dt:%d%H%MZ} {issue_dt:%d%H}/{(issue_dt+timedelta(hours=validity)):%d%H}")

    f = df.iloc[0]
    taf.append(f"{int(f.WD):03d}{int(f.WS):02d}KT 9999 {tcc_to_cloud(f.CC)}")

    for i in range(1, len(df)):
        p, c = df.iloc[i-1], df.iloc[i]
        t1, t2 = p.time.strftime("%d%H"), c.time.strftime("%d%H")

        if abs(c.WS-p.WS) >= 10:
            taf.append(f"BECMG {t1}/{t2} {int(c.WD):03d}{int(c.WS):02d}KT 5000 {tcc_to_cloud(c.CC)}")

        if c.CC >= 80 and c.RH >= 85:
            taf.append(f"TEMPO {t1}/{t2} 3000 TSRA SCT018CB")

    taf.append("RMK AUTO FUSION SYSTEM")
    return taf

# ---------------------------------------------------------
# NARASI OTOMATIS (INI KUNCI UTAMA)
# ---------------------------------------------------------
def taf_to_narrative(taf, issue_dt, validity):
    teks = []
    end = issue_dt + timedelta(hours=validity)

    teks.append(
        f"Prakiraan cuaca Bandara Juanda (WARR) berlaku tanggal "
        f"{issue_dt:%d %B %Y} pukul {issue_dt:%H.%M} UTC hingga "
        f"{end:%d %B %Y} pukul {end:%H.%M} UTC."
    )

    for ln in taf:
        if ln.startswith("TAF") or ln.startswith("RMK"):
            continue

        if ln.startswith("TEMPO"):
            _, w, _, wx, cloud = ln.split()
            t1, t2 = w.split("/")
            teks.append(
                f"\nPada pukul {t1[2:]} hingga {t2[2:]} UTC, "
                f"diprakirakan terjadi perubahan cuaca sementara berupa hujan "
                f"disertai badai petir, dengan jarak pandang menurun hingga 3 km "
                f"serta terdapat awan Cumulonimbus."
            )

        elif ln.startswith("BECMG"):
            _, w, *_ = ln.split()
            t1, t2 = w.split("/")
            teks.append(
                f"\nSelanjutnya, pada pukul {t1[2:]} hingga {t2[2:]} UTC, "
                f"kondisi cuaca berubah secara bertahap."
            )

        else:
            wd = ln[:3]
            ws = ln[3:5]
            teks.append(
                f"\nPada awal periode, angin bertiup dari arah {wd}° "
                f"dengan kecepatan {int(ws)} knot, jarak pandang lebih dari 10 km, "
                f"dan tidak terdapat cuaca signifikan."
            )

    return "\n".join(teks)

# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
if st.button("🚀 Generate TAFOR Otomatis"):
    issue_dt = datetime.combine(issue_date, datetime.min.time()) + timedelta(hours=issue_hour)

    gfs = openmeteo_df(fetch_openmeteo("gfs"), "GFS")
    ecmwf = openmeteo_df(fetch_openmeteo("ecmwf"), "ECMWF")
    icon = openmeteo_df(fetch_openmeteo("icon"), "ICON")

    df = gfs.merge(ecmwf, on="time").merge(icon, on="time")
    df_fused = fuse(df, validity)

    taf = build_taf(df_fused, issue_dt, validity)
    narasi = taf_to_narrative(taf, issue_dt, validity)

    st.subheader("📝 TAFOR")
    st.code("\n".join(taf))

    st.subheader("📰 Narasi Resmi Otomatis")
    st.text_area("Siap laporan / briefing", narasi, height=300)

    st.success("✅ TAFOR & Narasi berhasil dibuat — silakan VALIDASI sebelum rilis operasional.")
