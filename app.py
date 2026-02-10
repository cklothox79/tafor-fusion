import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta

# ===============================
# KONFIGURASI BANDARA (JUANDA)
# ===============================
AIRPORT = "WARR"
LAT = -7.379
LON = 112.787

# ===============================
# OPEN-METEO FETCH (SAFE)
# ===============================
@st.cache_data(show_spinner=False)
def fetch_openmeteo(model):
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={LAT}&longitude={LON}"
        "&hourly=winddirection_10m,windspeed_10m,visibility,weathercode"
        "&timezone=UTC"
        f"&models={model}"
    )
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.warning(f"⚠️ Model {model.upper()} tidak tersedia ({e})")
        return None


def openmeteo_df(data, label):
    if data is None:
        return None

    h = data["hourly"]
    df = pd.DataFrame({
        "time": pd.to_datetime(h["time"]),
        "wind_dir": h["winddirection_10m"],
        "wind_spd": h["windspeed_10m"],
        "vis": h["visibility"],
        "wx": h["weathercode"],
    })
    df["model"] = label
    return df


# ===============================
# FUSION ENGINE
# ===============================
def fuse_models(dfs):
    dfs = [d for d in dfs if d is not None]
    if not dfs:
        st.error("❌ Semua model gagal diambil. Tidak bisa membuat TAF.")
        st.stop()

    df = pd.concat(dfs)
    fused = (
        df.groupby("time")
        .agg({
            "wind_dir": "median",
            "wind_spd": "median",
            "vis": "median",
            "wx": "max",
        })
        .reset_index()
    )
    return fused


# ===============================
# TAF BUILDER
# ===============================
def build_taf_from_fused(df, issue_dt, validity):
    start = issue_dt
    end = issue_dt + timedelta(hours=validity)

    df = df[(df["time"] >= start) & (df["time"] <= end)]

    taf = []
    signif = []

    first = df.iloc[0]
    taf.append(
        f"TAF {AIRPORT} {issue_dt:%d%H%M}Z "
        f"{start:%d%H}/{end:%d%H} "
        f"{int(first.wind_dir):03d}{int(first.wind_spd):02d}KT "
        f"9999 FEW020"
    )

    for _, r in df.iterrows():
        if r.wx >= 95:
            signif.append(
                f"TEMPO {r.time:%d%H}/{(r.time + timedelta(hours=6)):%d%H} "
                "3000 TSRA SCT018CB"
            )

        if r.vis < 5000:
            signif.append(
                f"BECMG {r.time:%d%H}/{(r.time + timedelta(hours=2)):%d%H} "
                "4000 HZ"
            )

    return taf + signif, signif


# ===============================
# NARASI OTOMATIS
# ===============================
def taf_to_narrative(taf_lines, issue_dt, validity):
    end = issue_dt + timedelta(hours=validity)

    narasi = [
        f"Prakiraan cuaca Bandara Juanda (WARR) berlaku tanggal "
        f"{issue_dt:%d %B %Y} pukul {issue_dt:%H.%M} UTC hingga "
        f"{end:%d %B %Y} pukul {end:%H.%M} UTC."
    ]

    for l in taf_lines:
        if l.startswith("TAF"):
            continue

        if l.startswith("TEMPO"):
            narasi.append(
                "\nPada periode tertentu diprakirakan terjadi hujan "
                "disertai badai petir dengan jarak pandang menurun "
                "hingga sekitar 3 km serta awan Cumulonimbus."
            )

        if l.startswith("BECMG"):
            narasi.append(
                "\nSelanjutnya kondisi cuaca berubah secara bertahap "
                "dengan jarak pandang menurun hingga sekitar 4 km "
                "akibat kabut asap (haze)."
            )

    return "\n".join(narasi)


# ===============================
# STREAMLIT UI
# ===============================
st.title("🛫 TAFOR Fusion Pro — Operational (WARR / Juanda)")
st.caption("Auto TAFOR + Narasi Resmi | Fusion BMKG – ECMWF – ICON – GFS")

col1, col2, col3 = st.columns(3)
issue_date = col1.date_input("📅 Issue date (UTC)")
issue_hour = col2.number_input("🕓 Issue hour (UTC)", 0, 23, 6)
validity = col3.number_input("⏱ Validity (hours)", 6, 48, 24)

issue_dt = datetime.combine(issue_date, datetime.min.time()) + timedelta(hours=issue_hour)

st.divider()

# ===============================
# FETCH MODELS
# ===============================
ecmwf = openmeteo_df(fetch_openmeteo("ecmwf"), "ECMWF")
gfs = openmeteo_df(fetch_openmeteo("gfs"), "GFS")
icon = openmeteo_df(fetch_openmeteo("icon"), "ICON")

fused = fuse_models([ecmwf, gfs, icon])

taf_lines, signif = build_taf_from_fused(fused, issue_dt, validity)
narasi = taf_to_narrative(taf_lines, issue_dt, validity)

# ===============================
# OUTPUT
# ===============================
st.subheader("📄 TAFOR Otomatis")
st.code("\n".join(taf_lines), language="text")

st.subheader("📰 Narasi Prakiraan Resmi")
st.text_area("Siap laporan / briefing", narasi, height=250)
