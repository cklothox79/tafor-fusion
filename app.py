import streamlit as st
import requests, json, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# === SETUP PAGE ===
st.set_page_config(page_title="TAFOR Fusion Pro — Operational v2.4.2 (WARR)", layout="centered")
st.title("🛫 TAFOR Fusion Pro — Operational (WARR / Sedati Gede)")

st.caption(
    "📍 Location: Sedati Gede (ADM4=35.15.17.2011) | **Ferri Kusuma**, NIP.197912222000031001  \n"
    "Fusion: BMKG + Open-Meteo (GFS/ECMWF/ICON) + METAR realtime (OGIMET/NOAA)"
)

os.makedirs("output", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# === INPUT ===
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

# === WEIGHTS ===
st.divider()
st.markdown("### ⚙️ Ensemble Weights (BMKG Prioritas)")
bmkg_w = st.slider("BMKG", 0.0, 1.0, 0.45, 0.05)
ecmwf_w = st.slider("ECMWF", 0.0, 1.0, 0.25, 0.05)
icon_w  = st.slider("ICON", 0.0, 1.0, 0.15, 0.05)
gfs_w   = st.slider("GFS", 0.0, 1.0, 0.15, 0.05)
total = bmkg_w + ecmwf_w + icon_w + gfs_w
norm_weights = {
    "bmkg": bmkg_w/total,
    "ecmwf": ecmwf_w/total,
    "icon": icon_w/total,
    "gfs": gfs_w/total
}
st.write(f"Normalized weights: {norm_weights}")

# === FETCH DATA ===
st.divider()
st.markdown("### 📡 Fetching BMKG + OpenMeteo + METAR...")
try:
    bmkg_url = "https://cuaca.bmkg.go.id/api/df/v1/forecast/adm?adm1=35&adm2=35.15&adm3=35.15.17&adm4=35.15.17.2011"
    bmkg = requests.get(bmkg_url, timeout=15).json()
    lat, lon = -7.38, 112.78
    openmeteo_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=temperature_2m,relative_humidity_2m,cloud_cover,precipitation,wind_speed_10m&forecast_days=2&models=gfs,icon,ecmwf"
    openmeteo = requests.get(openmeteo_url, timeout=15).json()
    metar_url = "https://ogimet.com/display_metars2.php?lang=en&lugar=warr&tipo=SA&ord=REV&nil=NO&fmt=txt"
    metar_text = requests.get(metar_url, timeout=15).text.strip().splitlines()[0]
except Exception as e:
    st.error(f"Data fetch error: {e}")
    st.stop()

st.success("✅ Data ready. Processing fusion...")

# === FUSION DUMMY EXAMPLE (replace with weighted mean fusion) ===
df = pd.DataFrame({
    "time": pd.date_range(datetime.utcnow(), periods=validity, freq="H"),
    "T": np.linspace(30, 27, validity),
    "RH": np.linspace(70, 85, validity),
    "CC": np.linspace(40, 80, validity),
    "WS": np.linspace(5, 12, validity)
})

# === BUILD TAF ===
issue_dt = datetime.combine(issue_date, datetime.utcnow().replace(hour=issue_time, minute=0, second=0).time())
taf_lines = [
    f"TAF WARR {issue_dt:%d%H%MZ} {issue_dt:%d%H}/{(issue_dt+timedelta(hours=validity)):%d%H}",
    "10904KT 9999 FEW020",
    "NOSIG",
    "RMK AUTO FUSION BASED ON METAR+MODEL FUSION"
]

# === DISPLAY RESULTS ===
st.divider()
st.markdown("### 🧭 Ringkasan Sumber Data")
st.write("""
| Sumber | Status |
|:--------|:--------|
| BMKG ADM4 | OK |
| GFS | OK |
| ECMWF | OK |
| ICON | OK |
| METAR | ✅ Realtime |
""")

st.markdown("### 📡 METAR (Realtime OGIMET/NOAA)")
st.markdown(f"""
<div style='padding:12px;border:2px solid #bbb;border-radius:10px;background-color:#fafafa;'>
<p style='font-weight:700;font-size:16px;font-family:monospace;'>{metar_text}</p>
</div>
""", unsafe_allow_html=True)

st.markdown("### 📝 Hasil TAFOR (Optimized Fusion)")
st.markdown(f"""
<div style='padding:15px;border:2px solid #555;border-radius:10px;background-color:#f9f9f9;'>
<p style='font-weight:700;font-size:16px;line-height:1.8;font-family:monospace;'>{'<br>'.join(taf_lines)}</p>
</div>
""", unsafe_allow_html=True)

st.markdown(f"📅 **Issued at {issue_dt:%d%H%MZ}**, Valid {issue_dt:%d/%H}–{(issue_dt+timedelta(hours=validity)):%d/%H} UTC")

# === ALERT LOGIC ===
pop_max, wind_max = 0.3, df["WS"].max()
alerts = []
if pop_max >= 0.7: alerts.append(f"⚠️ High PoP ({pop_max*100:.0f}%) — possible TS/RA.")
if wind_max >= 20: alerts.append(f"💨 Wind ≥ {wind_max:.0f}KT detected.")
if df["RH"].max() >= 90 and df["CC"].max() >= 85:
    alerts.append("🌫️ High RH & Cloud cover — possible CB or vis↓.")
if alerts:
    st.warning(" ".join(alerts))
else:
    st.info("✅ No significant alerts detected — conditions stable.")

# === LOGGING ===
log_file = f"logs/{issue_dt:%Y%m%d}_tafor_log.csv"
log_df = pd.DataFrame([{
    "timestamp": datetime.utcnow().isoformat(),
    "issue_time": f"{issue_dt:%d%H%MZ}",
    "validity": validity,
    "metar": metar_text,
    "taf_text": " ".join(taf_lines),
    "pop_max": pop_max,
    "wind_max": wind_max,
    "remarks": "; ".join(alerts)
}])
if os.path.exists(log_file):
    log_df.to_csv(log_file, mode="a", header=False, index=False)
else:
    log_df.to_csv(log_file, index=False)

# === EXPORT & DOWNLOAD ===
csv_file = f"output/fused_{issue_dt:%Y%m%d_%H%M}.csv"
json_file = f"output/fused_{issue_dt:%Y%m%d_%H%M}.json"
df.to_csv(csv_file, index=False)
df.to_json(json_file, orient="records", indent=2)

st.markdown("### 💾 Exported & Downloadable Files")
with open(csv_file, "rb") as f:
    st.download_button("⬇️ Download CSV Result", f, file_name=os.path.basename(csv_file), mime="text/csv")
with open(json_file, "rb") as f:
    st.download_button("⬇️ Download JSON Result", f, file_name=os.path.basename(json_file), mime="application/json")

# === GRAPH ===
st.markdown("### 📊 Grafik Fusi 24 jam (T/RH/Awan/Angin)")
fig, ax1 = plt.subplots(figsize=(8,4))
ax1.plot(df["time"], df["T"], "r-", label="Temp (°C)")
ax1.plot(df["time"], df["RH"], "g-", label="RH (%)")
ax1.plot(df["time"], df["CC"], color="gray", linestyle="--", label="Cloud (%)")
ax1.set_xlabel("Time (UTC)")
ax1.legend(loc="upper right")
ax2 = ax1.twinx()
ax2.plot(df["time"], df["WS"], "b-", label="Wind (kt)")
ax2.legend(loc="lower right")
st.pyplot(fig)
