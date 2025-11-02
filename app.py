# app.py — tafor-fusion (BMKG + OpenMeteo: GFS, ECMWF, ICON)
import streamlit as st
import requests, json, numpy as np, pandas as pd, matplotlib.pyplot as plt
from datetime import datetime, timedelta
from math import radians, cos, sin, atan2, degrees

st.set_page_config(page_title="🌀 TAFOR Fusion — Sedati Gede", layout="centered")

st.title("🌀 Multi-Model Fusion (BMKG + GFS + ECMWF + ICON)")
st.caption("Lokasi: Sedati Gede, Sidoarjo — -7.379°, 112.787° (WARR vicinity)")
st.divider()

# === Konfigurasi dasar ===
LAT, LON = -7.379, 112.787
REFRESH_TTL = 1800  # 30 menit
weights = {"bmkg": 0.45, "ecmwf": 0.25, "icon": 0.15, "gfs": 0.15}

# === Helper: Wind conversion ===
def wind_to_uv(speed, deg):
    theta = radians((270 - deg) % 360)
    return speed * cos(theta), speed * sin(theta)

def uv_to_wind(u, v):
    spd = np.sqrt(u**2 + v**2)
    theta = degrees(atan2(v, u))
    deg = (270 - theta) % 360
    return spd, deg

def weighted_mean(vals, ws):
    arr, w = np.array(vals, dtype=float), np.array(ws, dtype=float)
    mask = ~np.isnan(arr)
    if not mask.any(): return np.nan
    return np.sum(arr[mask] * w[mask]) / np.sum(w[mask])

# === Fetcher BMKG ===
@st.cache_data(ttl=REFRESH_TTL)
def fetch_bmkg():
    url = "https://cuaca.bmkg.go.id/api/df/v1/forecast/adm"
    params = {"adm1":"35","adm2":"35.15","adm3":"35.15.17","adm4":"35.15.17.2001"}
    try:
        r = requests.get(url, params=params, timeout=20, verify=False)
        data = r.json()
        cuaca = data["data"][0]["cuaca"][0][0]
        return cuaca
    except Exception:
        st.warning("BMKG API gagal, mencoba offline fallback...")
        return None

# === Fetcher OpenMeteo (GFS/ECMWF/ICON) ===
def fetch_openmeteo_model(model):
    base = f"https://api.open-meteo.com/v1/{model}"
    params = {
        "latitude": LAT, "longitude": LON,
        "hourly": "temperature_2m,relative_humidity_2m,cloud_cover,"
                  "windspeed_10m,winddirection_10m",
        "forecast_days": 2, "timezone": "UTC"
    }
    try:
        r = requests.get(base, params=params, timeout=15)
        return r.json()
    except Exception:
        st.error(f"Gagal akses Open-Meteo {model.upper()}")
        return None

# === Ambil semua data ===
bmkg_data = fetch_bmkg()
gfs = fetch_openmeteo_model("gfs")
ecmwf = fetch_openmeteo_model("ecmwf")
icon = fetch_openmeteo_model("icon")

# === Parsing Open-Meteo ke DataFrame ===
def om_to_df(j, model_name):
    if not j or "hourly" not in j: return None
    h = j["hourly"]
    df = pd.DataFrame({
        "time": pd.to_datetime(h["time"]),
        f"T_{model_name}": h["temperature_2m"],
        f"RH_{model_name}": h["relative_humidity_2m"],
        f"CC_{model_name}": h["cloud_cover"],
        f"WS_{model_name}": h["windspeed_10m"],
        f"WD_{model_name}": h["winddirection_10m"]
    })
    return df

dfs = []
if gfs: dfs.append(om_to_df(gfs, "GFS"))
if ecmwf: dfs.append(om_to_df(ecmwf, "ECMWF"))
if icon: dfs.append(om_to_df(icon, "ICON"))

# === Gabungkan waktu acuan utama ===
df_all = dfs[0][["time"]].copy() if dfs else pd.DataFrame()
for d in dfs:
    df_all = pd.merge(df_all, d, on="time", how="outer")
df_all = df_all.sort_values("time").reset_index(drop=True)

# === Tambahkan BMKG (approx) ===
if bmkg_data:
    # ambil nilai sekitar jam UTC
    times, t, rh, cc, ws, wd = [], [], [], [], [], []
    for c in bmkg_data:
        times.append(datetime.fromisoformat(c["datetime"].replace("Z","+00:00")))
        t.append(c.get("t")); rh.append(c.get("hu")); cc.append(c.get("tcc"))
        ws.append(c.get("ws")); wd.append(c.get("wd_deg"))
    df_bmkg = pd.DataFrame({
        "time": times, "T_BMKG": t, "RH_BMKG": rh, "CC_BMKG": cc, "WS_BMKG": ws, "WD_BMKG": wd
    })
    df_all = pd.merge(df_all, df_bmkg, on="time", how="outer")

# === Hitung fusi numerik ===
fused_rows = []
for i, r in df_all.iterrows():
    # temperatur, kelembapan, cloud cover
    T_vals, RH_vals, CC_vals = [], [], []
    WS_vals, WD_vals, WU_vals, WV_vals = [], [], [], []
    srcs, w = [], []
    for key, wt in weights.items():
        if key.upper() in ["GFS","ECMWF","ICON"]:
            p = key.upper()
        elif key=="bmkg": p="BMKG"
        else: continue
        if pd.notna(r.get(f"T_{p}")):
            T_vals.append(r[f"T_{p}"]); RH_vals.append(r[f"RH_{p}"]); CC_vals.append(r[f"CC_{p}"])
            ws = r[f"WS_{p}"]; wd = r[f"WD_{p}"]
            if pd.notna(ws) and pd.notna(wd):
                u,v = wind_to_uv(ws,wd); WU_vals.append(u); WV_vals.append(v)
            w.append(wt); srcs.append(p)
    if not w: continue
    T_f = weighted_mean(T_vals, w)
    RH_f = weighted_mean(RH_vals, w)
    CC_f = weighted_mean(CC_vals, w)
    U_f = weighted_mean(WU_vals, w)
    V_f = weighted_mean(WV_vals, w)
    WS_f, WD_f = uv_to_wind(U_f, V_f)
    fused_rows.append([r["time"], T_f, RH_f, CC_f, WS_f, WD_f])

df_fused = pd.DataFrame(fused_rows, columns=["time","T","RH","CC","WS","WD"])
df_fused = df_fused[df_fused["time"] >= datetime.utcnow()].head(24)

# === Visualisasi ===
st.subheader("📈 Grafik Fusi Multi-Model 24 jam (Sedati Gede)")
fig, ax1 = plt.subplots()
ax1.plot(df_fused["time"], df_fused["T"], label="Temp (°C)")
ax1.plot(df_fused["time"], df_fused["RH"], label="RH (%)")
ax1.plot(df_fused["time"], df_fused["CC"], label="Cloud (%)")
ax1.plot(df_fused["time"], df_fused["WS"], label="Wind Speed (kt)")
ax1.set_ylabel("Nilai")
ax1.set_xlabel("Waktu (UTC)")
plt.xticks(rotation=45)
plt.legend()
st.pyplot(fig)

# === Analisis cepat ===
st.markdown("### 🧠 Analisis Otomatis (Fusi 24 jam)")
try:
    now = df_fused.iloc[0]
    signif = ""
    if now.CC > 80 and now.RH > 85:
        signif = "Potensi hujan / konveksi ringan"
    elif now.CC < 30 and now.RH < 60:
        signif = "Cerah dan kering"
    else:
        signif = "Berawan normal, kondisi stabil"
    st.markdown(f"""
    <div style='padding:10px;border:2px solid #888;border-radius:10px;background:#f6f6f6'>
    <b>Jam awal:</b> {now.time:%Y-%m-%d %H:%M UTC}<br>
    <b>T:</b> {now.T:.1f}°C, <b>RH:</b> {now.RH:.0f}%, <b>Awan:</b> {now.CC:.0f}%, <b>Angin:</b> {now.WD:.0f}°/{now.WS:.1f} kt<br>
    <b>Interpretasi:</b> {signif}
    </div>
    """, unsafe_allow_html=True)
except Exception:
    st.info("Belum ada data fusi yang bisa dianalisis.")

st.info("💡 Data fusi: BMKG + GFS + ECMWF + ICON (Open-Meteo). Gunakan hasil ini untuk analisis lokal atau TAFOR otomatis.")
