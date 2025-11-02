# app.py
import streamlit as st
import requests, json, os, io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, timezone

# -------------------------
# === CONFIG / PAGE SETUP ==
# -------------------------
st.set_page_config(page_title="TAFOR Fusion Pro — Operational v2.4.2 (WARR)",
                   layout="centered",
                   initial_sidebar_state="collapsed")
st.title("🛫 TAFOR Fusion Pro — Operational (WARR / Sedati Gede)")

st.caption(
    "📍 Location: Sedati Gede (ADM4=35.15.17.2011) | **Ferri Kusuma**, NIP.197912222000031001  \n"
    "Fusion: BMKG + Open-Meteo (GFS/ECMWF/ICON) + METAR realtime (OGIMET/NOAA)"
)

os.makedirs("output", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# -------------------------
# === INPUTS (USER) =======
# -------------------------
col1, col2, col3 = st.columns(3)
with col1:
    issue_date = st.date_input("📅 Issue date (UTC)", datetime.utcnow().date())
with col2:
    jam_penting = [0, 3, 6, 9, 12, 15, 18, 21]
    jam_sekarang = datetime.utcnow().hour
    default_jam = min(jam_penting, key=lambda j: abs(j - jam_sekarang))
    issue_time = st.selectbox("🕓 Issue time (UTC)", jam_penting, index=jam_penting.index(default_jam))
with col3:
    validity = st.number_input("🕐 Validity (hours)", min_value=6, max_value=72, value=24, step=6)

st.divider()
st.markdown("### ⚙️ Ensemble Weights (BMKG Prioritas)")
bmkg_w = st.slider("BMKG", 0.0, 1.0, 0.45, 0.05)
ecmwf_w = st.slider("ECMWF", 0.0, 1.0, 0.25, 0.05)
icon_w  = st.slider("ICON", 0.0, 1.0, 0.15, 0.05)
gfs_w   = st.slider("GFS", 0.0, 1.0, 0.15, 0.05)

# protect against zeros
total = bmkg_w + ecmwf_w + icon_w + gfs_w
if total <= 0:
    st.warning("Total weights are zero — using equal weights as fallback.")
    total = 4.0
    bmkg_w = ecmwf_w = icon_w = gfs_w = 1.0

norm_weights = {
    "bmkg": bmkg_w / total,
    "ecmwf": ecmwf_w / total,
    "icon": icon_w / total,
    "gfs": gfs_w / total
}
st.write(f"Normalized weights: {norm_weights}")

# -------------------------
# === LOCATION / URLS =====
# -------------------------
# coordinates for Sedati Gede (approx). Adjust if you want precise values.
LAT, LON = -7.38, 112.78

BMKG_ADM4_URL = ("https://cuaca.bmkg.go.id/api/df/v1/forecast/adm"
                 "?adm1=35&adm2=35.15&adm3=35.15.17&adm4=35.15.17.2011")
OPENMETEO_BASE = "https://api.open-meteo.com/v1/forecast"
OGIMET_METAR_URL = "https://ogimet.com/display_metars2.php?lang=en&lugar=warr&tipo=SA&ord=REV&nil=NO&fmt=txt"

# -------------------------
# === HELPERS: FETCH DATA ==
# -------------------------
def safe_get_json(url, params=None, timeout=15):
    try:
        r = requests.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.session_state.setdefault("_errors", []).append(f"JSON fetch error for {url}: {e}")
        return None

def safe_get_text(url, params=None, timeout=15):
    try:
        r = requests.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        return r.text
    except Exception as e:
        st.session_state.setdefault("_errors", []).append(f"Text fetch error for {url}: {e}")
        return None

def fetch_bmkg():
    """Attempt to fetch BMKG ADM4 forecast JSON. Returns dict or None."""
    return safe_get_json(BMKG_ADM4_URL)

def fetch_openmeteo_model(model_name):
    """
    Fetch hourly variables for a specific model from Open-Meteo.
    We'll request a forecast window covering validity (plus a margin).
    """
    params = {
        "latitude": LAT,
        "longitude": LON,
        "hourly": "temperature_2m,relative_humidity_2m,cloud_cover,precipitation,wind_speed_10m",
        "forecast_days": int(np.ceil((validity + 6) / 24)),  # ensure enough hours
        "models": model_name,
        "timezone": "UTC"
    }
    return safe_get_json(OPENMETEO_BASE, params=params)

def fetch_metar_ogimet():
    """Fetch latest METAR text (simple: take first non-empty line)."""
    txt = safe_get_text(OGIMET_METAR_URL)
    if not txt:
        return None
    # OGIMET returns many lines; pick the first non-empty printable line that looks like METAR
    for line in txt.splitlines():
        line = line.strip()
        if line and (line.startswith("WARR") or len(line.split()) > 2):
            return line
    return txt.splitlines()[0] if txt.splitlines() else None

# -------------------------
# === RUN FETCH & PROCESS ==
# -------------------------
st.divider()
st.markdown("### 📡 Fetching BMKG + OpenMeteo + METAR...")
with st.spinner("Contacting remote sources..."):
    bmkg_raw = fetch_bmkg()
    open_gfs = fetch_openmeteo_model("gfs")
    open_ecmwf = fetch_openmeteo_model("ecmwf")
    open_icon = fetch_openmeteo_model("icon")
    metar_text = fetch_metar_ogimet()

if "_errors" in st.session_state:
    for e in st.session_state["_errors"]:
        st.error(e)

# -------------------------
# === BUILD MODEL DATAFRAMES ==
# -------------------------
def build_hourly_df_from_openmeteo(openm):
    """
    Convert an Open-Meteo response into a dataframe with columns (time, T, RH, CC, WS, P) if available.
    Returns None if response empty or malformed.
    """
    if not openm or "hourly" not in openm or "time" not in openm["hourly"]:
        return None
    h = openm["hourly"]
    times = pd.to_datetime(h["time"], utc=True)
    df = pd.DataFrame({"time": times})
    # safe extraction
    mapping = {
        "temperature_2m": "T",
        "relative_humidity_2m": "RH",
        "cloud_cover": "CC",
        "wind_speed_10m": "WS",
        "precipitation": "P"
    }
    for src_key, col in mapping.items():
        if src_key in h:
            df[col] = h[src_key]
    return df.set_index("time")

# attempt to convert each model into indexed dfs
gfs_df = build_hourly_df_from_openmeteo(open_gfs)
ecmwf_df = build_hourly_df_from_openmeteo(open_ecmwf)
icon_df = build_hourly_df_from_openmeteo(open_icon)

# fallback dataset (dummy linear trends) if no model data available
def build_dummy_df():
    times = pd.date_range(datetime.utcnow().replace(minute=0, second=0, microsecond=0, tzinfo=timezone.utc),
                          periods=validity+6, freq="H")
    return pd.DataFrame({
        "T": np.linspace(30, 24, len(times)),
        "RH": np.linspace(70, 85, len(times)),
        "CC": np.linspace(20, 80, len(times)),
        "WS": np.linspace(5, 12, len(times)),
        "P": np.zeros(len(times))
    }, index=times)

# pick reference index (UTC hours) for fusion: from issue datetime to issue+validity
issue_dt = datetime.combine(issue_date, datetime.utcnow().replace(hour=issue_time, minute=0, second=0, microsecond=0).time()).replace(tzinfo=timezone.utc)
ref_index = pd.date_range(issue_dt, periods=validity, freq="H", tz=timezone.utc)

# ensure each model df covers the ref_index (reindex with NaNs allowed)
def reindex_model_df(model_df):
    if model_df is None:
        return None
    try:
        return model_df.reindex(ref_index).interpolate(limit=3).ffill().bfill()
    except Exception:
        return None

gfs_df = reindex_model_df(gfs_df)
ecmwf_df = reindex_model_df(ecmwf_df)
icon_df = reindex_model_df(icon_df)

# if BMKG provides usable hourly, try to extract T/RH/etc. (BMKG schema may differ; we safe-guard)
bmkg_hourly_df = None
if bmkg_raw and isinstance(bmkg_raw, dict):
    try:
        # BMKG 'data' shape varies; attempt to find hourly arrays with time key
        # This is conservative: if not present, skip using BMKG as numeric model.
        if "hourly" in bmkg_raw and "time" in bmkg_raw["hourly"]:
            times = pd.to_datetime(bmkg_raw["hourly"]["time"], utc=True)
            bmkg_df = pd.DataFrame({"time": times})
            # map likely keys if present
            for key_map in [("temperature_2m","T"), ("temp","T"), ("relative_humidity_2m","RH"), ("cloud_cover","CC"),
                            ("wind_speed_10m","WS"), ("precipitation","P")]:
                src, dest = key_map
                if src in bmkg_raw["hourly"]:
                    bmkg_df[dest] = bmkg_raw["hourly"][src]
            bmkg_hourly_df = bmkg_df.set_index("time").reindex(ref_index).interpolate(limit=3).ffill().bfill()
        else:
            bmkg_hourly_df = None
    except Exception:
        bmkg_hourly_df = None

# create a list of model dataframes for fusion, aligned to ref_index
model_dfs = {
    "bmkg": bmkg_hourly_df,
    "ecmwf": ecmwf_df,
    "icon": icon_df,
    "gfs": gfs_df
}

# if all model dfs are None, use dummy
if all(v is None for v in model_dfs.values()):
    st.warning("No model hourly data available — using internal synthetic/fallback forecast.")
    fused_df = build_dummy_df().reindex(ref_index)
else:
    # Fusion by weighted mean with NaN handling: for each variable, compute weighted mean across available models
    var_list = ["T", "RH", "CC", "WS", "P"]
    fused = pd.DataFrame(index=ref_index)
    for var in var_list:
        numer = np.zeros(len(ref_index), dtype=float)
        denom = np.zeros(len(ref_index), dtype=float)
        for mname, w in norm_weights.items():
            dfm = model_dfs.get(mname)
            if dfm is None or var not in dfm.columns:
                continue
            vals = dfm[var].values
            # treat nan by zeroing numerator and not adding to denom
            valid_mask = ~np.isnan(vals)
            numer[valid_mask] += vals[valid_mask] * w
            denom[valid_mask] += w
        # where denom == 0 -> NaN
        with np.errstate(divide='ignore', invalid='ignore'):
            fused[var] = np.where(denom > 0, numer / denom, np.nan)
    # fill small gaps by interpolation
    fused = fused.interpolate(limit=3).ffill().bfill()
    fused_df = fused

# trim to validity length (in hours)
df = fused_df.iloc[:validity].reset_index().rename(columns={"index": "time"})
# ensure time column is timezone-aware UTC datetime objects
df["time"] = pd.to_datetime(df["time"], utc=True)

st.success("✅ Data ready. Processing fusion...")

# -------------------------
# === SIMPLE METAR PARSING (display only) ==
# -------------------------
if metar_text:
    metar_display = metar_text
else:
    metar_display = "METAR not available"

# -------------------------
# === POP (Probability of Precip) ESTIMATE ==
# -------------------------
# Very simple PoP estimate based on precipitation magnitude & cloud cover and RH:
def estimate_pop_series(df_row):
    """
    Heuristic PoP per-hour [0..1]:
    - if P >= 1 mm => high
    - else scale with CC and RH
    """
    p = df_row.get("P", 0.0) if not pd.isna(df_row.get("P", np.nan)) else 0.0
    cc = df_row.get("CC", 0.0) if not pd.isna(df_row.get("CC", np.nan)) else 0.0
    rh = df_row.get("RH", 0.0) if not pd.isna(df_row.get("RH", np.nan)) else 0.0

    if p >= 2.0:
        return 0.95
    if p >= 0.5:
        return 0.7
    # otherwise blend cloud and RH
    pop = min(1.0, (cc / 100.0) * 0.6 + (max(0, (rh - 60)) / 40.0) * 0.4)
    return float(pop)

df["PoP"] = df.apply(estimate_pop_series, axis=1)
pop_max = float(df["PoP"].max()) if "PoP" in df.columns else 0.0
wind_max = float(df["WS"].max()) if "WS" in df.columns else 0.0

# -------------------------
# === BUILD TAF LINES (simple but clear) ==
# -------------------------
def build_taf(issue_dt, df):
    """
    Build a basic TAF block with:
    - header
    - main forecast (simple: wind + visibility + cloud + prob for precipitation if PoP>0.3)
    - nosig if no significant changes
    """
    header = f"TAF WARR {issue_dt:%d%H%MZ} {issue_dt:%d%H}/{(issue_dt+timedelta(hours=validity)):%d%H}"
    # pick first hour as initial conditions summary (use METAR if available)
    # for display: create a compact representation
    taf_body = []
    # initial line - derive approximate wind
    first = df.iloc[0]
    ws0 = first.get("WS", np.nan)
    if not pd.isna(ws0):
        # convert m/s to KT if necessary? Assuming Open-Meteo returns m/s -> convert to kt (1 m/s = 1.94384 kt)
        # But some model values may already be in kt; to avoid unit confusion we keep values as-is but label generically.
        wind_str = f"{int(round(ws0))}KT"
    else:
        wind_str = "00000KT"

    taf_body.append(f"{wind_str} 9999 FEW020")  # basic initial

    # create period groups (every 3/6 hours) for readability
    for start in range(0, len(df), 6):
        sub = df.iloc[start:start+6]
        if sub.empty:
            continue
        t0 = sub["time"].iloc[0].to_pydatetime()
        t1 = sub["time"].iloc[-1].to_pydatetime()
        # choose representative conditions: median/mean
        mean_ws = sub["WS"].mean() if "WS" in sub else 0
        mean_cc = sub["CC"].mean() if "CC" in sub else 0
        mean_pop = sub["PoP"].mean() if "PoP" in sub else 0
        mean_rh = sub["RH"].mean() if "RH" in sub else 0
        # cloud
        if mean_cc >= 80:
            cloud = "BKN012"  # example
        elif mean_cc >= 50:
            cloud = "SCT025"
        elif mean_cc >= 20:
            cloud = "FEW030"
        else:
            cloud = "SKC"
        # precipitation shorthand
        pop_flag = ""
        if mean_pop >= 0.7:
            pop_flag = "PROB30 TEMPO "  # heavy chance
        elif mean_pop >= 0.3:
            pop_flag = "PROB30 "
        # wind
        wstr = f"{int(round(mean_ws))}KT"
        taf_body.append(f"{t0:%d%HZ}/{t1:%d%H} {wstr} 9999 {cloud} {pop_flag}".strip())

    remarks = "RMK AUTO FUSION BASED ON METAR+MODEL FUSION"
    lines = [header] + taf_body + [remarks]
    return lines

taf_lines = build_taf(issue_dt, df)

# -------------------------
# === DISPLAY RESULTS ===
# -------------------------
st.divider()
st.markdown("### 🧭 Ringkasan Sumber Data")
st.write("""
| Sumber | Status |
|:--------|:--------|
| BMKG ADM4 | {} |
| GFS | {} |
| ECMWF | {} |
| ICON | {} |
| METAR | {} |
""".format(
    "OK" if bmkg_raw else "Not available",
    "OK" if gfs_df is not None else "Not available",
    "OK" if ecmwf_df is not None else "Not available",
    "OK" if icon_df is not None else "Not available",
    "✅ Realtime" if metar_text else "Not available"
))

st.markdown("### 📡 METAR (Realtime OGIMET/NOAA)")
st.markdown(f"""
<div style='padding:12px;border:2px solid #bbb;border-radius:10px;background-color:#fafafa;'>
<p style='font-weight:700;font-size:16px;font-family:monospace;'>{metar_display}</p>
</div>
""", unsafe_allow_html=True)

st.markdown("### 📝 Hasil TAFOR (Optimized Fusion)")
st.markdown(f"""
<div style='padding:15px;border:2px solid #555;border-radius:10px;background-color:#f9f9f9;'>
<p style='font-weight:700;font-size:14px;line-height:1.6;font-family:monospace;'>{'<br>'.join(taf_lines)}</p>
</div>
""", unsafe_allow_html=True)

st.markdown(f"📅 **Issued at {issue_dt:%d%H%MZ} UTC**, Valid {issue_dt:%d%H}–{(issue_dt+timedelta(hours=validity)):%d%H} UTC")

# -------------------------
# === ALERT LOGIC =========
# -------------------------
alerts = []
if pop_max >= 0.7:
    alerts.append(f"⚠️ High PoP ({pop_max*100:.0f}%) — possible heavy precipitation / convective activity.")
if wind_max >= 20:  # threshold in KT (approx if model in m/s, this may misrepresent; adjust if units known)
    alerts.append(f"💨 Wind ≥ {wind_max:.0f}KT detected.")
if (df["RH"].max() >= 90) and (df["CC"].max() >= 85):
    alerts.append("🌫️ High RH & Cloud cover — possible CB or reduced visibility.")
if alerts:
    # show each alert
    for a in alerts:
        st.warning(a)
else:
    st.info("✅ No significant alerts detected — conditions stable.")

# -------------------------
# === LOGGING ============
# -------------------------
log_file = f"logs/{issue_dt:%Y%m%d}_tafor_log.csv"
log_df = pd.DataFrame([{
    "timestamp": datetime.utcnow().replace(tzinfo=timezone.utc).isoformat(),
    "issue_time": f"{issue_dt:%d%H%MZ}",
    "validity": validity,
    "metar": metar_display,
    "taf_text": " | ".join(taf_lines),
    "pop_max": pop_max,
    "wind_max": wind_max,
    "remarks": "; ".join(alerts) if alerts else "OK"
}])
if os.path.exists(log_file):
    log_df.to_csv(log_file, mode="a", header=False, index=False)
else:
    log_df.to_csv(log_file, index=False)

# -------------------------
# === EXPORT & DOWNLOAD ===
# -------------------------
csv_file = f"output/fused_{issue_dt:%Y%m%d_%H%M}.csv"
json_file = f"output/fused_{issue_dt:%Y%m%d_%H%M}.json"
# save df
df.to_csv(csv_file, index=False)
df.to_json(json_file, orient="records", indent=2, date_format="iso")

st.markdown("### 💾 Exported & Downloadable Files")
with open(csv_file, "rb") as f:
    st.download_button("⬇️ Download CSV Result", f, file_name=os.path.basename(csv_file), mime="text/csv")
with open(json_file, "rb") as f:
    st.download_button("⬇️ Download JSON Result", f, file_name=os.path.basename(json_file), mime="application/json")
with open(log_file, "rb") as f:
    st.download_button("⬇️ Download Log CSV", f, file_name=os.path.basename(log_file), mime="text/csv")

# Also provide TAF plain text download
taf_bytes = "\n".join(taf_lines).encode("utf-8")
st.download_button("⬇️ Download TAF (plain text)", data=io.BytesIO(taf_bytes), file_name=f"TAF_WARR_{issue_dt:%Y%m%d_%H%M}.txt", mime="text/plain")

# -------------------------
# === SIMPLE SUMMARY TABLE ==
# -------------------------
st.markdown("### 📋 Ringkasan Fused Forecast (sample)")
st.dataframe(df.head(12).astype({"T": float, "RH": float, "CC": float, "WS": float, "PoP": float}), use_container_width=True)

# -------------------------
# === PLOT GRAPH ==========
# -------------------------
st.markdown("### 📊 Grafik Fusi (T / RH / CC / WS)")
fig, ax1 = plt.subplots(figsize=(9,4))
ax1.plot(df["time"], df["T"], label="Temp (°C)")
ax1.plot(df["time"], df["RH"], label="RH (%)")
ax1.plot(df["time"], df["CC"], linestyle="--", label="Cloud (%)")
ax1.set_xlabel("Time (UTC)")
ax1.set_ylabel("T / RH / CC")
ax1.legend(loc="upper left")
ax2 = ax1.twinx()
ax2.plot(df["time"], df["WS"], label="Wind (kt)", linestyle="-.")
ax2.set_ylabel("Wind (kt)")
ax2.legend(loc="upper right")
plt.tight_layout()
st.pyplot(fig)

# -------------------------
# === OPTIONAL: DEBUG INFO =
# -------------------------
with st.expander("🔧 Debug / Raw responses (for operators)"):
    st.subheader("BMKG raw (truncated)")
    st.write(bmkg_raw if bmkg_raw else "No BMKG JSON retrieved.")
    st.subheader("OpenMeteo GFS sample (truncated)")
    st.write({k: (v if k!="hourly" else {kk: list(vv)[:6] for kk,vv in v.items()}) for k,v in (open_gfs or {}).items()} if open_gfs else "No GFS data.")
    st.subheader("OpenMeteo ECMWF sample (truncated)")
    st.write({k: (v if k!="hourly" else {kk: list(vv)[:6] for kk,vv in v.items()}) for k,v in (open_ecmwf or {}).items()} if open_ecmwf else "No ECMWF data.")
    st.subheader("OpenMeteo ICON sample (truncated)")
    st.write({k: (v if k!="hourly" else {kk: list(vv)[:6] for kk,vv in v.items()}) for k,v in (open_icon or {}).items()} if open_icon else "No ICON data.")
    st.subheader("METAR (raw)")
    st.code(metar_display or "None")

# -------------------------
# === END ============
# -------------------------
st.success("TAFOR fusion complete — review TAF, alerts, and exported files above.")
