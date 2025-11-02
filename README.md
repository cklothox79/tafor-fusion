# 🌀 TAFOR-Fusion

Fusi numerik cuaca real-time untuk Sedati Gede (Juanda – WARR)  
Sumber data:
- BMKG API ADM4 (35.15.17.2001)
- Open-Meteo model endpoints (GFS, ECMWF, ICON)

## Jalankan lokal
```bash
pip install -r requirements.txt
streamlit run app.py

Fungsi utama

Mengambil data 4 model (BMKG, GFS, ECMWF, ICON)

Melakukan fusi berbobot (BMKG prioritas)

Menampilkan hasil numerik dan grafik 24 jam

Menyediakan interpretasi otomatis
