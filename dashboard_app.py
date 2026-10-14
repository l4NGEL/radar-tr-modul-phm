# =============================================================
# dashboard_app.py -- T/R modulu canli saglik izleme paneli (Streamlit)
#
# 00-03'teki ayri ayri script/model cikartilarini tek bir "operator ekrani"
# altinda birlestirir: telemetri + tespit + teshis + yanlis-alarm-olasiligi +
# RUL, hepsi ayni ekranda, zaman ekseninde gezinilebilir.
#
# Calistirmak icin: streamlit run dashboard_app.py
# (00, 01, 02, 03 daha once en az bir kez calistirilmis olmali -- sonuclar/
#  ve veri/ klasorlerindeki dosyalari okuyor, kendi basina veri uretmiyor.)
# =============================================================

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

BASE = Path(__file__).parent
VERI_DIR = BASE / "veri"
SONUC_DIR = BASE / "sonuclar"
MODEL_DIR = SONUC_DIR / "models"

st.set_page_config(page_title="T/R Modulu Saglik Izleme", layout="wide")


@st.cache_data
def veri_yukle():
    telemetri = pd.read_csv(VERI_DIR / "telemetri_ozellikli.csv")
    modul_tablosu = pd.read_csv(VERI_DIR / "modul_tablosu.csv")
    meta = json.loads((VERI_DIR / "meta.json").read_text(encoding="utf-8"))
    return telemetri, modul_tablosu, meta


@st.cache_resource
def modelleri_yukle():
    return {
        "rf_teshis": joblib.load(MODEL_DIR / "rf_teshis.joblib"),
        "yanlis_alarm": joblib.load(MODEL_DIR / "yanlis_alarm_model.joblib"),
        "ozellik_meta": joblib.load(MODEL_DIR / "ozellik_meta.joblib"),
    }


eksik = [p for p in [
    VERI_DIR / "telemetri_ozellikli.csv", VERI_DIR / "modul_tablosu.csv", VERI_DIR / "meta.json",
    MODEL_DIR / "rf_teshis.joblib", MODEL_DIR / "yanlis_alarm_model.joblib", MODEL_DIR / "ozellik_meta.joblib",
] if not p.exists()]
if eksik:
    st.error("Once pipeline'i calistir: `python 00_tr_modul_simulasyonu.py`, `01_...py`, `02_...py`.\n\n"
              "Eksik dosyalar:\n" + "\n".join(f"- {p}" for p in eksik))
    st.stop()

telemetri, modul_tablosu, meta = veri_yukle()
modeller = modelleri_yukle()
FEATURES = meta["features"]
FEATURE_COLS = modeller["ozellik_meta"]["feature_cols"]
Z_KOLONLARI = modeller["ozellik_meta"]["z_kolonlari"]
TREND_KOLONLARI = modeller["ozellik_meta"]["trend_kolonlari"]
nominal_ort = modeller["ozellik_meta"]["ort"]
nominal_std = modeller["ozellik_meta"]["std"]
N_ORNEK = meta["n_ornek"]

# %% Baslik
st.title("T/R Modulu Sagligi Izleme Paneli")
st.caption("radar-tr-modul-phm -- sentetik AESA T/R modulu BIT telemetrisi uzerinde tespit / teshis / kestirimci bakim demosu")

# %% Kontroller (ust bar -- st.sidebar degil, bazi tarayici/preview ortamlarinda
# gomulu iframe icinde sidebar guvenilir acilmayabiliyor; ust bar her yerde calisir)
modul_secenekleri = modul_tablosu.sort_values("modul_id")
col_secim, col_zaman = st.columns([1, 2])
with col_secim:
    secili_modul = st.selectbox(
        "Modul",
        modul_secenekleri["modul_id"],
        format_func=lambda mid: f"Modul {mid:02d}  --  gercek: {modul_secenekleri.set_index('modul_id').loc[mid, 'ariza_modu']}",
    )
with col_zaman:
    zaman_adimi = st.slider("Zaman adimi (izleme ani)", 0, N_ORNEK - 1, N_ORNEK - 1)
st.caption(
    "Gercek etiket secim listesinde gorunuyor cunku bu bir PORTFOY demosu -- "
    "amac modelin dogrulugunu seffaf gostermek. Gercek bir operator ekraninda "
    "bu bilgi elbette onceden bilinmez."
)
st.divider()

modul_df = telemetri[telemetri["modul_id"] == secili_modul].sort_values("zaman_adimi").reset_index(drop=True)
satir_index = modul_df[modul_df["zaman_adimi"] == zaman_adimi].index
if len(satir_index) == 0:
    st.stop()
i = satir_index[0]
satir = modul_df.iloc[i]
gecmis = modul_df.iloc[: i + 1]

modul_bilgi = modul_tablosu.set_index("modul_id").loc[secili_modul]
gercek_ariza_modu = modul_bilgi["ariza_modu"]
t_onset = modul_bilgi["t_onset"]

# %% Durum banner'i
anomali_var = bool(satir["tespit_rf"])
col_durum, col_teshis, col_alarm, col_rul = st.columns(4)

with col_durum:
    if anomali_var:
        st.error(f"ANOMALI TESPIT EDILDI\n\nModul {secili_modul:02d} / t={zaman_adimi}")
    else:
        st.success(f"SAGLIKLI\n\nModul {secili_modul:02d} / t={zaman_adimi}")

# %% Teshis (hangi bilesen/ariza)
teshis_olasilik = None
if anomali_var:
    X = satir[FEATURE_COLS].to_frame().T.astype(float)
    olasiliklar = modeller["rf_teshis"].predict_proba(X)[0]
    siniflar = modeller["rf_teshis"].classes_
    teshis_olasilik = dict(zip(siniflar, olasiliklar))
    en_olasi = max(teshis_olasilik, key=teshis_olasilik.get)
    with col_teshis:
        st.metric("Teshis (en olasi)", en_olasi, f"{teshis_olasilik[en_olasi]:.0%} olasilik")
else:
    with col_teshis:
        st.metric("Teshis", "-- (anomali yok)")

# %% Yanlis alarm olasiligi -- su ana kadarki ardisik tespit suresine bakarak
if anomali_var:
    bayrak = gecmis["tespit_rf"].to_numpy()
    sure = 1
    for v in bayrak[-2::-1]:
        if v:
            sure += 1
        else:
            break
    pencere = gecmis.iloc[-sure:]
    buyukluk = pencere[Z_KOLONLARI].abs().max(axis=1).mean()
    kac_kanal = (pencere[Z_KOLONLARI].abs() > 3.0).any(axis=0).sum()
    olay_ozellik = pd.DataFrame([{"sure": sure, "buyukluk": buyukluk, "kac_kanal": kac_kanal, "sonraki_egilim": 0.0}])
    ya_olasilik = modeller["yanlis_alarm"].predict_proba(olay_ozellik)[0]
    ya_siniflar = modeller["yanlis_alarm"].classes_  # [False, True] -> [yanlis alarm, gercek ariza]
    gercek_ariza_olasiligi = float(ya_olasilik[list(ya_siniflar).index(True)]) if True in list(ya_siniflar) else float("nan")
    with col_alarm:
        st.metric("Gercek ariza olasiligi", f"{gercek_ariza_olasiligi:.0%}",
                   help="Yanlis alarm siniflandiricisinin, bu olayin gecici bir "
                        "sicramadan degil gercek bir arizadan kaynaklandigina dair tahmini. "
                        "Canli izlemede 'olay bittikten sonraki egilim' ozelligi henuz "
                        "gozlenemedigi icin notr (0) varsayiliyor -- bu yuzden canli tahmin, "
                        "02_...py'deki geriye-donuk (retrospektif) degerlendirmeden biraz "
                        "daha dusuk kesinlikte olabilir.")
else:
    with col_alarm:
        st.metric("Gercek ariza olasiligi", "--")

# %% RUL (sadece kademeli arizalarda, onset sonrasinda anlamli)
KADEMELI = set(meta["kademeli_arizalar"])
if gercek_ariza_modu in KADEMELI and pd.notna(t_onset) and zaman_adimi > t_onset + 20:
    kanal = meta["rul_birincil_kanal"][gercek_ariza_modu]
    tam_sapma = meta["rul_tam_bozulma_sapmasi"][gercek_ariza_modu]
    ramp_uzunluk = meta["ramp_uzunluklari"][gercek_ariza_modu]
    kritik_ramp_orani = meta["kritik_ramp_orani"]
    kritik_deger = nominal_ort[kanal] + tam_sapma * kritik_ramp_orani

    egitim_pencere = gecmis[(gecmis["zaman_adimi"] >= t_onset)]
    if len(egitim_pencere) >= 5:
        X = egitim_pencere[["zaman_adimi"]].to_numpy()
        y = egitim_pencere[kanal].to_numpy()
        egim, kesisim = np.polyfit(X.flatten(), y, 1)
        if abs(egim) > 1e-4:
            kesisim_zamani = (kritik_deger - kesisim) / egim
            tahmini_rul = kesisim_zamani - zaman_adimi
            with col_rul:
                if tahmini_rul > 0:
                    st.metric("Kalan omur (RUL)", f"{tahmini_rul:.0f} ornek",
                               help=f"'{kanal}' kanalinin egilimine dogrusal ekstrapolasyon")
                else:
                    st.metric("Kalan omur (RUL)", "kritik esik asildi")
        else:
            with col_rul:
                st.metric("Kalan omur (RUL)", "trend henuz belirsiz")
    else:
        with col_rul:
            st.metric("Kalan omur (RUL)", "yetersiz veri")
else:
    with col_rul:
        st.metric("Kalan omur (RUL)", "-- (kademeli ariza degil / henuz erken)")

st.divider()

# %% Telemetri grafikleri
st.subheader("Telemetri Kanallari")
fig, eksenler = plt.subplots(2, 3, figsize=(15, 6))
for ax, f in zip(eksenler.flat, FEATURES):
    ax.plot(gecmis["zaman_adimi"], gecmis[f], color="#2563eb", linewidth=1.2)
    ort, std = nominal_ort[f], nominal_std[f]
    ax.axhspan(ort - 3 * std, ort + 3 * std, color="#22c55e", alpha=0.12, label="nominal bant (+-3 sigma)")
    if pd.notna(t_onset):
        ax.axvline(t_onset, color="#ef4444", linestyle=":", linewidth=1.0, label="gercek ariza baslangici")
    ax.axvline(zaman_adimi, color="#111827", linestyle="-", linewidth=0.8, alpha=0.5)
    ax.set_title(f, fontsize=10)
    ax.set_xlim(0, N_ORNEK - 1)
    ax.tick_params(labelsize=8)
handles, labels = eksenler.flat[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, 1.02))
fig.tight_layout()
st.pyplot(fig)

st.caption(
    f"Gercek etiket: **{gercek_ariza_modu}**"
    + (f" (baslangic t={int(t_onset)})" if pd.notna(t_onset) else "")
    + f" | Kural-tabanli tespit: {'ANOMALI' if bool(satir['tespit_kural']) else 'saglikli'}"
    + f" | IsolationForest: {'ANOMALI' if bool(satir['tespit_iforest']) else 'saglikli'}"
    + f" | RandomForest: {'ANOMALI' if bool(satir['tespit_rf']) else 'saglikli'}"
)

if anomali_var and teshis_olasilik:
    st.subheader("Teshis olasilik dagilimi")
    teshis_df = pd.DataFrame({"Ariza modu": list(teshis_olasilik.keys()), "Olasilik": list(teshis_olasilik.values())})
    st.bar_chart(teshis_df.set_index("Ariza modu"))
