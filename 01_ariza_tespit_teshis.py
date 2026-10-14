# =============================================================
# 01 -- Ariza tespiti (detection) + teshis/lokalizasyon (diagnosis/isolation)
#
# Tespit: "ariza var mi?"  Teshis: "hangi ariza modu?"
# CIT (Cihaz Ici Test / test edilebilirlik) tasarimindaki temel soru "hangi
# olcum, hangi arizayi, ne kadar guvenle ayirt edebiliyor" sorusudur; asagidaki
# teshis siniflandiricisi ve Fault Isolation metrigi tam olarak bunu olcuyor.
#
# Uc farkli tespit yontemi kasten karsilastiriliyor (hangisinin ne zaman
# secilecegini gerekcelendirmek icin):
#   1) Kural-tabanli esik (|z-skor| > 3)      -> en basit, gercek BIT'lerin
#      cogunun ilk katmani, referans/dip nokta.
#   2) IsolationForest (denetimsiz)            -> etiketli ariza gecmisi
#      olmadan da calisir, bilinmeyen/yeni ariza tiplerine genellenebilir.
#   3) RandomForest (denetimli)                 -> etiketli veri varsa en
#      guclu performans, ama sadece gorulmus ariza tiplerini taniyabilir.
# =============================================================

# %% Ayarlar
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

BASE = Path(__file__).parent
VERI_DIR = BASE / "veri"
SONUC_DIR = BASE / "sonuclar"
MODEL_DIR = SONUC_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
Z_ESIK = 3.0       # kural-tabanli BIT esigi
ROLL_PENCERE = 10  # egilim (trend) hesaplamak icin kayan pencere genisligi

meta = json.loads((VERI_DIR / "meta.json").read_text(encoding="utf-8"))
FEATURES = meta["features"]
TREND_KOLONLARI = ["kazanc_dB", "guc_dBm", "sicaklik_C", "faz_gurultusu_dBcHz"]
TRAIN_ID = set(meta["train_modul_id"])
TEST_ID = set(meta["test_modul_id"])

telemetri = pd.read_csv(VERI_DIR / "telemetri.csv")
modul_tablosu = pd.read_csv(VERI_DIR / "modul_tablosu.csv")

# %% Ozellik muhendisligi: nominal z-skor + egilim (trend)
# Nominal ortalama/std SADECE train-split + gercekten saglikli (gecici blip
# de olmayan) satirlardan hesaplaniyor -- gercek bir BIT kalibrasyonunda da
# esikler bilinen-saglam birimlerden turetilir, ariza verisi kalibrasyona
# karismaz.
telemetri = telemetri.sort_values(["modul_id", "zaman_adimi"]).reset_index(drop=True)

nominal_mask = (
    telemetri["modul_id"].isin(TRAIN_ID)
    & (telemetri["gercek_etiket"] == "saglikli")
    & (~telemetri["gecici_anomali"])
)
nominal_ort = telemetri.loc[nominal_mask, FEATURES].mean()
nominal_std = telemetri.loc[nominal_mask, FEATURES].std()

z_kolonlari = []
for f in FEATURES:
    col = f"{f}_z"
    telemetri[col] = (telemetri[f] - nominal_ort[f]) / nominal_std[f]
    z_kolonlari.append(col)

egilim_kolonlari = []
for c in TREND_KOLONLARI:
    roll = telemetri.groupby("modul_id")[c].transform(lambda s: s.rolling(ROLL_PENCERE, min_periods=1).mean())
    egilim = roll.groupby(telemetri["modul_id"]).diff(ROLL_PENCERE).fillna(0.0)
    telemetri[f"{c}_egilim"] = egilim
    egilim_kolonlari.append(f"{c}_egilim")

FEATURE_COLS = z_kolonlari + egilim_kolonlari
telemetri["anomali_var"] = (telemetri["gercek_etiket"] != "saglikli") | telemetri["gecici_anomali"]

train_df = telemetri[telemetri["modul_id"].isin(TRAIN_ID)]
test_df = telemetri[telemetri["modul_id"].isin(TEST_ID)]

# %% 1) Kural-tabanli tespit
telemetri["tespit_kural"] = (telemetri[z_kolonlari].abs() > Z_ESIK).any(axis=1)

# %% 2) IsolationForest (denetimsiz) -- sadece saglam train orneginde egitilir
iforest_egitim = train_df.loc[
    (train_df["gercek_etiket"] == "saglikli") & (~train_df["gecici_anomali"]), FEATURE_COLS
]
iforest = IsolationForest(n_estimators=200, contamination=0.08, random_state=SEED)
iforest.fit(iforest_egitim)
telemetri["tespit_iforest"] = iforest.predict(telemetri[FEATURE_COLS]) == -1

# %% 3) RandomForest (denetimli) tespit
rf_tespit = RandomForestClassifier(n_estimators=300, random_state=SEED, class_weight="balanced", n_jobs=-1)
rf_tespit.fit(train_df[FEATURE_COLS], train_df["anomali_var"])
telemetri["tespit_rf"] = rf_tespit.predict(telemetri[FEATURE_COLS])

# train_df/test_df, tespit_* kolonlari eklenmeden ONCE alinmisti (egitim icin
# yeterliydi) -- asagida bu kolonlari da icermesi gerektigi icin tazeleniyor.
train_df = telemetri[telemetri["modul_id"].isin(TRAIN_ID)]
test_df = telemetri[telemetri["modul_id"].isin(TEST_ID)]

# %% Teshis / lokalizasyon (CIT): sadece gercek ariza orneklerinde, hangi ariza
train_fault = train_df[train_df["gercek_etiket"] != "saglikli"]
test_fault = test_df[test_df["gercek_etiket"] != "saglikli"]

rf_teshis = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)
rf_teshis.fit(train_fault[FEATURE_COLS], train_fault["gercek_etiket"])
teshis_tahmin = rf_teshis.predict(test_fault[FEATURE_COLS])

teshis_dogruluk = accuracy_score(test_fault["gercek_etiket"], teshis_tahmin)
etiket_siralama = sorted(test_fault["gercek_etiket"].unique())
cm = confusion_matrix(test_fault["gercek_etiket"], teshis_tahmin, labels=etiket_siralama)


# %% Test edilebilirlik metrikleri: FDC, FAR, MTTD (her tespit yontemi icin)
def metrikleri_hesapla(detector_kolonu: str):
    gercek_ariza = test_df["gercek_etiket"] != "saglikli"
    gercek_nominal = (test_df["gercek_etiket"] == "saglikli") & (~test_df["gecici_anomali"])
    gecici_blip = test_df["gecici_anomali"]

    # FDC = gercek ariza orneklerinin ne kadari yakalandi (recall, pozitif sinif = ariza)
    fdc = float((test_df.loc[gercek_ariza, detector_kolonu]).mean()) if gercek_ariza.any() else float("nan")
    far = float((test_df.loc[gercek_nominal, detector_kolonu]).mean()) if gercek_nominal.any() else float("nan")
    blip_yakalama = float((test_df.loc[gecici_blip, detector_kolonu]).mean()) if gecici_blip.any() else float("nan")

    # MTTD: her arizali test modulunde, t_onset'ten ilk dogru tespite kadar gecen sure
    gecikmeler = []
    for _, satir in modul_tablosu[modul_tablosu["modul_id"].isin(TEST_ID) & (modul_tablosu["ariza_modu"] != "saglikli")].iterrows():
        alt = test_df[(test_df["modul_id"] == satir["modul_id"]) & (test_df["zaman_adimi"] >= satir["t_onset"])]
        tespit_edilenler = alt[alt[detector_kolonu]]
        if len(tespit_edilenler) > 0:
            gecikmeler.append(int(tespit_edilenler["zaman_adimi"].iloc[0] - satir["t_onset"]))
    mttd = float(np.mean(gecikmeler)) if gecikmeler else float("nan")
    kacirilan = int((modul_tablosu["modul_id"].isin(TEST_ID) & (modul_tablosu["ariza_modu"] != "saglikli")).sum() - len(gecikmeler))

    return {
        "fault_detection_coverage": round(fdc, 4),
        "false_alarm_rate": round(far, 4),
        "gecici_blip_yakalama_orani": round(blip_yakalama, 4),
        "mean_time_to_detect_ornek": round(mttd, 2) if not np.isnan(mttd) else None,
        "kacirilan_arizali_modul_sayisi": kacirilan,
    }


metrikler = {
    "kural_tabanli": metrikleri_hesapla("tespit_kural"),
    "isolation_forest": metrikleri_hesapla("tespit_iforest"),
    "random_forest": metrikleri_hesapla("tespit_rf"),
    "teshis_lokalizasyon": {
        "fault_isolation_accuracy": round(float(teshis_dogruluk), 4),
        "test_ariza_ornegi_sayisi": int(len(test_fault)),
        "siniflar": etiket_siralama,
        "confusion_matrix": cm.tolist(),
    },
}

# %% Kaydet
(SONUC_DIR / "01_tespit_teshis_metrikleri.json").write_text(
    json.dumps(metrikler, ensure_ascii=False, indent=2), encoding="utf-8"
)
telemetri.to_csv(VERI_DIR / "telemetri_ozellikli.csv", index=False)
joblib.dump(rf_tespit, MODEL_DIR / "rf_tespit.joblib")
joblib.dump(rf_teshis, MODEL_DIR / "rf_teshis.joblib")
joblib.dump(iforest, MODEL_DIR / "iforest.joblib")
joblib.dump({"ort": nominal_ort, "std": nominal_std, "feature_cols": FEATURE_COLS,
             "z_kolonlari": z_kolonlari, "egilim_kolonlari": egilim_kolonlari,
             "trend_kolonlari": TREND_KOLONLARI, "roll_pencere": ROLL_PENCERE},
            MODEL_DIR / "ozellik_meta.joblib")

# %% Ozet
print("=== Tespit metrikleri (test split) ===")
for yontem, m in metrikler.items():
    if yontem == "teshis_lokalizasyon":
        continue
    print(f"{yontem:18s} FDC={m['fault_detection_coverage']:.2%}  FAR={m['false_alarm_rate']:.2%}  "
          f"gecici_yakalama={m['gecici_blip_yakalama_orani']:.2%}  MTTD={m['mean_time_to_detect_ornek']} ornek  "
          f"kacirilan={m['kacirilan_arizali_modul_sayisi']}")
print(f"\n=== Teshis / lokalizasyon (CIT) ===")
print(f"Fault Isolation Accuracy: {teshis_dogruluk:.2%} ({len(test_fault)} test ornegi, {len(etiket_siralama)} sinif)")
print(f"Confusion matrix ({etiket_siralama}):\n{cm}")
print(f"\nYazildi: {SONUC_DIR / '01_tespit_teshis_metrikleri.json'}, veri/telemetri_ozellikli.csv, sonuclar/models/*.joblib")
