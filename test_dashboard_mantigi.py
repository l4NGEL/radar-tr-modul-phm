# =============================================================
# test_dashboard_mantigi.py
#
# dashboard_app.py'nin tespit/teshis/yanlis-alarm/RUL dallarini, Streamlit
# UI'siz (headless) calistirip dogrulayan kucuk bir duman testi (smoke test).
# `pytest` ile de calisir, dogrudan `python test_dashboard_mantigi.py` ile de.
#
# Streamlit widget'larini degil, panelin kullandigi hesaplama mantigini
# (ayni modeller, ayni ozellikler) her ariza dali icin dogrular: saglikli,
# kademeli ve ani ariza senaryolarinda tespit/teshis/yanlis-alarm/RUL
# ciktilarinin beklenen degerleri uretmesi.
# =============================================================

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE = Path(__file__).parent
VERI_DIR = BASE / "veri"
MODEL_DIR = BASE / "sonuclar" / "models"


def _pipeline_ciktilari_var_mi():
    gerekli = [VERI_DIR / "telemetri_ozellikli.csv", VERI_DIR / "modul_tablosu.csv", VERI_DIR / "meta.json",
               MODEL_DIR / "rf_teshis.joblib", MODEL_DIR / "yanlis_alarm_model.joblib", MODEL_DIR / "ozellik_meta.joblib"]
    return all(p.exists() for p in gerekli)


def modul_durumunu_hesapla(telemetri, modul_tablosu, meta, ozellik_meta, rf_teshis, yanlis_alarm_model,
                            modul_id: int, zaman_adimi: int):
    # dashboard_app.py'deki AYNI hesaplama adimlari -- Streamlit cagrilari
    # cikarilmis, sadece sayisal sonuc dondurulmus hali.
    FEATURE_COLS = ozellik_meta["feature_cols"]
    Z_KOLONLARI = ozellik_meta["z_kolonlari"]
    nominal_ort = ozellik_meta["ort"]
    KADEMELI = set(meta["kademeli_arizalar"])

    modul_df = telemetri[telemetri["modul_id"] == modul_id].sort_values("zaman_adimi").reset_index(drop=True)
    i = modul_df[modul_df["zaman_adimi"] == zaman_adimi].index[0]
    satir = modul_df.iloc[i]
    gecmis = modul_df.iloc[: i + 1]
    modul_bilgi = modul_tablosu.set_index("modul_id").loc[modul_id]
    gercek_ariza_modu = modul_bilgi["ariza_modu"]
    t_onset = modul_bilgi["t_onset"]

    sonuc = {"gercek_ariza_modu": gercek_ariza_modu, "anomali_var": bool(satir["tespit_rf"]),
             "teshis": None, "gercek_ariza_olasiligi": None, "rul": None}

    if sonuc["anomali_var"]:
        X = satir[FEATURE_COLS].to_frame().T.astype(float)
        olasiliklar = rf_teshis.predict_proba(X)[0]
        teshis_olasilik = dict(zip(rf_teshis.classes_, olasiliklar))
        sonuc["teshis"] = max(teshis_olasilik, key=teshis_olasilik.get)

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
        ya_olasilik = yanlis_alarm_model.predict_proba(olay_ozellik)[0]
        ya_siniflar = list(yanlis_alarm_model.classes_)
        sonuc["gercek_ariza_olasiligi"] = float(ya_olasilik[ya_siniflar.index(True)])

    if gercek_ariza_modu in KADEMELI and pd.notna(t_onset) and zaman_adimi > t_onset + 20:
        kanal = meta["rul_birincil_kanal"][gercek_ariza_modu]
        tam_sapma = meta["rul_tam_bozulma_sapmasi"][gercek_ariza_modu]
        kritik_deger = nominal_ort[kanal] + tam_sapma * meta["kritik_ramp_orani"]
        egitim_pencere = gecmis[gecmis["zaman_adimi"] >= t_onset]
        if len(egitim_pencere) >= 5:
            egim, kesisim = np.polyfit(egitim_pencere["zaman_adimi"].to_numpy(), egitim_pencere[kanal].to_numpy(), 1)
            if abs(egim) > 1e-4:
                sonuc["rul"] = float((kritik_deger - kesisim) / egim - zaman_adimi)

    return sonuc


def test_dashboard_dallari():
    assert _pipeline_ciktilari_var_mi(), (
        "Once pipeline'i calistir: python 00_tr_modul_simulasyonu.py / 01_...py / 02_...py"
    )

    telemetri = pd.read_csv(VERI_DIR / "telemetri_ozellikli.csv")
    modul_tablosu = pd.read_csv(VERI_DIR / "modul_tablosu.csv")
    meta = json.loads((VERI_DIR / "meta.json").read_text(encoding="utf-8"))
    ozellik_meta = joblib.load(MODEL_DIR / "ozellik_meta.joblib")
    rf_teshis = joblib.load(MODEL_DIR / "rf_teshis.joblib")
    yanlis_alarm_model = joblib.load(MODEL_DIR / "yanlis_alarm_model.joblib")
    N_ORNEK = meta["n_ornek"]

    # 1) Saglikli modul, varsayilan gorunum -- anomali OLMAMALI
    s = modul_durumunu_hesapla(telemetri, modul_tablosu, meta, ozellik_meta, rf_teshis, yanlis_alarm_model, 0, N_ORNEK - 1)
    assert s["anomali_var"] is False, f"Modul 0 saglikli olmali, ama anomali_var={s['anomali_var']}"

    # 2) PA_Bozulmasi -- anomali + dogru teshis + RUL sayisal bir deger
    s = modul_durumunu_hesapla(telemetri, modul_tablosu, meta, ozellik_meta, rf_teshis, yanlis_alarm_model, 22, 300)
    assert s["anomali_var"] is True
    assert s["teshis"] == "PA_Bozulmasi", f"Beklenen PA_Bozulmasi, gelen {s['teshis']}"
    assert s["gercek_ariza_olasiligi"] is not None and 0.0 <= s["gercek_ariza_olasiligi"] <= 1.0
    assert s["rul"] is not None, "Kademeli ariza + onset sonrasi icin RUL hesaplanmis olmali (negatif de olabilir)"

    # 3) Konnektor_Arizasi -- ani ariza, RUL kavrami YOK (None kalmali)
    konnektor_modul = int(modul_tablosu[modul_tablosu["ariza_modu"] == "Konnektor_Arizasi"]["modul_id"].iloc[0])
    onset = int(modul_tablosu.set_index("modul_id").loc[konnektor_modul, "t_onset"])
    s = modul_durumunu_hesapla(telemetri, modul_tablosu, meta, ozellik_meta, rf_teshis, yanlis_alarm_model, konnektor_modul, onset + 30)
    assert s["anomali_var"] is True
    assert s["teshis"] == "Konnektor_Arizasi"
    assert s["rul"] is None, "Ani arizada RUL hesaplanMAMALI"

    # 4) Osilator_Kaymasi -- anomali + dogru teshis + RUL sayisal bir deger
    s = modul_durumunu_hesapla(telemetri, modul_tablosu, meta, ozellik_meta, rf_teshis, yanlis_alarm_model, 31, 340)
    assert s["anomali_var"] is True
    assert s["teshis"] == "Osilator_Kaymasi"
    assert s["rul"] is not None

    print("test_dashboard_dallari: TUM DALLAR HATASIZ VE BEKLENDIGI GIBI CALISTI")


if __name__ == "__main__":
    test_dashboard_dallari()
