# =============================================================
# 02 -- Yanlis alarm tahminlemesi + kestirimci bakim (RUL)
#
# Iki ayri model:
#   A) Yanlis alarm siniflandiricisi -- 01'deki RF dedektorun isaretledigi
#      surekli "olay"lari (event) gercek ariza / gecici yanlis alarm olarak
#      ikiye ayirir. Ozellikler: olay suresi, buyukluk, kac kanalin birlikte
#      sapdigi, olay bittikten sonra sinyalin toparlanip toparlanmadigi.
#   B) RUL (kalan kullanilabilir omur) kestirimi -- sadece KADEMELI arizalarda
#      (PA_Bozulmasi, Termal_Ariza, Osilator_Kaymasi) anlamli: birincil kanalin
#      egilimine dogrusal regresyon uydurup kritik esige ne zaman ulasacagini
#      ekstrapole eder. Ani arizada (Konnektor_Arizasi) RUL kavrami yok --
#      o an olur, kademeli bir on-belirti yok, o yuzden bu grup RUL'e dahil
#      edilmiyor (gercekci bir sinirlama, gizlenmiyor).
# =============================================================

# %% Ayarlar
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report, f1_score

BASE = Path(__file__).parent
VERI_DIR = BASE / "veri"
SONUC_DIR = BASE / "sonuclar"
MODEL_DIR = SONUC_DIR / "models"

SEED = 42
Z_ESIK = 3.0
SONRAKI_PENCERE = 8  # olay bittikten sonra kac ornek daha izlenip "toparlaniyor mu" bakilacak

meta = json.loads((VERI_DIR / "meta.json").read_text(encoding="utf-8"))
TRAIN_ID = set(meta["train_modul_id"])
TEST_ID = set(meta["test_modul_id"])
KADEMELI_ARIZALAR = meta["kademeli_arizalar"]
RAMP_UZUNLUKLARI = meta["ramp_uzunluklari"]
KRITIK_RAMP_ORANI = meta["kritik_ramp_orani"]
RUL_BIRINCIL_KANAL = meta["rul_birincil_kanal"]
RUL_TAM_BOZULMA_SAPMASI = meta["rul_tam_bozulma_sapmasi"]

telemetri = pd.read_csv(VERI_DIR / "telemetri_ozellikli.csv")
modul_tablosu = pd.read_csv(VERI_DIR / "modul_tablosu.csv")
ozellik_meta = joblib.load(MODEL_DIR / "ozellik_meta.joblib")
Z_KOLONLARI = ozellik_meta["z_kolonlari"]
nominal_ort = ozellik_meta["ort"]


# %% A) Olay (event) cikarimi: tespit_rf'in ardisik True bloklarini grupla
def ardisik_bloklar(bool_dizi: np.ndarray):
    idx = np.where(bool_dizi)[0]
    if len(idx) == 0:
        return []
    bloklar, basla, onceki = [], idx[0], idx[0]
    for i in idx[1:]:
        if i == onceki + 1:
            onceki = i
        else:
            bloklar.append((basla, onceki))
            basla = onceki = i
    bloklar.append((basla, onceki))
    return bloklar


def modul_olaylarini_cikar(df_modul: pd.DataFrame):
    # df_modul: tek bir modulun zaman_adimi'na gore sirali satirlari.
    # zaman_adimi == 0..N-1 seklinde pozisyonel oldugu icin dogrudan index
    # olarak kullanilabiliyor (00_...'de bu sekilde uretildi).
    df_modul = df_modul.reset_index(drop=True)
    bayrak = df_modul["tespit_rf"].to_numpy()
    olaylar = []
    for basla, bitis in ardisik_bloklar(bayrak):
        pencere = df_modul.iloc[basla:bitis + 1]
        sonraki = df_modul.iloc[bitis + 1: bitis + 1 + SONRAKI_PENCERE]

        buyukluk_su_an = pencere[Z_KOLONLARI].abs().max(axis=1).mean()
        buyukluk_sonraki = sonraki[Z_KOLONLARI].abs().max(axis=1).mean() if len(sonraki) > 0 else 0.0
        kac_kanal = (pencere[Z_KOLONLARI].abs() > Z_ESIK).any(axis=0).sum()

        gercek_ariza = (pencere["gercek_etiket"] != "saglikli").mean() > 0.5

        olaylar.append({
            "modul_id": df_modul["modul_id"].iloc[0],
            "baslangic": int(df_modul["zaman_adimi"].iloc[basla]),
            "sure": bitis - basla + 1,
            "buyukluk": float(buyukluk_su_an),
            "kac_kanal": int(kac_kanal),
            "sonraki_egilim": float(buyukluk_sonraki - buyukluk_su_an),  # negatif = toparlaniyor
            "gercek_ariza": bool(gercek_ariza),
        })
    return olaylar


tum_olaylar = []
for modul_id, grup in telemetri.groupby("modul_id"):
    tum_olaylar.extend(modul_olaylarini_cikar(grup.sort_values("zaman_adimi")))
olaylar_df = pd.DataFrame(tum_olaylar)

OLAY_OZELLIKLERI = ["sure", "buyukluk", "kac_kanal", "sonraki_egilim"]
olaylar_train = olaylar_df[olaylar_df["modul_id"].isin(TRAIN_ID)]
olaylar_test = olaylar_df[olaylar_df["modul_id"].isin(TEST_ID)]

yanlis_alarm_model = RandomForestClassifier(n_estimators=200, random_state=SEED, class_weight="balanced")
yanlis_alarm_model.fit(olaylar_train[OLAY_OZELLIKLERI], olaylar_train["gercek_ariza"])
tahmin = yanlis_alarm_model.predict(olaylar_test[OLAY_OZELLIKLERI])

rapor = classification_report(olaylar_test["gercek_ariza"], tahmin, target_names=["yanlis_alarm", "gercek_ariza"], output_dict=True)
# Yanlis alarm bastirma orani: gercekten yanlis-alarm olan olaylarin ne kadari
# dogru sekilde "gercek ariza degil" olarak isaretlendi (specificity).
gercek_yanlis_alarmlar = olaylar_test[~olaylar_test["gercek_ariza"]]
if len(gercek_yanlis_alarmlar) > 0:
    bastirma_orani = float((yanlis_alarm_model.predict(gercek_yanlis_alarmlar[OLAY_OZELLIKLERI]) == False).mean())
else:
    bastirma_orani = float("nan")


# %% B) RUL (kestirimci bakim) -- sadece kademeli arizali test modullerinde
def rul_kestir(modul_id, ariza_modu, t_onset):
    kanal = RUL_BIRINCIL_KANAL[ariza_modu]
    tam_sapma = RUL_TAM_BOZULMA_SAPMASI[ariza_modu]
    ramp_uzunluk = RAMP_UZUNLUKLARI[ariza_modu]
    kritik_deger = nominal_ort[kanal] + tam_sapma * KRITIK_RAMP_ORANI

    gercek_kritik_zaman = t_onset + KRITIK_RAMP_ORANI * ramp_uzunluk

    # "Su an" = ramp'in yaridan bir miktar fazlasi (kademeli bozulma fark
    # edilmis ama kritik esige henuz varilmamis) -- gercek bir izleme
    # senaryosunu taklit ediyor: elimizde SADECE simdiye kadarki veri var.
    gozlem_ani = t_onset + 0.6 * ramp_uzunluk
    if gozlem_ani >= gercek_kritik_zaman:
        return None  # bu ariza turunde ramp cok kisa, anlamli bir gozlem penceresi yok

    gecmis = telemetri[(telemetri["modul_id"] == modul_id) & (telemetri["zaman_adimi"] >= t_onset)
                        & (telemetri["zaman_adimi"] <= gozlem_ani)]
    if len(gecmis) < 5:
        return None

    X = gecmis[["zaman_adimi"]].to_numpy()
    y = gecmis[kanal].to_numpy()
    reg = LinearRegression().fit(X, y)

    # egim=0'a yakinsa (henuz ayirt edilebilir bir trend yoksa) ekstrapolasyon
    # patolojik (sonsuza yakin) sonuc verir -- bu durumda "henuz guvenilir RUL
    # kestirilemiyor" demek daha durust, uydurma bir sayi uretmekten iyidir.
    egim = reg.coef_[0]
    if abs(egim) < 1e-4:
        return None
    kesisim_zamani = (kritik_deger - reg.intercept_) / egim

    tahmini_rul = kesisim_zamani - gozlem_ani
    gercek_rul = gercek_kritik_zaman - gozlem_ani
    return {
        "modul_id": int(modul_id),
        "ariza_modu": ariza_modu,
        "gozlem_ani": round(gozlem_ani, 1),
        "tahmini_rul_ornek": round(float(tahmini_rul), 1),
        "gercek_rul_ornek": round(float(gercek_rul), 1),
        "mutlak_hata_ornek": round(float(abs(tahmini_rul - gercek_rul)), 1),
    }


rul_sonuclari = []
for _, satir in modul_tablosu[modul_tablosu["modul_id"].isin(TEST_ID)
                               & modul_tablosu["ariza_modu"].isin(KADEMELI_ARIZALAR)].iterrows():
    sonuc = rul_kestir(satir["modul_id"], satir["ariza_modu"], satir["t_onset"])
    if sonuc is not None:
        rul_sonuclari.append(sonuc)

rul_df = pd.DataFrame(rul_sonuclari)
rul_mae = float(rul_df["mutlak_hata_ornek"].mean()) if len(rul_df) > 0 else float("nan")

# %% Kaydet
sonuc = {
    "yanlis_alarm_siniflandirici": {
        "test_olay_sayisi": int(len(olaylar_test)),
        "gercek_ariza_olay_sayisi": int(olaylar_test["gercek_ariza"].sum()),
        "yanlis_alarm_olay_sayisi": int((~olaylar_test["gercek_ariza"]).sum()),
        "f1_gercek_ariza": round(rapor["gercek_ariza"]["f1-score"], 4),
        "f1_yanlis_alarm": round(rapor["yanlis_alarm"]["f1-score"], 4),
        "yanlis_alarm_bastirma_orani": round(bastirma_orani, 4) if not np.isnan(bastirma_orani) else None,
    },
    "kestirimci_bakim_rul": {
        "degerlendirilen_modul_sayisi": int(len(rul_df)),
        "ortalama_mutlak_hata_ornek": round(rul_mae, 2) if not np.isnan(rul_mae) else None,
        "modul_detaylari": rul_sonuclari,
    },
}
(SONUC_DIR / "02_yanlis_alarm_rul_metrikleri.json").write_text(
    json.dumps(sonuc, ensure_ascii=False, indent=2), encoding="utf-8"
)
olaylar_df.to_csv(SONUC_DIR / "olaylar.csv", index=False)
joblib.dump(yanlis_alarm_model, MODEL_DIR / "yanlis_alarm_model.joblib")

# %% Ozet
print("=== Yanlis alarm siniflandirici (test split, olay bazli) ===")
print(f"Toplam olay: {len(olaylar_test)}  (gercek ariza: {int(olaylar_test['gercek_ariza'].sum())}, "
      f"yanlis alarm: {int((~olaylar_test['gercek_ariza']).sum())})")
print(f"F1 (gercek ariza): {rapor['gercek_ariza']['f1-score']:.2%}  F1 (yanlis alarm): {rapor['yanlis_alarm']['f1-score']:.2%}")
print(f"Yanlis alarm bastirma orani: {bastirma_orani:.2%}")
print(f"\n=== Kestirimci bakim (RUL) -- kademeli arizali test modulleri ===")
print(rul_df.to_string(index=False) if len(rul_df) else "(degerlendirilebilir modul yok)")
print(f"\nOrtalama mutlak RUL hatasi: {rul_mae:.1f} ornek ({len(rul_df)} modul uzerinden)")
print(f"\nYazildi: {SONUC_DIR / '02_yanlis_alarm_rul_metrikleri.json'}, sonuclar/olaylar.csv, sonuclar/models/yanlis_alarm_model.joblib")
