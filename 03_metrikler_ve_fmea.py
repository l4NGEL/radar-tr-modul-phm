# =============================================================
# 03 -- Metrik ozeti + FMEA tablosu
#
# FMEA (Failure Mode and
# Effects Analysis) tablosu, 00-02'de simule edilip tespit/teshis edilen 5
# arizanin (4 gercek ariza modu + 1 gecici/yanlis-alarm kaynagi) sistematik
# etki/siddet/tespit-yontemi analizidir.
#
# NOT: Bu FMEA, gercek bir T/R modulu uretici dokumanindan degil, T/R modulu
# ariza literaturunden esinlenerek bu portfoy projesi icin sentezlendi --
# egitici/gosterim amacli, gizli/tescilli hicbir veri icermez.
# =============================================================

# %% Ayarlar
import json
from pathlib import Path

import pandas as pd

BASE = Path(__file__).parent
VERI_DIR = BASE / "veri"
SONUC_DIR = BASE / "sonuclar"

# %% FMEA tablosu (elle tanimli -- domain bilgisi, hesaplanan bir sey degil)
FMEA = [
    {
        "bilesen": "Guc Amplifikatoru (PA)",
        "ariza_modu": "PA_Bozulmasi",
        "olasi_neden": "Transistor yaslanmasi, kumulatif termal stres",
        "yerel_etki": "Cikis gucu ve kazancta kademeli dusus, akim artisi",
        "sistem_etkisi": "Radar menzilinde/SNR'de azalma",
        "siddet_1_10": 7,
        "tespit_yontemi": "Kademeli egilim izleme + RUL ekstrapolasyonu",
        "proje_karsiligi": "01 (teshis), 02 (RUL kestirimi)",
    },
    {
        "bilesen": "RF Konnektor / Kablo",
        "ariza_modu": "Konnektor_Arizasi",
        "olasi_neden": "Titresim, gevseme, korozyon, temas bozulmasi",
        "yerel_etki": "VSWR'de ani yukselme, guc kaybi",
        "sistem_etkisi": "Modulun ani devre disi kalmasi",
        "siddet_1_10": 8,
        "tespit_yontemi": "Ani esik asimi (z-skor)",
        "proje_karsiligi": "01 (kural-tabanli + RF tespit)",
    },
    {
        "bilesen": "Sogutma Alt Sistemi",
        "ariza_modu": "Termal_Ariza",
        "olasi_neden": "Sogutucu akis azalmasi, fan/soguk plaka bozulmasi",
        "yerel_etki": "Sicaklikta kademeli artis, esik sonrasi kazanc kompresyonu (kaskad)",
        "sistem_etkisi": "Modul omrunun kisalmasi, olasi termal kapanma",
        "siddet_1_10": 6,
        "tespit_yontemi": "Sicaklik egilimi + kaskad kazanc etkisi izleme",
        "proje_karsiligi": "01 (teshis), 02 (RUL kestirimi)",
    },
    {
        "bilesen": "Yerel Osilator (LO)",
        "ariza_modu": "Osilator_Kaymasi",
        "olasi_neden": "Referans kaynagi yaslanmasi, sicaklik hassasiyeti",
        "yerel_etki": "Faz gurultusunde kademeli artis",
        "sistem_etkisi": "Doppler/hassas olcum performansinda bozulma",
        "siddet_1_10": 5,
        "tespit_yontemi": "Faz gurultusu egilimi + RUL ekstrapolasyonu",
        "proje_karsiligi": "01 (teshis), 02 (RUL kestirimi)",
    },
    {
        "bilesen": "Telemetri / Sensor Zinciri",
        "ariza_modu": "Gecici sensor gurultusu (gercek ariza degil)",
        "olasi_neden": "EMI, guc hatti gecici dalgalanmasi, sensor gurultusu",
        "yerel_etki": "Kisa sureli anormal okuma, kendiliginden nominale donus",
        "sistem_etkisi": "Gereksiz alarm / operator guvensizligi (yanlis alarm maliyeti)",
        "siddet_1_10": 3,
        "tespit_yontemi": "Olay suresi + toparlanma egilimi tabanli siniflandirma",
        "proje_karsiligi": "02 (yanlis alarm siniflandirici)",
    },
]

# %% Metrik ozetini 01 ve 02'den birlestir
tespit = json.loads((SONUC_DIR / "01_tespit_teshis_metrikleri.json").read_text(encoding="utf-8"))
alarm_rul = json.loads((SONUC_DIR / "02_yanlis_alarm_rul_metrikleri.json").read_text(encoding="utf-8"))

ozet_satirlari = []
for yontem in ["kural_tabanli", "isolation_forest", "random_forest"]:
    m = tespit[yontem]
    ozet_satirlari.append({
        "Yontem": yontem,
        "Fault Detection Coverage": f"{m['fault_detection_coverage']:.1%}",
        "False Alarm Rate": f"{m['false_alarm_rate']:.1%}",
        "Gecici Blip Yakalama": f"{m['gecici_blip_yakalama_orani']:.1%}",
        "MTTD (ornek)": m["mean_time_to_detect_ornek"],
    })
ozet_df = pd.DataFrame(ozet_satirlari)

teshis = tespit["teshis_lokalizasyon"]
ya = alarm_rul["yanlis_alarm_siniflandirici"]
rul = alarm_rul["kestirimci_bakim_rul"]

# %% Kaydet: FMEA + genel ozet, README'ye yapistirilabilir markdown olarak
fmea_df = pd.DataFrame(FMEA)
fmea_md = fmea_df.rename(columns={
    "bilesen": "Bilesen", "ariza_modu": "Ariza Modu", "olasi_neden": "Olasi Neden",
    "yerel_etki": "Yerel Etki", "sistem_etkisi": "Sistem Etkisi", "siddet_1_10": "Siddet (1-10)",
    "tespit_yontemi": "Tespit Yontemi", "proje_karsiligi": "Proje Karsiligi",
}).to_markdown(index=False)
(SONUC_DIR / "fmea_tablosu.md").write_text(fmea_md, encoding="utf-8")

genel_ozet_md = f"""# Sonuc ozeti (otomatik uretildi -- 03_metrikler_ve_fmea.py)

## Ariza tespiti (test split, 3 yontem karsilastirmasi)

{ozet_df.to_markdown(index=False)}

## Ariza teshisi / lokalizasyonu (CIT)

- Fault Isolation Accuracy: **{teshis['fault_isolation_accuracy']:.1%}** ({teshis['test_ariza_ornegi_sayisi']} test ornegi, {len(teshis['siniflar'])} sinif)
- Siniflar: {', '.join(teshis['siniflar'])}

## Yanlis alarm siniflandirici

- Test olay sayisi: {ya['test_olay_sayisi']} (gercek ariza: {ya['gercek_ariza_olay_sayisi']}, yanlis alarm: {ya['yanlis_alarm_olay_sayisi']})
- F1 (gercek ariza): {ya['f1_gercek_ariza']:.1%} | F1 (yanlis alarm): {ya['f1_yanlis_alarm']:.1%}
- Yanlis alarm bastirma orani: {ya['yanlis_alarm_bastirma_orani']:.1%}

## Kestirimci bakim (RUL)

- Degerlendirilen modul: {rul['degerlendirilen_modul_sayisi']}
- Ortalama mutlak RUL hatasi: {rul['ortalama_mutlak_hata_ornek']} ornek
"""
(SONUC_DIR / "metrikler_ozet.md").write_text(genel_ozet_md, encoding="utf-8")

# %% Ekrana yazdir
print(genel_ozet_md)
print("\n\n" + fmea_md)
print(f"\nYazildi: {SONUC_DIR / 'metrikler_ozet.md'}, {SONUC_DIR / 'fmea_tablosu.md'}")
