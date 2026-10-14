# =============================================================
# 00 -- T/R modulu telemetri simulasyonu + ariza enjeksiyonu
# CPU'da saniyeler icinde calisir; GPU veya dis veri gerekmez, telemetri
# burada uretilir. Dosya "# %%" hucrelerine bolundu (VSCode/Jupyter).
#
# Senaryo: AESA radar dizisindeki N_MODUL adet T/R (verici-alici) modulunun
# BIT (Built-In Test) telemetrisi -- cikis gucu, kazanc, VSWR, sicaklik, akim,
# faz gurultusu. Her modul ya saglikli kalir ya da 4 arizadan birini gelistirir
# (kademeli: PA_Bozulmasi, Termal_Ariza, Osilator_Kaymasi | ani: Konnektor_Arizasi).
# Saglikli modullerin bir kismina da GECICI (kalici olmayan, kendiliginden
# duzelen) anomali enjekte edilir -- bunlar "yanlis alarm" egitim verisi olacak.
#
# ONEMLI: Butun degerler (esikler, ramp sureleri, gurultu seviyeleri) gercek
# bir T/R modulu urun spesifikasyonundan degil, T/R modulu literaturunde
# (guc amplifikatoru bozulmasi, VSWR/konnektor arizasi, termal drift, LO faz
# gurultusu) genel bilinen davranis bicimlerinden esinlenerek EGITICI/PORTFOY
# amacli sentezlendi. Gercekci ama uydurma veri.
# =============================================================

# %% Ayarlar
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

BASE = Path(__file__).parent
VERI_DIR = BASE / "veri"
VERI_DIR.mkdir(exist_ok=True)

SEED = 42
N_MODUL_SAGLIKLI = 20
N_MODUL_ARIZA_PER_MOD = 5  # 4 ariza modu x 5 = 20 arizali modul -> toplam 40
N_ORNEK = 400  # modul basina BIT telemetri ornegi (~ operasyon saati)
TEST_ORANI = 0.30

FEATURES = ["guc_dBm", "kazanc_dB", "vswr", "sicaklik_C", "akim_A", "faz_gurultusu_dBcHz"]
KADEMELI_ARIZALAR = ["PA_Bozulmasi", "Termal_Ariza", "Osilator_Kaymasi"]
ANI_ARIZALAR = ["Konnektor_Arizasi"]
FAULT_MODES = KADEMELI_ARIZALAR + ANI_ARIZALAR

# Kademeli arizalarin "tam bozulma" ramp uzunluklari (ornek sayisi) -- RUL
# hesaplarinda (02_...) kritik-esik-gecis anini turetmek icin ayni sabitler
# tekrar kullaniliyor, o yuzden meta.json'a da yaziliyor.
RAMP_UZUNLUKLARI = {"PA_Bozulmasi": 150, "Termal_Ariza": 120, "Osilator_Kaymasi": 180}
KRITIK_RAMP_ORANI = 0.9  # ramp bu orana ulasinca "kritik/arizaya yakin" kabul edilir

# Kademeli arizalarin RUL (kalan omur) hesabinda izlenecek "birincil" kanali ve
# ramp=1.0'da (tam bozulma) o kanalda ne kadarlik toplam sapma olustugu --
# modul_uret()'teki katsayilarla AYNI olmak zorunda (02_...'de kritik esik
# turetmek icin tekrar kullaniliyor, tek yerden okunsun diye meta.json'a yazilir).
RUL_BIRINCIL_KANAL = {"PA_Bozulmasi": "kazanc_dB", "Termal_Ariza": "sicaklik_C", "Osilator_Kaymasi": "faz_gurultusu_dBcHz"}
RUL_TAM_BOZULMA_SAPMASI = {"PA_Bozulmasi": -9.0, "Termal_Ariza": 25.0, "Osilator_Kaymasi": 12.0}


# %% Tek modul telemetrisi ureten fonksiyon
def modul_uret(modul_id: int, rng: np.random.Generator, ariza_modu: str, t_onset):
    t = np.arange(N_ORNEK)
    # Modul basina rastgele fazli, yavas "yuk" dongusu (gunluk operasyon
    # temposunu taklit eder) -- tum kanallari hafifce birlikte hareket ettirir,
    # boylece kanallar arasi gercekci bir korelasyon olusur.
    yuk = 0.5 + 0.5 * np.sin(2 * np.pi * t / 96 + rng.uniform(0, 2 * np.pi))

    sinyaller = {
        "guc_dBm": 37.0 + 0.6 * yuk + rng.normal(0, 0.3, N_ORNEK),
        "kazanc_dB": 30.0 + 0.3 * yuk + rng.normal(0, 0.25, N_ORNEK),
        "vswr": 1.15 + 0.05 * yuk + rng.normal(0, 0.03, N_ORNEK),
        "sicaklik_C": 48.0 + 8.0 * yuk + rng.normal(0, 1.5, N_ORNEK),
        "akim_A": 2.5 + 0.15 * yuk + rng.normal(0, 0.08, N_ORNEK),
        "faz_gurultusu_dBcHz": -95.0 + rng.normal(0, 1.2, N_ORNEK),
    }

    etiket = np.array(["saglikli"] * N_ORNEK, dtype=object)
    gecici = np.zeros(N_ORNEK, dtype=bool)

    if ariza_modu == "PA_Bozulmasi":
        ramp = np.clip((t - t_onset) / RAMP_UZUNLUKLARI[ariza_modu], 0, 1)
        sinyaller["kazanc_dB"] -= 9.0 * ramp
        sinyaller["guc_dBm"] -= 6.0 * ramp
        sinyaller["akim_A"] += 0.6 * ramp  # amplifikator telafi etmeye calisir
        etiket[t >= t_onset] = ariza_modu

    elif ariza_modu == "Konnektor_Arizasi":
        adim = (t >= t_onset).astype(float)
        sinyaller["vswr"] += rng.uniform(0.6, 1.3) * adim
        sinyaller["guc_dBm"] -= rng.uniform(3.0, 5.0) * adim
        etiket[t >= t_onset] = ariza_modu

    elif ariza_modu == "Termal_Ariza":
        ramp = np.clip((t - t_onset) / RAMP_UZUNLUKLARI[ariza_modu], 0, 1)
        sinyaller["sicaklik_C"] += 25.0 * ramp
        # kaskad etki: sicaklik esigini asinca kazanc da kompresyona ugrar
        asiri = np.clip(sinyaller["sicaklik_C"] - 75.0, 0, None)
        sinyaller["kazanc_dB"] -= 0.3 * asiri
        etiket[t >= t_onset] = ariza_modu

    elif ariza_modu == "Osilator_Kaymasi":
        ramp = np.clip((t - t_onset) / RAMP_UZUNLUKLARI[ariza_modu], 0, 1)
        sinyaller["faz_gurultusu_dBcHz"] += 12.0 * ramp
        etiket[t >= t_onset] = ariza_modu

    else:
        # Saglikli modul: bazilarina kisa sureli, kendiliginden duzelen
        # (GERCEK ariza OLMAYAN) telemetri sicramasi enjekte edilir. Bunlar
        # naif esik-tabanli bir BIT'i tetikler ama gercek arizanin aksine
        # birkac ornek sonra nominale doner -- 02_...'deki yanlis alarm
        # siniflandiricisinin ayirt etmesi gereken tam da bu.
        n_blip = rng.integers(0, 3)
        blip_secenekleri = {
            "vswr": (0.4, 0.9, 1),
            "guc_dBm": (-4.0, -2.0, 1),
            "sicaklik_C": (6.0, 12.0, 1),
        }
        for _ in range(n_blip):
            kanal = rng.choice(list(blip_secenekleri))
            lo, hi, _ = blip_secenekleri[kanal]
            genislik = int(rng.integers(1, 3))
            t_blip = int(rng.integers(20, N_ORNEK - genislik - 1))
            buyukluk = rng.uniform(lo, hi)
            sinyaller[kanal][t_blip:t_blip + genislik] += buyukluk
            gecici[t_blip:t_blip + genislik] = True

    df = pd.DataFrame(sinyaller)
    df.insert(0, "zaman_adimi", t)
    df.insert(0, "modul_id", modul_id)
    df["vswr"] = np.clip(df["vswr"], 1.0, None)
    df["gercek_etiket"] = etiket
    df["gecici_anomali"] = gecici
    return df


# %% Modul planini olustur (kac modul, hangi ariza, ne zaman baslar)
rng = np.random.default_rng(SEED)

plan = []
modul_id = 0
for _ in range(N_MODUL_SAGLIKLI):
    plan.append({"modul_id": modul_id, "ariza_modu": "saglikli", "t_onset": None})
    modul_id += 1
for mode in FAULT_MODES:
    for _ in range(N_MODUL_ARIZA_PER_MOD):
        # Onset erken/gec ucta olmasin diye pay birakiyoruz: kademeli arizalarda
        # ramp'in N_ORNEK icinde tamamlanabilmesi, ani arizada da yeterli
        # "arizali" gozlem penceresi kalmasi icin.
        t_onset = int(rng.integers(60, N_ORNEK - RAMP_UZUNLUKLARI.get(mode, 40) - 40))
        plan.append({"modul_id": modul_id, "ariza_modu": mode, "t_onset": t_onset})
        modul_id += 1

modul_tablosu = pd.DataFrame(plan)

# %% Tum moduller icin telemetri uret ve birlestir
parcalar = []
for _, satir in modul_tablosu.iterrows():
    alt_rng = np.random.default_rng(SEED + 1000 + satir["modul_id"])
    parcalar.append(modul_uret(satir["modul_id"], alt_rng, satir["ariza_modu"], satir["t_onset"]))

telemetri = pd.concat(parcalar, ignore_index=True)

# %% Modul-bazli (grup-farkinda) train/test ayrimi, ariza moduna gore stratifiye
# Segment/ornek bazli rastgele split degil -- sonar-hedef-siniflandirma
# projesindeki ayni ilkeyle (GroupShuffleSplit): bir modulun ornekleri ya
# tamamen train'de ya tamamen test'te olmali, yoksa model "ariza turunu"
# degil o modulun kendi gurultu imzasini ezberleyebilir.
train_id, test_id = train_test_split(
    modul_tablosu["modul_id"],
    test_size=TEST_ORANI,
    stratify=modul_tablosu["ariza_modu"],
    random_state=SEED,
)
modul_tablosu["split"] = np.where(modul_tablosu["modul_id"].isin(set(train_id)), "train", "test")

# %% Diske yaz
telemetri.to_csv(VERI_DIR / "telemetri.csv", index=False)
modul_tablosu.to_csv(VERI_DIR / "modul_tablosu.csv", index=False)

meta = {
    "seed": SEED,
    "n_ornek": N_ORNEK,
    "features": FEATURES,
    "fault_modes": FAULT_MODES,
    "kademeli_arizalar": KADEMELI_ARIZALAR,
    "ani_arizalar": ANI_ARIZALAR,
    "ramp_uzunluklari": RAMP_UZUNLUKLARI,
    "kritik_ramp_orani": KRITIK_RAMP_ORANI,
    "rul_birincil_kanal": RUL_BIRINCIL_KANAL,
    "rul_tam_bozulma_sapmasi": RUL_TAM_BOZULMA_SAPMASI,
    "train_modul_id": sorted(int(m) for m in train_id),
    "test_modul_id": sorted(int(m) for m in test_id),
}
(VERI_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

# %% Ozet
print(f"Toplam modul: {len(modul_tablosu)} | ornek/modul: {N_ORNEK} | toplam satir: {len(telemetri)}")
print(modul_tablosu["ariza_modu"].value_counts())
print(f"Train modul: {len(train_id)} | Test modul: {len(test_id)}")
print(f"Gecici anomali iceren satir sayisi (saglikli modullerde): {int(telemetri['gecici_anomali'].sum())}")
print(f"\nYazildi: {VERI_DIR / 'telemetri.csv'}, modul_tablosu.csv, meta.json")
