# Sonuc ozeti (otomatik uretildi -- 03_metrikler_ve_fmea.py)

## Ariza tespiti (test split, 3 yontem karsilastirmasi)

| Yontem           | Fault Detection Coverage   | False Alarm Rate   | Gecici Blip Yakalama   |   MTTD (ornek) |
|:-----------------|:---------------------------|:-------------------|:-----------------------|---------------:|
| kural_tabanli    | 91.5%                      | 1.2%               | 81.8%                  |          14.17 |
| isolation_forest | 86.3%                      | 7.6%               | 36.4%                  |           7.33 |
| random_forest    | 93.3%                      | 0.9%               | 90.9%                  |          12.33 |

## Ariza teshisi / lokalizasyonu (CIT)

- Fault Isolation Accuracy: **96.8%** (1384 test ornegi, 4 sinif)
- Siniflar: Konnektor_Arizasi, Osilator_Kaymasi, PA_Bozulmasi, Termal_Ariza

## Yanlis alarm siniflandirici

- Test olay sayisi: 51 (gercek ariza: 16, yanlis alarm: 35)
- F1 (gercek ariza): 54.5% | F1 (yanlis alarm): 87.5%
- Yanlis alarm bastirma orani: 100.0%

## Kestirimci bakim (RUL)

- Degerlendirilen modul: 4
- Ortalama mutlak RUL hatasi: 4.7 ornek
