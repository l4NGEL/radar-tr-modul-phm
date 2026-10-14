% =============================================================
% tr_modul_sim.m
%
% Tek bir T/R modulunun PA_Bozulmasi (kademeli kazanc/guc bozulmasi)
% senaryosunun MATLAB'da SADELESTIRILMIS tekrari + esik tabanli BIT
% (Built-In Test) karar mantigi.
%
% BILEREK Simulink degil, duz bir script: MATLAB deneyimi henuz temel
% seviyede oldugu icin, acilip calistirilamayan bir .slx dosyasi yerine,
% dogrudan okunup dogrulanabilen bir script tercih edildi. Ana pipeline
% (00-03, Python -- ayni klasorun ust dizininde) burada tekrar edilmiyor;
% bu script sadece "ayni sistemi MATLAB'da da modelleyip temel bir BIT
% karari uretebiliyorum" seviyesini gosteriyor.
%
% Calistirmak icin kurulum GEREKMEZ: MATLAB Online'da (matlab.mathworks.com,
% ucretsiz MathWorks hesabiyla, tarayicidan) bu dosyayi yukleyip Run'a
% basmak yeterli.
% =============================================================

clear; clc; close all;

%% Ayarlar (Python tarafindaki 00_tr_modul_simulasyonu.py ile ayni buyukluk
%  mertebeleri kullanildi, ama bu BAGIMSIZ, kucultulmus bir tekrar)
N = 400;              % ornek sayisi (BIT telemetri orneklemesi)
t_onset = 150;         % ariza baslangic zaman adimi
ramp_uzunluk = 150;    % kademeli bozulmanin tamamlanma suresi (ornek)
kazanc_nominal = 30.0; % dB
toplam_dususu_dB = 9.0;
esik_dB = 5.0;         % nominalden bu kadar dB dusuk olursa BIT FAIL

t = 0:(N-1);

%% Nominal (saglikli) kazanc sinyali + gurultu
rng(42);
kazanc = kazanc_nominal + 0.3*sin(2*pi*t/96) + 0.25*randn(1, N);

%% PA bozulmasi enjeksiyonu (0 -> 1 kademeli ramp, t_onset'ten itibaren)
ramp = max(0, min(1, (t - t_onset) / ramp_uzunluk));
kazanc = kazanc - toplam_dususu_dB * ramp;

%% Esik tabanli BIT mantigi
sapma = kazanc_nominal - kazanc;
bit_fail = sapma > esik_dB;   % true = FAIL, false = PASS

ilk_tespit = find(bit_fail, 1);
if isempty(ilk_tespit)
    fprintf('BIT: Bu operasyon penceresinde esik hic asilmadi.\n');
else
    ilk_tespit_zamani = ilk_tespit - 1;   % MATLAB 1-tabanli index -> 0-tabanli zaman
    gecikme = ilk_tespit_zamani - t_onset;
    fprintf('BIT: Ilk FAIL zaman adimi = %d (gercek ariza baslangici = %d, tespit gecikmesi = %d ornek)\n', ...
        ilk_tespit_zamani, t_onset, gecikme);
end

%% Gorsellestirme
figure('Position', [100, 100, 800, 500]);

subplot(2,1,1);
plot(t, kazanc, 'b-', 'LineWidth', 1.0); hold on;
yline(kazanc_nominal - esik_dB, 'r--', 'LineWidth', 1.2);
xline(t_onset, 'k:', 'LineWidth', 1.0);
xlabel('Zaman adimi'); ylabel('Kazanc (dB)');
title('T/R Modulu Kazanc Trendi -- PA Bozulmasi Senaryosu');
legend('Olculen kazanc', 'BIT esigi', 'Gercek ariza baslangici', 'Location', 'southwest');
grid on;

subplot(2,1,2);
stairs(t, bit_fail, 'k-', 'LineWidth', 1.5);
ylim([-0.2, 1.2]);
xlabel('Zaman adimi'); ylabel('BIT durumu (0=PASS, 1=FAIL)');
title('Esik Tabanli BIT Karari');
grid on;

saveas(gcf, 'tr_modul_sim_sonuc.png');
fprintf('Grafik kaydedildi: tr_modul_sim_sonuc.png\n');
