# Gate B Status — PASS

Date/time: 2026-08-26 16:43 +03:00  
Tool: Codex  
Model, if known: GPT-5  
Operation ID: `f02-scientific-augmentation-20260826`

Tam koşu 10/10 atomik birim, 240 bilimsel hücre ve 7.440 ham metrik satırıyla
998.67 saniyede tamamlandı. Kilitli B1--B7 ölçütlerinin tamamı geçti; ayrı
validator 20/20 PASS verdi.

Birincil non-null normalized vector-MSE oranları:

- paired/independent LIME: 0.71, %95 cluster-bootstrap CI [0.63, 0.83];
- paired/independent sampling KernelSHAP: 0.33 [0.26, 0.52];
- paired LIME/independent GLIME-Binomial: 0.77 [0.71, 0.83];
- S-LIME toplam sorgu maliyetiyle eşli paired LIME/independent S-LIME: 0.60
  [0.41, 0.71].

No-shift FAR paired kanalda hem LIME hem KernelSHAP için 0.000; independent
kanalda sırasıyla 0.3125 ve 0.4850. Bu FAR sonucu yapısal exact no-shift
kontrolüdür; deployment alarm faydası olarak yorumlanmaz. Gate C bu ayrımı
held-out threshold/power protokolüyle sınayacaktır.

Gate C yetkilidir. Manuscript/PDF bu run tarafından değiştirilmedi.
