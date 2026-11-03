# Yalnız görüntüyle araç konumu ve hareketi — 31 Ekim 2026

## GaussianCaR kararı

[Resmî model kodunda](https://github.com/santimontiel/gaussiancar/blob/c0e748b9e95a6a6062579ddebdcd1c1e79500c2e/gaussiancar/modeling/model.py)
`forward_features_camera` ayrı bir kamera koludur. Ancak tam `forward`, koşulsuz olarak
`forward_features_radar(batch)` çağırır; radar kolu `batch["radar_points"]` kullanır. Sonrasında kamera/radar BEV
özellikleri birleştirilir. Yayımlanan konfigürasyonlarda hazır, doğrulanmış bir kamera-only çıkarım yolu bulunmadı.

Kamera özelliklerini çıkarmak mümkündür; bu, radar olmadan aynı hazır modelin doğru 3D konum vereceğini göstermez.
Radar kolunu atlamak veya sıfır radar vermek için doğrulanmış bir sonuç sunulmuyor. Kullanıcının tercihi doğrultusunda
bu model için radar taklidi ya da yeniden eğitim yapılmadı.

## Tek ileri bakan kamera için adaylar

| Model | Çıkarım girdisi ve çıktısı | Bu projeye katkısı | Deneme durumu |
|---|---|---|---|
| [WildDet3D](https://github.com/allenai/WildDet3D) | RGB; metin, 2D kutu veya nokta istemi; isteğe bağlı kamera matrisi; 3D kutu/poz | Mevcut SAM araç kutularını 3D yerleştirmeye destek olarak kullanıyor. Ek derinlik sensörü zorunlu değil. | Kaggle T4'te gerçekten çalıştırıldı ve beş vakaya entegre edildi: 50 kare, 309 tahmin, 182 uygun tahmin, 48 desteklenen araç izi. [Kurulum, eşikler ve sınırlar](wilddet3d.md). |
| [SpaTrackerV2](https://github.com/henry123-boy/SpaTrackerV2) | RGB video; `--data_type RGB` modu; 3D nokta izleri ve açık kamera hareketi | Kareler arası süreklilik ve kamera/nesne hareketini ayırmak için destek. Araç kimliği, rijit gövde pozu ve metrik ölçek ayrıca bağlanmalı. | Resmî RGB yolu incelendi; çalıştırılmadı. |
| [MonoDGP](https://github.com/PuFanqi23/MonoDGP) | Tek RGB + kamera kalibrasyonu; 3D konum, boyut ve yön | Yol araçlarına odaklı, KITTI üzerinde eğitilmiş alternatif bir 3D dedektör. Farklı kamera/gece/örtülmeler ayrıca sınanmalı. | Kod ve hazır checkpoint bağlantıları incelendi; çalıştırılmadı. Eski PyTorch/deformable-attention kurulumu ayrı ortam gerektiriyor. |
| [UrbanOmniDetect-2](https://github.com/mkturkcan/urbanomnidetect) | RGB; COCO kutuları ve sekiz **görüntüye izdüşmüş** küp köşesi; takip/BEV/refinement araçları | Hafif, doğrudan video akışı var. Metrik yerleştirme için köşelerin zemine bağlanması ve ölçek çözümü hâlâ gerekli. | L modeli Kaggle T4'te, 15 özgün karede, 640 ve 960 çözünürlükte gerçekten çalıştırıldı. Bu kliplerde ana konumlandırıcı olarak seçilmedi. |

**Uygulanan yol:** WildDet3D'nin uygun RGB tahminleri, kaynak araç kutuları ve normal trafik için ortak yol/hareket
kısıtları birlikte kullanılıyor. Statik Gaussian ortamı ayrı kalıyor. SpaTrackerV2 zaman/kamera desteği için adaydır;
burada henüz çalıştırılmadı. WildDet3D'nin reddedilen veya eksik tahminlerinde kaynak kutu geometrisine dönülüyor.

### İlgili ama doğrudan tak-çalıştır olmayanlar

- [MonoSOWA](https://github.com/jskvrna/MonoSOWA): yerel nesne hareketi modeliyle kamera/nesne hareketini ayıran
  video tabanlı otomatik etiketleme ve MonoDETR eğitimi. Resmî akış KITTI/KITTI-360 biçimi, kalibrasyon ve araç
  pozları bekliyor; rastgele MP4 için görsel odometri/veri adaptörü gerekir. Yalnız görüntüyle çalışan son dedektör
  ile bu etiketleme altyapısının gereksinimleri aynı şey değildir. Burada çalıştırılmadı.
- [GaussianBeV makalesi](https://arxiv.org/abs/2407.14108): GaussianCaR'a mimari olarak yakın, kameradan BEV
  üreten bir yaklaşım. Çoklu kamera görüntüleri/koordinat dönüşümleri için tasarlanmış; tek dashcam için hazır çözüm
  olarak seçilmedi. Burada çalıştırılmadı.

## Gerçek GPU denemesi

UrbanOmniDetect kaynak sürümü `caa8537803f61b5d04602a25432575f36fce085d`, Hugging Face ağırlık sürümü
`be4f69a300c43f6d2c7dd35f99b6c2a828912817`, dosya `urbanomnidetect_v2_l_640.pt`.

Özel Kaggle setup çıktısı `helizac/mercy-rgb-vehicle-runtime-v1`, `ultralytics==8.4.61`, gerekli ek paket, kaynak kod
ve checkpoint'i bir kez kaydeder. Sonraki iki çıkarım işi internet kapalıyken aynı cache'i kullandı. Kurulum GPU
smoke testi ve her iki çıkarım işi tamamlandı; donanım Tesla T4, Torch 2.11.0+cu128. Yerel GPU'da bu ağ çalıştırılmadı.

Beş videodan toplam 15 özgün kare seçildi. Karşılaştırma referansı mevcut SAM kutularıdır; insan tarafından
etiketlenmiş 3D gerçeklik değildir. Eşleşme için 2D IoU >= 0.5 kullanıldı. Basit köşe kontrolü: küpün 2D dışbükey zarf
alanı / dedektör kutu alanı 0.35..2.5 ve medyan köşe güveni >= 0.25. Bu kontrol 3D yön/konum doğruluğunu ölçmez.

| Ağ girdi boyutu | Kaynak kutu eşleşmesi | Basit köşe kontrolünden geçen eşleşmeler | Kritik gözlem |
|---|---:|---:|---|
| 960 | 62 / 96 | 47 | cut-in kare 121'de hiç araç çıkmadı; bazı yakın araçlarda küpler küçücük bir bölgeye çöktü |
| 640 | 55 / 96 | 39 | aynı karede bir araç çıktı, ancak yakın gövdenin küpü yine çöktü |

Bu nedenle tahminler canlı 3D yerleştirmeye otomatik eklenmedi. Görüntüler ve ham JSON çıktılar:
`data/kaggle-vehicle-pose/output/e000e779ac/pose-predictions/` (960),
`data/kaggle-vehicle-pose/output/d1f40ec8ef/pose-predictions/` (640).
Özet: `data/research/vision-only-models/benchmark.json`.

Tekrar çalıştırma:

```bash
python scripts/kaggle_vehicle_pose.py --imgsz 960 --case cutin_rearend hero lanechange night_rain snow_pileup
python scripts/kaggle_vehicle_pose.py --imgsz 640 --case cutin_rearend hero lanechange night_rain snow_pileup
python scripts/evaluate_vehicle_pose.py
```

## Oynatıcıdaki hareket düzeltmesi

Eski kare-bağımsız kutu uydurması, normal araçlara büyük yanal sapmalar ve yön/hareket uyuşmazlığı veriyordu.
`mercy/ingest/road_motion.py` normal trafik için bütün zaman aralığını birlikte hesaplar:

- Sabit yol yönü ve sabit yanal şerit konumu.
- Negatif olamayan, sürekli hız eğrisi; yavaşlama ve durma mümkün, gövde geriye kaydırılmıyor.
- Karşı yönden trafik ancak yüksek güvenli tam-video kanıtıyla ayrı yön alıyor; tek yan/ön görünüşlü crop yeterli değil.
- Kaza/ego katılımcıları ve yüksek güvenli manevralar esnek kalıyor.
- Kalibre edilmiş kaza aralığı varsa uyum orada hesaplanıyor. Dışarıdaki belirsiz kamera devamı bu hesabı bozamıyor.
- Kaynak tespit aralıkları `observed_range` olarak korunuyor; fiziksel araçlar bütün klipte kalıcı, veri olmayan
  aralıklar varsayımlı devam olarak işaretleniyor. Anlar arasında aynı model nesnesi kullanılıyor.

Beş vakada 82 normal kaynak izi kontrol edildi: yanal hareket ve gövde yönüne ters hareket yok. Beş kaza izi artık
ortak zaman uyumu ve hız/ivme sınırlarıyla hesaplanıyor; ego/kamera hareketi de kesintisizleştirildi. Tam-video yön
incelemesi perspektif kaynaklı yanlış 90 derece dönüşleri düzeltiyor. Bu bir **hareket varsayımıdır**; mutlak konum/şerit genişliği/hız ölçümünün
doğrulandığı anlamına gelmez. Kaynak kutuya uyum bazı izlerde kötüleşebilir; önce/sonra artıklar dosyalarda saklanır.

```bash
python scripts/refine_vehicles.py --motion-only --case cutin_rearend hero lanechange night_rain snow_pileup
```

Varsayımsız önceki izler `vehicles/unconstrained-motion.json`, davranış incelemesi `vehicles/motion-review.json`,
hareket kısıtları ve hatalar `vehicles/road-motion.json` içindedir. Tekrar çalıştırma zaten kısıtlanmış izleri bir daha
uydurmak yerine kaydedilmiş başlangıç izlerini kullanır. Tam yeniden üretme akışı da bu aşamayı içerir.
