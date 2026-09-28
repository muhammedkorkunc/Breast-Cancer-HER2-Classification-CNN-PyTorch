# 🔬 Breast Cancer HER2 Status Multi-Class Classification with CNN (PyTorch)

### BCI Veri Seti Üzerinde Özel Konvolüsyonel Sinir Ağı (CNN) ile HER2 Skorlama Analizi

Bu depo; Fatih Sultan Mehmet Vakıf Üniversitesi Bilgisayar Mühendisliği **Derin Öğrenme (Deep Learning)** dersi kapsamında geliştirilen; histopatolojik meme kanseri görüntülerinde **HER2 (Human Epidermal Growth Factor Receptor 2)** durumunun 4 farklı sınıfa (`0`, `1+`, `2+`, `3+`) sınıflandırılmasına yönelik uçtan uca CNN modelleme hattını ve teknik proje raporunu içerir.

---

## 🏗️ Model Mimarisi (Custom CNN Pipeline)

Model, 256x256x3 boyutundaki girdi tensörünü 4 aşamalı konvolüsyon-havuzlama ve ardından 3 tam bağlantılı (FC) katmandan geçirerek sınıflandırır:

```text
Girdi (256x256x3)
  ──> Conv2D(3 -> 8, kernel 3x3) + ReLU + MaxPool(2x2) ──> (128x128x8)
  ──> Conv2D(8 -> 16, kernel 3x3) + ReLU + MaxPool(2x2) ──> (64x64x16)
  ──> Conv2D(16 -> 32, kernel 3x3) + ReLU + MaxPool(2x2) ──> (32x32x32)
  ──> Conv2D(32 -> 64, kernel 3x3) + ReLU + MaxPool(2x2) ──> (16x16x64)
  ──> Flatten (16x16x64 = 16,384)
  ──> Linear(16384 -> 64) + ReLU
  ──> Linear(64 -> 128) + ReLU
  ──> Linear(128 -> 256) + ReLU
  ──> Linear(256 -> 4)  [Çıktı Sınıfları: 0, 1+, 2+, 3+]

  📊 Hiperparametreler ve Eğitim ÖzetiHiperparametreDeğer / AçıklamaFrameworkPyTorch & TorchvisionOptimizasyon AlgoritmasıAdam Optimizer (lr = 0.001)Kayıp FonksiyonuCrossEntropyLossBatch Size & Epoch32 batch size, 10 epochGirdi Ön İşlemeResize(256, 256), ToTensor, ImageNet NormalizasyonuModel İzlemeTensorBoard (Loss/Train, Loss/Val, Accuracy Metrikleri)

  📈 Başarım ve Karmaşıklık Matrisi (Confusion Matrix)
Modelin test kümesi üzerindeki sınıf bazlı başarı oranları:

Class 0: Model veri dengesizliği nedeniyle 0 sınıfında zorlanmıştır.

Class 1+: %27.23 doğruluk

Class 2+: %78.92 doğruluk (En yüksek doğru tespit oranı)

Class 3+: %67.83 doğruluk (Yüksek pozitiflik ayrımı)

Detaylı hata analizi ve epoch-loss eğrileri korumali_rapor/ altındaki teknik raporda belgelenmiştir.

Copyright (c) 2026 Muhammed Emin Korkunç. All Rights Reserved.

Yazarın açık yazılı izni olmaksızın kısmen veya tamamen kopyalanması,
dağıtılması, çoğaltılması veya ticari/akademik amaçla izinsiz kullanımı kesinlikle yasaktır.

👨‍💻 Geliştirici / Author
Muhammed Emin Korkunç

GitHub: @muhammedkorkunc

LinkedIn: Muhammed Emin Korkunç

Email: muhammedemin.korkunc@gmail.com
```
