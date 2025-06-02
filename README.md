# PDF Chatbot (Türkçe RAG - Streamlit)

Bu proje, PDF dosyalarınızdan Türkçe ve ayrıntılı cevaplar üreten, modern arayüze sahip bir Soru-Cevap (RAG) uygulamasıdır. Büyük dil modeli (LLM) ve vektör tabanlı arama (FAISS) ile çalışır. Kullanıcı dostu, tamamen Türkçe ve kolayca genişletilebilir bir yapıya sahiptir.

---

## Özellikler
- PDF yükleyip, belgeye özel Türkçe ve ayrıntılı sorular sorabilirsiniz.
- Büyük dil modeli (LLM) ve embedding modeli seçimi yapabilirsiniz.
- Modern, kullanıcı dostu ve tamamen Türkçe arayüz.
- Hızlı ve verimli vektör arama (FAISS).
- Uzun, analitik ve kaynaklı cevaplar.
- Profesyonel yazılım mühendisliği prensipleri ve design pattern entegrasyonu:
  - **Factory Pattern:** Model ve embedding seçimi.
  - **Strategy Pattern:** Farklı embedding stratejileri.
  - **Singleton Pattern:** FAISS index ve embedding modelinin tekil tutulması.
  - **Adapter Pattern:** PDF okuma işlemlerinde esneklik.
  - **Observer Pattern:** UI güncellemeleri (Streamlit session state).

---

## 1. Gereksinimler
- Python 3.8 veya üzeri
- CUDA destekli GPU (tavsiye edilir, CPU ile de çalışır fakat yavaş olabilir)
- [HuggingFace hesabı](https://huggingface.co/join) ve erişim token'ı (LLM için gereklidir)

---

## 2. Kurulum

### a) Depoyu Klonlayın
```bash
git clone https://github.com/kullaniciadi/PDF-CHATBOT-main.git
cd PDF-CHATBOT-main
```

### b) Sanal Ortam Oluşturun (Önerilir)
```bash
python -m venv venv
source venv/bin/activate  # Windows için: venv\Scripts\activate
```

### c) Gereksinimleri Yükleyin
```bash
pip install -r requirements.txt
```

### d) Spacy Modelini İndirin
```bash
python -m spacy download en_core_web_sm
```

---

## 3. HuggingFace Token Ayarı
`llm.py` dosyasında, kendi HuggingFace erişim tokenınızı girin:
```python
login(token="hf_xxx...")
```
Token almak için: https://huggingface.co/settings/tokens

---

## 4. Kullanım

### a) Uygulamayı Başlatın
```bash
streamlit run main.py
```

### b) PDF Yükleyin ve Soru Sorun
- Sol menüden PDF dosyanızı yükleyin ve "İşle"ye tıklayın.
- Soru kutusuna istediğiniz soruyu yazıp "Sor" butonuna basın.
- Cevaplar, ilgili PDF sayfa görselleriyle birlikte ekranda gösterilecektir.

---

## 5. Testler
Projeyi fonksiyonel olarak test etmek için:
```bash
pytest -s test_functional.py
```
> Not: Testler için `test_dataset.pdf` dosyasının klasörde bulunması gerekir.

---

## 6. Sık Karşılaşılan Sorunlar & Çözümler
- **CUDA Hatası:** GPU yoksa veya CUDA kurulumu eksikse, model CPU'da çalışır fakat yavaş olur.
- **HuggingFace Token Hatası:** Token'ı doğru girdiğinizden emin olun.
- **Model İndirme Sorunu:** İnternet bağlantınızı ve HuggingFace erişiminizi kontrol edin.
- **PDF Okuma Hatası:** PDF dosyanızın bozuk olmadığından emin olun.

---

## 7. Geliştirici Notları
- Yeni bir model veya embedding eklemek için sadece ilgili sınıfa ekleme yapmanız yeterli.
- Kodun tamamı SOLID prensiplerine uygun, modüler ve genişletilebilir şekilde tasarlanmıştır.
- Büyük PDF'ler ve uzun cevaplar için optimize edilmiştir.

---

## 8. Katkı
Pull request ve issue açabilirsiniz. Her türlü katkı ve öneriye açığız!

---

## 9. Dosya ve Klasör Yapısı
- `main.py` : Streamlit arayüzü ve ana uygulama
- `llm.py` : LLM yükleme ve konfigürasyon
- `generation_llm.py` : Prompt oluşturma ve cevap üretimi
- `pdf_preprocessing.py` : PDF metin çıkarımı ve ön işleme
- `retrieval.py` : Vektör arama ve sayfa kaydetme
- `faiss_vd.py` : FAISS vektör veritabanı işlemleri
- `html_templates.py` : Arayüz şablonları ve CSS
- `test_functional.py` : Otomatik fonksiyonel testler
- `requirements.txt` : Gerekli Python paketleri
- `test_dataset.pdf` : Test PDF dosyası
- `static/` : Geçici görsellerin kaydedildiği klasör

---

## 10. Lisans
MIT
