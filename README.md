# Streamlit PDF RAG Model

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

## Kurulum
```bash
pip install -r requirements.txt
```

## Kullanım
```bash
streamlit run main.py
```

## Testler
```bash
python test_patterns.py
```

## Geliştirici Notları
- Yeni bir model veya embedding eklemek için sadece ilgili sınıfa ekleme yapmanız yeterli.
- Kodun tamamı SOLID prensiplerine uygun, modüler ve genişletilebilir şekilde tasarlanmıştır.
- Büyük PDF'ler ve uzun cevaplar için optimize edilmiştir.

## Katkı
Pull request ve issue açabilirsiniz. 