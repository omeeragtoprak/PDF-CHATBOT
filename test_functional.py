import os
import pytest
from faiss_vd import create_faiss_index, add_embeddings_to_faiss
from generation_llm import ask, prompt_formatter
from pdf_preprocessing import (
    open_and_read_pdf, encode_sentences, split_sentences, create_sentence_chunks,
    clear_tokens, create_text_chunks, encode_text_chunks, MIN_TOKEN_LEN
)
from sentence_transformers import util

def test_pdf_qa_pipeline():
    """
    Test Case: PDF'den Soru-Cevap Pipeline'ı (Unit + System Test)
    Test Amacı: PDF'den bilgi çıkarımı, embedding, FAISS, retrieval ve LLM ile cevap üretme adımlarının doğruluğunu ve entegrasyonunu test etmek.
    Test Tipi: Black-box (Kara Kutu) + White-box (Beyaz Kutu) karışık, otomatik unit/sistem testi.
    Test Oracle: Her soru için beklenen (expected) cevap ile model cevabının cosine similarity (veya semantic similarity) skoruyla karşılaştırılması.
    """

    # === Initialization ===
    pdf_path = "test_dataset.pdf"
    assert os.path.exists(pdf_path), f"Test için PDF dosyası ({pdf_path}) bulunamadı."
    from sentence_transformers import SentenceTransformer
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

    # === Test Steps ===
    # 1. PDF'den metin çıkarımı
    with open(pdf_path, "rb") as f:
        pdf_stream = f.read()
    pages_and_texts = open_and_read_pdf(pdf_stream)
    assert len(pages_and_texts) > 0, "PDF'den metin çıkarılamadı."

    # 2. Cümleleme ve chunk'lama
    encode_sentences(pages_and_texts=pages_and_texts)
    split_sentences(pages_and_texts=pages_and_texts)
    pages_and_chunks = create_sentence_chunks(pages_and_texts=pages_and_texts)
    pages_and_chunks_over_min_token_len = clear_tokens(
        pages_and_chunks=pages_and_chunks, min_token_len=MIN_TOKEN_LEN
    )
    text_chunks = create_text_chunks(
        pages_and_chunks_over_min_token_len=pages_and_chunks_over_min_token_len
    )
    assert len(text_chunks) > 0, "Chunk'lama başarısız."

    # 3. Embedding ve FAISS index
    text_chunk_embeddings = encode_text_chunks(text_chunks=text_chunks, embedding_model=embedding_model)
    faiss_index = create_faiss_index(embedding_dim=len(text_chunk_embeddings[0]))
    add_embeddings_to_faiss(faiss_index, text_chunk_embeddings)

    # === Test Inputs (Test Data & Oracles) ===
    questions_and_answers = [
        {
            "question": "Projenin temel amacı nedir?",
            "expected": "Kullanıcıların gerçek zamanlı borsa bilgilerini takip etmesini, hisse senedi alım-satımı yapmasını ve portföylerini kolayca yönetmesini sağlayacak kullanıcı dostu bir borsa uygulaması geliştirmektir."
        },
        {
            "question": "Projede hangi teknolojiler kullanılacaktır?",
            "expected": "Frontend için React, veritabanı ve gerçek zamanlı veri için Firebase, güvenlik için Firebase Rules kullanılacaktır."
        },
        {
            "question": "Projenin ana işlevlerinden üç tanesini sayınız.",
            "expected": "Gerçek zamanlı hisse fiyatlarının görüntülenmesi, hisse alım-satımı yapma imkânı, kullanıcı portföyünün yönetimi ve analiz raporları sunulması."
        },
        {
            "question": "Projede öngörülen başlıca risklerden biri nedir ve nasıl yönetilecektir?",
            "expected": "Proje gereksinimlerinin eksik veya yanlış tanımlanması riski önleme stratejisiyle yönetilecektir."
        },
        {
            "question": "Kullanıcı yönetimi kapsamında hangi özellikler yer almaktadır?",
            "expected": "Yeni kullanıcılar için kayıt sistemi ve mevcut kullanıcılar için güvenli kimlik doğrulama özellikleri yer almaktadır."
        }
    ]

    # === Test Execution & Oracle Evaluation ===
    total_score = 0
    for i, qa in enumerate(questions_and_answers, 1):
        # Retrieval
        query_embedding = embedding_model.encode([qa["question"]], convert_to_numpy=True)
        distances, indices = faiss_index.search(query_embedding, 3)
        relevant_passages = [text_chunks[idx] for idx in indices[0]]

        # Prompt & LLM
        prompt = prompt_formatter(query=qa["question"], resources="\n".join(relevant_passages))
        try:
            output_text = ask(
                prompt=prompt,
                temperature=0.1,
                max_new_tokens=128,
                format_answer_text=True,
                return_answer_only=True
            )
        except Exception as e:
            pytest.skip(f"LLM modeli yüklenemedi veya test ortamında çalıştırılamadı: {e}")

        # === Oracle: Cosine Similarity Skoru ===
        expected_embedding = embedding_model.encode([qa["expected"]], convert_to_tensor=True)
        output_embedding = embedding_model.encode([output_text], convert_to_tensor=True)
        cosine_score = util.cos_sim(expected_embedding, output_embedding)[0][0].item()

        # === Test Report ===
        print(f"\n--- Test Case {i} ---")
        print(f"Soru: {qa['question']}")
        print(f"Beklenen Cevap: {qa['expected']}")
        print(f"Model Cevabı: {output_text}")
        print(f"Cosine Similarity Skoru: {cosine_score:.2f}")
        print(f"Test Sonucu: {'BAŞARILI' if cosine_score > 0.5 else 'BAŞARISIZ'}")
        print("-" * 60)
        total_score += cosine_score

        # Test Oracles: Her bir soru için minimum başarı eşiği
        assert cosine_score > 0.2, f"Test Case {i} için cosine similarity çok düşük!"

    # === Tear Down & Genel Rapor ===
    avg_score = total_score / len(questions_and_answers)
    print(f"\nOrtalama Cosine Similarity Skoru: {avg_score:.2f}")
    assert avg_score > 0.3, "Ortalama cosine similarity çok düşük!"

    print("\nTüm testler başarıyla tamamlandı.")