# Importing necessarry libraries
import streamlit as st
from retrieval import retrieve_relevant_passages, get_relevant_pages, save_pages
from pdf_preprocessing import open_and_read_pdf, encode_sentences, split_sentences, create_sentence_chunks, clear_tokens, create_text_chunks, encode_text_chunks, MIN_TOKEN_LEN
from sentence_transformers import SentenceTransformer
from html_templates import css, user_template, bot_template
import torch
from faiss_vd import create_faiss_index, add_embeddings_to_faiss
from generation_llm import ask, prompt_formatter
from time import perf_counter as timer
from llm import ModelFactory, SentenceTransformerEmbedding, FaissIndexSingleton, FitzPDFReader

st.set_page_config(page_title = "PDF'inizle Sohbet Edin", page_icon = ":books:")
st.write(css, unsafe_allow_html=True)

SAYFA_BASLIGI = "PDF'inizle Sohbet Edin"
SAYFA_IKON = ":books:"
SORU_METNI = "Belgeniz hakkında bir soru sorun:"
embedding_model = SentenceTransformer(model_name_or_path = "all-mpnet-base-v2", device = "cuda" if torch.cuda.is_available() else "cpu")

def main():
    st.header(SAYFA_BASLIGI + " " + SAYFA_IKON)

    if "faiss_index_bool" not in st.session_state:
        st.session_state.faiss_index_bool = False 

    if "pdf_stream" not in st.session_state:
        st.session_state.pdf_stream = None 

    # Model ve embedding seçenekleri
    model_options = {
        "Gemma 7B (Google)": "gemma-7b-it",
        "Gemma 2B (Google)": "gemma-2-2b-it"
        # Gerekirse başka modeller eklenebilir
    }
    embedding_options = {
        "SentenceTransformer (mpnet-base-v2)": "all-mpnet-base-v2"
        # Gerekirse başka embedding modelleri eklenebilir
    }

    with st.sidebar:
        st.subheader("Belgeniz")
        selected_model = st.selectbox("Dil Modeli Seçiniz", list(model_options.keys()), key="model_select")
        selected_embedding = st.selectbox("Embedding Modeli Seçiniz", list(embedding_options.keys()), key="embedding_select")
        pdf_doc = st.file_uploader("PDF dosyanızı buraya yükleyin ve 'İşle'ye tıklayın", type=["pdf"])
        if st.session_state.get("selected_model") != selected_model:
            st.session_state.selected_model = selected_model
            st.session_state.llm_model = ModelFactory.get_model(model_options[selected_model])
        if st.session_state.get("selected_embedding") != selected_embedding:
            st.session_state.selected_embedding = selected_embedding
            st.session_state.embedding_strategy = SentenceTransformerEmbedding(embedding_options[selected_embedding])
        if pdf_doc and st.button("İşle"):
            with st.spinner("İşleniyor..."):
                st.session_state.pdf_stream = pdf_doc.read()
                # Adapter Pattern ile PDF okuma
                pdf_reader = FitzPDFReader()
                doc = pdf_reader.read(st.session_state.pdf_stream)
                pages_and_texts = open_and_read_pdf(st.session_state.pdf_stream)
                encode_sentences(pages_and_texts=pages_and_texts)
                split_sentences(pages_and_texts=pages_and_texts)
                pages_and_chunks = create_sentence_chunks(pages_and_texts=pages_and_texts)
                pages_and_chunks_over_min_token_len = clear_tokens(pages_and_chunks=pages_and_chunks, min_token_len=MIN_TOKEN_LEN)
                text_chunks = create_text_chunks(pages_and_chunks_over_min_token_len=pages_and_chunks_over_min_token_len)

                # Strategy Pattern ile embedding
                text_chunk_embeddings = st.session_state.embedding_strategy.encode(text_chunks)

                # Singleton Pattern ile FAISS index
                faiss_index = FaissIndexSingleton.get_instance(dim=len(text_chunk_embeddings[0]))
                faiss_index.add(text_chunk_embeddings.cpu().numpy() if hasattr(text_chunk_embeddings, 'cpu') else text_chunk_embeddings)
                
                st.session_state.faiss_index = faiss_index
                st.session_state.text_chunks = text_chunks
                st.session_state.faiss_index_bool = True
                st.success("Belge işlendi. Artık sorular sorabilirsiniz!")

    if st.session_state.faiss_index_bool:
        user_question = st.text_input(SORU_METNI, key="user_question_input")

        if user_question and st.button("Sor"):
            st.write(user_template.replace("{{MSG}}", f"{user_question.capitalize()}"), unsafe_allow_html=True)
            with st.spinner("Cevap oluşturuluyor..."):
                relevant_passages, distances, retrieve_time, indices = retrieve_relevant_passages(
                    user_question,
                    st.session_state.embedding_strategy,
                    st.session_state.faiss_index,
                    st.session_state.text_chunks,
                    top_k=5
                )
                answer = ""
                st.write(f"Getirme Süresi : {retrieve_time:.2f} saniye.")
                for passage, distance in zip(relevant_passages, distances[0]):
                    answer += f"{passage}\n"
                page_numbers = get_relevant_pages(pdf_stream = st.session_state.pdf_stream, page_indices = indices)
                save_pages(pdf_stream = st.session_state.pdf_stream, page_numbers = page_numbers)
                for i in range(1, 4):
                    st.image(f"static/images/page{i}.png")
                prompt = prompt_formatter(query=user_question, resources=answer)
                start_time = timer()
                output_text = ask(prompt=prompt, temperature=0.7, max_new_tokens=1024, format_answer_text=True, return_answer_only=True)
                end_time = timer()
                generation_time = end_time - start_time
                st.write(f"Cevap Oluşturma Süresi : {generation_time:.2f} saniye.")
                st.write(bot_template.replace("{{MSG}}", f"{output_text}"), unsafe_allow_html=True)

    # Kullanıcı farkındalığı için imleç ve tooltip desteği
    st.markdown("""
    <style>
    .stButton>button { cursor: pointer; }
    .stTextInput>div>input { cursor: text; }
    .stFileUploader>div { cursor: pointer; }
    .stSpinner { cursor: progress; }
    .stImage { transition: box-shadow 0.3s; }
    .stImage:hover { box-shadow: 0 0 10px #4F8BF9; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<small>İpucu: Bir butonun veya alanın üzerine geldiğinizde açıklama görebilirsiniz.</small>", unsafe_allow_html=True)

if __name__ == '__main__':
    main()