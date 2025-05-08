# Import necessary libraries
import streamlit as st
import os
from retrieval import retrieve_relevant_passages, get_relevant_pages, save_pages
from pdf_preprocessing import open_and_read_pdf, encode_sentences, split_sentences, create_sentence_chunks, clear_tokens, create_text_chunks, encode_text_chunks, MIN_TOKEN_LEN
from sentence_transformers import SentenceTransformer
from html_templates import css, user_template, bot_template
import torch
from faiss_vd import create_faiss_index, add_embeddings_to_faiss
from generation_llm import ask, prompt_formatter
from time import perf_counter as timer

torch.cuda.is_available()
torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)]
torch.classes.__path__ = []

st.set_page_config(page_title="PDF ile Sohbet Et", page_icon=":books:")
st.write(css, unsafe_allow_html=True)

PAGE_TITLE = "PDF ile Sohbet Et"
PAGE_ICON = ":books:"
TEXT_INPUT = "Belgeniz hakkında bir soru sorun:"
embedding_model = SentenceTransformer(model_name_or_path="all-mpnet-base-v2", device="cuda" if torch.cuda.is_available() else "cpu")

def main():
    st.header(PAGE_TITLE + PAGE_ICON)

    if "faiss_index_bool" not in st.session_state:
        st.session_state.faiss_index_bool = False 

    if "pdf_stream" not in st.session_state:
        st.session_state.pdf_stream = None 

    if "pdf_name" not in st.session_state:
        st.session_state.pdf_name = None

    with st.sidebar:
        st.subheader("Belgeniz")
        pdf_doc = st.file_uploader("PDF dosyanızı buraya yükleyin ve 'İşle'ye tıklayın")
        if pdf_doc and st.button("İşle"):
            with st.spinner("İşleniyor..."):
                st.session_state.pdf_stream = pdf_doc.read()
                st.session_state.pdf_name = pdf_doc.name.split(".")[0]
                pages_and_texts = open_and_read_pdf(st.session_state.pdf_stream)
                encode_sentences(pages_and_texts=pages_and_texts)
                split_sentences(pages_and_texts=pages_and_texts)
                pages_and_chunks = create_sentence_chunks(pages_and_texts=pages_and_texts)
                pages_and_chunks_over_min_token_len = clear_tokens(pages_and_chunks=pages_and_chunks, min_token_len=MIN_TOKEN_LEN)
                text_chunks = create_text_chunks(pages_and_chunks_over_min_token_len=pages_and_chunks_over_min_token_len)

                text_chunk_embeddings = encode_text_chunks(text_chunks=text_chunks, embedding_model=embedding_model)

                faiss_index = create_faiss_index(embedding_dim=len(text_chunk_embeddings[0]))
                add_embeddings_to_faiss(faiss_index, text_chunk_embeddings)
                
                st.session_state.faiss_index = faiss_index
                st.session_state.text_chunks = text_chunks
                st.session_state.faiss_index_bool = True
                st.success("Belge işlendi. Artık sorular sorabilirsiniz!")

    if st.session_state.faiss_index_bool:
        user_question = st.text_input(TEXT_INPUT, key="user_question_input")

        if user_question and st.button("Sor"):
            st.write(user_template.replace("{{MSG}}", f"{user_question.capitalize()}"), unsafe_allow_html=True)
            with st.spinner("Yanıt oluşturuluyor..."):
                relevant_passages, distances, retrieve_time, indices = retrieve_relevant_passages(
                    user_question,
                    embedding_model,
                    st.session_state.faiss_index,
                    st.session_state.text_chunks,
                    top_k=5
                )
                answer = ""
                st.write(f"Getirme Süresi : {retrieve_time:.2f} saniye.")
                for passage, distance in zip(relevant_passages, distances[0]):
                    answer += f"{passage}\n"
                page_numbers = get_relevant_pages(pdf_stream=st.session_state.pdf_stream, page_indices=indices)
                save_pages(pdf_stream=st.session_state.pdf_stream, page_numbers=page_numbers, pdf_name=st.session_state.pdf_name)

                image_dir = f"static/{st.session_state.pdf_name}/"
                for image_file in sorted(os.listdir(image_dir))[:2]:  
                    st.image(os.path.join(image_dir, image_file))

                prompt = prompt_formatter(query=user_question, resources=answer)
                start_time = timer()
                output_text = ask(prompt=prompt, temperature=0.7, max_new_tokens=2048, format_answer_text=True, return_answer_only=True)
                end_time = timer()
                generation_time = end_time - start_time
                st.write(f"Oluşturma Süresi : {generation_time:.2f} saniye.")
                st.write(bot_template.replace("{{MSG}}", f"{output_text}"), unsafe_allow_html=True)

if __name__ == '__main__':
    main()
