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

st.set_page_config(page_title = "Chat With Your PDF", page_icon = ":books:")
st.write(css, unsafe_allow_html=True)

PAGE_TITLE = "Chat With Your PDF"
PAGE_ICON = ":books:"
TEXT_INPUT = "Ask a question about your document:"
embedding_model = SentenceTransformer(model_name_or_path = "all-mpnet-base-v2", device = "cuda" if torch.cuda.is_available() else "cpu")

def main():
    st.header(PAGE_TITLE + PAGE_ICON)

    if "faiss_index_bool" not in st.session_state:
        st.session_state.faiss_index_bool = False 

    if "pdf_stream" not in st.session_state:
        st.session_state.pdf_stream = None 

    with st.sidebar:
        st.subheader("Your document")
        pdf_doc = st.file_uploader("Upload your PDF here and click on 'Process'")
        if pdf_doc and st.button("Process"):
            with st.spinner("Processing..."):
                st.session_state.pdf_stream = pdf_doc.read()
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
                st.success("Document processed. Now you can ask questions!")

    if st.session_state.faiss_index_bool:
        user_question = st.text_input(TEXT_INPUT, key="user_question_input")

        if user_question and st.button("Ask"):
            st.write(user_template.replace("{{MSG}}", f"{user_question.capitalize()}"), unsafe_allow_html=True)
            with st.spinner("Generating the answer..."):
                relevant_passages, distances, retrieve_time, indices = retrieve_relevant_passages(
                    user_question,
                    embedding_model,
                    st.session_state.faiss_index,
                    st.session_state.text_chunks,
                    top_k=5
                )
                answer = ""
                st.write(f"Retrieve Time : {retrieve_time:.2f} seconds.")
                for passage, distance in zip(relevant_passages, distances[0]):
                    answer += f"{passage}\n"
                page_numbers = get_relevant_pages(pdf_stream = st.session_state.pdf_stream, page_indices = indices)
                save_pages(pdf_stream = st.session_state.pdf_stream, page_numbers = page_numbers)
                for i in range(1, 4):
                    st.image(f"static/images/page{i}.png")
                prompt = prompt_formatter(query=user_question, resources=answer)
                start_time = timer()
                output_text = ask(prompt=prompt, temperature=0.7, max_new_tokens=256, format_answer_text=True, return_answer_only=True)
                end_time = timer()
                generation_time = end_time - start_time
                st.write(f"Generation Time : {generation_time:.2f} seconds.")
                st.write(bot_template.replace("{{MSG}}", f"{output_text}"), unsafe_allow_html=True)

if __name__ == '__main__':
    main()