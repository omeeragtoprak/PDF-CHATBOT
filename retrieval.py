# Import necessarry libraries
from time import perf_counter as timer
import fitz
import streamlit as st
from pdf_preprocessing import open_and_read_pdf, encode_sentences, split_sentences, create_sentence_chunks

# Retrieve functions
def retrieve_relevant_passages(query, embedding_model, index, text_chunks, top_k=5):
    start_time = timer()
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)
    relevant_passages = [text_chunks[idx] for idx in indices[0]]
    end_time = timer()

    retrieve_time = end_time - start_time

    return relevant_passages, distances, retrieve_time, indices

def get_relevant_pages(pdf_stream, page_indices: list):
    pages_and_texts = open_and_read_pdf(pdf_stream)
    encode_sentences(pages_and_texts=pages_and_texts)
    split_sentences(pages_and_texts=pages_and_texts)
    pages_and_chunks = create_sentence_chunks(pages_and_texts=pages_and_texts)
    page_indices = page_indices[0]

    page_numbers = [pages_and_chunks[page_index]["page_number"] for page_index in page_indices]

    return page_numbers

def save_pages(pdf_stream, page_numbers):
    pdf_document = fitz.open(stream = pdf_stream, filetype = "pdf")
    print(page_numbers)
    for page_idx, page_number in enumerate(page_numbers):
        if 0 <= page_number < pdf_document.page_count:
            page = pdf_document.load_page(page_number + 41)
            pix = page.get_pixmap()

            png_file_path = f"static/images/page{page_idx}.png"
            pix.save(png_file_path)