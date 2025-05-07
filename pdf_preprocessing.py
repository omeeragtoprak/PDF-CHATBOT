# Importing necessarry libraries
import re
import pandas as pd
from tqdm.auto import tqdm
import random
import fitz
from spacy.lang.en import English
import streamlit as st

# Global Variables
NLP_model = English()
NLP_model.add_pipe("sentencizer")
NUM_SENTENCE_CHUNK_SIZE = 10
MIN_TOKEN_LEN = 30

# PDF preprocessing functions 
def text_formatter(text: str) -> str:
    cleaned_text = text.replace("\n", " ").strip()

    return cleaned_text

def open_and_read_pdf(pdf_stream) -> list[dict]:
    doc = fitz.open(stream=pdf_stream, filetype = "pdf")
    pages_and_texts = []
    print("Reading the PDF and creating the dictionary...")
    for page_number, page in tqdm(enumerate(doc)):
        text = page.get_text()
        text = text_formatter(text = text)

        pages_and_texts.append({"page_number": page_number - 41,
                                "page_char_count": len(text),
                                "page_word_count": len(text.split(" ")),
                                "page_sentence_count": len(text.split(". ")),
                                "page_token_count": len(text) / 4,
                                "text": text})
        
    return pages_and_texts
    
def random_samples(pages_and_texts: dict, num_examples: int):
    return random.sample(pages_and_texts, num_examples)

def encode_sentences(pages_and_texts: dict):
    print("Encoding the sentences...")
    for item in tqdm(pages_and_texts):
        item["sentences"] = list(NLP_model(item["text"]).sents)
        item["sentences"] = [str(sentence) for sentence in item["sentences"]]
        item["page_sentence_count_spacy"] = len(item["sentences"])

def split_list(input_list: list,
               slice_size: int = NUM_SENTENCE_CHUNK_SIZE) -> list[list[str]]:
    return [input_list[i:i+slice_size] for i in range(0, len(input_list), slice_size)]

def split_sentences(pages_and_texts: dict):
    print("Splitting the sentences...")
    for item in tqdm(pages_and_texts):
        item["sentence_chunks"] = split_list(input_list = item["sentences"],
                                             slice_size = NUM_SENTENCE_CHUNK_SIZE)
        item["num_chunks"] = len(item["sentence_chunks"])

def create_sentence_chunks(pages_and_texts):
    pages_and_chunks = []
    print("Creating the sentence chunks...")
    for item in tqdm(pages_and_texts):
        for sentence_chunk in item["sentence_chunks"]:
            chunk_dict = {}
            chunk_dict["page_number"] = item["page_number"]

            joined_sentence_chunk = "".join(sentence_chunk).replace("  ", " ").strip()
            joined_sentence_chunk = re.sub(r"\.([A-Z])", r'. \1', joined_sentence_chunk)

            chunk_dict["sentence_chunk"] = joined_sentence_chunk

            chunk_dict["chunk_char_count"] = len(joined_sentence_chunk)
            chunk_dict["chunk_word_count"] = len([word for word in joined_sentence_chunk.split(" ")])
            chunk_dict["chunk_token_count"] = len(joined_sentence_chunk) / 4

            pages_and_chunks.append(chunk_dict)

    return pages_and_chunks

def clear_tokens(pages_and_chunks: dict, min_token_len: int = MIN_TOKEN_LEN):
    df = pd.DataFrame(pages_and_chunks)

    pages_and_chunks_over_min_token_len = df[df["chunk_token_count"] > min_token_len].to_dict(orient = "records")

    return pages_and_chunks_over_min_token_len

def embed_text_chunks(pages_and_chunks_over_min_token_len, embedding_model):
    for item in tqdm(pages_and_chunks_over_min_token_len):
        item["embedding"] = embedding_model.encode(item["sentence_chunk"])

def create_text_chunks(pages_and_chunks_over_min_token_len):
    text_chunks = [item["sentence_chunk"] for item in pages_and_chunks_over_min_token_len]
    
    return text_chunks

def encode_text_chunks(text_chunks, embedding_model):
    text_chunk_embeddings = embedding_model.encode(text_chunks, batch_size = 32, convert_to_tensor = True)

    return text_chunk_embeddings