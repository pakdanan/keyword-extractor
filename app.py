import streamlit as st
from keybert import KeyBERT

chosen_model = st.selectbox(
    'Choose model',
    ('all-MiniLM-L6-v2', 'paraphrase-multilingual-MiniLM-L12-v2', 'paraphrase-mpnet-base-v2','paraphrase-multilingual-mpnet-base-v2'))

sample_text = """
Changes in carbon dioxide levels in the atmosphere in the past played a key role in determining the timing and location of early human species mating. Neanderthal and Denisovan species that had different environmental preferences were able to meet, interbreed, and produce mixed offspring.
"""

text = st.text_area(label="Enter text", value=sample_text)

btnResult = st.button('Extract Keywords')
if btnResult:
    kw_model = KeyBERT(chosen_model)
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2),use_maxsum=True, nr_candidates=20, top_n=5)
    st.write(keywords)
