import streamlit as st
from keybert import KeyBERT

st.header('Keywords Extraction')

chosen_model = st.selectbox(
    'Model',
    ('all-MiniLM-L6-v2', 'paraphrase-multilingual-MiniLM-L12-v2', 'paraphrase-mpnet-base-v2','paraphrase-multilingual-mpnet-base-v2'))

top_n = st.slider('Number of keywords', min_value=5, max_value=20, value=10, step=1)
ngram_range = st.slider('ngram range',min_value=1, max_value=3, value=(1, 2))

sample_text = """Changes in carbon dioxide levels in the atmosphere in the past played a key role in determining the timing and location of early human species mating. Neanderthal and Denisovan species that had different environmental preferences were able to meet, interbreed, and produce mixed offspring.
This was told by the international team in the journal Science which was published on August 10, 2023. This research responded to the results of research in 2018. At that time a number of researchers announced the discovery of a female individual -nicknamed Denny-who lived 90,000 years ago."""

text = st.text_area(label="Enter text", value=sample_text)

btnResult = st.button('Extract Keywords')
if btnResult:
    kw_model = KeyBERT(chosen_model)
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=ngram_range,use_maxsum=True, nr_candidates=20, top_n=top_n)
    st.write(keywords)
