import streamlit as st
from keybert import KeyBERT

chosen_model = st.selectbox(
    'Choose model',
    ('all-MiniLM-L6-v2', 'paraphrase-multilingual-MiniLM-L12-v2', 'paraphrase-mpnet-base-v2','paraphrase-multilingual-mpnet-base-v2'))

sample_text = 
    """Computer Science is the study of computers and computational systems.
    Unlike electrical and computer engineers, computer scientists deal mostly
    with software and software systems; this includes their theory, design,
    development, and application. Principal areas of study within Computer
    Science include artificial intelligence, computer systems and networks,
    security, database systems, human computer interaction, vision and graphics,
    numerical analysis, programming languages, software engineering, bioinformatics
    and theory of computing. Although knowing how to program is essential to
    the study of computer science, it is only one element of the field. Computer
    scientists design and analyze algorithms to solve programs and study the
    performance of computer hardware and software. The problems that computer
    scientists encounter range from the abstract-- determining what problems
    can be solved with computers and the complexity of the algorithms that
    solve them – to the tangible – designing applications that perform well
    on handheld devices, that are easy to use, and that uphold security measures."""

text = st.text_area(label="Enter text", value=sample_text)

btnResult = st.button('Extract Keywords')
if btnResult:
    kw_model = KeyBERT(chosen_model)
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2),use_maxsum=True, nr_candidates=20, top_n=5)
    st.write(keywords)
