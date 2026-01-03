# app.py

import streamlit as st
import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim import corpora, models
from transformers import T5ForConditionalGeneration, T5Tokenizer
import re

# --- INITIAL SETUP & MODEL LOADING ---
st.set_page_config(page_title="H-FLDA Biomedical Summarizer", layout="wide")

# --- FIXED NLTK DOWNLOAD ---
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('tokenizers/punkt_tab')
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('punkt')
        nltk.download('punkt_tab')
        nltk.download('stopwords')

download_nltk_data()

from nltk.corpus import stopwords

# --- CACHE MODEL LOADING ---
@st.cache_resource
def load_models_and_data():
    """Loads all necessary models and data."""
    model_path = 't5-small-bio-summarizer'
    try:
        tokenizer = T5Tokenizer.from_pretrained(model_path)
        model = T5ForConditionalGeneration.from_pretrained(model_path)
    except Exception as e:
        st.error(f"Failed to load fine-tuned T5 model from '{model_path}'. Error: {e}")
        return None, None, None, None

    # Load dataset for LDA
    try:
        df_pubmed = pd.read_csv('datasets/pubmed_cancer_with_abstracts.csv')
        if "abstract" not in df_pubmed.columns:
            st.error("CSV must contain a column named 'abstract'")
            return None, None, None, None
        df_pubmed.dropna(subset=["abstract"], inplace=True)
        docs = df_pubmed["abstract"].tolist()
    except FileNotFoundError:
        st.error("Error: 'datasets/pubmed_cancer_with_abstracts.csv' not found.")
        return None, None, None, None

    stop_words = set(stopwords.words('english'))
    texts = [[word for word in doc.lower().split() if word not in stop_words and word.isalpha()] for doc in docs]

    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    lda_model = models.LdaModel(corpus, num_topics=10, id2word=dictionary, passes=5)

    return tokenizer, model, lda_model, dictionary

# --- CORE SUMMARIZATION LOGIC ---

# Step 1: Rule-based Extractive Summarization
def rule_based_extractive_summary(text, top_n=5):
    sentences = nltk.sent_tokenize(text)
    if len(sentences) <= top_n:
        return sentences

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(sentences)

    sentence_scores = tfidf_matrix.sum(axis=1)
    ranked_sentence_indices = sentence_scores.argsort(axis=0)[::-1]
    top_indices = sorted(ranked_sentence_indices[:top_n].A1)

    return [sentences[i] for i in top_indices]

# Step 2: LDA Topic Modeling re-ranking
def score_sentences_with_lda(sentences, lda_model, dictionary):
    scores = []
    stop_words = set(stopwords.words("english"))
    for sentence in sentences:
        words = [word for word in sentence.lower().split() if word not in stop_words and word.isalpha()]
        bow = dictionary.doc2bow(words)
        if not bow:
            scores.append(0)
            continue
        topic_distribution = lda_model.get_document_topics(bow, minimum_probability=0.0)
        sentence_score = max([prob for _, prob in topic_distribution])
        scores.append(sentence_score)
    return scores

# Step 3: Abstractive Summarization
def abstractive_summary(text, tokenizer, model):
    preprocess_text = "biomedical summary: " + text.strip().replace("\n", "")
    tokenized_text = tokenizer.encode(preprocess_text, return_tensors="pt")

    summary_ids = model.generate(
        tokenized_text,
        max_length=200,
        min_length=60,
        num_beams=8,
        length_penalty=1.2,
        no_repeat_ngram_size=4,
        early_stopping=True,
        repetition_penalty=2.5,
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# --- HYBRID PIPELINE ---
def clean_sentence(sentence: str) -> str:
    """Remove bad starts and capitalize."""
    sentence = sentence.strip()
    for bad_start in [",", ".", "and ", "but ", "or "]:
        if sentence.lower().startswith(bad_start.strip()):
            sentence = sentence[len(bad_start):].strip()
    if sentence:
        sentence = sentence[0].upper() + sentence[1:]
    return sentence

def generate_hybrid_summary(text, lda_model, dictionary, t5_tokenizer, t5_model):
    extractive_sentences = rule_based_extractive_summary(text, top_n=10)
    lda_scores = score_sentences_with_lda(extractive_sentences, lda_model, dictionary)

    ranked_sentences = sorted(zip(extractive_sentences, lda_scores), key=lambda x: x[1], reverse=True)
    
    # ✅ clean BEFORE abstractive step
    top_sentences = [clean_sentence(sentence) for sentence, score in ranked_sentences[:5]]

    extractive_input_for_t5 = " ".join(top_sentences)
    final_summary = abstractive_summary(extractive_input_for_t5, t5_tokenizer, t5_model)

    # ✅ clean AFTER abstractive step too
    final_summary = clean_sentence(final_summary)

    # Format into 4-5 lines
    summary_sentences = re.split(r'(?<=[.!?])\s+', final_summary)
    return "\n".join(summary_sentences[:5])

# --- STREAMLIT UI ---
def main_app():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Summarizer"])

    if page == "Summarizer":
        st.title("🔬 H-FLDA: Hybrid Biomedical Text Summarizer")
        st.markdown("""
        This tool implements the hybrid summarization pipeline from the paper: 
        **"H-FLDA & Biomedical Text summarization using a Hybrid Approach of Rule Based Filtering, LDA Topic Modeling and Abstractive NLP"**.
        Enter a biomedical abstract below OR upload a file to generate a concise summary.
        """)

        input_text = st.text_area("Enter Biomedical Abstract Here:", height=250, placeholder="Paste the abstract...")

        uploaded_file = st.file_uploader("Or upload a document (TXT, DOCX, PDF)", type=["txt", "docx", "pdf"])
        if uploaded_file is not None:
            if uploaded_file.type == "text/plain":
                input_text = uploaded_file.read().decode("utf-8")
            elif uploaded_file.type == "application/pdf":
                from PyPDF2 import PdfReader
                pdf_reader = PdfReader(uploaded_file)
                input_text = " ".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                from docx import Document
                doc = Document(uploaded_file)
                input_text = " ".join([para.text for para in doc.paragraphs])

        if st.button("Generate Summary", type="primary"):
            if input_text.strip():
                with st.spinner("Summarizing... Please wait."):
                    t5_tokenizer, t5_model, lda_model, dictionary = load_models_and_data()
                    if t5_model is None:
                        st.stop()
                    final_summary = generate_hybrid_summary(input_text, lda_model, dictionary, t5_tokenizer, t5_model)
                st.subheader("Generated Summary")
                st.write(final_summary)
            else:
                st.warning("Please enter text or upload a file to summarize.")

# --- LOGIN SYSTEM ---
USERS_CSV = "users.csv"

try:
    users = pd.read_csv(USERS_CSV)
except FileNotFoundError:
    users = pd.DataFrame(columns=["username", "password"])
    users.to_csv(USERS_CSV, index=False)

if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False

if not st.session_state['authenticated']:
    st.title("Login / Register")
    st.write("Please enter your credentials to access the summarizer.")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Login"):
            if ((users["username"] == username) & (users["password"] == password)).any():
                st.session_state['authenticated'] = True
                st.success(f"Welcome back, {username}!")
                st.rerun()
            else:
                st.error("Invalid username or password ❌")

    with col2:
        if st.button("Register"):
            if username.strip() == "" or password.strip() == "":
                st.error("Username and password cannot be empty ❌")
            elif username in users["username"].values:
                st.warning("Username already exists. Please choose another.")
            else:
                new_user = pd.DataFrame([[username, password]], columns=["username", "password"])
                users = pd.concat([users, new_user], ignore_index=True)
                users.to_csv(USERS_CSV, index=False)
                st.success("Account created ✅ Please login now.")
else:
    main_app()
