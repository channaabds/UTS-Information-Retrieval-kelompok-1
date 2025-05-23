import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import TreebankWordTokenizer
from gensim.models import Word2Vec
import faiss

# Download stopwords sekali saja
nltk.download('stopwords')

# --- Preprocessing ---
stop_words = set(stopwords.words('english'))
tokenizer = TreebankWordTokenizer()

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = tokenizer.tokenize(text)
    tokens = [w for w in tokens if w not in stop_words]
    return tokens

# --- Load & preprocess data ---
@st.cache_data
def load_data():
    df = pd.read_json('News_Category_Dataset_v3.json', lines=True)
    df['text'] = df['headline'] + ' ' + df['short_description']
    df = df[['text', 'category']]
    df['tokens'] = df['text'].apply(preprocess)
    return df

df = load_data()

# --- Train Word2Vec model (cache resource agar hanya load sekali) ---
@st.cache_resource(show_spinner=False)
def train_word2vec(sentences):
    model = Word2Vec(sentences, vector_size=100, window=5, min_count=2, workers=4, sg=1, epochs=10)
    return model

w2v_model = train_word2vec(df['tokens'])

# --- Vectorize Documents ---
def document_vector(tokens, model):
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    if len(vectors) == 0:
        return np.zeros(model.vector_size, dtype='float32')
    return np.mean(vectors, axis=0).astype('float32')

@st.cache_data
def get_doc_vectors(tokens_series):
    # Gunakan w2v_model global yang sudah dicache sebagai resource
    return np.vstack(tokens_series.apply(lambda x: document_vector(x, w2v_model)))

doc_vectors = get_doc_vectors(df['tokens'])

# --- Build Faiss index ---
@st.cache_data(show_spinner=False)
def build_faiss_index(doc_vecs):
    dim = doc_vecs.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner Product similarity
    # Normalize vectors supaya cocok untuk cosine similarity
    faiss.normalize_L2(doc_vecs)
    index.add(doc_vecs)
    return index

faiss_index = build_faiss_index(doc_vectors)

# --- Search Function ---
def search(query, model, index, top_k=10):
    query_tokens = preprocess(query)
    query_vec = document_vector(query_tokens, model).reshape(1, -1)
    faiss.normalize_L2(query_vec)
    distances, indices = index.search(query_vec, top_k)
    return indices.flatten(), distances.flatten()

# --- Evaluation Function ---
def evaluate(results_idx, df, relevant_category, total_relevant):
    retrieved_categories = df.iloc[results_idx]['category']
    TP = sum(retrieved_categories == relevant_category)
    retrieved = len(results_idx)
    precision = TP / retrieved if retrieved > 0 else 0
    recall = TP / total_relevant if total_relevant > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1, TP

# --- Streamlit UI ---
st.title("🔍 Information Retrieval dengan Word2Vec + Faiss")
st.write("Dataset: HuffPost News Category (headline + short description)")

query = st.text_input("Masukkan query pencarian (contoh: 'omicron covid')")

if query:
    top_indices, scores = search(query, w2v_model, faiss_index)
    st.subheader(f"Hasil pencarian untuk: '{query}'")

    if all(score == 0 for score in scores):
        st.warning("Tidak ditemukan dokumen yang relevan.")
    else:
        for i, score in zip(top_indices, scores):
            if score > 0:
                st.markdown(f"**Skor Kemiripan:** {score:.4f}")
                st.markdown(f"**Kategori:** {df.iloc[i]['category']}")
                st.write(df.iloc[i]['text'])
                st.markdown("---")

        top_categories = df.iloc[top_indices]['category']
        most_common_category = top_categories.mode()[0]
        total_relevant_in_dataset = sum(df['category'] == most_common_category)

        precision, recall, f1, TP = evaluate(top_indices, df, most_common_category, total_relevant_in_dataset)
        st.markdown("📊 **Evaluasi Kinerja untuk Query Ini:**")
        st.write(f"Precision: {precision:.4f}")
        st.write(f"Recall: {recall:.4f}")
        st.write(f"F1-Score: {f1:.4f}")
        st.write(f"Jumlah Dokumen Relevan di Dataset: {total_relevant_in_dataset}")
        st.write(f"Jumlah Dokumen True Positives: {TP}")
