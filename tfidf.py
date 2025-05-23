import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TreebankWordTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('stopwords')

# --- Preprocessing Function ---
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
tokenizer = TreebankWordTokenizer()

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = tokenizer.tokenize(text)
    tokens = [stemmer.stem(w) for w in tokens if w not in stop_words]
    return ' '.join(tokens)

# --- Load & Preprocess Data ---
@st.cache_data
def load_data():
    df = pd.read_json('News_Category_Dataset_v3.json', lines=True)
    df['text'] = df['headline'] + ' ' + df['short_description']
    df = df[['text', 'category']]
    df['clean_text'] = df['text'].apply(preprocess)
    return df

df = load_data()

# --- Representasi TF-IDF ---
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(df['clean_text'])

# --- Query Function ---
def search_tfidf(query, top_k=10):
    query = query.lower()
    query = re.sub(r'[^a-z\s]', '', query)
    tokens = tokenizer.tokenize(query)
    stemmed = [stemmer.stem(w) for w in tokens if w not in stop_words]
    query_clean = ' '.join(stemmed)
    
    query_vec = vectorizer.transform([query_clean])
    similarity_scores = cosine_similarity(query_vec, X_tfidf).flatten()
    
    top_indices = similarity_scores.argsort()[::-1][:top_k]
    return top_indices, similarity_scores

# --- Evaluation Function ---
def evaluate_system(query, top_indices, df):
    query = query.lower()
    query = re.sub(r'[^a-z\s]', '', query)
    tokens = tokenizer.tokenize(query)
    stemmed_query = [stemmer.stem(w) for w in tokens if w not in stop_words]

    def is_relevant(text):
        return any(word in text for word in stemmed_query)

    relevant_indices = df[df['clean_text'].apply(is_relevant)].index

    total_relevant = len(relevant_indices)
    retrieved = top_indices
    retrieved_set = set(retrieved)
    relevant_set = set(relevant_indices)

    true_positives = len(retrieved_set.intersection(relevant_set))
    retrieved_count = len(retrieved)

    precision = true_positives / retrieved_count if retrieved_count else 0
    recall = true_positives / total_relevant if total_relevant else 0
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1, total_relevant, true_positives

# --- Streamlit UI ---
st.title("🔍 Information Retrieval dengan TF-IDF + Cosine Similarity")
st.write("Dataset: HuffPost News Category (headline + short description)")

query = st.text_input("Masukkan query pencarian (contoh: 'omicron covid')")

if query:
    top_indices, scores = search_tfidf(query)
    st.subheader(f"Hasil pencarian untuk: '{query}'")

    if scores[top_indices[0]] == 0:
        st.warning("Tidak ditemukan dokumen yang relevan.")
    else:
        for i in top_indices:
            if scores[i] > 0:
                st.markdown(f"**Skor Kemiripan:** {scores[i]:.4f}")
                st.markdown(f"**Kategori:** {df.iloc[i]['category']}")
                st.write(df.iloc[i]['text'])
                st.markdown("---")

        precision, recall, f1, total_relevant, true_positives = evaluate_system(query, top_indices, df)

        st.markdown("📊 **Evaluasi Kinerja untuk Query Ini:**")
        st.markdown(f"- Precision: {precision:.4f}")
        st.markdown(f"- Recall: {recall:.4f}")
        st.markdown(f"- F1-Score: {f1:.4f}")
        st.markdown(f"- Jumlah Dokumen Relevan di Dataset: {total_relevant}")
        st.markdown(f"- Jumlah Dokumen True Positives: {true_positives}")



# import streamlit as st
# import pandas as pd
# import re
# import nltk
# import matplotlib.pyplot as plt
# from nltk.corpus import stopwords
# from nltk.stem import PorterStemmer
# from nltk.tokenize import TreebankWordTokenizer
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity

# nltk.download('stopwords')

# # --- Preprocessing Function ---
# stop_words = set(stopwords.words('english'))
# stemmer = PorterStemmer()
# tokenizer = TreebankWordTokenizer()

# def preprocess(text):
#     text = text.lower()
#     text = re.sub(r'[^a-z\s]', '', text)
#     tokens = tokenizer.tokenize(text)
#     tokens = [stemmer.stem(w) for w in tokens if w not in stop_words]
#     return ' '.join(tokens)

# # --- Load & Preprocess Data ---
# @st.cache_data
# def load_data():
#     df = pd.read_json('News_Category_Dataset_v3.json', lines=True)
#     df['text'] = df['headline'] + ' ' + df['short_description']
#     df = df[['text', 'category']]
#     df['clean_text'] = df['text'].apply(preprocess)
#     return df

# df = load_data()

# # --- TF-IDF Representation ---
# vectorizer = TfidfVectorizer()
# X_tfidf = vectorizer.fit_transform(df['clean_text'])

# # --- Search Function ---
# def search_tfidf(query, top_k=10):
#     query = query.lower()
#     query = re.sub(r'[^a-z\s]', '', query)
#     tokens = tokenizer.tokenize(query)
#     stemmed = [stemmer.stem(w) for w in tokens if w not in stop_words]
#     query_clean = ' '.join(stemmed)
    
#     query_vec = vectorizer.transform([query_clean])
#     similarity_scores = cosine_similarity(query_vec, X_tfidf).flatten()
    
#     top_indices = similarity_scores.argsort()[::-1][:top_k]
#     return top_indices, similarity_scores

# # --- Streamlit UI ---
# st.title("🔍 Information Retrieval dengan TF-IDF + Cosine Similarity")
# st.write("Dataset: HuffPost News Category (headline + short description)")

# query = st.text_input("Masukkan query pencarian (contoh: 'omicron covid')")

# if query:
#     top_indices, scores = search_tfidf(query)
#     st.subheader(f"Hasil pencarian untuk: '{query}'")

#     if scores[top_indices[0]] == 0:
#         st.warning("Tidak ditemukan dokumen yang relevan.")
#     else:
#         top_scores = scores[top_indices]
#         top_titles = [df.iloc[i]['text'][:60] + '...' for i in top_indices]

#         # --- Visualisasi Skor Similarity ---
#         st.write("### 📊 Visualisasi Skor Kemiripan (Top 10)")
#         fig1, ax1 = plt.subplots()
#         ax1.barh(top_titles[::-1], top_scores[::-1], color='skyblue')
#         ax1.set_xlabel("Similarity Score")
#         ax1.set_title("Top 10 Hasil Pencarian")
#         plt.tight_layout()
#         st.pyplot(fig1)

#         # --- Tampilkan Dokumen + Index ---
#         st.write("### 📄 Dokumen Top 10")
#         for i in top_indices:
#             if scores[i] > 0:
#                 st.markdown(f"**Index Dokumen:** `{i}`")
#                 st.markdown(f"**Skor Kemiripan:** {scores[i]:.4f}")
#                 st.markdown(f"**Kategori:** {df.iloc[i]['category']}")
#                 st.write(df.iloc[i]['text'])
#                 st.markdown("---")

#         # --- Pilih dokumen relevan secara manual ---
#         st.write("### ✍️ Tandai Dokumen yang Relevan")
#         relevant_selected = st.multiselect(
#             "Pilih index dokumen relevan dari hasil di atas:",
#             options=[int(i) for i in top_indices]
#         )
#         relevant_docs = set(relevant_selected)
#         retrieved_docs = set(top_indices)

#         # --- Evaluasi Sistem ---
#         st.write("### ✅ Evaluasi Sistem")

#         true_positive = len(relevant_docs & retrieved_docs)
#         false_positive = len(retrieved_docs - relevant_docs)
#         false_negative = len(relevant_docs - retrieved_docs)

#         precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0
#         recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0
#         f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

#         # --- Tampilkan nilai numerik ---
#         st.write(f"**Precision:** {precision:.2f}")
#         st.write(f"**Recall:** {recall:.2f}")
#         st.write(f"**F1-Score:** {f1:.2f}")

#         # --- Visualisasi Evaluasi Metrik ---
#         st.write("### 📈 Visualisasi Evaluasi Metrik")
#         metrics = ['Precision', 'Recall', 'F1-Score']
#         values = [precision, recall, f1]

#         fig2, ax2 = plt.subplots()
#         bars = ax2.barh(metrics, values, color=['orange', 'green', 'blue'])
#         ax2.set_xlim(0, 1)
#         ax2.set_title("Evaluasi IR System")
#         for bar in bars:
#             width = bar.get_width()
#             ax2.text(width + 0.02, bar.get_y() + bar.get_height()/2, f"{width:.2f}", va='center')
#         plt.tight_layout()
#         st.pyplot(fig2)
