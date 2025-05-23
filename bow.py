import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TreebankWordTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict

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

# --- Representasi BoW ---
vectorizer = CountVectorizer()
X_bow = vectorizer.fit_transform(df['clean_text'])

# --- Build Inverted Index ---
inverted_index = defaultdict(set)
for idx, text in enumerate(df['clean_text']):
    for word in set(text.split()):
        inverted_index[word].add(idx)

# --- Query Function ---
def search_query(query):
    query = query.lower()
    query = re.sub(r'[^a-z\s]', '', query)
    tokens = tokenizer.tokenize(query)
    stemmed = [stemmer.stem(w) for w in tokens if w not in stop_words]
    result_sets = [inverted_index.get(word, set()) for word in stemmed]
    if result_sets:
        common_docs = set.intersection(*result_sets)
    else:
        common_docs = set()
    return list(common_docs)

# --- Evaluation Function ---
def evaluate_query(query, retrieved_indices):
    query_clean = preprocess(query)
    query_words = set(query_clean.split())

    relevant_indices = set()
    for word in query_words:
        relevant_indices |= inverted_index.get(word, set())

    retrieved_set = set(retrieved_indices)
    true_positives = retrieved_set & relevant_indices

    if not retrieved_set:
        precision = 0.0
    else:
        precision = len(true_positives) / len(retrieved_set)

    if not relevant_indices:
        recall = 0.0
    else:
        recall = len(true_positives) / len(relevant_indices)

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1

# --- Streamlit UI ---
st.title("🧠 Information Retrieval dengan BoW + Inverted Index")
st.write("Dataset: HuffPost News Category (headline + short description)")

query = st.text_input("Masukkan query pencarian (contoh: 'mental health')")

if query:
    result_indices = search_query(query)
    precision, recall, f1 = evaluate_query(query, result_indices)

    st.subheader(f"Hasil pencarian untuk: '{query}'")

    if result_indices:
        for i in result_indices[:10]:  # Tampilkan hanya 10 hasil teratas
            st.markdown(f"**Kategori:** {df.iloc[i]['category']}")
            st.write(df.iloc[i]['text'])
            st.markdown("---")
    else:
        st.warning("Tidak ditemukan dokumen yang relevan.")

    st.subheader("📊 Evaluasi Kinerja untuk Query Ini:")
    st.markdown(f"**Precision**: {precision:.2f}")
    st.markdown(f"**Recall**: {recall:.2f}")
    st.markdown(f"**F1-Score**: {f1:.2f}")



# import streamlit as st
# import pandas as pd
# import re
# import nltk
# from nltk.corpus import stopwords
# from nltk.stem import PorterStemmer
# from nltk.tokenize import TreebankWordTokenizer
# from sklearn.feature_extraction.text import CountVectorizer
# from collections import defaultdict
# import matplotlib.pyplot as plt

# nltk.download('stopwords')

# # --- Preprocessing ---
# stop_words = set(stopwords.words('english'))
# stemmer = PorterStemmer()
# tokenizer = TreebankWordTokenizer()

# def preprocess(text):
#     text = text.lower()
#     text = re.sub(r'[^a-z\s]', '', text)
#     tokens = tokenizer.tokenize(text)
#     tokens = [stemmer.stem(w) for w in tokens if w not in stop_words]
#     return ' '.join(tokens)

# # --- Load Data ---
# @st.cache_data
# def load_data():
#     df = pd.read_json('News_Category_Dataset_v3.json', lines=True)
#     df['text'] = df['headline'] + ' ' + df['short_description']
#     df = df[['text', 'category']]
#     df['clean_text'] = df['text'].apply(preprocess)
#     return df

# df = load_data()

# # --- Build BoW + Inverted Index ---
# vectorizer = CountVectorizer()
# X_bow = vectorizer.fit_transform(df['clean_text'])

# inverted_index = defaultdict(set)
# for idx, text in enumerate(df['clean_text']):
#     for word in set(text.split()):
#         inverted_index[word].add(idx)

# def search_query(query):
#     query = query.lower()
#     query = re.sub(r'[^a-z\s]', '', query)
#     tokens = tokenizer.tokenize(query)
#     stemmed = [stemmer.stem(w) for w in tokens if w not in stop_words]
#     result_sets = [inverted_index.get(word, set()) for word in stemmed]
#     if result_sets:
#         common_docs = set.intersection(*result_sets)
#     else:
#         common_docs = set()
#     return list(common_docs)

# # --- Evaluasi: Precision, Recall, F1 ---
# def evaluate_ir_system(query, relevant_category, top_k=10):
#     retrieved_indices = search_query(query)[:top_k]
#     relevant_indices = df[df['category'] == relevant_category].index.tolist()
#     true_positives = len(set(retrieved_indices).intersection(set(relevant_indices)))
#     precision = true_positives / len(retrieved_indices) if retrieved_indices else 0
#     recall = true_positives / len(relevant_indices) if relevant_indices else 0
#     f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
#     return precision, recall, f1, retrieved_indices

# # --- Streamlit UI ---
# st.title("🧠 Information Retrieval dengan Evaluasi (BoW + Inverted Index)")

# query = st.text_input("Masukkan query pencarian (contoh: 'omicron covid')")

# # Dropdown kategori sebagai ground truth
# kategori_list = df['category'].unique().tolist()
# relevant_category = st.selectbox("Pilih kategori relevan (ground truth) untuk evaluasi", kategori_list)

# if query and relevant_category:
#     precision, recall, f1, result_indices = evaluate_ir_system(query, relevant_category)

#     st.subheader(f"Hasil pencarian untuk: '{query}' (top 10)")

#     if result_indices:
#         for i in result_indices:
#             st.markdown(f"**Kategori:** {df.iloc[i]['category']}")
#             st.write(df.iloc[i]['text'])
#             st.markdown("---")
#     else:
#         st.warning("Tidak ditemukan dokumen yang relevan.")

#     # Visualisasi metrik
#     metrics = {'Precision': precision, 'Recall': recall, 'F1-Score': f1}
#     st.subheader("Evaluasi Sistem IR")
#     st.write(f"Precision: {precision:.2f}")
#     st.write(f"Recall: {recall:.2f}")
#     st.write(f"F1-Score: {f1:.2f}")

#     # Bar chart menggunakan matplotlib
#     fig, ax = plt.subplots()
#     ax.bar(metrics.keys(), metrics.values(), color=['skyblue', 'lightgreen', 'salmon'])
#     ax.set_ylim(0,1)
#     ax.set_ylabel('Score')
#     ax.set_title('Precision, Recall, dan F1-Score')
#     for i, v in enumerate(metrics.values()):
#         ax.text(i, v + 0.02, f"{v:.2f}", ha='center', fontsize=12)
#     st.pyplot(fig)
