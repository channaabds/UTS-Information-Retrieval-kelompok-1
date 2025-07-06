# 📰 Smart News Retrieval System

📚 Academic Project – Mid Term Exam, Information Retrieval Course  
🧑‍💻 Universitas Islam Sultan Agung

---

## 📌 Overview

The Smart News Retrieval System is an academic project developed to explore and compare keyword-based and semantic-based information retrieval techniques. Designed to search and retrieve relevant news articles from a large dataset (200,000+ HuffPost news articles), the system evaluates different approaches in terms of precision, recall, and F1-score.

---

## 🔧 Tech Stack

- Python  
- NLTK  
- scikit-learn  
- Pandas & NumPy  
- Word2Vec (Gensim)  
- FAISS (Facebook AI Similarity Search)

---

## 🧠 Techniques Used

- Text Preprocessing (tokenization, stopwords removal, stemming)  
- Keyword-based retrieval:
  - Bag of Words (BoW)
  - TF-IDF + Cosine Similarity
- Semantic-based retrieval:
  - Word2Vec embeddings + FAISS approximate nearest neighbor indexing  
- Inverted index construction for fast lookup

---

## ✅ Evaluation

| Method                      | Strength                                         | Result                                     |
|----------------------------|--------------------------------------------------|--------------------------------------------|
| TF-IDF + Cosine Similarity | High precision for direct keyword queries        | Achieved up to 100% precision              |
| Word2Vec + FAISS           | Better semantic understanding and flexibility    | Good recall, useful in vague queries       |
| BoW                        | Simple and efficient                             | Limited semantic capacity                  |

Metrics evaluated:
- 🔹 Precision  
- 🔹 Recall  
- 🔹 F1-Score

---

## 📈 Key Contribution

- Implemented and tested three IR methods  
- Designed vector space models for semantic comparison  
- Built evaluation scripts to compare model performance on real queries  
- Identified trade-offs and proposed hybrid solution for practical IR use cases

---

## 🧪 Outcome

The study concluded that:
- TF-IDF is effective for exact matches but weak for fuzzy/semantic queries  
- Word2Vec + FAISS performs well for semantic similarity but less efficient for exact keywords  
- A hybrid approach is recommended to achieve balance between relevance and accuracy

---

## 📁 Dataset

- Dataset used: [HuffPost News Dataset](https://www.kaggle.com/datasets/rmisra/news-category-dataset)  
- Total Articles: ~200,000  
- Categories: Politics, Entertainment, Wellness, etc.

---

## 📷 Preview

<table>
  <tr>
    <td><img src="./view sistem/0.png" width="700"/></td>
    <td><img src="./view sistem/0 (1).png" width="700"/></td>
  </tr>
  <tr>
    <td><img src="./view sistem/0 (2).png" width="700"/></td>
    <td><img src="./view sistem/0 (3).png" width="700"/></td>
  </tr>
  <tr>
   <td><img src="./view sistem/0 (4).png" width="700"/></td>
    <td><img src="./view sistem/0 (5).png" width="700"/></td>
  </tr>
</table>

---
