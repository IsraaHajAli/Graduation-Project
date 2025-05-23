import re, string, unicodedata
from ftfy import fix_text
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import spacy
import numpy as np
from gensim.models import Word2Vec
import tensorflow as tf
import joblib

# التحميل لمرة واحدة فقط
nlp = spacy.load("en_core_web_sm")
w2v_model = Word2Vec.load('C:\\Users\\future\\Desktop\\System5-20250412T203641Z-001\\System5\\Feature_Extraction\\word2vec_model.bin')
model = tf.saved_model.load("my_model")
infer = model.signatures["serving_default"]

# تنظيف النص
def clean_text(text):
    text = fix_text(text)
    text = unicodedata.normalize("NFKD", text)
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    text = re.sub(r"\[.*?\]", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>+", "", text)
    text = re.sub(r"[%s]" % re.escape(string.punctuation), " ", text)
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"\w*\d\w*", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.lower().strip()

# إزالة stopwords
def remove_stopwords_from_tokens(tokens):
    stop_words = set(stopwords.words("english"))
    return [word for word in tokens if word.lower() not in stop_words]

# Lemmatization
def lemmatize_tokens(tokens):
    doc = nlp(" ".join(tokens))
    return [token.lemma_ for token in doc if token.is_alpha]

def preprocess_for_word2vec_BiLSTM(text, save_tokens_path='tokens.pkl'):
    tokens = word_tokenize(text)
    tokens = remove_stopwords_from_tokens(tokens)
    tokens = lemmatize_tokens(tokens)

    # حفظ التوكنز لو مطلوب
    if save_tokens_path:
        joblib.dump(tokens, save_tokens_path)

    # تحويل التوكنز إلى Word2Vec vectors
    vectors = [w2v_model.wv[word] for word in tokens if word in w2v_model.wv]

    return np.array(vectors), tokens


# تحويل لـ Word2Vec
def word2vec_featureextraction(w2v_model, tokens, embedding_dim=100):
    vectors = []
    for word in tokens:
        if word in w2v_model.wv:
            vectors.append(w2v_model.wv[word])
        else:
            vectors.append(np.zeros(embedding_dim))
    if len(vectors) == 0:
        return np.zeros((1, embedding_dim))
    else:
        return np.array(vectors)

# التنبؤ
def predict_using_w2v_bilstm(model, vectors):
    MAX_SEQUENCE_LEN = 150
    EMBEDDING_DIM = 100

    # لو أقل من الطول المطلوب، نكمّل بزيرو
    padded = np.zeros((1, MAX_SEQUENCE_LEN, EMBEDDING_DIM), dtype='float32')
    length = min(len(vectors), MAX_SEQUENCE_LEN)
    if length > 0:
        padded[0, :length, :] = vectors[:length]

    prediction = model.predict(padded, verbose=0)
    return prediction[0][0]



def predict_using_w2v_bilstm(model, vectors):
    MAX_SEQUENCE_LEN = 150
    EMBEDDING_DIM = 100

    # إذا vectors شكلها (n_words, 100)، لازم نحطها داخل مصفوفة padded شكلها (1, 150, 100)
    padded = np.zeros((1, MAX_SEQUENCE_LEN, EMBEDDING_DIM), dtype='float32')
    length = min(len(vectors), MAX_SEQUENCE_LEN)
    if length > 0:
        padded[0, :length, :] = vectors[:length]

    prediction = model.predict(padded, verbose=0)
    return prediction[0][0]

