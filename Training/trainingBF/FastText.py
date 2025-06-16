import pandas as pd
import numpy as np
import fasttext
import ast

# === 1. تجهيز بيانات التدريب بصيغة fastText ===
def prepare_fasttext_file(input_csv, output_txt):
    print("📥 Loading and formatting data...")
    df = pd.read_csv(input_csv)
    df.dropna(subset=["tokens", "label"], inplace=True)
    df["tokens"] = df["tokens"].apply(ast.literal_eval)
    df["formatted"] = "__label__" + df["label"].astype(str) + " " + df["tokens"].apply(lambda x: ' '.join(x))
    df["formatted"].to_csv(output_txt, index=False, header=False)
    print(f"✅ Training text file saved to: {output_txt}")
    return df  # نرجع الداتا لنستخدمها في التمثيل العددي

# === 2. تدريب موديل FastText ===
def train_fasttext_model(txt_path, model_output="fasttext_model.bin"):
    print("⚙️ Training FastText model...")
    model = fasttext.train_supervised(
        input=txt_path,
        epoch=25,
        lr=1.0,
        wordNgrams=2,
        minCount=1,
        verbose=2
    )
    model.save_model(model_output)
    print(f"✅ FastText model saved to: {model_output}")
    return model

# === 3. استخراج التمثيلات العددية باستخدام FastText ===
def extract_embeddings(data, model, max_len=150, embedding_dim=100):
    print("🔍 Extracting embeddings...")
    all_vectors = []
    labels = []

    for tokens, label in zip(data["tokens"], data["label"]):
        sentence_vecs = []

        for word in tokens[:max_len]:
            vec = model.get_word_vector(word)
            sentence_vecs.append(vec)

        # pad with zeros
        while len(sentence_vecs) < max_len:
            sentence_vecs.append(np.zeros(embedding_dim))

        all_vectors.append(sentence_vecs)
        labels.append(label)

    X = np.array(all_vectors)
    y = np.array(labels)
    return X, y

# === 4. حفظ embeddings إلى CSV ===
def save_embeddings(X, y, output_file="train_embeddings.csv"):
    flat_X = X.reshape((X.shape[0], -1))  # reshape to (samples, 15000)
    df = pd.DataFrame(flat_X)
    df["label"] = y
    df.to_csv(output_file, index=False)
    print(f"✅ Embeddings saved to: {output_file}")

# === MAIN ===
if __name__ == "__main__":
    input_csv = "FastTextData.csv"
    output_txt = "fasttext_train.txt"
    model_path = "fasttext_model.bin"
    embeddings_output = "train_embeddings.csv"

    # 1. تجهيز البيانات
    df = prepare_fasttext_file(input_csv, output_txt)

    # 2. تدريب FastText
    model = train_fasttext_model(output_txt, model_path)

    # 3. استخراج التمثيلات العددية
    X_train, y_train = extract_embeddings(df, model)

    # 4. حفظها في ملف
    save_embeddings(X_train, y_train, embeddings_output)
