import pandas as pd
import numpy as np
import fasttext
import ast
import os
import tensorflow as tf
import psutil
import shutil
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

def print_ram():
    print("🧠 Available RAM:", round(psutil.virtual_memory().available / 1e9, 2), "GB")

# === 1. دالة استخراج التمثيلات العددية ===
def extract_embeddings(df, model, from_tokens=False, max_len=150, embedding_dim=100):
    all_vectors, labels = [], []

    for i in range(len(df)):
        tokens = df.iloc[i]["tokens"] if from_tokens else df.iloc[i]["text"].split()
        label = df.iloc[i]["label"]

        sentence_vecs = []
        for word in tokens[:max_len]:
            vec = model.get_word_vector(word)
            sentence_vecs.append(vec)

        while len(sentence_vecs) < max_len:
            sentence_vecs.append(np.zeros(embedding_dim))

        all_vectors.append(sentence_vecs)
        labels.append(label)

    X = np.array(all_vectors)
    y = to_categorical(labels, num_classes=2)
    return X, y

# === 2. تحميل موديل FastText ===
fasttext_model = fasttext.load_model("fasttext_model.bin")

# === 3. تحميل أو تجهيز بيانات التدريب ===
if os.path.exists("X_train.npy") and os.path.exists("y_train.npy"):
    print("🔁 Loading cached training data...")
    X_train = np.load("X_train.npy")
    y_train = np.load("y_train.npy")
else:
    print("💾 Generating and saving training data...")
    train_df = pd.read_csv("FastTextData.csv")
    train_df["tokens"] = train_df["tokens"].apply(ast.literal_eval)
    X_train, y_train = extract_embeddings(train_df, fasttext_model, from_tokens=True)
    np.save("X_train.npy", X_train)
    np.save("y_train.npy", y_train)

# === 4. تحميل أو تجهيز بيانات التحقق ===
if os.path.exists("X_val.npy") and os.path.exists("y_val.npy"):
    print("🔁 Loading cached validation data...")
    X_val = np.load("X_val.npy")
    y_val = np.load("y_val.npy")
else:
    print("💾 Generating and saving validation data...")
    val_df = pd.read_csv("val (1).csv")
    X_val, y_val = extract_embeddings(val_df, fasttext_model, from_tokens=False)
    np.save("X_val.npy", X_val)
    np.save("y_val.npy", y_val)

# === 5. تحميل أو تجهيز بيانات الاختبار ===
if os.path.exists("X_test.npy") and os.path.exists("y_test.npy"):
    print("🔁 Loading cached test data...")
    X_test = np.load("X_test.npy")
    y_test = np.load("y_test.npy")
else:
    print("💾 Generating and saving test data...")
    test_df = pd.read_csv("test (1).csv")
    X_test, y_test = extract_embeddings(test_df, fasttext_model, from_tokens=False)
    np.save("X_test.npy", X_test)
    np.save("y_test.npy", y_test)

# === 6. بناء نموذج BiLSTM ===
model = Sequential([
    Bidirectional(LSTM(128, return_sequences=True, dropout=0.3), input_shape=(150, 100)),
    Bidirectional(LSTM(64, dropout=0.3)),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(2, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# === 7. تجهيز البيانات باستخدام tf.data.Dataset ===
AUTOTUNE = tf.data.AUTOTUNE

train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)) \
    .shuffle(1000) \
    .batch(16) \
    .cache() \
    .prefetch(AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)) \
    .batch(16) \
    .cache() \
    .prefetch(AUTOTUNE)

# === 8. التدريب باستخدام mini-epochs ومراقبة الذاكرة ===
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history_all = []
for i in range(10):
    print(f"\n🚀 Training Mini Epoch {i+1}/10")
    print_ram()
    history = model.fit(train_ds, validation_data=val_ds, epochs=1, callbacks=[early_stopping])
    print_ram()
    history_all.append(history)

# === 9. تقييم النموذج
loss, acc = model.evaluate(X_test, y_test)
print(f"\n✅ Test Loss: {loss:.4f}")
print(f"✅ Test Accuracy: {acc:.4f}")


# === 10. المقاييس النهائية
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

accuracy  = accuracy_score(y_true_classes, y_pred_classes)
precision = precision_score(y_true_classes, y_pred_classes)
recall    = recall_score(y_true_classes, y_pred_classes)
f1        = f1_score(y_true_classes, y_pred_classes)
conf_mat  = confusion_matrix(y_true_classes, y_pred_classes)
report    = classification_report(y_true_classes, y_pred_classes)

print("\n📊 Final Evaluation Metrics:")
print(f"✅ Accuracy:  {accuracy:.4f}")
print(f"✅ Precision: {precision:.4f}")
print(f"✅ Recall:    {recall:.4f}")
print(f"✅ F1-Score:  {f1:.4f}")

print("\n📊 Confusion Matrix:")
print(conf_mat)

print("\n📊 Classification Report:")
print(report)

# ========== STEP 10: Save and zip model ========== #
print("💾 Saving model...")
model.save("my_bilstm_model.h5")
print("✅ Model saved as my_bilstm_model.h5")

shutil.make_archive("my_bilstm_model_h5", 'zip', '.', "my_bilstm_model.h5")
print("📦 Model zipped as my_bilstm_model_h5.zip")