import tensorflow as tf
import pandas as pd
import numpy as np
from keras import Sequential
from keras.src.callbacks import EarlyStopping
from keras.src.layers import Bidirectional, LSTM, Dropout, Dense
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.utils.np_utils import to_categorical
import fasttext

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# **تحميل البيانات من CSV (يتضمن النص والتصنيفات)**
train_data = pd.read_csv(r"C:\\Users\future\\Desktop\\System5-20250412T203641Z-001\System5\\PreProcessing\\train.csv", usecols=['text', 'label'])
test_data = pd.read_csv(r"C:\\Users\future\\Desktop\\System5-20250412T203641Z-001\System5\\PreProcessing\\test.csv", usecols=['text', 'label'])

# **التأكد من عدم وجود قيم فارغة**
train_data.dropna(inplace=True)
test_data.dropna(inplace=True)

# **تحويل التصنيفات من 'fake' و 'true' إلى 0 و 1**
train_data["label"] = train_data["label"].str.lower().map({'fake': 0, 'true': 1})
train_data.dropna(subset=["label"], inplace=True)  # 🧹 تنظيف الصفوف اللي فش فيها label صالح

test_data["label"] = test_data["label"].str.lower().map({'fake': 0, 'true': 1})
test_data.dropna(subset=["label"], inplace=True)  # 🧹 تنظيف الصفوف اللي فش فيها label صالح

# **تحميل نموذج FastText المدرب مسبقًا**

fasttext_model = fasttext.load_model("C:\\Users\\future\\Desktop\\System5-20250412T203641Z-001\\System5\\Feature_Extraction\\fasttext_model.bin")


# **استخراج الميزات باستخدام FastText**
train_features = np.array([fasttext_model.get_sentence_vector(text) for text in train_data["text"]])
test_features = np.array([fasttext_model.get_sentence_vector(text) for text in test_data["text"]])

# **تحويل التصنيفات إلى One-Hot Encoding**
y_train = to_categorical(train_data["label"].values)
y_test = to_categorical(test_data["label"].values)

# **تقسيم البيانات إلى تدريب وتحقق**
X_train, X_val, y_train, y_val = train_test_split(train_features, y_train, test_size=0.2, random_state=42)

# **إعادة تشكيل البيانات ليتوافق مع LSTM**
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
X_test = test_features.reshape((test_features.shape[0], test_features.shape[1], 1))

# **بناء نموذج BiLSTM**
model = Sequential([
    Bidirectional(LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.3), input_shape=(X_train.shape[1], 1)),
    Bidirectional(LSTM(64, dropout=0.3, recurrent_dropout=0.3)),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(2, activation='softmax')  # لأن لدينا تصنيفين (0: زائف، 1: حقيقي)
])

# **تجميع النموذج**
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# **إضافة EarlyStopping لتجنب التدريب الزائد**
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# **تدريب النموذج**
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32, callbacks=[early_stopping])

# **تقييم النموذج**
loss, accuracy = model.evaluate(X_test, y_test)
print(f"✅ Test Accuracy: {accuracy * 100:.2f}%")

# **التنبؤ بالفئات على بيانات الاختبار**
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)  # تحويل الاحتمالات إلى تصنيفات (0 أو 1)
y_true_classes = np.argmax(y_test, axis=1)  # تحويل One-Hot إلى تصنيفات (0 أو 1)

# **حساب المقاييس المختلفة**
precision = precision_score(y_true_classes, y_pred_classes)
recall = recall_score(y_true_classes, y_pred_classes)
f1 = f1_score(y_true_classes, y_pred_classes)
conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)
classification_rep = classification_report(y_true_classes, y_pred_classes)

# **طباعة النتائج**
print(f"📊 Precision: {precision:.4f}")
print(f"📊 Recall: {recall:.4f}")
print(f"📊 F1-Score: {f1:.4f}")
print("📊 Confusion Matrix:")
print(conf_matrix)
print("📊 Classification Report:")
print(classification_rep)

# **حفظ النموذج المدرب**
model.save("bilstm_fasttext_model.h5")