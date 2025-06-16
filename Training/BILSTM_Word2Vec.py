import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Bidirectional, LSTM
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns

# === تحميل البيانات ===
X = np.load("C:\\Users\\future\\Desktop\\System5-20250412T203641Z-001\\System5\\word2vec_features.npy")
y = np.load("C:\\Users\\future\\Desktop\\System5-20250412T203641Z-001\\System5\\word2vec_labels.npy")

# === تقسيم البيانات ===
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

# === إعداد النموذج ===
MAX_SEQUENCE_LEN = X.shape[1]   # = 150
EMBEDDING_DIM = X.shape[2]      # = 100

model = Sequential([
    Bidirectional(LSTM(128, return_sequences=True), input_shape=(MAX_SEQUENCE_LEN, EMBEDDING_DIM)),
    Dropout(0.3),
    Bidirectional(LSTM(64)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# === تدريب النموذج ===
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=15,
    batch_size=32,
    callbacks=[early_stop]
)

# === حفظ النموذج ===
model.save("C:\\Users\\future\\Desktop\\System5-20250412T203641Z-001\\System5\\my_model.h5")
model.export("BW_model")
print("✅ Model saved as my_model.h5")

# === الرسم البياني للأداء ===
plt.figure(figsize=(10, 4))
plt.plot(history.history['accuracy'], label='Train Accuracy', marker='o')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='o')
plt.title('🔹 Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(history.history['loss'], label='Train Loss', marker='o')
plt.plot(history.history['val_loss'], label='Validation Loss', marker='o')
plt.title('🔻 Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# === التقييم ===
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)

print("📊 Accuracy:", accuracy_score(y_test, y_pred))
print("📊 Precision:", precision_score(y_test, y_pred))
print("📊 Recall:", recall_score(y_test, y_pred))
print("📊 F1 Score:", f1_score(y_test, y_pred))

# === مصفوفة الارتباك ===
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

