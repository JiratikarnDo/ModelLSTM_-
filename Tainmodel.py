import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from pythainlp.tokenize import word_tokenize

# ตั้งค่าพารามิเตอร์
MAX_LENGTH = 7  
NUM_CLASSES = 32  

# โหลดไฟล์ Excel
file_path = "QuestionAnswer_AI.xlsx"
sheets = pd.read_excel(file_path, sheet_name=None)

# โหลดข้อมูลจากแต่ละ Sheet
df_questions = sheets["QuestionAnswer"]
df_answers = sheets["Answer"]

# ป้องกัน NaN ใน sub_questions
df_questions["sub_questions"] = df_questions["sub_questions"].fillna('')

# ตัดคำภาษาไทย
df_questions["tokenized"] = df_questions["sub_questions"].apply(lambda x: word_tokenize(str(x), keep_whitespace=False))

# ใช้ Tokenizer แปลงคำเป็นตัวเลข
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df_questions["tokenized"])
sequences = tokenizer.texts_to_sequences(df_questions["tokenized"])

# ทำ Padding
X = pad_sequences(sequences, maxlen=MAX_LENGTH, padding="post")

# แปลง answer เป็น One-hot Encoding
y = to_categorical(df_questions["answer"], num_classes=NUM_CLASSES)

# แบ่งข้อมูล train/test (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# กำหนดขนาด Vocabulary
VOCAB_SIZE = len(tokenizer.word_index) + 1

# ใช้ EarlyStopping เพื่อลด Overfitting
early_stopping = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

# สร้างโมเดล LSTM ใหม่ (เพิ่ม Dropout และใช้ Bidirectional LSTM)
model = Sequential([
    Embedding(VOCAB_SIZE, 128, input_length=MAX_LENGTH),
    SpatialDropout1D(0.4),
    Bidirectional(LSTM(128, dropout=0.4, recurrent_dropout=0.4, return_sequences=True)),
    Bidirectional(LSTM(64, dropout=0.4, recurrent_dropout=0.4)),
    Dense(128, activation="relu"),
    Dropout(0.4), 
    Dense(NUM_CLASSES, activation="softmax")
])


# คอมไพล์โมเดล
optimizer = Adam(learning_rate=0.0005, decay=1e-6)
model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

# เทรนโมเดล
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])

# บันทึกโมเดล
model.save("trained_model.keras")

# บันทึก Tokenizer
import pickle
with open("tokenizer.pkl", "wb") as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
