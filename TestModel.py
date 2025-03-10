import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import deepcut
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# โหลดโมเดล
model = load_model("trained_model.keras")

# โหลด Tokenizer
with open("tokenizer.pkl", "rb") as handle:
    tokenizer = pickle.load(handle)

# โหลดไฟล์คำตอบ
file_path = "QuestionAnswer_AI.xlsx"
sheets = pd.read_excel(file_path, sheet_name=None)
df_answers = sheets["Answer"]

# ใช้ deepcut เพื่อให้การตัดคำแม่นยำขึ้น
def custom_tokenize(text):
    return deepcut.tokenize(text)

# คำนวณ Cosine Similarity เพื่อเลือกคำตอบที่ใกล้เคียงที่สุด
def find_best_answer(question, possible_answers):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([question] + possible_answers)
    cosine_sim = cosine_similarity(vectors[0], vectors[1:])  # วัดความคล้ายกัน
    best_match_index = cosine_sim.argmax()
    return possible_answers[best_match_index]

# ฟังก์ชันทำนายผล
def predict_answer(text, threshold=0.4):  # ปรับ threshold ลงเล็กน้อย
    tokens = custom_tokenize(text)  # ใช้ deepcut แทน word_tokenize
    seq = tokenizer.texts_to_sequences([tokens])
    padded = pad_sequences(seq, maxlen=7, padding="post")
    pred = model.predict(padded)

    # คำนวณค่าความมั่นใจ
    confidence_scores = pred[0]
    predicted_class = np.argmax(confidence_scores)
    max_confidence = np.max(confidence_scores)

    # ถ้าความมั่นใจต่ำกว่า threshold → ตอบว่า "ไม่เข้าใจคำถาม"
    if max_confidence < threshold:
        return "unknown", confidence_scores, "ขอโทษ ฉันไม่เข้าใจคำถามนี้"

    # ดึงคำตอบทั้งหมดที่อยู่ในหมวดเดียวกัน
    response_options = df_answers[df_answers["answer"] == predicted_class]["detail"].tolist()

    # ใช้ Cosine Similarity เลือกคำตอบที่เหมาะสมที่สุด
    best_response = find_best_answer(text, response_options)

    return predicted_class, confidence_scores, best_response

# ทดสอบโมเดล
question = "ประกอบอาชีพอะไรได้บ้าง"  # เปลี่ยนเป็นคำถามที่ต้องการทดสอบ
predicted_answer, confidence, response = predict_answer(question)

# แสดงผลลัพธ์
print(f"คำตอบที่ทำนายได้: {predicted_answer}")
print(f"ค่าความมั่นใจสูงสุด: {np.max(confidence)}")
print(f"คำตอบจาก Answer Sheet: {response}")
