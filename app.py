from flask import Flask, request
import requests
import numpy as np
import pandas as pd
import pickle
import random
import deepcut
import tensorflow as tf
load_model = tf.keras.models.load_model
pad_sequences = tf.keras.preprocessing.sequence.pad_sequences
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from pythainlp.tokenize import word_tokenize


#  โหลดโมเดล
model = load_model("C:/Users/User/Desktop/งาน ปี 3 ต้องจบ/Data Chatbot/Modeling/trained_model.keras")

#  โหลด Tokenizer
with open("Modeling/tokenizer.pkl", "rb") as handle:
    tokenizer = pickle.load(handle)

#  โหลดข้อมูลจากไฟล์ Excel
file_path = "Modeling/QuestionAnswer_AI.xlsx"
sheets = pd.read_excel(file_path, sheet_name=None)
df_answers = sheets["Answer"]
df_question = sheets["QuestionAnswer"]
merged_df = df_question.merge(df_answers, on="answer", how="left")

# แปลงเป็น dictionary โดยใช้ sub_questions เป็น key และ detail เป็น value
question_answer_dict = dict(zip(merged_df["sub_questions"], merged_df["detail"]))

#  ตั้งค่า Facebook API
PAGE_ACCESS_TOKEN = "EAAZAnUGahizwBO1blzD18zkZCtFZAdLIOqJlvqJ8e45YgggZAUrX7SICFtykZAc3iZCi85HZC4otsV7QWpz4ZCBnN9ZCbU8Im1OKroZB9UHVIDISVMCUTofGzemqTuo8BMRMzuNspUSOLea0Fs1i6CUwmZAvjwegwG3YV4gWxH08VmRxPI5t3uL9EeTlnYVoEHJTlPIaQZDZD"
VERIFY_TOKEN = "b_heeeeem"

#  ป้องกันการตอบข้อความซ้ำ
processed_messages = set()

app = Flask(__name__)

#  ฟังก์ชันตัดคำโดยใช้ deepcut
def custom_tokenize(text):
    return deepcut.tokenize(text)

#  ฟังก์ชันใช้ `cosine similarity` เพื่อเลือกคำตอบที่แม่นยำที่สุด
#def find_best_answer(question, possible_answers):
    vectorizer = TfidfVectorizer(ngram_range=(1,2), tokenizer=custom_tokenize)
    vectors = vectorizer.fit_transform([question] + possible_answers)
    cosine_sim = cosine_similarity(vectors[0], vectors[1:])
    best_match_index = cosine_sim.argmax()
    return possible_answers[best_match_index]

#  ฟังก์ชันทำนายผล
def preprocess_question(text):
    # ลบ "สาขานี้" ออกจากประโยคเพื่อป้องกัน Bias
    #text = text.replace("สาขานี้", "").strip()
    return text

def predict_answer(text, threshold=0.4):
    original_text = text  # เก็บคำถามต้นฉบับไว้
    text = preprocess_question(text)  # ปรับแต่งข้อความ
    
    if text in question_answer_dict:
        return "direct_match", None, question_answer_dict[text]

    tokens = custom_tokenize(text)
    print(f"📌 DEBUG: คำที่ตัดได้จาก deepcut = {tokens}")

    seq = tokenizer.texts_to_sequences([tokens])
    padded = pad_sequences(seq, maxlen=10, padding="post")
    pred = model.predict(padded)

    confidence_scores = pred[0]
    predicted_class = np.argmax(confidence_scores)
    max_confidence = np.max(confidence_scores)

    print(f"📌 DEBUG: โมเดลพยากรณ์ class = {predicted_class}, ความมั่นใจ = {max_confidence}")
    print(f"📌 DEBUG: ค่าความน่าจะเป็นทั้งหมด = {confidence_scores}")

    if max_confidence < threshold:
        return "unknown", confidence_scores, f"ขอโทษ ฉันไม่เข้าใจคำถาม '{original_text}'"

    response_options = df_answers[df_answers["answer"] == predicted_class]["detail"].tolist()

    best_response = random.choice(response_options) if response_options else "ขอโทษ ฉันไม่มีข้อมูลเกี่ยวกับเรื่องนี้"
    
    return predicted_class, confidence_scores, best_response

#  ฟังก์ชัน Webhook สำหรับรับข้อความจาก Messenger
@app.route("/webhook", methods=["GET", "POST"])
def webhook():
    if request.method == "GET":
        mode = request.args.get("hub.mode")
        token = request.args.get("hub.verify_token")
        challenge = request.args.get("hub.challenge")

        if mode == "subscribe" and token == VERIFY_TOKEN:
            print(" Webhook Verified!")
            return challenge, 200
        return "Verification token mismatch", 403

    elif request.method == "POST":
        data = request.json
        print("📌 DEBUG: Received Data:", data)

        for entry in data.get("entry", []):
            for messaging_event in entry.get("messaging", []):
                if "message" in messaging_event and "is_echo" not in messaging_event["message"]:
                    sender_id = messaging_event["sender"]["id"]
                    message_text = messaging_event["message"].get("text", "(ไม่มีข้อความ)")
                    message_id = messaging_event["message"]["mid"]

                    #  ป้องกันการตอบข้อความซ้ำ
                    if message_id in processed_messages:
                        print(f"⚠️ Duplicate message detected: {message_text}")
                        continue
                    processed_messages.add(message_id)

                    print(f"📩 Received message from {sender_id}: {message_text}")

                    #  ใช้ AI chatbot หาคำตอบ
                    _, _, response = predict_answer(message_text)

                    #  ส่งข้อความกลับไปที่ผู้ใช้
                    send_message(sender_id, response)

        return "OK", 200

#  ฟังก์ชันส่งข้อความกลับไปที่ Messenger
def send_message(recipient_id, text):
    url = f"https://graph.facebook.com/v17.0/me/messages?access_token={PAGE_ACCESS_TOKEN}"
    data = {"recipient": {"id": recipient_id}, "message": {"text": text}}
    headers = {"Content-Type": "application/json"}

    response = requests.post(url, json=data, headers=headers)
    print("📌 Response from Facebook:", response.json())

if __name__ == "__main__":
    app.run(port=5000, debug=True)
