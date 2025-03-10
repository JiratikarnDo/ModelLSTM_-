import pandas as pd
import numpy as np

# โหลดไฟล์ Excel
file_path = "QuestionAnswer.xlsx"  # หรือใช้พาธเต็ม เช่น r"C:\Users\...\QuestionAnswer.xlsx"

# โหลดทุก Sheet
sheets = pd.read_excel(file_path, sheet_name=None)

# แสดงชื่อของทุก Sheets ที่อยู่ในไฟล์
print("Sheets ที่พบในไฟล์:", sheets.keys())

# โหลดแต่ละ Sheet ตามชื่อที่ถูกต้อง
df_questions = sheets["Question"]  # ชื่อ Sheet ที่มีคำถาม
df_answers = sheets["Answer"]  # ชื่อ Sheet ที่มีคำตอบ

# แสดงตัวอย่างข้อมูลของแต่ละ Sheet
print("ตัวอย่างคำถาม:")
print(df_questions.head())

print("ตัวอย่างคำตอบ:")
print(df_answers.head())

# เช็คความยาวของ `sub_questions`
df_questions["num_words"] = df_questions["sub_questions"].apply(lambda x: len(str(x).split()))

# คำนวณค่าสถิติที่สำคัญ
max_length = df_questions["num_words"].max()  # ค่ามากที่สุด
mean_length = df_questions["num_words"].mean()  # ค่าเฉลี่ย
percentile_95 = np.percentile(df_questions["num_words"], 95)  # ค่าความยาวที่ครอบคลุม 95%

# แสดงผล
print(f"ค่ามากที่สุดของความยาวคำถามย่อย: {max_length}")
print(f"ค่าเฉลี่ยของความยาวคำถามย่อย: {mean_length:.2f}")
print(f"95th Percentile (ค่าที่ครอบคลุม 95% ของคำถามทั้งหมด): {percentile_95}")
