import os
from typing import List, Dict

import streamlit as st
import pandas as pd

from openai import OpenAI

# =========================
# Helper: Get OpenAI API Key
# =========================

def get_api_key() -> str:
    """
    يحاول يأخذ الـ API key من:
    1) st.secrets["OPENAI_API_KEY"]
    2) أو من خانة في الـ sidebar لو ما كان موجود في secrets
    """
    key = None

    # 1) من secrets في Streamlit Cloud (مفضل عند النشر)
    try:
        key = st.secrets.get("OPENAI_API_KEY", None)
    except Exception:
        key = None

    # 2) لو ما فيه key في secrets، نخلي المستخدم يكتبه
    if not key:
        key = st.sidebar.text_input(
            "🔑 أدخلي OpenAI API Key (لن يتم حفظه)",
            type="password",
            help="للإستخدام المحلي فقط. في Streamlit Cloud يفضّل استخدام Secrets.",
        )

    return key


# ===================================
# GPT: توليد ورقة عمل بسيطة للطالب
# ===================================

def generate_worksheet(
    client: OpenAI,
    student_name: str,
    grade: str,
    skill: str,
    level: str,
    num_questions: int = 5,
) -> str:
    system_prompt = (
        "You are an educational content generator for primary school English in Qatar. "
        "Create a short reading passage and multiple-choice questions for the given student "
        "based on grade, skill, and performance level."
    )

    user_prompt = f"""
Student name: {student_name}
Grade: {grade}
Skill: {skill}
Performance level: {level}

Task:
1. Write a short passage (80–120 words) appropriate for this grade and skill.
2. Create {num_questions} multiple-choice questions (A–D) based on the passage.
3. Indicate the correct option for each question.

Return the result in a clear plain text format:
PASSAGE:
...
QUESTIONS:
1) ...
   A) ...
   B) ...
   C) ...
   D) ...
   Correct: X
...
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.7,
    )

    return response.choices[0].message.content


# =========================
# Pandas helpers
# =========================

REQUIRED_COLUMNS = ["student_id", "student_name", "grade", "skill", "score"]


def ingest_and_validate(df: pd.DataFrame) -> pd.DataFrame:
    """
    هذه الدالة تدعم شكل جدول الثيسس:
    StudentNumber, StudentName, LanguageFunction, ReadingComprehension, Grammar, Writing, Total
    وتحوله تلقائيًا إلى الشكل المطلوب:
    student_id, student_name, grade, skill, score
    """

    # إذا كان الملف من نوع الثيسس (الأعمدة الموجودة في الصورة)
    thesis_cols = {"StudentNumber", "StudentName",
                   "LanguageFunction", "ReadingComprehension",
                   "Grammar", "Writing"}

    if thesis_cols.issubset(df.columns):
        # نحول الجدول من wide إلى long: صف لكل مهارة
        df_long = df.melt(
            id_vars=["StudentNumber", "StudentName"],
            value_vars=["LanguageFunction", "ReadingComprehension", "Grammar", "Writing"],
            var_name="skill",
            value_name="score",
        )

        # نعيد تسمية الأعمدة لتطابق ما يستخدمه باقي الكود
        df_long = df_long.rename(
            columns={
                "StudentNumber": "student_id",
                "StudentName": "student_name",
            }
        )

        # نفترض أن كلهم من نفس الصف (Grade 3) – يمكنك تعديلها لاحقًا أو قراءتها من ملف آخر
        df_long["grade"] = 3

        df = df_long

    # من هنا فصاعدًا نطبق نفس التحقق القديم على الأعمدة الموحدة
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns after transform: {missing}")

    df = df.copy()

    # تحويل score إلى رقم
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df = df.dropna(subset=["score"])

    # نفترض الدرجات من 0 إلى 100 (أو 0 إلى 25 حسب مقياسك، تقدرين تغيرينها)
    df = df[(df["score"] >= 0) & (df["score"] <= 100)]

    # إزالة المكررات
    df = df.drop_duplicates()

    return df


    df = df.copy()

    # تحويل score إلى رقم
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df = df.dropna(subset=["score"])

    # نفترض الدرجة من 0 إلى 100
    df = df[(df["score"] >= 0) & (df["score"] <= 100)]

    # إزالة المكررات
    df = df.drop_duplicates()

    return df


def apply_leveling(df: pd.DataFrame, mastery_threshold: float = 75.0) -> pd.DataFrame:
    df = df.copy()

    def classify(score):
        if score < mastery_threshold:
            return "Low"
        elif score < mastery_threshold + 15:
            return "Medium"
        else:
            return "High"

    df["level"] = df["score"].apply(classify)
    return df


# =========================
# Streamlit UI
# =========================

def main():
    st.set_page_config(
        page_title="English Worksheets Generator",
        layout="wide",
    )

    st.title("📚 English Worksheets Generator")
    st.write(
        "Prototype for generating AI-powered remedial worksheets using Pandas + GPT API."
    )

    # --- Sidebar settings ---
    st.sidebar.header("Settings")

    mastery_threshold = st.sidebar.slider(
        "Mastery threshold (Low/Medium/High)",
        min_value=0,
        max_value=100,
        value=75,
    )

    num_questions = st.sidebar.slider(
        "Number of questions per worksheet",
        min_value=3,
        max_value=6,
        value=5,
    )

    # Get API key
    api_key = get_api_key()

    if not api_key:
        st.warning("🔑 الرجاء إدخال OpenAI API Key من الـ sidebar أو من Secrets.")
        st.stop()

    client = OpenAI(api_key=api_key)

    # --- File uploader ---
    st.subheader("1️⃣ Upload student performance CSV")

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is None:
        st.info("⬆️ رجاءً ارفعي ملف CSV لبدء المعالجة.")
        return

    # --- Process data ---
    try:
        df_raw = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")
        return

    st.markdown("### Raw Data Preview")
    st.dataframe(df_raw.head())

    try:
        df_clean = ingest_and_validate(df_raw)
    except Exception as e:
        st.error(f"Validation error: {e}")
        return

    st.markdown("### Validated Data")
    st.dataframe(df_clean.head())

    df_leveled = apply_leveling(df_clean, mastery_threshold=mastery_threshold)
    st.markdown("### Leveled Data (Low / Medium / High)")
    st.dataframe(df_leveled.head())

    # --- Filters ---
    st.subheader("2️⃣ Select group to generate worksheets for")

    grades = sorted(df_leveled["grade"].astype(str).unique())
    selected_grade = st.selectbox("Grade", grades)

    filtered_grade = df_leveled[df_leveled["grade"].astype(str) == selected_grade]

    skills = sorted(filtered_grade["skill"].astype(str).unique())
    selected_skill = st.selectbox("Skill", skills)

    levels = sorted(filtered_grade["level"].unique())
    selected_level = st.selectbox("Performance level", levels)

    target_df = filtered_grade[
        (filtered_grade["skill"].astype(str) == selected_skill)
        & (filtered_grade["level"] == selected_level)
    ]

    st.write(f"Number of students in this group: {len(target_df)}")

    st.subheader("3️⃣ Generate worksheets")

    if st.button("Generate worksheets for this group"):
        if target_df.empty:
            st.warning("No students match this filter.")
            return

        worksheets: List[Dict] = []

        with st.spinner("Generating worksheets using GPT..."):
            for _, row in target_df.iterrows():
                student_name = str(row["student_name"])
                grade = str(row["grade"])
                skill = str(row["skill"])
                level = str(row["level"])

                try:
                    ws_text = generate_worksheet(
                        client=client,
                        student_name=student_name,
                        grade=grade,
                        skill=skill,
                        level=level,
                        num_questions=num_questions,
                    )
                except Exception as e:
                    st.error(f"Error generating worksheet for {student_name}: {e}")
                    continue

                worksheets.append(
                    {
                        "student_name": student_name,
                        "content": ws_text,
                    }
                )

        if not worksheets:
            st.error("No worksheets were generated.")
            return

        st.success("✅ Worksheets generated successfully!")

        # Show sample
        st.markdown("### Sample Worksheet")
        sample = worksheets[0]
        st.markdown(f"#### {sample['student_name']}")
        st.text(sample["content"])


if __name__ == "__main__":
    main()
