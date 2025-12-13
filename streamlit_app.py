import os
import io

import pandas as pd
import streamlit as st
from openai import OpenAI
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


# =====================================================
# 1. Helpers: API key
# =====================================================


def get_api_key() -> str:
    """Get OpenAI API key from environment or Streamlit secrets."""
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        try:
            key = st.secrets.get("OPENAI_API_KEY", None)
        except Exception:
            key = None
    return key


# =====================================================
# 2. Classification helpers
# =====================================================


def classify_level(score: float) -> str:
    """
    Thresholds (per skill score out of 25):
    Low:    0–11
    Medium: 12–21
    High:   22–25
    """
    if score <= 11:
        return "Low"
    elif score <= 21:
        return "Medium"
    elif score <= 25:
        return "High"
    else:
        return "Unknown"


def map_level_and_score_to_grade(level: str, score: float) -> int:
    """
    Map performance level + raw score to a curriculum grade
    (all students are in Grade 5 in school, لكن المنهج المستخدم
    في الورقة يعتمد على مستواهم):

    - Low  → Grade 3–4  (نقسّم Low إلى درجتين):
        * 0–7   → Grade 3
        * 8–11  → Grade 4
    - Medium (12–21) → Grade 5
    - High   (22–25) → Grade 6
    """
    if level == "Low":
        if score <= 7:
            return 3
        else:  # 8–11
            return 4
    elif level == "Medium":
        return 5
    elif level == "High":
        return 6
    else:
        # default fallback
        return 5


def transform_thesis_format(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert thesis dataset into long format (one row per student & skill).

    Expected columns:
    StudentNumber, StudentName, LanguageFunction, ReadingComprehension, Grammar, Writing
    """
    thesis_cols = {
        "StudentNumber",
        "StudentName",
        "LanguageFunction",
        "ReadingComprehension",
        "Grammar",
        "Writing",
    }

    if thesis_cols.issubset(df.columns):
        df_long = df.melt(
            id_vars=["StudentNumber", "StudentName"],
            value_vars=[
                "LanguageFunction",
                "ReadingComprehension",
                "Grammar",
                "Writing",
            ],
            var_name="skill",
            value_name="score",
        )

        df_long = df_long.rename(
            columns={
                "StudentNumber": "student_id",
                "StudentName": "student_name",
            }
        )
        return df_long

    # If already in long format, just return as is
    return df


# =====================================================
# 3. RAG helpers: curriculum_bank.csv
# =====================================================


@st.cache_data
def load_curriculum_bank():
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(base_dir, "curriculum_bank.csv")
        return pd.read_csv(csv_path)
    except Exception as e:
        st.error(f"Failed to load curriculum_bank.csv: {e}")
        return None



def build_rag_context(
    curriculum_df: pd.DataFrame | None, skill: str, curriculum_grade: int
) -> str:
    """
    Build RAG context from curriculum_bank.csv.

    Expected columns:
    grade, module, skill, topic, standard, teaching_point,
    grammar_ref (optional), vocab_list (optional), example (optional)
    """
    if curriculum_df is None:
        return ""

    required_cols = {"grade", "skill", "topic", "standard", "teaching_point"}
    if not required_cols.issubset(curriculum_df.columns):
        return ""

    try:
        subset = curriculum_df[
            (curriculum_df["grade"] == curriculum_grade)
            & (curriculum_df["skill"].str.lower() == str(skill).lower())
        ]

        if subset.empty:
            return ""

        bullets: list[str] = []
        for _, row in subset.iterrows():
            line = (
                f"- Grade {row['grade']} | Module: {row.get('module', '')} | "
                f"Topic: {row['topic']}\n"
                f"  Standard: {row['standard']}\n"
                f"  Teaching point: {row['teaching_point']}\n"
            )
            grammar_ref = row.get("grammar_ref", None)
            vocab_list = row.get("vocab_list", None)
            example = row.get("example", None)

            if isinstance(grammar_ref, str) and grammar_ref.strip():
                line += f"  Grammar reference: {grammar_ref}\n"
            if isinstance(vocab_list, str) and vocab_list.strip():
                line += f"  Key vocabulary: {vocab_list}\n"
            if isinstance(example, str) and example.strip():
                line += f"  Example sentence: {example}\n"

            bullets.append(line)

        # نكتفي بأول 10 عناصر حتى لا يصبح الـ prompt طويلاً
        return "\n".join(bullets[:10])
    except Exception:
        return ""


def build_exam_style_task(skill: str, curriculum_grade: int, num_questions: int) -> str:
    s = str(skill).lower()

    # ---- Language Functions: Read & Match (A/B) ----
    if "languagefunction" in s or "language function" in s:
        return f"""
TASK (School test style):
Create "Language Functions" section like: Read and match (A/B).
- Provide 4 items in column A (questions / prompts).
- Provide 4 answers in column B (a–d) in shuffled order.
- Use simple classroom English appropriate for Grade {curriculum_grade}.
Format EXACTLY:

LANGUAGE FUNCTIONS:
Read and match.

A                          B
1- ...
2- ...
3- ...
4- ...

a. ...
b. ...
c. ...
d. ...

ANSWER KEY:
1) c
2) a
3) d
4) b
"""

    # ---- Reading: Passage + 2 MCQ + 2 short answers ----
    if "reading" in s:
        return f"""
TASK (School test style):
Create a reading comprehension section (Grade {curriculum_grade}).
- Write ONE passage (90–140 words for grades 5–6, 60–100 words for grades 3–4).
- Then write:
  Q1) MAIN idea MCQ (4 options)
  Q2) Vocabulary in context MCQ (underlined word) (4 options)
  Q3) Short answer (1–2 lines)
  Q4) Short answer (1–2 lines)

Format EXACTLY:

READING COMPREHENSION:
Directions: Read the passage and answer the questions.

PASSAGE:
<text with ONE underlined word>

QUESTIONS:
1- What is the text MAINLY about?
A) ...
B) ...
C) ...
D) ...

2- What does the underlined word ____ mean?
A) ...
B) ...
C) ...
D) ...

3- <short answer question>
____________________________________

4- <short answer question>
____________________________________

ANSWER KEY:
1) B
2) A
3) <expected answer>
4) <expected answer>
"""

    # ---- Vocabulary: Fill gaps from a word box ----
    if "vocab" in s or "vocabulary" in s:
        return f"""
TASK (School test style):
Create a vocabulary section like: Fill in the gaps with suitable words from the box.
- Provide a word box with 4 words.
- Provide 4 sentences with blanks.

Format EXACTLY:

VOCABULARY:
Fill in the gaps with suitable words from the box.

WORD BOX: word1 - word2 - word3 - word4

1- _________
2- _________
3- _________
4- _________

ANSWER KEY:
1) wordX
2) wordY
3) wordZ
4) wordW
"""

    # ---- Writing (Grammar): MCQ + Correct the underlined word/verb ----
    if "grammar" in s:
        return f"""
TASK (School test style):
Create a Grammar section (Grade {curriculum_grade}) with:
- 2 MCQ questions (choose correct answer A–D)
- 2 "Do as shown between brackets" items where the student corrects the underlined verb/word.

Format EXACTLY:

GRAMMAR:
Read and choose the correct answer.

1- <sentence>
A) ...
B) ...
C) ...
D) ...

2- <sentence>
A) ...
B) ...
C) ...
D) ...

Do as shown between brackets.

3- <sentence with UNDERLINED verb/word> (Correct the underlined verb/word)
____________________________________

4- <sentence with UNDERLINED verb/word> (Correct the underlined verb/word)
____________________________________

ANSWER KEY:
1) D
2) A
3) <correct form>
4) <correct form>
"""

    # ---- Writing: prompt + helping questions + checklist/rubric ----
    if "writing" in s:
        # عدد الجمل حسب المرحلة (مثل نماذجكم)
        sentences = 4 if curriculum_grade == 3 else (5 if curriculum_grade == 4 else (6 if curriculum_grade == 5 else 7))
        return f"""
TASK (School test style):
Create a writing prompt like the school test.
- Ask for a paragraph of {sentences} sentences.
- Provide 3–4 helping questions.
- Add a simple rubric lines (Spelling / Punctuation / Content / Grammar & Vocab).

Format EXACTLY:

WRITING PROMPT:
Write a paragraph of {sentences} sentences about: <topic>

Helping questions:
- ...
- ...
- ...

RUBRIC (Total 6):
Writing conventions: Spelling (1), Punctuation (1)
Language Use (Grammar & Vocab): (2)
Content (Ideas & organization): (2)

ANSWER KEY:
(Provide a short model paragraph of {sentences} sentences.)
"""

    # fallback
    return f"""
TASK:
Create {num_questions} suitable questions for Grade {curriculum_grade} focusing on the skill: {skill}.
Provide ANSWER KEY.
"""



# =====================================================
# 4. Worksheet generation helpers
# =====================================================


def generate_worksheet(
    client: OpenAI,
    student_name: str,
    student_grade: int,
    curriculum_grade: int,
    skill: str,
    level: str,
    num_questions: int,
    rag_context: str,
) -> str:
    """Generate worksheet using GPT based on grade, skill, level and RAG context."""

    system_prompt = (
        "You are a primary school English assessment generator aligned with the Qatar National Curriculum. "
        "You MUST follow the exact task format provided. "
        "Do NOT create a reading passage unless the selected skill is ReadingComprehension/Reading. "
        "Keep language clear, age-appropriate, and culturally suitable."
    )

    # ✅ خليه نص بسيط بدون تكرار headings داخل headings
    rag_section = ""
    if rag_context:
        rag_section = f"""
CURRICULUM BANK (RAG) — use these points to align your questions:
{rag_context}
"""

    # ✅ هذا اللي يحدد شكل الاختبار حسب skill
    task_block = build_exam_style_task(
        skill=skill,
        curriculum_grade=curriculum_grade,
        num_questions=num_questions,
    )

    user_prompt = f"""
Student name: {student_name}
Actual school grade: {student_grade}
Target curriculum grade: {curriculum_grade}
Skill: {skill}
Performance level: {level}

{rag_section}

IMPORTANT RULES:
- Follow the EXACT section headings and format in the TASK block.
- If Skill is Grammar: DO NOT write a reading passage.
- If Skill is Writing: DO NOT write a reading passage unless the task says so.
- If Skill is LanguageFunction: use matching format A/B.
- Provide ANSWER KEY exactly as required.

{task_block}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.4,
    )

    return response.choices[0].message.content



# =====================================================
# 5. UI CSS
# =====================================================



CUSTOM_CSS = """
<style>

/* HEADER */
.app-header {
    width: 100%;
    padding: 1.6rem 2rem;
    background: linear-gradient(135deg, #8A1538, #5e0d24);
    border-radius: 0 0 20px 20px;
    color: #ffffff;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.20);
}

.header-title {
    font-size: 2.2rem;
    font-weight: 800;
}

.header-sub {
    font-size: 1rem;
    opacity: 0.95;
}

</style>
"""

# =====================================================
# 6. Streamlit App
# =====================================================


def main():
    st.set_page_config(
        page_title="English Worksheets Generator",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # HEADER
    st.markdown(
        """
        <div class="app-header">
            <div class="header-title">English Worksheets Generator</div>
            <div class="header-sub">
                Prototype for adaptive remedial worksheets using Pandas + RAG + GPT API
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # API KEY
    api_key = get_api_key()
    if not api_key:
        st.error("OPENAI_API_KEY is missing. Add it in Settings → Secrets.")
        return
    client = OpenAI(api_key=api_key)

    # Session state
    if "df_raw" not in st.session_state:
        st.session_state["df_raw"] = None
    if "processed_df" not in st.session_state:
        st.session_state["processed_df"] = None
    if "curriculum_df" not in st.session_state:
        st.session_state["curriculum_df"] = load_curriculum_bank()

    curriculum_df = st.session_state["curriculum_df"]

    # Tabs
    tab_overview, tab_data, tab_generate, tab_help = st.tabs(
        ["Overview", "Data & RAG", "Generate Worksheets", "Help & Tools"]
    )

    # -------- OVERVIEW TAB --------
    with tab_overview:
        st.markdown(
            """
            <div class="card">
                <div class="step-title">How the prototype works</div>
                <p class="step-help">
                This prototype follows three main steps:
                </p>
                <ol class="step-help">
                  <li><b>Upload & process student performance data</b> (Pandas) to classify students into Low / Medium / High for each skill.</li>
                  <li><b>Attach curriculum knowledge</b> via a structured curriculum bank (Grades 3–6) using a simple <b>RAG</b> layer.</li>
                  <li><b>Generate personalised worksheets</b> for each student using the GPT API, aligned with the selected skill and curriculum grade.</li>
                </ol>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # -------- DATA & RAG TAB --------
    with tab_data:
        # STEP 1: upload students CSV
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(
            '<div class="step-title">Step 1 — Upload student performance CSV</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <span class="tool-tag">Pandas</span>
            <span class="tool-tag">Data validation</span>
            <p class="step-help">
            Expected format (from thesis dataset):
            <code>StudentNumber, StudentName, LanguageFunction, ReadingComprehension, Grammar, Writing</code>
            </p>
            """,
            unsafe_allow_html=True,
        )

        uploaded = st.file_uploader("Upload Students.csv", type=["csv"])
        if uploaded is not None:
            try:
                df_raw = pd.read_csv(uploaded)
                st.session_state["df_raw"] = df_raw
                st.write("Raw data preview:")
                st.dataframe(df_raw.head(), use_container_width=True)
            except Exception as e:
                st.error(f"Could not read CSV: {e}")

        st.markdown("</div>", unsafe_allow_html=True)

        # STEP 2: Curriculum bank (RAG) status
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(
            '<div class="step-title">Step 2 — Curriculum bank (RAG)</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <span class="tool-tag">RAG</span>
            <p class="step-help">
            The app automatically loads <code>curriculum_bank.csv</code> (Grades 3–6),
            which summarises the Qatar National Curriculum standards and Top Stars-style
            teaching points for each grade and skill. This structured bank is used as
            retrieval context when generating worksheets, so the content stays aligned
            with the real curriculum.
            </p>
            """,
            unsafe_allow_html=True,
        )

        if curriculum_df is None:
            st.error("curriculum_bank.csv not found in the app folder.")
        else:
            st.success("Curriculum bank loaded successfully ✔")
            try:
                summary = curriculum_df.groupby(["grade", "skill"]).size()
                st.markdown("**Available entries by grade and skill:**")
                st.dataframe(summary.to_frame("count"))
            except Exception:
                st.write("Preview:")
                st.dataframe(curriculum_df.head(), use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # STEP 3: process and classify
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(
            '<div class="step-title">Step 3 — Process data & classify levels</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <span class="tool-tag">Rule-based classifier</span>
            <p class="step-help">
            This step automatically analyzes student scores and assigns performance levels
            (Low / Medium / High) based on fixed thresholds (0–11, 12–21, 22–25) per skill.
            Then it maps each level + score to a <b>recommended curriculum grade</b>:
            Low → Grade 3–4, Medium → Grade 5, High → Grade 6.
            </p>
            """,
            unsafe_allow_html=True,
        )

        if st.button("Process student data"):
            df_raw_state = st.session_state.get("df_raw", None)
            if df_raw_state is None:
                st.error("Please upload the student performance CSV first.")
            else:
                try:
                    df_proc = transform_thesis_format(df_raw_state)

                    # Level classification
                    df_proc["level"] = df_proc["score"].apply(classify_level)

                    # Recommended curriculum grade (for RAG & worksheet generation)
                    df_proc["recommended_grade"] = df_proc.apply(
                        lambda row: map_level_and_score_to_grade(
                            level=row["level"],
                            score=row["score"],
                        ),
                        axis=1,
                    )

                    st.session_state["processed_df"] = df_proc

                    st.success("Student data processed successfully ✔")

                    counts = df_proc["level"].value_counts()
                    st.markdown("**Classification summary (by level):**")
                    st.markdown(
                        f"- Low: {counts.get('Low', 0)} students  \n"
                        f"- Medium: {counts.get('Medium', 0)} students  \n"
                        f"- High: {counts.get('High', 0)} students"
                    )

                    st.markdown("**Skills detected in the dataset:**")
                    skills_counts = df_proc["skill"].value_counts()
                    for sk, cnt in skills_counts.items():
                        st.markdown(f"- {sk}: {cnt} records")

                    st.write(
                        "Processed data preview (sorted by student & skill):"
                    )
                    st.dataframe(
                        df_proc.sort_values(
                            ["student_id", "skill"]
                        ).head(20),
                        use_container_width=True,
                    )

                except Exception as e:
                    st.error(f"Error while processing data: {e}")

        st.markdown("</div>", unsafe_allow_html=True)

    # -------- GENERATE WORKSHEETS TAB --------
    with tab_generate:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(
            '<div class="step-title">Step 4 — Generate worksheets (PDF only)</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <span class="tool-tag">GPT API</span>
            <span class="tool-tag">RAG</span>
            <span class="tool-tag">PDF export</span>
            <p class="step-help">
            For each student in the selected skill and level, the system generates a personalised worksheet
            and a separate answer key, aligned with the curriculum bank for Grades 3–6.
            </p>
            """,
            unsafe_allow_html=True,
        )

        df = st.session_state.get("processed_df", None)
        curriculum_df = st.session_state.get("curriculum_df", None)

        if df is None:
            st.info(
                "Please go to the 'Data & RAG' tab and process the student data first."
            )
        elif curriculum_df is None:
            st.error("Curriculum bank is not loaded. Please add curriculum_bank.csv.")
        else:
            skills = sorted(df["skill"].unique())
            selected_skill = st.selectbox("Choose skill", skills)

            levels = ["Low", "Medium", "High"]
            selected_level = st.selectbox("Choose performance level", levels)

            num_q = st.slider(
                "Number of questions per worksheet", 3, 10, 5
            )

            target_df = df[
                (df["skill"] == selected_skill)
                & (df["level"] == selected_level)
            ]

            st.markdown(f"Students in this group: **{len(target_df)}**")

            if st.button("Generate PDFs for this group"):
                if target_df.empty:
                    st.error("No students match this skill + level.")
                else:
                    with st.spinner(
                        "Generating worksheets and answer keys…"
                    ):
                        try:
                            for _, row in target_df.iterrows():
                                grade_for_rag = int(
                                    row.get("recommended_grade", 5)
                                )

                                rag_context = build_rag_context(
                                    curriculum_df=curriculum_df,
                                    skill=row["skill"],
                                    curriculum_grade=grade_for_rag,
                                )

                                full_text = generate_worksheet(
                                    client=client,
                                    student_name=row["student_name"],
                                    student_grade=5,  # الصف الحقيقي (طلاب الصف الخامس)
                                    curriculum_grade=grade_for_rag,
                                    skill=row["skill"],
                                    level=row["level"],
                                    num_questions=num_q,
                                    rag_context=rag_context,
                                )

                                worksheet_body, answer_key = (
                                    split_worksheet_and_answer(
                                        full_text
                                    )
                                )

                                ws_pdf = text_to_pdf(
                                    title=f"Worksheet for {row['student_name']}",
                                    content=worksheet_body,
                                )
                                ak_pdf = text_to_pdf(
                                    title=f"Answer Key for {row['student_name']}",
                                    content=answer_key,
                                )

                                st.markdown(f"#### {row['student_name']}")
                                c1, c2 = st.columns(2)
                                with c1:
                                    st.download_button(
                                        label="Download worksheet PDF",
                                        data=ws_pdf,
                                        file_name=f"worksheet_{row['student_name']}.pdf",
                                        mime="application/pdf",
                                    )
                                with c2:
                                    st.download_button(
                                        label="Download answer key PDF",
                                        data=ak_pdf,
                                        file_name=f"answer_key_{row['student_name']}.pdf",
                                        mime="application/pdf",
                                    )

                            st.success("All PDFs generated successfully ✅")
                        except Exception as e:
                            st.error(
                                f"Error while generating worksheets: {e}"
                            )

        st.markdown("</div>", unsafe_allow_html=True)

    # -------- HELP TAB --------
    with tab_help:
        st.markdown(
            """
            <div class="card">
                <div class="step-title">Help & implementation notes</div>
                <p class="step-help">
                    This tab summarises the main tools used in the prototype:
                </p>
                <ul class="step-help">
                    <li><b>Pandas</b> — reading the CSV file, reshaping the thesis dataset into long format, and classifying students.</li>
                    <li><b>Rule-based classifier</b> — thresholds (Low / Medium / High) mapped to recommended curriculum grades (3–6) for adaptive support.</li>
                    <li><b>RAG</b> — a structured curriculum bank (<code>curriculum_bank.csv</code>) with standards, grammar points and vocabulary for Grades 3–6 is used as retrieval context.</li>
                    <li><b>GPT API</b> — generates passages, questions, and answer keys aligned with the skill, level, and curriculum grade.</li>
                    <li><b>PDF export</b> — the final worksheets and answer keys are exported as A4 PDFs so the teacher can download and print them.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
