import os
import io

import pandas as pd
import streamlit as st
from openai import OpenAI
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


# ==============================
# Helper functions
# ==============================


def get_api_key() -> str:
    """Get OpenAI API key from environment or Streamlit secrets."""
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        try:
            key = st.secrets.get("OPENAI_API_KEY", None)
        except Exception:
            key = None
    return key


def classify_level(score: float) -> str:
    """
    Classification based on the actual test out of 25:
    - High   : > 23
    - Medium : 15–22
    - Low    : < 15
    """
    if score > 23:
        return "High"
    elif score >= 15:
        return "Medium"
    return "Low"

def score_to_curriculum_grade(score: float) -> int:
    """
    Map raw score (out of 25) to target curriculum grade:
    - > 23       → Grade 6
    - 15–22      → Grade 5
    - 10–14      → Grade 4
    - < 10       → Grade 3
    يمكنك تعديل حدود 10 / 14 لاحقًا حسب رؤيتك.
    """
    if score > 23:
        return 6
    elif score >= 15:
        return 5
    elif score >= 10:
        return 4
    else:
        return 3


def transform_thesis_format(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert thesis dataset into long format (one row per student & skill).
    Expected columns:
    StudentNumber, StudentName, LanguageFunction, ReadingComprehension, Grammar, Writing
    """
    thesis_cols = {
        "StudentNumber", "StudentName",
        "LanguageFunction", "ReadingComprehension",
        "Grammar", "Writing"
    }

    if thesis_cols.issubset(df.columns):
        df_long = df.melt(
            id_vars=["StudentNumber", "StudentName"],
            value_vars=[
                "LanguageFunction", "ReadingComprehension",
                "Grammar", "Writing"
            ],
            var_name="skill",
            value_name="score",
        )

        df_long = df_long.rename(columns={
            "StudentNumber": "student_id",
            "StudentName": "student_name",
        })
        return df_long

    # If already in long format, just return as is
    return df


def build_skill_instruction(skill: str) -> str:
    """Return detailed instructions depending on the skill name."""
    s = str(skill).lower()

    if "grammar" in s:
        return (
            "The worksheet must explicitly assess GRAMMAR. "
            "Use multiple-choice questions that focus on: verb tenses, subject–verb agreement, "
            "prepositions, articles, comparative/superlative forms, and sentence structure. "
            "Use short sentences similar to those in the Qatari English tests. "
            "DO NOT ask general reading questions here; every question must check a grammar rule."
        )

    if "reading" in s:
        return (
            "The worksheet must explicitly assess READING COMPREHENSION. "
            "Write one short passage and then create questions about: main idea, supporting details, "
            "true/false, inference, and vocabulary in context. "
            "Questions should follow the style of school tests: 'What is the main idea of the text?', "
            "'Why does ...?', 'The underlined word means ...'. "
            "Do NOT test grammar rules here unless they are part of understanding the text."
        )

    if "writing" in s:
        return (
            'The worksheet must explicitly assess WRITING. '
            "Focus on sentence and short-paragraph level tasks, for example: choosing the best topic sentence, "
            "ordering jumbled sentences to form a paragraph, completing a sentence with a suitable connector, "
            "or choosing the sentence with correct punctuation and capitalisation. "
            "Questions are still MCQ (A–D) but always about how to write better sentences."
        )

    if "languagefunction" in s or "language function" in s:
        return (
            "The worksheet must explicitly assess LANGUAGE FUNCTIONS. "
            "Each question should present a short situation or dialogue, and the student chooses the best "
            "response or expression (e.g. making requests, giving advice, inviting, apologising, agreeing, "
            "disagreeing, asking for information). "
            "Use prompts like: 'What would you say?', 'Choose the correct reply', "
            "and short dialogues with a missing line."
        )

    return (
        "Make sure all questions clearly practise the given skill in an age-appropriate way."
    )


def build_rag_context(curriculum_df: pd.DataFrame, skill: str, curriculum_grade: int) -> str:
    """
    Very simple RAG: filter curriculum bank by grade & skill and
    convert rows into short bullet points.
    Expected columns: grade, skill, plus any other descriptive columns.
    """
    if curriculum_df is None:
        return ""

    required_cols = {"grade", "skill"}
    if not required_cols.issubset(curriculum_df.columns):
        return ""

    try:
        temp = curriculum_df.copy()
        temp["grade_str"] = temp["grade"].astype(str)
        mask = (
            (temp["grade_str"] == str(curriculum_grade)) &
            (temp["skill"].astype(str).str.lower() == str(skill).lower())
        )
        subset = temp[mask]
        if subset.empty:
            return ""

        bullets = []
        for _, row in subset.iterrows():
            row_dict = row.to_dict()
            row_dict.pop("grade_str", None)
            g = row_dict.pop("grade", None)
            sk = row_dict.pop("skill", None)
            # Keep other fields as description
            rest = " | ".join(
                f"{k}: {v}"
                for k, v in row_dict.items()
                if pd.notna(v)
            )
            bullets.append(f"- Grade {g}, Skill {sk}: {rest}")

        return "\n".join(bullets[:8])  # limit context
    except Exception:
        return ""


def generate_worksheet(
    client: OpenAI,
    student_name: str,
    student_grade: int,
    curriculum_grade: int,
    skill: str,
    level: str,
    num_questions: int = 5,
    rag_context: str = ""
) -> str:
    """Generate worksheet using GPT based on mapped curriculum grade, skill, and RAG context."""

    skill_instruction = build_skill_instruction(skill)

    system_prompt = (
        "You are an educational content generator for primary school English "
        "within the Qatari National Curriculum. Adjust difficulty and language "
        "based on the TARGET curriculum grade. Keep content clear, culturally appropriate, "
        "and suitable for students."
    )

    rag_section = ""
    if rag_context:
        rag_section = f"""
Curriculum RAG context (reference material from the official curriculum bank):
{rag_context}

Use this information to align the passage topic, vocabulary, and question focus
with the curriculum expectations for this grade and skill.

"""

    user_prompt = f"""
Student name: {student_name}
Actual school grade: {student_grade}
Target curriculum grade: {curriculum_grade}
Skill: {skill}
Performance level: {level} (Low / Medium / High)

Additional instructions about the skill:
{skill_instruction}

{rag_section}

Task:
1. Write a short reading passage (80–120 words) appropriate for the target grade.
2. The passage and ALL questions must clearly and explicitly practise the given SKILL described above. Do NOT mix other skills.
3. Create {num_questions} multiple-choice questions (A–D).
4. Provide an answer key clearly.

Required format (use exactly these headings):

PASSAGE:
<your passage>

QUESTIONS:
1) ...
A) ...
B) ...
C) ...
D) ...
2) ...
...

ANSWER KEY:
1) C
2) A
...
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.6,
    )

    return response.choices[0].message.content


def split_worksheet_and_answer(text: str):
    """Split GPT output into worksheet body (no answers) and answer key."""
    marker = "ANSWER KEY:"
    idx = text.upper().find(marker)
    if idx == -1:
        return text.strip(), "ANSWER KEY:\n(Not clearly provided by the model.)"
    body = text[:idx].strip()
    answer = text[idx:].strip()
    return body, answer


def text_to_pdf(title: str, content: str) -> bytes:
    """
    Convert text to a simple A4 PDF (in memory).
    Returns the PDF as bytes so it can be downloaded in Streamlit.
    """
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    x = 40
    y = height - 60

    c.setFont("Helvetica-Bold", 14)
    c.drawString(x, y, title)
    y -= 30

    c.setFont("Helvetica", 11)

    for line in content.split("\n"):
        while len(line) > 110:
            part = line[:110]
            c.drawString(x, y, part)
            line = line[110:]
            y -= 14
            if y < 40:
                c.showPage()
                y = height - 60
                c.setFont("Helvetica", 11)
        c.drawString(x, y, line)
        y -= 14
        if y < 40:
            c.showPage()
            y = height - 60
            c.setFont("Helvetica", 11)

    c.showPage()
    c.save()
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes


# ==============================
# Custom CSS (professional UI)
# ==============================

CUSTOM_CSS = """
<style>

/* Hide Streamlit default header/footer */
header, footer {visibility: hidden;}

/* Global app styles */
body, .stApp {
    background: #f6f7fb;
    font-family: "Cairo", sans-serif;
    color: #111827;
}

/* =============================
   TOP NAVBAR (HEADER)
   ============================= */
.navbar {
    position: sticky;
    top: 0;
    z-index: 999;
    width: 100%;
    background: #ffffff;
    border-bottom: 1px solid #e5e7eb;
    padding: 0.9rem 2.2rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: 0 4px 12px rgba(15, 23, 42, 0.04);
}

.nav-left {
    display: flex;
    flex-direction: column;
    gap: 2px;
}

.nav-logo {
    font-size: 1.35rem;
    font-weight: 800;
    letter-spacing: .3px;
    color: #8A1538;
}

.nav-subtitle {
    font-size: .85rem;
    color: #6b7280;
}

/* RIGHT SIDE BUTTONS */
.nav-right {
    display: flex;
    gap: 0.6rem;
    align-items: center;
}

.nav-btn {
    padding: 0.4rem 1.1rem;
    border-radius: 999px;
    font-size: .8rem;
    font-weight: 600;
    border: 1px solid #8A1538;
    background: #ffffff;
    color: #8A1538;
    cursor: pointer;
}

.nav-btn:hover {
    background: #fdf2f6;
}

.nav-btn-primary {
    background: #8A1538;
    color: #ffffff;
    border: none;
}

.nav-btn-primary:hover {
    background: #6e0f2c;
}

/* =============================
   TABS
   ============================= */
.stTabs {
    margin-top: 1.0rem;
    margin-bottom: 1.2rem;
}

.stTabs [data-baseweb="tab-list"] {
    gap: .6rem;
}

.stTabs [data-baseweb="tab"] {
    background: #e5e7eb;
    color: #4b5563;
    border-radius: 999px;
    padding: .4rem 1.3rem;
    font-size: .9rem;
    border: none;
}

.stTabs [data-baseweb="tab"]:hover {
    background: #d1d5db;
    color: #111827;
}

.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, #8A1538, #b11b49);
    color: #ffffff !important;
    font-weight: 700;
    box-shadow: 0 4px 12px rgba(139, 20, 54, 0.35);
}

/* =============================
   CARDS
   ============================= */
.card {
    background: #ffffff;
    padding: 1.5rem 1.7rem;
    border-radius: 16px;
    margin-bottom: 1.2rem;
    border: 1px solid #e5e7eb;
    box-shadow: 0 6px 18px rgba(15, 23, 42, 0.06);
}

.step-title {
    color: #8A1538;
    font-size: 1.25rem;
    font-weight: 700;
    margin-bottom: .25rem;
}

.step-help {
    color: #4b5563;
    font-size: .95rem;
}

/* Tool tags */
.tool-tag {
    display: inline-block;
    background: #fde7f0;
    color: #8A1538;
    border-radius: 999px;
    padding: 4px 12px;
    font-size: .75rem;
    margin-top: 4px;
    margin-right: 4px;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #8A1538, #b11b49);
    color: #ffffff;
    border-radius: 999px;
    border: none;
    padding: .5rem 1.4rem;
    font-weight: 600;
    font-size: .9rem;
    box-shadow: 0 4px 12px rgba(139, 20, 54, 0.35);
}

.stButton > button:hover {
    background: #7a0e31;
}

/* Download button */
.stDownloadButton > button {
    background: #ffffff;
    color: #374151;
    border: 1px solid #d1d5db;
    border-radius: 999px;
    padding: .45rem 1.2rem;
    font-size: .85rem;
}

.stDownloadButton > button:hover {
    background: #f3f4ff;
    border-color: #c4c7ff;
}

/* Code style */
.stMarkdown code, code {
    background: #fde7f0;
    color: #8A1538;
    padding: 3px 8px;
    border-radius: 6px;
    font-family: "JetBrains Mono", monospace;
    font-size: .85rem;
}

/* Dataframe / text color fix */
.stDataFrame, .stMarkdown, .stText {
    color: #111827 !important;
}

</style>
"""


# ==============================
# Streamlit App
# ==============================
def main():
    st.set_page_config(
        page_title="English Worksheets Generator",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # Apply global CSS
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # ---------------- TOP NAVBAR (UI only) ----------------
    st.markdown(
        """
        <div class="navbar">
            <div class="nav-left">
                <div class="nav-logo">English Worksheets Generator</div>
                <div class="nav-subtitle">
                    Adaptive English worksheets aligned with the Qatari curriculum
                </div>
            </div>
            <div class="nav-right">
                <button class="nav-btn">Sign in</button>
                <button class="nav-btn nav-btn-primary">Sign up</button>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---------------- API KEY ----------------
    api_key = get_api_key()
    if not api_key:
        st.error("OPENAI_API_KEY is missing. Add it in Settings → Secrets.")
        return

    client = OpenAI(api_key=api_key)

    # ---------------- SESSION STATE ----------------
    if "df_raw" not in st.session_state:
        st.session_state["df_raw"] = None
    if "processed_df" not in st.session_state:
        st.session_state["processed_df"] = None
    if "curriculum_df" not in st.session_state:
        st.session_state["curriculum_df"] = None

    # ---------------- TABS ----------------
    tab_overview, tab_data, tab_generate, tab_help = st.tabs(
        ["Overview", "Data & RAG", "Generate Worksheets", "Help & Tools"]
    )

    # ========== OVERVIEW TAB ==========
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
                  <li><b>Attach curriculum knowledge</b> via a small curriculum bank CSV. This is used as a simple <b>RAG</b> layer to ground GPT in real topics.</li>
                  <li><b>Generate personalised worksheets</b> for each student using the GPT API, aligned with the selected skill and curriculum grade.</li>
                </ol>
            </div>
            """,
            unsafe_allow_html=True,
        )
    # -------- DATA & RAG TAB --------
    with tab_data:

        # STEP 1 — upload CSV
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="step-title">Step 1 — Upload student performance CSV</div>', unsafe_allow_html=True)

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

        # STEP 2 — Curriculum RAG
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="step-title">Step 2 — Curriculum bank (RAG)</div>', unsafe_allow_html=True)

        curriculum_file = st.file_uploader("Upload curriculum CSV (optional)", type=["csv"], key="curriculum_csv")
        if curriculum_file is not None:
            try:
                cur_df = pd.read_csv(curriculum_file)
                st.session_state["curriculum_df"] = cur_df
                st.write("Curriculum bank preview:")
                st.dataframe(cur_df.head(), use_container_width=True)
            except Exception as e:
                st.error(f"Could not read curriculum bank: {e}")

        st.markdown("</div>", unsafe_allow_html=True)

        # STEP 3 — Process Data
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="step-title">Step 3 — Process data & classify levels</div>', unsafe_allow_html=True)

        if st.button("Process student data"):
            df_raw_state = st.session_state.get("df_raw", None)
            if df_raw_state is None:
                st.error("Please upload the student performance CSV first.")
            else:
                try:
                    df_proc = transform_thesis_format(df_raw_state)
                    df_proc["level"] = df_proc["score"].apply(classify_level)
                    df_proc["target_curriculum_grade"] = df_proc["score"].apply(score_to_curriculum_grade)

                    st.session_state["processed_df"] = df_proc
                    st.success("Student data processed successfully ✓")
                    st.dataframe(df_proc, use_container_width=True)

                except Exception as e:
                    st.error(f"Error while processing data: {e}")

        st.markdown("</div>", unsafe_allow_html=True)

    # -------- GENERATE WORKSHEETS TAB --------
    with tab_generate:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="step-title">Step 4 — Generate worksheets (PDF only)</div>', unsafe_allow_html=True)

        df = st.session_state.get("processed_df", None)
        curriculum_df = st.session_state.get("curriculum_df", None)

        if df is None:
            st.info("Please process the student data first.")
        else:
            skills = sorted(df["skill"].unique())
            selected_skill = st.selectbox("Choose skill", skills)
            levels = ["Low", "Medium", "High"]
            selected_level = st.selectbox("Choose performance level", levels)
            num_q = st.slider("Number of questions", 3, 10, 5)

            target_df = df[(df["skill"] == selected_skill) & (df["level"] == selected_level)]
            st.markdown(f"Students in this group: **{len(target_df)}**")

    #     ...
    #
    # with tab_help:
    #     ...

    # -------- GENERATE WORKSHEETS TAB --------
# -------- STEP 4: Generate worksheets (PDF only) --------
    with tab_generate:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="step-title">Step 4 — Generate worksheets (PDF only)</div>', unsafe_allow_html=True)
    st.markdown("""
    <p class="step-help">
    For each student in the selected skill and level, the system generates a personalised worksheet
    and a separate answer key. Only PDF download buttons are shown.
    </p>
    """, unsafe_allow_html=True)

    # Choose skill
    skill = st.selectbox(
        "Choose skill",
        ["LanguageFunction", "ReadingComprehension", "Grammar", "Writing"]
    )

    # Choose performance level
    selected_level = st.selectbox("Choose performance level", ["Low", "Medium", "High"])

    processed_df = st.session_state.get("processed_df", None)

    if processed_df is None:
        st.warning("Please process the student data in Step 3 before generating worksheets.")
    else:
        group_df = processed_df[processed_df["level"] == selected_level]

        st.write(f"Students in this group: **{len(group_df)}**")

        # Generate PDFs
        if st.button("Generate PDFs for this group"):
            if len(group_df) == 0:
                st.warning("There are no students in this category.")
            else:
                with st.spinner("Generating personalised PDF worksheets…"):
                    pdf_results = generate_pdfs_for_group(group_df, skill)

                st.success("PDF worksheets are ready!")

                for item in pdf_results:
                    col1, col2 = st.columns(2)

                    with col1:
                        st.download_button(
                            label=f"Download Worksheet — {item['name']}",
                            data=item["worksheet"],
                            file_name=f"{item['name']}_worksheet.pdf",
                            mime="application/pdf"
                        )
                    with col2:
                        st.download_button(
                            label=f"Download Answer Key — {item['name']}",
                            data=item["answer_key"],
                            file_name=f"{item['name']}_answers.pdf",
                            mime="application/pdf"
                        )

    st.markdown('</div>', unsafe_allow_html=True)

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
                    <li><b>Rule-based classifier</b> — fixed thresholds (Low / Medium / High) mapped to curriculum grades for differentiation.</li>
                    <li><b>RAG</b> — a small curriculum bank CSV is used as a retrieval layer to give GPT concrete topics, objectives, and examples.</li>
                    <li><b>GPT API</b> — generates passages, questions, and answer keys aligned with the skill and curriculum grade.</li>
                    <li><b>PDF export</b> — the final worksheets and answer keys are exported as A4 PDFs so the teacher can download and print them.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
