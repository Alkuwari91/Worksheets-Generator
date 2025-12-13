import os
import io
import pandas as pd
import streamlit as st
from openai import OpenAI
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from PyPDF2 import PdfReader   # لقراءة ملفات الـ PDF

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

# =====================================================
# 3. RAG helpers: load curriculum & textbooks from PDF
# =====================================================

def load_pdf_text(pdf_path: str) -> str:
    """Read all text from a PDF (simple extract)."""
    if not os.path.exists(pdf_path):
        return ""
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    except Exception:
        return ""


def build_curriculum_memory() -> dict:
    """
    Load curriculum standards PDFs for grades 3–6.
    Keys are numeric grades: 3, 4, 5, 6.
    """
    return {
        3: load_pdf_text("Grade 3 Curriculum Standards.pdf"),
        4: load_pdf_text("Grade 4 Curriculum Standards.pdf"),
        5: load_pdf_text("Grade 5 Curriculum Standards.pdf"),
        6: load_pdf_text("Grade 6 Curriculum Standards.pdf"),
    }


def build_textbook_memory() -> dict:
    """
    Load Top Stars textbooks PDFs for grades 3–6.
    """
    return {
        3: load_pdf_text("TopStars_3A_QTR_21-22_SB_001-088_LOW.pdf"),
        4: load_pdf_text("TopStars_4A_ed21-22_SB.pdf"),
        5: load_pdf_text("Top Stars_5A_SB.pdf"),
        6: load_pdf_text("TopStars_6A_QTR_2021-2022_TB_001-160_LOW.pdf"),
    }


def build_rag_context(grade: int, skill: str,
                      curriculum_memory: dict,
                      textbook_memory: dict) -> str:
    """
    Build a simple RAG context using:
    - Curriculum standards PDF for this grade.
    - Top Stars textbook PDF for this grade.

    We don't have structured tagging inside the PDFs,
    so we provide short excerpts plus the skill label
    and let GPT align the content.
    """
    cur_text = curriculum_memory.get(grade, "") or ""
    book_text = textbook_memory.get(grade, "") or ""

    if not cur_text and not book_text:
        return ""

    # نحدّ من عدد الحروف حتى لا يكون الـ prompt طويل جدًا
    cur_excerpt = cur_text[:2000]
    book_excerpt = book_text[:2000]

    return f"""
Curriculum standards excerpt for Grade {grade} (raw text from official PDF).
Focus on parts related to the skill: {skill}.

{cur_excerpt}

Textbook excerpt (Top Stars) for Grade {grade}.
Use topics, vocabulary, and task styles that fit this grade and skill.

{book_excerpt}
"""

# =====================================================
# 4. Worksheet generation helpers
# =====================================================

def build_skill_instruction(skill: str) -> str:
    s = str(skill).lower()
    if "grammar" in s:
        return (
            "Focus the questions on grammar usage, sentence structure, verb tenses, "
            "and error-correction style MCQs, appropriate for the target grade."
        )
    if "reading" in s:
        return (
            "Focus the questions on reading comprehension: main idea, details, "
            "inference, and vocabulary in context related to the passage."
        )
    if "writing" in s:
        return (
            "Focus the questions on writing skills: organising ideas, choosing "
            "correct connectors, and building clear sentences."
        )
    if "languagefunction" in s or "language function" in s:
        return (
            "Focus the questions on language functions such as making requests, "
            "giving advice, asking for information, agreeing and disagreeing, etc."
        )
    return (
        "Make sure the questions clearly practise the given skill in an "
        "age-appropriate way."
    )


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
Curriculum-aligned standards and textbook excerpts for this grade and skill:
Use the following teaching standards, objectives, topics, and examples when generating the passage and questions.
Ensure that ALL worksheet content aligns with these curriculum requirements:

{rag_context}
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
2. The passage and questions must clearly practise the given skill.
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

# =====================================================
# 5. UI CSS
# =====================================================

CUSTOM_CSS = """
<style>
header, footer {visibility: hidden;}

body, .stApp {
    background: #f4f5f7;
    font-family: "Cairo", sans-serif;
    color: #1f2937;
}

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
    letter-spacing: .3px;
}
.header-sub {
    font-size: 1rem;
    opacity: .95;
}

/* TABS */
.stTabs {
    margin-top: .5rem;
    margin-bottom: 1.2rem;
}
.stTabs [data-baseweb="tab-list"] {
    gap: .6rem;
}
.stTabs [data-baseweb="tab"] {
    background: #e8eaf0;
    color: #4b5563;
    border-radius: 999px;
    padding: .45rem 1.3rem;
    font-size: .9rem;
    border: none;
}
.stTabs [data-baseweb="tab"]:hover {
    background: #d5d7df;
    color: #111827;
}
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, #8A1538, #b11b49);
    color: #ffffff !important;
    font-weight: 700;
    box-shadow: 0 4px 12px rgba(139, 20, 54, 0.35);
}

/* CARDS */
.card {
    background: white;
    padding: 1.5rem 1.7rem;
    border-radius: 16px;
    margin-bottom: 1.2rem;
    border: 1px solid #e5e7eb;
    box-shadow: 0 6px 20px rgba(0,0,0,0.06);
}
.step-title {
    color: #8A1538;
    font-size: 1.3rem;
    font-weight: 700;
}
.step-help {
    color: #555;
    font-size: .95rem;
}

/* TOOL TAGS */
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

/* BUTTONS */
.stButton > button {
    background: linear-gradient(135deg, #8A1538, #b11b49);
    color: white;
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
    background: white;
    color: #374151;
    border: 1px solid #d1d5db;
    border-radius: 999px;
    padding: .45rem 1.2rem;
    font-size: .85rem;
}
.stDownloadButton > button:hover {
    background: #f3eeff;
    border-color: #c4c7ff;
}

/* CODE STYLE */
.stMarkdown code, code {
    background: #fde7f0;
    color: #8A1538;
    padding: 3px 8px;
    border-radius: 6px;
    font-family: "JetBrains Mono", monospace;
    font-size: .85rem;
}

/* DATAFRAME / TEXT */
.stDataFrame, .stMarkdown, .stText {
    color: #1f2937 !important;
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
    if "curriculum_memory" not in st.session_state:
        st.session_state["curriculum_memory"] = build_curriculum_memory()
    if "textbook_memory" not in st.session_state:
        st.session_state["textbook_memory"] = build_textbook_memory()

    curriculum_memory = st.session_state["curriculum_memory"]
    textbook_memory = st.session_state["textbook_memory"]

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
                  <li><b>Attach curriculum knowledge</b> via Qatar National Curriculum standards and Top Stars textbooks (Grades 3–6) using a simple <b>RAG</b> layer.</li>
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
        st.markdown('<div class="step-title">Step 1 — Upload student performance CSV</div>', unsafe_allow_html=True)
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

        # STEP 2: Curriculum & textbooks RAG status
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="step-title">Step 2 — Curriculum standards & Top Stars (RAG)</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <span class="tool-tag">RAG</span>
            <p class="step-help">
            The app automatically loads Qatar National Curriculum standards and Top Stars textbooks
            for Grades 3–6 from local PDF files. These texts are used as a retrieval context
            when generating worksheets, so the content stays aligned with the real curriculum.
            </p>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("**RAG loading status:**")
        for g in [3, 4, 5, 6]:
            cur_ok = "Loaded" if curriculum_memory.get(g) else "Missing"
            book_ok = "Loaded" if textbook_memory.get(g) else "Missing"
            st.markdown(f"- Grade {g}: Curriculum = **{cur_ok}**, Top Stars = **{book_ok}**")

        st.markdown("</div>", unsafe_allow_html=True)

        # STEP 3: process and classify
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="step-title">Step 3 — Process data & classify levels</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <span class="tool-tag">Rule-based classifier</span>
            <p class="step-help">
            This step automatically analyzes student scores and assigns performance levels
            (Low / Medium / High) based on fixed thresholds (0–11, 12–21, 22–25) per skill.
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

                    df_proc["level"] = df_proc["score"].apply(classify_level)

                    df_proc["target_curriculum_grade"] = df_proc["level"].map(
                        {"Low": 3, "Medium": 5, "High": 6}
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

                    st.write("Processed data preview (sorted by student & skill):")
                    st.dataframe(
                        df_proc.sort_values(["student_id", "skill"]).head(20),
                        use_container_width=True,
                    )

                except Exception as e:
                    st.error(f"Error while processing data: {e}")

        st.markdown("</div>", unsafe_allow_html=True)

    # -------- GENERATE WORKSHEETS TAB --------
    with tab_generate:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="step-title">Step 4 — Generate worksheets (PDF only)</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <span class="tool-tag">GPT API</span>
            <span class="tool-tag">RAG</span>
            <span class="tool-tag">PDF export</span>
            <p class="step-help">
            For each student in the selected skill and level, the system generates a personalised worksheet
            and a separate answer key, aligned with Qatar National Curriculum standards and Top Stars textbooks.
            </p>
            """,
            unsafe_allow_html=True,
        )

        df = st.session_state.get("processed_df", None)

        if df is None:
            st.info("Please go to the 'Data & RAG' tab and process the student data first.")
        else:
            skills = sorted(df["skill"].unique())
            selected_skill = st.selectbox("Choose skill", skills)

            levels = ["Low", "Medium", "High"]
            selected_level = st.selectbox("Choose performance level", levels)

            num_q = st.slider("Number of questions per worksheet", 3, 10, 5)

            target_df = df[
                (df["skill"] == selected_skill) &
                (df["level"] == selected_level)
            ]

            st.markdown(f"Students in this group: **{len(target_df)}**")

            if st.button("Generate PDFs for this group"):
                if target_df.empty:
                    st.error("No students match this skill + level.")
                else:
                    with st.spinner("Generating worksheets and answer keys…"):
                        try:
                            for _, row in target_df.iterrows():
                                grade_for_rag = int(row["target_curriculum_grade"])

                                rag_context = build_rag_context(
                                    grade=grade_for_rag,
                                    skill=row["skill"],
                                    curriculum_memory=curriculum_memory,
                                    textbook_memory=textbook_memory,
                                )

                                full_text = generate_worksheet(
                                    client=client,
                                    student_name=row["student_name"],
                                    student_grade=5,  # تقدرين تعدلينها لو أضفتِ عمود grade حقيقي
                                    curriculum_grade=grade_for_rag,
                                    skill=row["skill"],
                                    level=row["level"],
                                    num_questions=num_q,
                                    rag_context=rag_context,
                                )

                                worksheet_body, answer_key = split_worksheet_and_answer(full_text)

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
                            st.error(f"Error while generating worksheets: {e}")

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
                    <li><b>Rule-based classifier</b> — thresholds (Low / Medium / High) mapped to curriculum grades (3, 5, 6) for differentiation.</li>
                    <li><b>RAG</b> — curriculum standards and Top Stars textbooks (Grades 3–6) are loaded from PDF files and used as retrieval context.</li>
                    <li><b>GPT API</b> — generates passages, questions, and answer keys aligned with the skill, level, and curriculum grade.</li>
                    <li><b>PDF export</b> — the final worksheets and answer keys are exported as A4 PDFs so the teacher can download and print them.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
