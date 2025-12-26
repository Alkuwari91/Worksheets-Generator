import base64
from reportlab.lib.utils import ImageReader
import unicodedata
import os
import io
import re
from typing import Optional

import pandas as pd
import streamlit as st
from openai import OpenAI
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


# =========================
# PDF TEXT NORMALIZATION
# =========================
def normalize_pdf_text(t: str) -> str:
    if not t:
        return ""

    # Normalize newlines
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = t.replace("\u2028", "\n").replace("\u2029", "\n")

    # Convert literal "\n" to real newlines
    t = t.replace("\\n", "\n")

    # Replace tabs with spaces
    t = t.replace("\t", " ")

    # ✅ Remove ASCII control characters (they show as squares in PDF viewers)
    # Keep only newline \n
    t = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", t)

    # Normalize quotes/dashes
    t = (
        t.replace("’", "'").replace("‘", "'")
         .replace("“", '"').replace("”", '"')
         .replace("–", "-").replace("—", "-")
    )

    # Improve exam-style spacing
    t = re.sub(r"\n{3,}", "\n\n", t)
    t = re.sub(r"[ ]{2,}", " ", t)

    return t.strip()
def normalize_pdf_text(t: str) -> str:
    if not t:
        return ""

    # Normalize newlines
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = t.replace("\u2028", "\n").replace("\u2029", "\n")

    # Convert literal "\n" to real newlines
    t = t.replace("\\n", "\n")

    # Replace tabs with spaces
    t = t.replace("\t", " ")

    # ✅ Remove ASCII control characters (they show as squares in PDF viewers)
    # Keep only newline \n
    t = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", t)

    # Normalize quotes/dashes
    t = (
        t.replace("’", "'").replace("‘", "'")
         .replace("“", '"').replace("”", '"')
         .replace("–", "-").replace("—", "-")
    )

    # Improve exam-style spacing
    t = re.sub(r"\n{3,}", "\n\n", t)
    t = re.sub(r"[ ]{2,}", " ", t)

    return t.strip()



def generate_support_image(client: OpenAI, prompt: str, size: str = "1024x1024") -> bytes:
    import base64

    result = client.images.generate(
        model="gpt-image-1",
        prompt=prompt,
        size=size,
    )
    b64 = result.data[0].b64_json
    return base64.b64decode(b64)





# =====================================================
# 1. Helpers: API key
# =====================================================

def get_api_key() -> Optional[str]:
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
    """Low: 0–11, Medium: 12–21, High: 22–25 (per skill score out of 25)."""
    try:
        score = float(score)
    except Exception:
        return "Unknown"

    if score <= 11:
        return "Low"
    if score <= 21:
        return "Medium"
    if score <= 25:
        return "High"
    return "Unknown"


def map_level_and_score_to_grade(level: str, score: float) -> int:
    """
    Map performance level + raw score to a curriculum grade.

    - Low  → Grade 3–4:
        * 0–7   → Grade 3
        * 8–11  → Grade 4
    - Medium (12–21) → Grade 5
    - High   (22–25) → Grade 6
    """
    try:
        score = float(score)
    except Exception:
        return 5

    if level == "Low":
        return 3 if score <= 7 else 4
    if level == "Medium":
        return 5
    if level == "High":
        return 6
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
        ).rename(
            columns={
                "StudentNumber": "student_id",
                "StudentName": "student_name",
            }
        )
        return df_long

    return df


# =====================================================
# 3. RAG helpers: curriculum_bank.csv
# =====================================================

@st.cache_data(show_spinner=False)
def load_curriculum_bank() -> Optional[pd.DataFrame]:
    """Load curriculum_bank.csv from the same folder as this file."""
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(base_dir, "curriculum_bank.csv")
        return pd.read_csv(csv_path)
    except Exception:
        return None


def build_rag_context(
    curriculum_df: Optional[pd.DataFrame],
    skill: str,
    curriculum_grade: int,
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
            (curriculum_df["grade"] == int(curriculum_grade))
            & (curriculum_df["skill"].astype(str).str.lower() == str(skill).lower())
        ]

        if subset.empty:
            return ""

        bullets = []
        for _, row in subset.iterrows():
            line = (
                f"- Grade {row['grade']} | Module: {row.get('module', '')} | Topic: {row.get('topic', '')}\\n"
                f"  Standard: {row.get('standard', '')}\\n"
                f"  Teaching point: {row.get('teaching_point', '')}\\n"
            )

            grammar_ref = row.get("grammar_ref", "")
            vocab_list = row.get("vocab_list", "")
            example = row.get("example", "")

            if isinstance(grammar_ref, str) and grammar_ref.strip():
                line += f"  Grammar reference: {grammar_ref}\\n"
            if isinstance(vocab_list, str) and vocab_list.strip():
                line += f"  Key vocabulary: {vocab_list}\\n"
            if isinstance(example, str) and example.strip():
                line += f"  Example sentence: {example}\\n"

            bullets.append(line)

        return "\\n".join(bullets[:10])
    except Exception:
        return ""


# =====================================================
# 4. Exam-style task templates
# =====================================================

def build_exam_style_task(skill: str, curriculum_grade: int, num_questions: int) -> str:
    s = str(skill).strip().lower()

    if "languagefunction" in s or "language function" in s:
        return f"""
Create a LANGUAGE FUNCTIONS section exactly like a school test.

STRICT RULES:
- Use ONE item per line.
- Use clear line breaks.
- DO NOT use symbols like ■ or bullets.
- Keep columns separated by new lines only.

FORMAT EXACTLY LIKE THIS:

LANGUAGE FUNCTIONS:
Read and match.

A:
1- I like pizza, but I don’t like broccoli.
2- Can I have a glass of water, please?
3- Would you like some ice cream?
4- I think swimming is great!

B:
a- Yes, please! I love ice cream.
b- My favourite colour is blue because it’s calm.
c- My favourite sport is football because it’s fun.
d- I agree. It is very healthy.

ANSWER KEY:
1) b
2) a
3) d
4) c
"""


    if "reading" in s:
        return f"""
READING COMPREHENSION:
Directions: Read the passage and answer the questions.

PASSAGE:
<Write ONE passage appropriate for Grade {curriculum_grade}. Include ONE underlined word.>

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

3- <Short answer question>
____________________________________

4- <Short answer question>
____________________________________

ANSWER KEY:
1) B
2) A
3) <expected answer>
4) <expected answer>
"""

    if "vocab" in s or "vocabulary" in s:
        return f"""
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

    if "grammar" in s:
        return f"""
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

    if "writing" in s:
        sentences = 4 if curriculum_grade == 3 else (5 if curriculum_grade == 4 else (6 if curriculum_grade == 5 else 7))
        return f"""
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

    return f"""
TASK:
Create {num_questions} suitable questions for Grade {curriculum_grade} focusing on the skill: {skill}.
Provide ANSWER KEY.
"""


# =====================================================
# 5. Worksheet generation (GPT)
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
    """
    Generate worksheet using GPT based on grade, skill, level and RAG context.
    IMPORTANT: Do not generate a reading passage unless skill is reading.
    """
    system_prompt = (
    "You are a primary school English assessment generator aligned with the Qatar National Curriculum. "
    "You MUST follow the exact section headings and format in the TASK block. "
    "Do NOT create a reading passage unless the selected skill is Reading/ReadingComprehension. "
    "Keep language clear, age-appropriate, and culturally suitable.\n\n"

    "IMPORTANT FORMATTING RULES:\n"
    "- Do NOT use symbols, bullets, boxes, or decorative characters.\n"
    "- Use plain text only (Word-style exam format).\n"
    "- Leave one blank line between sections.\n"
    "- Put section titles on their own lines.\n"
    "- Put A and B on separate lines.\n"
    "- Number questions exactly like this: 1-, 2-, 3-.\n"
    "- Do NOT use ■, ●, •, or similar symbols.\n\n"

    "GRAMMAR QUESTION RULES:\n"
    "- Do NOT use the word 'blank'.\n"
    "- Use only underscores ( ______ ) to indicate missing words.\n"
    "- Do NOT use <u>, </u>, or any HTML or markup tags.\n"
    "- Do NOT underline words using any method.\n"
    "- For correction questions, write the full sentence normally, then write the instruction on a new line:\n"
    "  (Correct the verb) or (Correct the sentence).\n"
)



    rag_section = ""
    if rag_context:
        rag_section = f"CURRICULUM BANK (RAG):\\n{rag_context}\\n"

    task_block = build_exam_style_task(
        skill=skill,
        curriculum_grade=int(curriculum_grade),
        num_questions=int(num_questions),
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
- Do not add extra sections.
- Provide ANSWER KEY as shown.

TASK BLOCK:
{task_block}
"""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.4,
    )
    return resp.choices[0].message.content


def split_worksheet_and_answer(text: str):
    marker = "ANSWER KEY:"
    idx = text.upper().find(marker)
    if idx == -1:
        return text.strip(), "ANSWER KEY:\\n(Not clearly provided by the model.)"
    body = text[:idx].strip()
    answer = text[idx:].strip()
    return body, answer
    
def clean_text_for_pdf(text: str) -> str:
    replacements = {
        "■": "",
        "\t": " ",
        "  ": " ",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)

    # ضمان سطر فاضي بين الأقسام
    text = text.replace("A:", "\nA:\n")
    text = text.replace("B:", "\nB:\n")
    text = text.replace("ANSWER KEY:", "\nANSWER KEY:\n")

    return text.strip()


def text_to_pdf(title: str, content: str, font_size: int = 11, line_height: int = 14,
                image_bytes: bytes | None = None) -> bytes:
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    x = 40
    y = height - 60

    # Title
    c.setFont("Helvetica-Bold", 14)
    c.drawString(x, y, title)
    y -= 22

    # Optional image (top-right)
    if image_bytes:
        try:
            img = ImageReader(io.BytesIO(image_bytes))
            img_w = 140
            img_h = 140
            c.drawImage(img, width - x - img_w, y - img_h + 10, img_w, img_h, mask='auto')
            y -= (img_h - 10)
        except Exception:
            pass

    y -= 10
    c.setFont("Helvetica", font_size)

    for line in content.split("\n"):
        # basic wrapping
        while len(line) > 110:
            c.drawString(x, y, line[:110])
            line = line[110:]
            y -= line_height
            if y < 40:
                c.showPage()
                y = height - 60
                c.setFont("Helvetica", font_size)

        c.drawString(x, y, line)
        y -= line_height
        if y < 40:
            c.showPage()
            y = height - 60
            c.setFont("Helvetica", font_size)

    c.showPage()
    c.save()
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes



# =====================================================
# 6. UI CSS (inside triple quotes to avoid SyntaxError)
# =====================================================

CUSTOM_CSS = """
<style>
/* Hide Streamlit header/footer */
header, footer {visibility: hidden;}

/* App base */
body, .stApp {
    background: #f4f5f7;
    font-family: "Cairo", sans-serif;
    color: #1f2937;
}

/* Remove extra top spacing (helps with the grey band above header too) */
section.main > div { padding-top: 0.5rem !important; }
.block-container { padding-top: 0.5rem !important; }

/* HEADER */
.app-header {
    width: 100%;
    padding: 1.6rem 2rem;
    background: linear-gradient(135deg, #8A1538, #5e0d24);
    border-radius: 0 0 20px 20px;
    color: #ffffff;
    margin-bottom: 0.8rem; /* أقل عشان التابات تقرب */
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.20);
}
.header-title { font-size: 2.2rem; font-weight: 800; letter-spacing: .3px; }
.header-sub { font-size: 1rem; opacity: 0.95; }

/* CARDS */
.card {
    background: white;
    padding: 1.5rem 1.7rem;
    border-radius: 16px;
    margin-bottom: 1.2rem;
    border: 1px solid #e5e7eb;
    box-shadow: 0 6px 20px rgba(0,0,0,0.06);
}
.step-title { color: #8A1538; font-size: 1.3rem; font-weight: 700; }
.step-help { color: #555; font-size: .95rem; }

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
.stButton > button:hover { background: #7a0e31; }

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

/* =========================
   TABS (Fix the white/grey band + style tabs)
   ========================= */

/* container area around tabs (kills the grey/white bar) */
div[data-testid="stTabs"]{
    background: transparent !important;
    margin-top: -10px !important;  /* يرفع التابات شوي لأعلى */
    padding-top: 0 !important;
}
div[data-testid="stTabs"] > div{
    background: transparent !important;
    padding-top: 0 !important;
}

/* tab list bar */
div[data-testid="stTabs"] div[role="tablist"]{
    background: linear-gradient(135deg, #8A1538, #5e0d24) !important;
    padding: 10px 12px !important;
    border-radius: 999px !important;
    box-shadow: 0 10px 25px rgba(0,0,0,0.18) !important;
    gap: .6rem !important;
}

/* remove default underline/border */
div[data-testid="stTabs"] div[data-baseweb="tab-border"]{
    display: none !important;
}
div[data-testid="stTabs"] div[data-baseweb="tab-highlight"]{
    background: transparent !important;
}

/* tab buttons */
div[data-testid="stTabs"] button[role="tab"]{
    background: transparent !important;
    color: rgba(255,255,255,0.85) !important;
    border-radius: 999px !important;
    border: none !important;
    padding: 8px 16px !important;
    font-size: .9rem !important;
}

/* hover */
div[data-testid="stTabs"] button[role="tab"]:hover{
    background: rgba(255,255,255,0.14) !important;
    color: #ffffff !important;
}

/* selected tab */
div[data-testid="stTabs"] button[role="tab"][aria-selected="true"]{
    background: #ffffff !important;
    color: #8A1538 !important;
    font-weight: 700 !important;
    box-shadow: 0 6px 16px rgba(0,0,0,0.22) !important;
}
</style>
"""



# =====================================================
# 7. Streamlit App
# =====================================================

def main():
    st.set_page_config(
        page_title="English Worksheets Generator",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.caption("APP VERSION: 2025-12-21-01")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

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

    api_key = get_api_key()
    if not api_key:
        st.error("OPENAI_API_KEY is missing. Add it in Render → Environment Variables.")
        return

    client = OpenAI(api_key=api_key)

    st.session_state.setdefault("df_raw", None)
    st.session_state.setdefault("processed_df", None)
    st.session_state.setdefault("curriculum_df", load_curriculum_bank())

    curriculum_df = st.session_state["curriculum_df"]

    tab_overview, tab_data, tab_generate, tab_help = st.tabs(
        ["Overview", "Data & RAG", "Generate Worksheets", "Help & Tools"]
    )

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

    with tab_data:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="step-title">Step 1 — Upload student performance CSV</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <span class="tool-tag">Pandas</span>
            <span class="tool-tag">Data validation</span>
            <p class="step-help">
            Expected format:
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

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="step-title">Step 2 — Curriculum bank (RAG)</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <span class="tool-tag">RAG</span>
            <p class="step-help">
            The app loads <code>curriculum_bank.csv</code> from the same folder as <code>streamlit_app.py</code>.
            </p>
            """,
            unsafe_allow_html=True,
        )

        if curriculum_df is None:
            st.error("curriculum_bank.csv not found in the app folder.")
            st.info("Make sure the file name is exactly: curriculum_bank.csv (same folder as streamlit_app.py).")
        else:
            st.success("Curriculum bank loaded successfully ✔")
            try:
                summary = curriculum_df.groupby(["grade", "skill"]).size()
                st.markdown("**Available entries by grade and skill:**")
                st.dataframe(summary.to_frame("count"))
            except Exception:
                st.dataframe(curriculum_df.head(), use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="step-title">Step 3 — Process data & classify levels</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <span class="tool-tag">Rule-based classifier</span>
            <p class="step-help">
            Classifies scores into Low/Medium/High and maps to recommended curriculum grade (3–6).
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
                    df_proc["recommended_grade"] = df_proc.apply(
                        lambda row: map_level_and_score_to_grade(row["level"], row["score"]),
                        axis=1,
                    )
                    st.session_state["processed_df"] = df_proc
                    st.success("Student data processed successfully ✔")
                    counts = df_proc["level"].value_counts()
                    st.markdown("**Classification summary (by level):**")
                    st.markdown(
                        f"- Low: {counts.get('Low', 0)}  \n"
                        f"- Medium: {counts.get('Medium', 0)}  \n"
                        f"- High: {counts.get('High', 0)}"
                    )

                    st.write("Processed data preview:")
                    st.dataframe(df_proc.sort_values(["student_id", "skill"]).head(20), use_container_width=True)
                except Exception as e:
                    st.error(f"Error while processing data: {e}")

        st.markdown("</div>", unsafe_allow_html=True)

    with tab_generate:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="step-title">Step 4 — Generate worksheets (PDF)</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <span class="tool-tag">GPT API</span>
            <span class="tool-tag">RAG</span>
            <span class="tool-tag">PDF export</span>
            <p class="step-help">
            Generates a worksheet + separate answer key for each student in the selected skill + level.
            </p>
            """,
            unsafe_allow_html=True,
        )

        df = st.session_state.get("processed_df", None)
        curriculum_df2 = st.session_state.get("curriculum_df", None)

        if df is None:
            st.info("Please go to the 'Data & RAG' tab and process the student data first.")
        elif curriculum_df2 is None:
            st.error("Curriculum bank is not loaded. Please add curriculum_bank.csv.")
        else:
            skills = sorted(df["skill"].astype(str).unique())
            selected_skill = st.selectbox("Choose skill", skills)
            selected_level = st.selectbox("Choose performance level", ["Low", "Medium", "High"])
            num_q = st.slider("Number of questions per worksheet", 3, 10, 5)

            target_df = df[(df["skill"] == selected_skill) & (df["level"] == selected_level)]
            st.markdown(f"Students in this group: **{len(target_df)}**")

            if st.button("Generate PDFs for this group"):
                if target_df.empty:
                    st.error("No students match this skill + level.")
                else:
                    with st.spinner("Generating worksheets and answer keys…"):
                        try:
                            for _, row in target_df.iterrows():
                                grade_for_rag = int(row.get("recommended_grade", 5))

                                rag_context = build_rag_context(
                                    curriculum_df=curriculum_df2,
                                    skill=str(row["skill"]),
                                    curriculum_grade=grade_for_rag,
                                )

                                full_text = generate_worksheet(
                                    client=client,
                                    student_name=str(row["student_name"]),
                                    student_grade=5,
                                    curriculum_grade=grade_for_rag,
                                    skill=str(row["skill"]),
                                    level=str(row["level"]),
                                    num_questions=int(num_q),
                                    rag_context=rag_context,
                                )

                                worksheet_body, answer_key = split_worksheet_and_answer(full_text)

                                worksheet_body = normalize_pdf_text(worksheet_body)
                                suspects = [(ch, hex(ord(ch))) for ch in worksheet_body[:500] if ord(ch) > 127]
                                st.write("NON-ASCII SAMPLE:", suspects[:80])
                                st.write("PREVIEW:", worksheet_body[:300])

                                answer_key = normalize_pdf_text(answer_key)
                                # --- Additional Support Mode (auto for Low) ---
                                support_mode = (row["level"] == "Low")

                                # Optional image for support mode (Reading/Writing only)
                                img_bytes = None
                                skill_lower = str(row["skill"]).lower()

                                if support_mode and ("reading" in skill_lower or "writing" in skill_lower):
                                    img_prompt = (
                                        "Simple black-and-white cartoon icon, no text, kid-friendly, print-friendly. "
                                        "Theme: school, bus, children learning."
                                    )
                                    try:
                                        img_bytes = generate_support_image(client, img_prompt, size="auto")
                                    except Exception as e:
                                        st.error(f"Image generation failed: {e}")
                                        img_bytes = None


                                # -------------------------------
                                # -------------------------------
                                # Generate worksheet PDF safely
                                # -------------------------------
                                try:
                                    ws_pdf = text_to_pdf(
                                        title=f"Worksheet for {row['student_name']}",
                                        content=worksheet_body,
                                        font_size=16 if support_mode else 11,
                                        line_height=20 if support_mode else 14,
                                        image_bytes=img_bytes,
                                    )
                                    st.success(
                                        f"Worksheet PDF created ✅ ({row['student_name']}) "
                                        f"size={len(ws_pdf)} bytes"
                                    )
                                except Exception as e:
                                    st.error(
                                        f"PDF generation failed for {row['student_name']}: {e}"
                                    )
                                    ws_pdf = None

                                # -------------------------------
                                # Generate answer key PDF
                                # -------------------------------
                                ak_pdf = text_to_pdf(
                                    title=f"Answer Key for {row['student_name']}",
                                    content=answer_key,
                                )

                                # -------------------------------
                                # Download buttons (ONLY if PDF exists)
                                # -------------------------------
                                if ws_pdf:
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

    with tab_help:
        st.markdown(
            """
            <div class="card">
                <div class="step-title">Help & implementation notes</div>
                <ul class="step-help">
                    <li><b>Pandas</b> — reads the CSV, reshapes data, and classifies students.</li>
                    <li><b>Rule-based classifier</b> — maps Low/Medium/High to Grades 3–6.</li>
                    <li><b>RAG</b> — uses <code>curriculum_bank.csv</code> to align outputs with standards.</li>
                    <li><b>GPT API</b> — generates exam-style sections per selected skill.</li>
                    <li><b>PDF export</b> — downloadable worksheet + answer key.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
