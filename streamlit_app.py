import os
import io
from typing import Tuple, Optional

import pandas as pd
import streamlit as st
from openai import OpenAI

# PDF generation
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm


# ==============================
# Helper functions
# ==============================

def get_api_key() -> Optional[str]:
    """Get OpenAI API key from environment or Streamlit secrets."""
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        try:
            key = st.secrets.get("OPENAI_API_KEY", None)
        except Exception:
            key = None
    return key


def transform_thesis_format(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert Marwa's thesis dataset into long format if it matches
    the expected columns (StudentNumber, StudentName, LanguageFunction, ReadingComprehension, Grammar, Writing).
    Otherwise returns the dataframe as is.
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
        ).reset_index(drop=True)

        df_long = df_long.rename(
            columns={
                "StudentNumber": "student_id",
                "StudentName": "student_name",
            }
        )
        return df_long

    # Assume already in long format with columns:
    # student_id, student_name, skill, score
    return df


def map_score_to_curriculum_grade(score: float) -> int:
    """
    Map student's score in a given skill to a curriculum grade:
    - 23 or above  -> Grade 6
    - 22 to 15     -> Grade 5
    - 14 or below  -> Grade 4 (we will also mention Grade 3 in the prompt for scaffolding)
    """
    if score >= 23:
        return 6
    elif score >= 15:
        return 5
    else:
        return 4


def classify_performance_band(score: float) -> str:
    """
    Simple band label just for display:
    High   -> score >= 23
    Medium -> 15 <= score <= 22
    Low    -> score < 15
    """
    if score >= 23:
        return "High"
    elif score >= 15:
        return "Medium"
    return "Low"


def build_skill_template(skill: str, curriculum_grade: int) -> str:
    """
    Return detailed instructions for GPT about the exact question types
    and format to use for this skill and curriculum grade.
    """

    s = skill.lower().strip()

    # --------- READING ----------
    if "reading" in s:
        return f"""
Reading Comprehension (based on Qatar English curriculum, Grade {curriculum_grade}):

- Write ONE reading passage of about 80–120 words.
- The topic and difficulty must be suitable for Grade {curriculum_grade}.
- Then create EXACTLY 4 questions in this format:

1) Multiple-choice question about the MAIN IDEA of the whole text
   - four options (A, B, C, D), only one correct.

2) Multiple-choice vocabulary question about the meaning of an underlined word from the text
   - four options (A, B, C, D).

3) Short-answer WH-question (Why / How / Where / Who) based on specific information.

4) Short-answer WH-question based on another detail in the passage.

Number the questions as 1), 2), 3), 4) and label the options A), B), C), D).
"""

    # --------- GRAMMAR ----------
    if "grammar" in s:
        return f"""
Grammar section (Grade {curriculum_grade}):

Create EXACTLY 4 questions in the SAME style as ministry mid-term tests:

13) Multiple-choice grammar question with four options (A, B, C, D).
14) Multiple-choice verb tense question (choose the correct verb form) with four options.
15) "Do as shown between brackets": give a sentence with a verb in brackets
    and ask the student to correct the verb form.
16) Another "Do as shown between brackets" sentence to correct the verb.

Use present simple / past simple / continuous forms appropriate for Grade {curriculum_grade}.
Number the items 13), 14), 15), 16) and keep the stem in clear exam style.
"""

    # --------- VOCABULARY ----------
    if "vocab" in s:
        return f"""
Vocabulary section (Grade {curriculum_grade}):

- Create a word box with 4–6 words from the Grade {curriculum_grade} curriculum.
- Then write 4 sentences with ONE gap each.
- The instruction must be: "Fill in the gaps with suitable words from the box."
- Number them 9), 10), 11), 12).

Example structure (you must generate new content, not copy):
VOCABULARY

Fill in the gaps with suitable words from the box:
(word1 – word2 – word3 – word4)

9) ......
10) ......
11) ......
12) ......
"""

    # --------- LANGUAGE FUNCTIONS ----------
    if "language" in s or "function" in s:
        return f"""
Language Functions section (Grade {curriculum_grade}):

Create a MATCHING activity A/B, same as ministry style:

- Column A: four questions or sentence starters (1–4).
- Column B: four answers (a–d).
- The instruction must be: "Read and match."

Example of structure (you must generate new content, not copy):
LANGUAGE FUNCTIONS
Read and match:

A                               B
1. How often do you ... ?       a. twice a day
2. What was he doing ... ?      b. watering the plants
3. Whose toys are these?        c. hers
4. What did you do yesterday?   d. I went to the park
"""

    # --------- WRITING ----------
    if "writing" in s:
        if curriculum_grade <= 3:
            sentences = 4
        elif curriculum_grade == 4:
            sentences = 5
        elif curriculum_grade == 5:
            sentences = 6
        else:
            sentences = 7

        return f"""
Writing task (Grade {curriculum_grade}):

- Ask the student to write a paragraph of about {sentences} sentences.
- The topic must be suitable for Grade {curriculum_grade} and linked to daily life or school.
- Give 3–5 guiding questions to support weaker learners.
- The instruction must be clear, for example:

"Write a paragraph of about {sentences} sentences about your favourite hobby.
Use these guiding questions:
• What is your hobby?
• When do you usually do it?
• Where do you do it?
• Why do you like it?
• Who do you do it with?"

Only write the TASK instructions, do NOT write the model answer.
"""

    # Default fallback
    return f"""
Create a short, curriculum-aligned English activity for Grade {curriculum_grade}
with 4 clear questions. Use a style similar to Qatar English mid-term exam questions.
"""


def build_rag_context(
    curriculum_df: Optional[pd.DataFrame],
    skill: str,
    curriculum_grade: int,
    max_rows: int = 6,
) -> str:
    """
    Simple RAG context builder from a local curriculum_bank.csv.
    Expected columns (flexible): grade, skill, topic, objective, example
    """
    if curriculum_df is None:
        return ""

    df = curriculum_df.copy()

    # Optional filters if columns exist
    if "grade" in df.columns:
        df = df[df["grade"] == curriculum_grade]

    skill_lower = skill.lower()
    if "skill" in df.columns:
        df = df[df["skill"].str.lower().str.contains(skill_lower, na=False)]

    if df.empty:
        return ""

    df = df.head(max_rows)

    parts = []
    for _, row in df.iterrows():
        topic = row.get("topic", "")
        obj = row.get("objective", "")
        example = row.get("example", "")
        piece = f"- Topic: {topic} | Objective: {obj} | Example: {example}"
        parts.append(piece)

    return "\n".join(parts)


def split_worksheet_and_answer_key(full_text: str) -> Tuple[str, str]:
    """
    Split GPT output into worksheet vs answer key using the 'ANSWER KEY' marker.
    If not found, returns the full text as worksheet and empty answer key.
    """
    if not full_text:
        return "", ""

    upper = full_text.upper()
    marker = "ANSWER KEY"
    idx = upper.find(marker)

    if idx == -1:
        return full_text.strip(), ""

    worksheet = full_text[:idx].strip()
    answer_key = full_text[idx:].strip()
    return worksheet, answer_key


def create_pdf_bytes(title: str, body_text: str) -> bytes:
    """
    Create a simple multi-line PDF in memory using ReportLab.
    """
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    # Title
    c.setFont("Helvetica-Bold", 14)
    c.drawString(2 * cm, height - 2.5 * cm, title)

    # Body
    c.setFont("Helvetica", 11)
    text_obj = c.beginText()
    text_obj.setTextOrigin(2 * cm, height - 3.5 * cm)
    text_obj.setLeading(14)

    for line in body_text.splitlines():
        if text_obj.getY() < 2 * cm:
            c.drawText(text_obj)
            c.showPage()
            c.setFont("Helvetica", 11)
            text_obj = c.beginText()
            text_obj.setTextOrigin(2 * cm, height - 2.5 * cm)
            text_obj.setLeading(14)
        text_obj.textLine(line)

    c.drawText(text_obj)
    c.showPage()
    c.save()

    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes


def generate_worksheet(
    client: OpenAI,
    student_name: str,
    student_grade: int,
    curriculum_grade: int,
    skill: str,
    level: str,
    score: float,
    rag_context: str = "",
) -> str:
    """
    Generate worksheet text using GPT based on curriculum grade & skill template.
    Includes an ANSWER KEY section at the end.
    """

    if curriculum_grade == 4:
        remediation_note = (
            "This student is low-achieving in this skill. "
            "Use Grade 4 objectives but also draw on simpler Grade 3 concepts "
            "to provide extra scaffolding and support."
        )
    else:
        remediation_note = (
            "The difficulty should match the curriculum grade only, "
            "with natural language support for the student."
        )

    skill_template = build_skill_template(skill, curriculum_grade)

    system_prompt = (
        "You are an English curriculum expert and exam writer for Qatar primary schools. "
        "You design remedial worksheets that follow the official mid-term test style. "
        "Always follow the requested structure exactly (numbering, options, instructions). "
        "Do NOT include explanations to the teacher; only write what the student will see on the worksheet."
    )

    rag_block = ""
    if rag_context:
        rag_block = f"""
Additional curriculum context (from the official bank). Use it to guide topics and vocabulary:

{rag_context}

"""

    user_prompt = f"""
Student name: {student_name}
Actual school grade: {student_grade}
Skill: {skill}
Score in this skill: {score}
Mapped curriculum grade for the worksheet: {curriculum_grade}
Performance level: {level}

Remediation guidance:
{remediation_note}

{rag_block}
=== REQUIRED WORKSHEET FORMAT FOR THIS SKILL ===
{skill_template}
=== END OF FORMAT DESCRIPTION ===

Now generate ONLY the worksheet content for {student_name}.
At the end, add a clear section titled "ANSWER KEY:" listing the correct answers.
Do NOT add extra commentary. Do NOT mention 'curriculum grade' explicitly in the text.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.5,
    )

    return response.choices[0].message.content


# ==============================
# Custom CSS (header + tabs + cards)
# ==============================

CUSTOM_CSS = """
<style>
header, footer {visibility: hidden;}

body, .stApp {
    background-color: #f6f7fb;
    font-family: "Cairo", sans-serif;
}

/* Header */
.app-header {
    width: 100%;
    padding: 1.6rem 2.2rem;
    background: linear-gradient(135deg, #8A1538, #600d26);
    border-radius: 0 0 20px 20px;
    color: white;
    margin-bottom: 1.5rem;
    box-shadow: 0 6px 20px rgba(0,0,0,0.25);
}

.header-title {
    font-size: 2.0rem;
    font-weight: 800;
    margin-bottom: 0.2rem;
}

.header-sub {
    font-size: 0.95rem;
    opacity: 0.9;
}

/* Tabs */
.stTabs [role="tablist"] {
    gap: 0.75rem;
}

.stTabs [role="tab"] {
    padding: 0.55rem 1.4rem;
    border-radius: 999px;
    border: 1px solid #e0d7de;
    color: #7a304a;
    background-color: #ffffff;
    font-weight: 600;
}

.stTabs [aria-selected="true"] {
    background: #8A1538 !important;
    color: #ffffff !important;
    border-color: #8A1538 !important;
}

/* Cards */
.card {
    background: white;
    padding: 1.4rem 1.6rem;
    border-radius: 18px;
    margin-bottom: 1.2rem;
    box-shadow: 0 6px 18px rgba(0, 0, 0, 0.06);
}

.step-title {
    color: #8A1538;
    font-size: 1.2rem;
    font-weight: 700;
    margin-bottom: 0.4rem;
}

.step-help {
    color: #555;
    font-size: 0.9rem;
    margin-bottom: 0.6rem;
}

/* Tool tags */
.tool-tag {
    display: inline-block;
    background: #f3e3ea;
    color: #8A1538;
    border-radius: 999px;
    padding: 0.2rem 0.65rem;
    font-size: 0.75rem;
    margin-right: 0.4rem;
    margin-bottom: 0.3rem;
}

/* Download buttons margin */
.download-row {
    margin-top: 0.5rem;
    margin-bottom: 0.8rem;
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

    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # HEADER
    st.markdown(
        """
        <div class="app-header">
            <div class="header-title">English Worksheets Generator</div>
            <div class="header-sub">
                Adaptive remedial worksheets aligned with Qatar English curriculum using Pandas + RAG + GPT API
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # API key
    api_key = get_api_key()
    if not api_key:
        st.error("OPENAI_API_KEY is missing. Please add it in Settings → Secrets.")
        return

    client = OpenAI(api_key=api_key)

    # Load curriculum bank once (optional)
    if "curriculum_df" not in st.session_state:
        try:
            # You can change the path if you store it in a /data folder
            cur_df = pd.read_csv("curriculum_bank.csv")
            st.session_state["curriculum_df"] = cur_df
        except Exception:
            st.session_state["curriculum_df"] = None

    curriculum_df = st.session_state.get("curriculum_df", None)

    tab_overview, tab_data, tab_generate, tab_help = st.tabs(
        ["Overview", "Data & RAG", "Generate Worksheets", "Help & Tools"]
    )

    # ----------------- OVERVIEW -----------------
    with tab_overview:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="step-title">Welcome</div>
            <p class="step-help">
            This prototype analyses student performance data and generates
            <b>curriculum-aligned remedial worksheets</b> for English skills
            (Reading, Grammar, Vocabulary, Language Functions, Writing).
            </p>
            <ul>
              <li>Upload your mid-term skill scores in CSV format.</li>
              <li>The system maps each score to an appropriate curriculum grade:
                <ul>
                  <li><b>23 and above</b> → Grade 6</li>
                  <li><b>22 to 15</b> → Grade 5</li>
                  <li><b>14 and below</b> → Grade 4 (with Grade 3 support)</li>
                </ul>
              </li>
              <li>For each student and skill, GPT generates exam-style questions and an answer key.</li>
              <li>You can download the worksheet and answer key as PDFs.</li>
            </ul>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # ----------------- DATA & RAG -----------------
    with tab_data:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="step-title">Step 1 — Upload student performance CSV</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <span class="tool-tag">Pandas</span>
            <span class="tool-tag">Data preparation</span>
            <p class="step-help">
            Upload the CSV file containing your students’ skill scores. The prototype expects
            the thesis-style format:
            <code>StudentNumber, StudentName, LanguageFunction, ReadingComprehension, Grammar, Writing</code>.
            </p>
            """,
            unsafe_allow_html=True,
        )

        uploaded = st.file_uploader("Upload Students.csv", type=["csv"])

        if uploaded is None:
            st.info("Please upload a CSV file to continue.")
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            df_raw = pd.read_csv(uploaded)
            st.write("Raw data preview:")
            st.dataframe(df_raw.head(), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # Step 2: curriculum bank for RAG
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="step-title">Step 2 — Curriculum bank for RAG</div>', unsafe_allow_html=True)
            st.markdown(
                """
                <span class="tool-tag">RAG</span>
                <span class="tool-tag">Curriculum bank</span>
                <p class="step-help">
                The curriculum bank is loaded automatically from <code>curriculum_bank.csv</code> (if available).
                It is used to align topics and objectives with the Qatar English curriculum.
                </p>
                """,
                unsafe_allow_html=True,
            )

            if curriculum_df is None:
                st.error("No curriculum bank loaded. Please ensure 'curriculum_bank.csv' exists if you want RAG support.")
            else:
                st.write("Curriculum bank preview:")
                st.dataframe(curriculum_df.head(), use_container_width=True)

            st.markdown("</div>", unsafe_allow_html=True)

            # Step 3: Process & classify
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="step-title">Step 3 — Process data & map curriculum levels</div>', unsafe_allow_html=True)
            st.markdown(
                """
                <span class="tool-tag">Rule-based mapping</span>
                <span class="tool-tag">Score bands</span>
                <p class="step-help">
                The system converts the data into long format (one row per student per skill)
                and applies your score rules to decide which curriculum grade to use for each worksheet.
                </p>
                """,
                unsafe_allow_html=True,
            )

            df_long = transform_thesis_format(df_raw)

            # Just for information: assume all students are in same actual grade? optional
            df_long["grade"] = df_long.get("grade", 5)

            df_long["performance_band"] = df_long["score"].apply(classify_performance_band)
            df_long["target_curriculum_grade"] = df_long["score"].apply(map_score_to_curriculum_grade)

            st.write("Processed data:")
            st.dataframe(df_long.head(20), use_container_width=True)

            st.session_state["df_processed"] = df_long
            st.markdown("</div>", unsafe_allow_html=True)

    # ----------------- GENERATE WORKSHEETS -----------------
    with tab_generate:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="step-title">Generate remedial worksheets</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <span class="tool-tag">GPT API</span>
            <span class="tool-tag">PDF export</span>
            <p class="step-help">
            Select a skill, and the system will generate exam-style worksheets and answer keys
            for all students mapped to that skill.
            The content is not shown on-screen; instead you get direct download links as PDFs.
            </p>
            """,
            unsafe_allow_html=True,
        )

        df_processed = st.session_state.get("df_processed", None)
        if df_processed is None:
            st.info("Please upload and process a CSV file in the 'Data & RAG' tab first.")
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            skills_available = sorted(df_processed["skill"].unique())
            selected_skill = st.selectbox("Choose a skill", skills_available)

            # Optionally filter by band
            band_filter = st.selectbox("Filter by performance band (optional)", ["All", "High", "Medium", "Low"])
            target_df = df_processed[df_processed["skill"] == selected_skill]

            if band_filter != "All":
                target_df = target_df[target_df["performance_band"] == band_filter]

            st.markdown(f"Students in this selection: **{len(target_df)}**")

            if st.button("Generate worksheets (PDF)"):
                if target_df.empty:
                    st.error("No students match the selected criteria.")
                else:
                    with st.spinner("Generating worksheets... please wait ⏳"):
                        for idx, (_, row) in enumerate(target_df.iterrows(), start=1):
                            student_name = str(row["student_name"])
                            student_id = row.get("student_id", idx)
                            skill = str(row["skill"])
                            score = float(row["score"])
                            band = row.get("performance_band", "")
                            cur_grade = int(row["target_curriculum_grade"])
                            actual_grade = int(row.get("grade", 5))

                            rag_ctx = build_rag_context(curriculum_df, skill, cur_grade)

                            full_text = generate_worksheet(
                                client=client,
                                student_name=student_name,
                                student_grade=actual_grade,
                                curriculum_grade=cur_grade,
                                skill=skill,
                                level=band,
                                score=score,
                                rag_context=rag_ctx,
                            )

                            worksheet_text, answer_key_text = split_worksheet_and_answer_key(full_text)

                            ws_title = f"Worksheet for {student_name} ({skill})"
                            ws_pdf = create_pdf_bytes(ws_title, worksheet_text)

                            if answer_key_text:
                                ak_title = f"Answer Key for {student_name} ({skill})"
                                ak_pdf = create_pdf_bytes(ak_title, answer_key_text)
                            else:
                                ak_pdf = None

                            st.markdown(f"#### {idx}. {student_name} — {skill} (Curriculum G{cur_grade}, {band})")

                            col1, col2 = st.columns(2)
                            with col1:
                                st.download_button(
                                    label="📄 Download worksheet (PDF)",
                                    data=ws_pdf,
                                    file_name=f"worksheet_{student_id}_{student_name}_{skill}.pdf",
                                    mime="application/pdf",
                                    key=f"ws_{idx}",
                                )
                            with col2:
                                if ak_pdf:
                                    st.download_button(
                                        label="🗝️ Download answer key (PDF)",
                                        data=ak_pdf,
                                        file_name=f"answer_key_{student_id}_{student_name}_{skill}.pdf",
                                        mime="application/pdf",
                                        key=f"ak_{idx}",
                                    )

                    st.success("Worksheet generation finished.")
            st.markdown("</div>", unsafe_allow_html=True)

    # ----------------- HELP & TOOLS -----------------
    with tab_help:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="step-title">Help & tools</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <p class="step-help">
            This tab documents the main tools and concepts used in the prototype:
            </p>
            <ul>
              <li><b>Pandas</b> – reshaping the thesis dataset into long format and computing score bands.</li>
              <li><b>Rule-based mapping</b> – applying your scoring rule to decide the target curriculum grade.</li>
              <li><b>RAG (Retrieval-Augmented Generation)</b> – optional curriculum bank stored in
                  <code>curriculum_bank.csv</code> to guide GPT with real objectives and examples.</li>
              <li><b>GPT-4o-mini</b> – generating exam-style content that mirrors the official mid-term templates
                  for Reading, Grammar, Vocabulary, Language Functions, and Writing.</li>
              <li><b>ReportLab</b> – exporting the generated worksheets and answer keys as downloadable PDFs.</li>
            </ul>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
