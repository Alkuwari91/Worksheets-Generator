
# Worksheets Generator  
AI-Based Student Worksheet Generation System

## Overview
This repository contains the implementation of an AI-based prototype developed as part of an MSc Computing dissertation at the University of Essex.  
The system supports the generation of personalised English worksheets for primary school students by combining student performance data with curriculum-aligned content using a Retrieval-Augmented Generation (RAG) approach.

The application is implemented using Python and Streamlit and is designed to support teachers in creating differentiated instructional materials aligned with curriculum expectations.

---

## System Design Overview
The system adopts a tab-based navigation layout to organise its core functionalities, including data upload, worksheet generation, settings, and help. This structure enables teachers to transition smoothly between tasks while maintaining a clear and guided workflow.

The central content area defaults to the data upload interface, intentionally directing users toward the first mandatory step in the process. To enhance transparency and usability, dynamic system indicators are integrated into the interface to display the real-time status of key backend components, including the OpenAI API connection and the readiness of the Retrieval-Augmented Generation (RAG) process. These indicators provide immediate feedback and ensure that worksheet generation is initiated only when required configurations are in place.

---

## Project Scope
The system provides the following functionality:

- Uploading student performance data in CSV format
- Classifying students into performance levels (Low / Medium / High) using fixed rule-based thresholds
- Mapping performance levels to target curriculum grades
- Combining validated student data with curriculum-aligned content through Retrieval-Augmented Generation
- Generating personalised English worksheets using the OpenAI API
- Producing downloadable PDF files for:
  - Student worksheets
  - Corresponding answer keys

This repository is provided to support transparency, verification, and reproducibility of the reported dissertation work.

---

## Technologies Used
- Python  
- Streamlit  
- Pandas  
- OpenAI API  
- ReportLab (PDF generation)

---

## Repository Structure
```

Worksheets-Generator/
│
├── streamlit_app.py
├── requirements.txt
├── curriculum_bank.csv
├── README.md
│
├── Grade 3 Curriculum Standards.pdf
├── Grade 4 Curriculum Standards.pdf
├── Grade 5 Curriculum Standards.pdf
└── Grade 6 Curriculum Standards.pdf

```

---

## Input Data

### Student Performance Data
The system accepts student performance data in CSV format. Uploaded datasets are validated before processing and are used to determine student proficiency levels across English skill domains.

### Curriculum-Aligned Content
Curriculum-aligned content is used to support Retrieval-Augmented Generation during worksheet creation, ensuring that generated materials align with curriculum expectations for the selected grade level and skill domain.

---

## Worksheet Generation
During worksheet generation, validated student data is combined with curriculum-aligned content through Retrieval-Augmented Generation to produce personalised worksheets. Teachers can filter students by skill domain and proficiency level, enabling targeted instructional interventions.

Generation parameters, such as difficulty level and number of questions, are applied during the generation process. For each request, the system produces two PDF outputs: a student worksheet and a corresponding answer key. Generated files are made available for download through the interface.

---

## Use Cases
The dissertation reports three use cases demonstrating the application of the system across different student performance distributions and English skill domains. Representative datasets and output files are included to support verification of system functionality.

---

## Reproducibility
All source code and supporting files required to reproduce the reported system behaviour are provided in this repository. The repository enables independent review of the system structure and implementation as described in the dissertation.

---

## Author
**Marwa Alkuwari**  
MSc Computing  
University of Essex
