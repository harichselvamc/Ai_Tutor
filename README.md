
# **Ai_Tutor** 🧠📚  
AI-powered Tutor with OCR and Vector Storage  

[![FastAPI](https://img.shields.io/badge/FastAPI-0.95+-brightgreen)](https://fastapi.tiangolo.com/)  
[![Streamlit](https://img.shields.io/badge/Streamlit-1.27+-red)](https://streamlit.io/)  
[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://www.python.org/)  
[![Qdrant](https://img.shields.io/badge/Vector%20DB-Qdrant-orange)](https://qdrant.tech/)  

## **📌 Overview**  
**Ai_Tutor** is an AI-powered tutoring system that leverages OCR (Optical Character Recognition), embedding techniques, and a **FastAPI backend** for efficient text extraction and semantic search. The system integrates with **Qdrant** (a vector storage backend) to enable fast retrieval of knowledge from documents. A **Streamlit frontend** provides an intuitive user interface.  

---

## **🛠 Features**
✅ **OCR-Based Document Processing** – Extracts text from PDFs and images  
✅ **Embedding Generation** – Converts text into vector representations  
✅ **FastAPI Backend** – Handles API requests and text retrieval  
✅ **Streamlit Frontend** – Provides an interactive web UI  
✅ **Vector Storage with Qdrant** – Enables efficient similarity search  
✅ **Dockerized Deployment** – Runs in a containerized environment  

---

## **📂 Project Structure**
```
Ai_Tutor/
│── aiagentmain.py            # Main AI agent logic  
│── env                       # Environment config file  
│── frontend.py               # Streamlit UI  
│── main.py                   # FastAPI backend  
│── ocr_extract.py            # OCR text extraction  
│── ocr_to_embeddings.py      # Converts OCR text to vector embeddings  
│── outputs/                  # Stores extracted outputs  
│── Pipfile                   # Dependency manager (Pipenv)  
│── Pipfile.lock              # Dependency lock file  
│── record_manager/           # Record management module  
│── requirements.txt          # Python dependencies  
│── test.pdf                  # Sample PDF file  
│── vector_storage_backend.py # Handles Qdrant vector storage  
```

---

## **🚀 Getting Started**
### **1️⃣ Setup Environment**
- Install **Python 3.12**  
- Clone the repository:
  ```bash
  git clone https://github.com/harichselvamc/Ai_Tutor.git
  cd Ai_Tutor
  ```

- Create a virtual environment:
  ```bash
  python -m venv env
  source env/bin/activate  # For Linux/Mac
  env\Scripts\activate     # For Windows
  ```

- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

---

### **2️⃣ Set API Keys**  
Export your API keys before running the project:
```bash
export GROQ_API_KEY=""
export MISTRAL_OCR_API_KEY=""
export MISTRAL_API_KEY=""
export EXA_API_KEY=""
```

Alternatively, create a **.env** file and add:
```
GROQ_API_KEY=""
MISTRAL_OCR_API_KEY=""
MISTRAL_API_KEY=""
EXA_API_KEY=""
```

---

### **3️⃣ Run the Backend (FastAPI)**
Start the FastAPI server:
```bash
uvicorn main:app --reload
```
The API will be available at: **http://127.0.0.1:8000**

---

### **4️⃣ Run the Frontend (Streamlit)**
Launch the UI:
```bash
streamlit run frontend.py
```
This will open the web interface in your browser.

---

### **5️⃣ OCR Processing & Embedding**
To process a document:
```bash
python ocr_extract.py
python ocr_to_embeddings.py
```
This extracts text and converts it into embeddings for semantic search.

---

### **6️⃣ Running with Docker (Qdrant)**
- Install **Docker** if not already installed.  
- Pull and run **Qdrant**:
  ```bash
  docker pull qdrant/qdrant
  docker run -p 6333:6333 qdrant/qdrant
  ```
- Once Qdrant is running, start FastAPI and Streamlit.

---


## **🔗 Connect**
GitHub Repo: [Ai_Tutor](https://github.com/harichselvamc/Ai_Tutor)  

---
