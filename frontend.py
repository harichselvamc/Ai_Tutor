import streamlit as st
import requests
import json
from typing import List, Dict, Any, Optional
import pandas as pd

# Configuration
API_URL = "http://127.0.0.1:8000"  

# Set page config
st.set_page_config(
    page_title="AI Quiz Application",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Choose a feature:", ["Ask a Question", "Generate Quiz", "About"])

# Header styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4B8BBE;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #306998;
        text-align: center;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Check API health
try:
    response = requests.get(f"{API_URL}/health")
    api_status = "✅ Connected" if response.status_code == 200 else "❌ Not Connected"
except:
    api_status = "❌ Not Connected"

st.sidebar.write(f"API Status: {api_status}")

def ask_question(question: str, enable_web_search: bool = False) -> str:
    """Send a question to the API and get a response."""
    try:
        response = requests.post(
            f"{API_URL}/ask",
            json={"question": question, "enable_web_search": enable_web_search}
        )
        if response.status_code == 200:
            return response.json()
        else:
            return f"Error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Error: {str(e)}"

def generate_quiz(topic: str, num_questions: int) -> Optional[Dict[str, Any]]:
    """Generate a quiz from the API."""
    try:
        response = requests.post(
            f"{API_URL}/quiz",
            json={"topic": topic, "num_questions": num_questions}
        )
        if response.status_code == 200:
            # The API returns JSON directly, not a string that needs parsing
            return response.json()
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

def evaluate_answer(question: str, selected_answer: str) -> Dict[str, Any]:
    """Evaluate an answer using the API."""
    try:
        response = requests.post(
            f"{API_URL}/evaluate",
            json={"question": question, "selected_answer": selected_answer}
        )
        if response.status_code == 200:
            data = response.json()
            # Extract the evaluation from the response
            # Format: {"evaluation": {"correctness": "Correct", "explanation": "..."}}
            if "evaluation" in data:
                eval_data = data["evaluation"]
                is_correct = eval_data.get("correctness", "").lower() == "correct"
                return {
                    "correct": is_correct,
                    "feedback": eval_data.get("explanation", "")
                }
            return {"correct": False, "feedback": "Unexpected response format"}
        else:
            return {"correct": False, "feedback": f"Error: {response.status_code} - {response.text}"}
    except Exception as e:
        return {"correct": False, "feedback": f"Error: {str(e)}"}

# Session state initialization
if 'quiz_data' not in st.session_state:
    st.session_state.quiz_data = None
if 'current_question' not in st.session_state:
    st.session_state.current_question = 0
if 'user_answers' not in st.session_state:
    st.session_state.user_answers = {}
if 'quiz_results' not in st.session_state:
    st.session_state.quiz_results = None

def reset_quiz():
    """Reset quiz state."""
    st.session_state.quiz_data = None
    st.session_state.current_question = 0
    st.session_state.user_answers = {}
    st.session_state.quiz_results = None

# Pages
if page == "Ask a Question":
    st.markdown("<h1 class='main-header'>Ask a Question</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Get answers from the AI assistant</p>", unsafe_allow_html=True)
    
    with st.form("question_form"):
        question = st.text_area("Enter your question:", height=100)
        enable_web_search = st.checkbox("Enable web search")
        submitted = st.form_submit_button("Submit")
        
        if submitted and question:
            with st.spinner("Getting answer..."):
                answer = ask_question(question, enable_web_search)
                st.markdown("### Answer")
                st.write(answer)

elif page == "Generate Quiz":
    st.markdown("<h1 class='main-header'>Interactive Quiz</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Test your knowledge with AI-generated quizzes</p>", unsafe_allow_html=True)
    
    # Quiz generation form
    if st.session_state.quiz_data is None:
        with st.form("quiz_form"):
            topic = st.text_input("Enter a topic:")
            num_questions = st.slider("Number of questions:", min_value=1, max_value=10, value=5)
            submitted = st.form_submit_button("Generate Quiz")
            
            if submitted and topic:
                with st.spinner("Generating quiz..."):
                    quiz_data = generate_quiz(topic, num_questions)
                    if quiz_data:
                        st.session_state.quiz_data = quiz_data
                        st.rerun()
    
    # Quiz taking interface
    elif st.session_state.quiz_results is None:
        quiz_data = st.session_state.quiz_data
        # Updated to match the actual API response format
        questions = quiz_data.get("quiz", [])
        
        if questions and st.session_state.current_question < len(questions):
            current_q = questions[st.session_state.current_question]
            question_text = current_q.get("question", "")
            options = current_q.get("options", [])
            
            st.markdown(f"### Question {st.session_state.current_question + 1}/{len(questions)}")
            st.markdown(f"**{question_text}**")
            
            # Display answer options as radio buttons
            selected_answer = st.radio("Select your answer:", options, key=f"q{st.session_state.current_question}")
            
            col1, col2 = st.columns([1, 5])
            
            with col1:
                if st.button("Submit Answer"):
                    st.session_state.user_answers[question_text] = selected_answer
                    st.session_state.current_question += 1
                    # If this was the last question, evaluate the quiz
                    if st.session_state.current_question >= len(questions):
                        st.session_state.quiz_results = {}
                        for q, a in st.session_state.user_answers.items():
                            result = evaluate_answer(q, a)
                            st.session_state.quiz_results[q] = result
                    st.rerun()
            
            with col2:
                if st.button("Reset Quiz"):
                    reset_quiz()
                    st.rerun()
    
    # Quiz results
    else:
        st.markdown("### Quiz Results")
        
        # Calculate score
        correct_count = sum(1 for result in st.session_state.quiz_results.values() 
                           if result.get("correct", False))
        total_questions = len(st.session_state.quiz_results)
        score_percentage = (correct_count / total_questions) * 100 if total_questions > 0 else 0
        
        # Display score
        st.markdown(f"**Score: {correct_count}/{total_questions} ({score_percentage:.1f}%)**")
        
        # Display detailed results
        results_data = []
        for question, result in st.session_state.quiz_results.items():
            results_data.append({
                "Question": question,
                "Your Answer": st.session_state.user_answers.get(question, ""),
                "Correct": "✅" if result.get("correct", False) else "❌",
                "Feedback": result.get("feedback", "")
            })
        
        results_df = pd.DataFrame(results_data)
        st.table(results_df[["Question", "Your Answer", "Correct", "Feedback"]])
        
        if st.button("Take Another Quiz"):
            reset_quiz()
            st.rerun()

elif page == "About":
    st.markdown("<h1 class='main-header'>About This App</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    ## AI Quiz Application
    
    This application connects to a FastAPI backend that provides:
    
    - **AI-powered Q&A**: Ask any question and get intelligent answers
    - **Quiz Generation**: Create custom quizzes on any topic
    - **Answer Evaluation**: Get instant feedback on your quiz answers
    
    ### How to Use
    
    1. **Ask a Question**: Navigate to the "Ask a Question" page, type your query, and get an AI-generated response.
    2. **Generate Quiz**: Go to the "Generate Quiz" page, enter a topic and number of questions, then test your knowledge.
    
    ### Technology Stack
    
    - Frontend: Streamlit
    - Backend: FastAPI with AI-powered services
    
    """)
