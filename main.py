from fastapi import FastAPI
from pydantic import BaseModel
import asyncio
import motor.motor_asyncio  # ✅ MongoDB async driver
from aiagentmain import tutoring_workflow, tutoring_system  # ✅ Import AI agent functions

# Initialize FastAPI app
app = FastAPI()

# ✅ Connect to MongoDB (local instance)
MONGO_URI = "mongodb://localhost:27017/"
client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI)
db = client["chatbot_db"]  # ✅ Database name
history_collection = db["chat_history"]  # ✅ Collection name


# Define input model for answering user questions
class UserInput(BaseModel):
    question: str
    enable_web_search: bool = False

@app.post("/ask")
async def ask_question(user_input: UserInput):
    """Handles user questions, checks history, and returns AI-generated responses."""
    
    # ✅ Check if the question was previously asked
    existing_entry = await history_collection.find_one({"question": user_input.question})

    if existing_entry:
        return {"response": existing_entry["answer"], "from_cache": True}  # ✅ Return cached response

    # ✅ If not in history, process the question
    result = await tutoring_workflow(user_input.question, enable_web_search=user_input.enable_web_search)

    # ✅ Store the new response in MongoDB
    await history_collection.insert_one({"question": user_input.question, "answer": result})

    return {"response": result, "from_cache": False}


# Define input model for quiz generation request
class QuizInput(BaseModel):
    topic: str
    num_questions: int

@app.post("/quiz")
async def generate_quiz(quiz_input: QuizInput):
    """Generates a multiple-choice quiz based on the provided topic and number of questions."""
    
    quiz_prompt = f"Generate {quiz_input.num_questions} multiple-choice quiz questions on the topic: {quiz_input.topic}"
    
    quiz_result = await tutoring_system.quiz_generator.run(user_prompt=quiz_prompt)
    
    return {"quiz": [q.model_dump() for q in quiz_result.data]}  # ✅ Ensuring structured output

# Define input model for evaluating quiz answers
class EvaluationInput(BaseModel):
    question: str
    selected_answer: str

@app.post("/evaluate")
async def evaluate_answer(evaluation_input: EvaluationInput):
    """Evaluates a quiz answer and determines if it's correct."""
    
    evaluation_prompt = (
        f"Evaluate the following quiz answer:\n"
        f"Question: {evaluation_input.question}\n"
        f"User's Answer: {evaluation_input.selected_answer}\n"
        f"Provide a structured JSON response with correctness and explanation."
    )
    
    evaluation_result = await tutoring_system.quiz_evaluator.run(user_prompt=evaluation_prompt)

    return {"evaluation": evaluation_result.data.model_dump()}  # ✅ Returns structured output


@app.get("/health")
async def health_check():
    """Checks if all API endpoints are running and AI agent is loaded."""
    try:
        # ✅ Check if MongoDB is connected
        mongo_status = "Connected" if client else "Disconnected"

        # ✅ Check if AI model is loaded
        ai_model_status = "Loaded" if tutoring_system else "Not Loaded"

        return {
            "status": "OK",
            "endpoints": {
                "/ask": "Available",
                "/quiz": "Available",
                "/evaluate": "Available",
                "/health": "Available"
            },
            "ai_model": ai_model_status,
            "mongo_status": mongo_status
        }
    except Exception as e:
        return {"status": "ERROR", "message": str(e)}
