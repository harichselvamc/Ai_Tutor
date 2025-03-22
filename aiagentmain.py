from dataclasses import dataclass
from typing import List, Dict, Any, Union
from pydantic_ai import agent
from pydantic_ai.models.groq import GroqModel
from qdrant_client import QdrantClient
from qdrant_client.models import ScoredPoint
from sentence_transformers import SentenceTransformer
from exa_py import Exa
import os
from dotenv import load_dotenv
from pydantic import BaseModel

# Load API key from environment
load_dotenv()
EXA_API_KEY=""
exa = Exa(api_key=EXA_API_KEY)

# Define models
model = GroqModel('llama-3.1-8b-instant', api_key='')
vector_model = SentenceTransformer('all-mpnet-base-v2')

# Qdrant Configuration
COLLECTION_NAME = "chapter7_sections"
client = QdrantClient(host="localhost", port=6333)

# Define Structured Output Models
class TeachingOutput(BaseModel):
    heading: str
    explanation: str
    example: str

class TutorOutput(BaseModel):
    explanation: str
    example: str

class QuizOutput(BaseModel):
    question: str
    options: List[str]

class EvaluationOutput(BaseModel):
    correctness: str
    explanation: str

@dataclass
class TutoringSystem:
    intent_agent: agent.Agent[None, str]
    teaching_plan_agent: agent.Agent[None, TeachingOutput]
    tutoring_agent: agent.Agent[None, TutorOutput]
    quiz_generator: agent.Agent[None, List[QuizOutput]]
    quiz_evaluator: agent.Agent[None, EvaluationOutput]

# Agents setup
intent_agent = agent.Agent(
    model=model,
    system_prompt="""
        Detect intent: Definition, Explanation, Example, Application, Quiz Request, Quiz Evaluation.
        Return only classification label.
    """,
    retries=2
)

teaching_plan_agent = agent.Agent(
    model=model,
    result_type=TeachingOutput,
    system_prompt="""
        Generate structured teaching plans for Chapter 7 topics based on provided context.
        Return response as a structured JSON with keys: heading, explanation, and example.
    """,
    retries=2
)

tutoring_agent = agent.Agent(
    model=model,
    result_type=TutorOutput,
    system_prompt="""
        Provide step-by-step explanations from Chapter 7 based on provided context with citations.
        Return response as a structured JSON with keys: explanation and example.
    """,
    retries=2
)

quiz_generator = agent.Agent(
    model=model,
    result_type=List[QuizOutput],  # 
    system_prompt="""
        Generate {num_questions} multiple-choice quiz questions for Chapter 7 based on the provided topic.
        Return a structured JSON list where each item has:
        - "question": The multiple-choice question.
        - "options": A list of 4 possible answers (1 correct, 3 incorrect).
        Ensure the questions are varied and relevant to the topic.
    """,
    retries=2
)


quiz_evaluator = agent.Agent(
    model=model,
    result_type=EvaluationOutput,
    system_prompt="""
        Evaluate the provided quiz answer based on the given question.
        Return a structured JSON with:
        - "correctness": "Correct" or "Incorrect"
        - "explanation": A short explanation of why the answer is correct or incorrect.
        Ensure fairness and accuracy in evaluation.
    """,
    retries=2
)

# Tutoring system instance
tutoring_system = TutoringSystem(
    intent_agent=intent_agent,
    teaching_plan_agent=teaching_plan_agent,
    tutoring_agent=tutoring_agent,
    quiz_generator=quiz_generator,
    quiz_evaluator=quiz_evaluator
)

# Corrected Qdrant search function
def search_qdrant(query: str, k: int = 3) -> List[str]:
    query_vector = vector_model.encode(query).tolist()

    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,  # 
        limit=k,
        with_payload=["text"]  
    )

    return [result.payload["text"] for result in results if "text" in result.payload]

# Improved web search function without domain restrictions
def web_search(query: str) -> str:
    results = exa.search_and_contents(query, highlights=True)
    if results and hasattr(results, "results") and results.results:
        best_result = results.results[0]
        highlights = best_result.highlights if hasattr(best_result, 'highlights') else ''
        return f"{best_result.title}: {highlights} [Source: {best_result.url}]"
    return "No relevant web source found."

# Simplified Tutoring Workflow
async def tutoring_workflow(user_question: str, enable_web_search: bool = False) -> Dict[str, Union[str, Dict]]:
    context_results = search_qdrant(user_question)
    context = "\n".join(context_results)
    print("Qdrant Query Results:", context_results)

    intent_result = await tutoring_system.intent_agent.run(user_prompt=f"Context: {context}\nQuestion: {user_question}")
    intent = intent_result.data.strip()
    print("Intent Results:", intent_result)

    structured_response = {"question": user_question, "intent": intent, "response": None}

    if intent == "Quiz Request":
        quiz_result = await tutoring_system.quiz_generator.run(user_prompt=f"Context: {context}\nQuestion: {user_question}")
        print("Quiz Request",quiz_result)
        structured_response["response"] = [quiz.model_dump() for quiz in quiz_result.data]  #

    elif intent == "Quiz Evaluation":
        evaluation_result = await tutoring_system.quiz_evaluator.run(user_prompt=f"Context: {context}\nQuestion: {user_question}")
        structured_response["response"] = evaluation_result.data.model_dump()  #

    elif intent in {"Definition", "Explanation", "Example", "Application"}:
        teaching_plan = await tutoring_system.teaching_plan_agent.run(user_prompt=f"Context: {context}\nQuestion: {user_question}")
        tutoring_result = await tutoring_system.tutoring_agent.run(user_prompt=f"Context: {context}\nQuestion: {user_question}")

        structured_response["response"] = {
            "teaching_plan": teaching_plan.data.model_dump(),  
            "explanation": tutoring_result.data.model_dump()  #
        }

    else:
        tutoring_result = await tutoring_system.tutoring_agent.run(user_prompt=f"Context: {context}\nQuestion: {user_question}")
        structured_response["response"] = tutoring_result.data.model_dump()  #

    if enable_web_search:
        web_info = web_search(user_question)
        structured_response["additional_context"] = web_info

    return structured_response

# Test Inputs
import asyncio
import nest_asyncio

nest_asyncio.apply()  # Apply workaround for running event loop

async def main():
    test_questions = [
        ("What is entropy?", False),
        ("Generate a quiz on section 7.2.", False),
    ]

    for question, enable_web in test_questions:
        result = await tutoring_workflow(question, enable_web_search=enable_web)
        print(f"\n**Structured Response:**\n{result}\n")

if __name__ == "__main__":  
    import asyncio
    asyncio.run(main())
