from typing import List, Dict, Any
from langchain_groq import ChatGroq
from pydantic import BaseModel
from langchain.schema import HumanMessage
import os
import json


class InterviewQuestions(BaseModel):
    """
    Pydantic model that defines the schema for interview questions output.

    Attributes:
        questions (List[str]): A list of interview questions generated based on candidate insights.
    """
    questions: List[str]


# Initialize the Groq-hosted large language model (LLM) via LangChain
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="gemma2-9b-it",
    temperature=0.3
)


def generate_interview_questions(insights: List[str]) -> Dict[str, List[str]]:
    """
    Generates a list of personalized interview questions based on candidate insights.

    This function:
    - Accepts a list of candidate-specific insights (skills, roles, achievements, etc.).
    - Sends a prompt to an LLM to generate five tailored interview questions.
    - Parses and validates the LLM response using a Pydantic schema.
    - Returns the validated questions or an error message.

    Args:
        insights (List[str]): A list of bullet-pointed insights about the candidate.

    Returns:
        Dict[str, List[str]]: A dictionary with key `"questions"` containing the list of questions,
                              or an error message if generation fails.
    """

    # Prompt instructs the model to convert candidate insights into relevant interview questions
    prompt = f"""
    Given the candidate insights below, generate a JSON list of 5 interview questions tailored to their profile.

    Return:
    {{
    "questions": [
        "...",
        "..."
    ]
    }}

    Insights:
    {insights}

    Only return valid JSON. No explanations or formatting. No markdown or triple backticks.
    """

    try:
        # Invoke the LLM with the constructed prompt
        response = llm.invoke([HumanMessage(content=prompt)])

        # Parse the JSON response string into a Python dictionary
        parsed = json.loads(response.content.strip())

        # Validate the dictionary using the InterviewQuestions Pydantic model
        validated = InterviewQuestions(**parsed)

        # Return the validated data as a dictionary
        return validated.dict()

    except Exception as e:
        # Catch any parsing/validation/LLM errors and return a structured error message
        return {"error": f"Error generating interview questions: {str(e)}"}
