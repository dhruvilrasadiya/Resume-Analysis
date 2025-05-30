from typing import Dict, Any
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage
import os

# Instantiate a ChatGroq model with the following parameters:
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="gemma2-9b-it",
    temperature=0.3
)


def generate_summary(structured_data: Dict[str, Any]) -> str:
    """
    Generates a professional summary paragraph from structured resume data.

    This function uses a language model to convert structured resume data into
    a clean, concise, and professional summary that highlights:
      - The candidate's experience,
      - Education,
      - Key skills and achievements.

    Args:
        structured_data (Dict[str, Any]): Dictionary containing parsed resume components
                                          like education, work experience, and skills.

    Returns:
        str: A single string containing the generated summary, or an error message if the
             generation fails.

    Example structured_data input:
    {
        "education": [...],
        "work_experiences": [...],
        "skills": [...]
    }
    """

    # Prompt asks the LLM to generate only summary text (not JSON or additional formatting)
    prompt = f"""
    Generate a professional, concise summary of this candidate's work experience and education:

    Structured Resume Data:
    {structured_data}

    Return only the summary text.
    """

    try:
        # Send prompt to the LLM and receive a response
        response = llm.invoke([HumanMessage(content=prompt)])

        # Return clean, stripped summary text
        return response.content.strip()
    except Exception as e:
        # In case of failure (LLM issues, parsing errors, etc.), return a readable error string
        return f"Error generating summary: {str(e)}"
