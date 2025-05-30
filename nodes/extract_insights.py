from typing import Dict, Any, List
from langchain_groq import ChatGroq
from pydantic import BaseModel
from langchain.schema import HumanMessage
import os
import json



class ResumeInsights(BaseModel):
    """
    Pydantic model to represent a list of high-level career insights extracted from a resume.

    Attributes:
        insights (List[str]): A list of bullet-pointed insights such as experience,
                              skills, roles, and educational highlights.
    """
    insights: List[str]


# The 'gemma2-9b-it' model is optimized for instruction-following tasks.
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),# Securely load the API key from environment variables
    model="gemma2-9b-it",# Use the Gemma 2 9B instruction-tuned model
    temperature=0 # Controls randomness; 0 allows some flexibility
)


def extract_insights(summary_or_data: str) -> Dict[str, List[str]]:
    """
    Extracts meaningful career-related insights from a resume summary or structured resume data.

    This function uses a language model to process raw or summarized resume content and extract
    bullet-point style insights. These insights can include:
        - Total years of experience
        - Technical skills or achievements
        - Leadership roles
        - Education level and relevance

    Args:
        summary_or_data (str): A text summary or structured resume content.

    Returns:
        Dict[str, List[str]]: A dictionary containing a list of insights under the key "insights",
                              or an error message if extraction or validation fails.

    Workflow:
        1. Build a natural language prompt that instructs the model to extract specific insights.
        2. Send the prompt to the language model.
        3. Parse the response string into a Python dictionary using `json.loads()`.
        4. Validate the parsed data against the ResumeInsights Pydantic model.
        5. Return the structured dictionary or an error message if any step fails.
    """

    # Prompt to instruct the LLM to return a JSON list of bullet-point insights
    prompt = f"""
    From the resume below, extract a JSON list of insights like:
    - Total years of experience
    - Key technical skills or achievements
    - Leadership roles
    - Education level and relevance
    Do not make these points as keys of the dictionary.
    Ensure to add maximum numbers of insights.
    Return the result in this format:
    {{
    "insights": [
        "...",
        "..."    
    ]
    }}

    Resume:
    \"\"\" 
    {summary_or_data}
    \"\"\"

    Only return valid JSON. No explanations or formatting. No markdown or triple backticks.
    """

    try:
        # Send the prompt to the LLM and get the response
        response = llm.invoke([HumanMessage(content=prompt)])

        # Convert the raw JSON response string into a Python dictionary
        parsed = json.loads(response.content.strip())

        # Validate and enforce structure using Pydantic
        validated = ResumeInsights(**parsed)

        # Return the final validated dictionary
        return validated.dict()

    except Exception as e:
        # Handle errors (e.g., bad JSON, missing keys, etc.)
        return {"error": f"Error extracting insights: {str(e)}"}
