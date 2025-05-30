from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage
import json
import os

class Education(BaseModel):
    """
    A Pydantic model representing a single educational qualification.
    """
    institution: str
    degree: str
    field: Optional[str] = ""
    start_date: Optional[str] = Field(default=None, description="YYYY-MM")
    end_date: Optional[str] = Field(default=None, description="YYYY-MM or Present")


class EducationList(BaseModel):
    """
    A Pydantic model for a list of educational qualifications.
    """
    education: List[Education]



# Initialize the Groq model using LangChain's ChatGroq class.
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"), # Fetch API key securely from environment variable
    model="gemma2-9b-it",# Use the fine-tuned gemma2-9b-it model from Groq
    temperature=0 # Set temperature to 0 for deterministic outputs
)



def extract_education(resume_text: str) -> Dict[str, Any]:
    """
    Extracts structured education history from unstructured resume text using a language model.

    Args:
        resume_text (str): Raw resume content as a string.

    Returns:
        Dict[str, Any]: A dictionary representation of the extracted education data,
                        or an error message if extraction or validation fails.
    
    Workflow:
    - Prompt the language model with specific instructions to extract education data in JSON format.
    - Parse the model's JSON response into a Python dictionary.
    - Validate the parsed data against the defined Pydantic models.
    - Return the structured and validated data.
    """

    # Prompt to instruct the LLM to extract education details and return only valid JSON
    prompt = f"""
    Extract all education details from the following resume in this JSON format:
    {{
    "education": [
        {{
        "institution": "...",
        "degree": "...",
        "field": "...",
        "start_date": "YYYY-MM",
        "end_date": "YYYY-MM or Present"
        }}
    ]
    }}
    Resume:
    \"\"\"
    {resume_text}
    \"\"\"
    Only return valid JSON. No explanations or formatting. No markdown or triple backticks."""

    try:
        # Send prompt to the LLM and get the raw response
        response = llm.invoke([HumanMessage(content=prompt)])

        # Parse the LLM response from string to Python dictionary
        parsed = json.loads(response.content.strip())

        # Validate the parsed dictionary against Pydantic schema
        validated = EducationList(**parsed)

        # Return the validated data as a standard dictionary
        return validated.dict()

    except Exception as e:
        # Handle any exceptions and return an error message
        return {"error": f"Education extraction failed: {str(e)}"}
