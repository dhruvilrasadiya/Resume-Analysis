from typing import Dict, Any, Optional, List
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage
from pydantic import BaseModel, Field
import json
import os


class WorkExperience(BaseModel):
    """
    A Pydantic model that defines the schema for a single work experience entry.

    Attributes:
        company (str): Name of the company where the individual worked.
        role (str): Job title or position held at the company.
        start_date (Optional[str]): Employment start date in 'YYYY-MM' format.
        end_date (Optional[str]): Employment end date in 'YYYY-MM' format or 'Present'.
        description (str): Summary of responsibilities, achievements, or work performed.
    """
    company: str
    role: str
    start_date: Optional[str] = Field(default=None, description="YYYY-MM")
    end_date: Optional[str] = Field(default=None, description="YYYY-MM or Present")
    description: str


class WorkExperienceList(BaseModel):
    """
    A wrapper Pydantic model that defines a list of work experience entries.

    Attributes:
        work_experiences (List[WorkExperience]): List of structured work experience records.
    """
    work_experiences: List[WorkExperience]


# Initialize the language model using the Groq API via LangChain.
# This model will generate JSON-structured work experience information from raw resume text.
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="gemma2-9b-it",               
    temperature=0
)


def extract_work_experience(resume_text: str) -> Dict[str, Any]:
    """
    Extracts structured work experience data from unstructured resume text using an LLM.

    This function:
    - Prompts a language model to extract work experience in a specific JSON format.
    - Parses the model's output.
    - Validates the parsed JSON against the defined Pydantic schema.
    - Returns structured data or an error message if the process fails.

    Args:
        resume_text (str): Raw resume content as a string.

    Returns:
        Dict[str, Any]: A dictionary containing extracted work experience in a structured format,
                        or an error message if parsing/validation fails.
    """

    # Craft a prompt to instruct the LLM to output only JSON-formatted work experience data
    prompt = f"""
    Extract all work experience details from the following resume in this JSON format:
    {{
    "work_experiences": [
        {{
        "company": "...",
        "role": "...",
        "start_date": "YYYY-MM",
        "end_date": "YYYY-MM or Present",
        "description": "..."
        }}
    ]
    }}
    Resume:
    \"\"\" 
    {resume_text}
    \"\"\"

    Only return valid JSON. No explanations or formatting. No markdown or triple backticks.
    """

    try:
        # Send the crafted prompt to the LLM and receive a response
        response = llm.invoke([HumanMessage(content=prompt)])

        # Convert the LLM's string output to a Python dictionary
        parsed = json.loads(response.content)

        # Validate the parsed JSON against the Pydantic schema
        validated = WorkExperienceList(**parsed)

        # Return the structured data as a dictionary
        return validated.dict()

    except Exception as e:
        # If any error occurs during parsing or validation, return a structured error
        return {"error": f"Work experience extraction failed: {str(e)}"}
