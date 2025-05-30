from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional
import uuid
import asyncio

# Import the function to generate interview questions
from nodes.generate_questions import generate_interview_questions

# Import the LangGraph-based DAG builder
from graph import build_graph


# Initialize FastAPI app
app = FastAPI(
    title="Resume Summary API"
)

# Compile and load the LangGraph graph
graph_app = build_graph()


class ResumeRequest(BaseModel):
    """
    Request model for analyzing a resume.
    """
    resume_text: str


class ResumeAnalysisResponse(BaseModel):
    """
    Response model for the /analyze-resume endpoint.
    Returns:
        - thread_id: A UUID used to track the workflow session
        - summary: Summary generated from resume (optional)
        - question: First generated interview question (optional)
    """
    thread_id: str
    summary: Optional[str]
    question: Optional[str]


class ResumeCheckpointRequest(BaseModel):
    """
    Request model for generating interview questions from a checkpointed summary.
    """
    thread_id: str
    resume_summary: Optional[str] = None


@app.post("/analyze-resume", response_model=ResumeAnalysisResponse, tags=["Resume analysis"])
async def analyze_resume(request: ResumeRequest):
    """
    POST /analyze-resume

    Description:
        Accepts raw resume text, runs it through the LangGraph workflow, and returns:
            - A generated summary
            - A sample interview question
            - A unique thread_id for checkpointing or resuming

    Args:
        request (ResumeRequest): Incoming resume text from the client.

    Returns:
        ResumeAnalysisResponse: Includes summary and one interview question.
    """
    # Generate a unique ID for tracking workflow execution
    thread_id = str(uuid.uuid4())

    # Initial workflow state containing the resume text
    state = {"resume_text": request.resume_text}

    # Stream workflow execution step-by-step
    stream = graph_app.stream(
        state,
        stream_mode="values",  # Only returns updated values from each node
        config={"thread_id": thread_id}
    )

    # Initialize response variables
    summary = None
    question = None

    # Iterate through streamed steps and extract summary and question
    for step in stream:
        if "summary" in step:
            summary = step["summary"]
        if "questions" in step:
            question = step["questions"][0]  # Only return the first question

    return ResumeAnalysisResponse(
        thread_id=thread_id,
        summary=summary,
        question=question
    )


@app.post("/resume-question", tags=["Generate question"])
async def resume_question(req: ResumeCheckpointRequest):
    """
    POST /resume-question

    Description:
        Resumes question generation based on an already generated resume summary.

    Args:
        req (ResumeCheckpointRequest): Includes `thread_id` and optionally the `resume_summary`.

    Returns:
        JSONResponse: Contains generated interview questions or error message.
    """
    # Prepare initial state from existing summary
    state = {}
    if req.resume_summary:
        state["summary"] = req.resume_summary

    # Attempt to generate interview questions directly from the summary
    try:
        result = generate_interview_questions({"summary": req.resume_summary})
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
