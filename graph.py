from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver

# Custom node functions for each processing step
from nodes.extract_work import extract_work_experience
from nodes.extract_education import extract_education
from nodes.generate_summary import generate_summary
from nodes.extract_insights import extract_insights
from nodes.generate_questions import generate_interview_questions

# Define the shared state type for LangGraph
State = dict

# Memory-based checkpointing for resuming workflows (in-memory only, not persistent across sessions)
checkpointer = InMemorySaver()


def extract_work_and_education(state: State) -> State:
    """
    Node Function: Extracts work experience and education information from the resume text.

    Parameters:
        state (dict): The shared state containing the raw resume text under 'resume_text'.

    Returns:
        dict: Updates state with:
            - 'work': Extracted work experience data
            - 'education': Extracted education data
    """
    resume_text = state.get("resume_text")
    if not resume_text:
        raise ValueError("Missing 'resume_text' in state for extract_work_and_education")

    return {
        "work": extract_work_experience(resume_text),
        "education": extract_education(resume_text)
    }


def summary_node(state: State) -> State:
    """
    Node Function: Generates a summary based on structured work and education data.

    Parameters:
        state (dict): State containing 'work' and 'education' dicts.

    Returns:
        dict: Updates state with:
            - 'summary': A generated summary string describing work and education background
    """
    structured = {
        "work_experiences": state.get("work", {}).get("work_experiences", []),
        "education": state.get("education", {}).get("education", [])
    }
    return {"summary": generate_summary(structured)}


def insight_node(state: State) -> State:
    """
    Node Function: Extracts insights from the generated summary.

    Parameters:
        state (dict): State containing the 'summary' string.

    Returns:
        dict: Updates state with:
            - 'insights': Key points or takeaways extracted from the summary
    """
    return extract_insights(state["summary"])


def question_node(state: State) -> State:
    """
    Node Function: Generates interview questions from the extracted insights.

    Parameters:
        state (dict): State containing 'insights'.

    Returns:
        dict: Updates state with:
            - 'questions': A list of tailored interview questions
    """
    return generate_interview_questions(state["insights"])


def build_graph():
    """
    Constructs and compiles the LangGraph DAG for resume analysis.

    Nodes:
        - extract_resume_data: Extract work and education from resume text
        - generate_summary: Generate a summary from extracted data
        - extract_insights: Extract insights from the summary
        - generate_questions: Generate interview questions from insights

    Edges:
        extract_resume_data -> generate_summary -> extract_insights -> generate_questions -> END

    Returns:
        Runnable DAG app with checkpointing enabled.
    """
    # Initialize the DAG builder with state type
    builder = StateGraph(State)

    # Add processing nodes to the graph
    builder.add_node("extract_resume_data", extract_work_and_education)
    builder.add_node("generate_summary", summary_node)
    builder.add_node("extract_insights", insight_node)
    builder.add_node("generate_questions", question_node)

    # Define the flow of the graph
    builder.set_entry_point("extract_resume_data")
    builder.add_edge("extract_resume_data", "generate_summary")
    builder.add_edge("generate_summary", "extract_insights")
    builder.add_edge("extract_insights", "generate_questions")
    builder.add_edge("generate_questions", END)

    # Compile and return the graph with in-memory checkpointing
    return builder.compile(checkpointer=checkpointer)
