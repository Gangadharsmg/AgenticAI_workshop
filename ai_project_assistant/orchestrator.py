
import json
import re
from typing import TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langgraph.graph import StateGraph, START, END

from llm import get_llm
from agents.research_agent import build_research_agent
from agents.planning_agent import build_planning_agent
from agents.review_agent import build_review_agent

from knowledge_retriever import retrieve_knowledge


# ---------------------------------------------------------
# 1. Graph State
# ---------------------------------------------------------

class ProjectState(TypedDict, total=False):
    project_idea: str
    user_request: str
    existing_proposal: str

    task_type: str

    knowledge_context: str
    research_report: str
    implementation_plan: str
    review_report: str
    final_proposal: str


# ---------------------------------------------------------
# 2. Initialize LLM and Agents
# ---------------------------------------------------------

llm = get_llm()

research_agent = build_research_agent(llm)
planning_agent = build_planning_agent(llm)
review_agent = build_review_agent(llm)


# ---------------------------------------------------------
# 3. Coordinator
# ---------------------------------------------------------

router_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are the Coordinator of an AI Project Assistant.

        Your responsibility is to understand the student's request
        and select the appropriate task.

        Available task types:

        1. research
           Use when the student wants:
           - Project idea research
           - Problem background
           - Concepts and approaches
           - Dataset suggestions
           - Research directions

        2. planning
           Use when the student wants:
           - Technology recommendations
           - Architecture
           - Implementation steps
           - Development milestones
           - Project planning

           Planning requires research first.

        3. review
           Use when the student provides an existing proposal or plan
           and wants feedback, feasibility analysis, gap identification,
           risk assessment or improvements.

        4. full
           Use when the student wants a complete project proposal
           involving research, planning and review.

        Return ONLY valid JSON in this format:

        {"task_type": "research"}

        The task_type must be one of:
        research, planning, review, full.
        """
    ),
    (
        "human",
        """
        Project idea:
        {project_idea}

        Student request:
        {user_request}

        Existing proposal, if provided:
        {existing_proposal}
        """
    )
])

router_chain = router_prompt | llm | StrOutputParser()


def classify_request(state: ProjectState) -> str:
    """Classify the request and return the required workflow."""

    response = router_chain.invoke({
        "project_idea": state.get("project_idea", ""),
        "user_request": state.get("user_request", ""),
        "existing_proposal": state.get("existing_proposal", ""),
    })

    # Extract JSON, including when the model wraps it in code fences.
    match = re.search(r"\{.*?\}", response, re.DOTALL)

    if match:
        try:
            result = json.loads(match.group())
            task_type = result.get("task_type", "").lower()

            if task_type in {"research", "planning", "review", "full"}:
                return task_type

        except (json.JSONDecodeError, AttributeError):
            pass

    # Safe fallback if the coordinator returns invalid output.
    request = (
        state.get("user_request", "") + " " +
        state.get("project_idea", "")
    ).lower()

    if any(word in request for word in [
        "review", "evaluate", "critique", "check feasibility",
        "identify gaps", "assess risks"
    ]):
        return "review"

    if any(word in request for word in [
        "complete proposal", "full proposal", "end-to-end",
        "complete project"
    ]):
        return "full"

    if any(word in request for word in [
        "plan", "architecture", "milestone",
        "implementation", "technology", "technologies"
    ]):
        return "planning"

    return "research"


# ---------------------------------------------------------
# 4. Coordinator Node
# ---------------------------------------------------------

def coordinator_node(state: ProjectState):

    print("\n[COORDINATOR] Analyzing student request...")

    task_type = classify_request(state)

    print(f"[COORDINATOR] Selected workflow: {task_type.upper()}")

    return {"task_type": task_type}


# ---------------------------------------------------------
# 5. Research Agent Node
# ---------------------------------------------------------

def research_node(state: ProjectState):

    print("\n[AGENT CALLED] Research Agent")

    project_idea = state.get("project_idea", "")
    user_request = state.get("user_request", "")

    # Retrieve relevant information from existing ChromaDB.
    print("[RAG] Retrieving relevant knowledge from ChromaDB...")

    knowledge_context = retrieve_knowledge(
        query=f"{project_idea}\n{user_request}",
        top_k=3,
    )

    print("[RAG] Knowledge retrieval completed.")

    research_report = research_agent.invoke({
        "project_idea": project_idea,
        "knowledge_context": knowledge_context,
    })

    print("[Research Agent] Research completed.")

    return {
        "knowledge_context": knowledge_context,
        "research_report": research_report,
    }


# ---------------------------------------------------------
# 6. Planning Agent Node
# ---------------------------------------------------------

def planning_node(state: ProjectState):

    print("\n[AGENT CALLED] Planning Agent")

    implementation_plan = planning_agent.invoke({
        "project_idea": state.get("project_idea", ""),
        "research_report": state.get("research_report", ""),
    })

    print("[Planning Agent] Planning completed.")

    return {
        "implementation_plan": implementation_plan,
    }


# ---------------------------------------------------------
# 7. Review Agent Node
# ---------------------------------------------------------

def review_node(state: ProjectState):

    print("\n[AGENT CALLED] Review Agent")

    # For a review-only request, review the student's supplied proposal.
    # Otherwise, review the plan generated by the Planning Agent.
    proposal_to_review = (
        state.get("existing_proposal")
        or state.get("implementation_plan")
        or state.get("research_report")
        or state.get("project_idea", "")
    )

    review_report = review_agent.invoke({
        "project_idea": state.get("project_idea", ""),
        "implementation_plan": proposal_to_review,
    })

    print("[Review Agent] Review completed.")

    return {
        "review_report": review_report,
    }


# ---------------------------------------------------------
# 8. Final Proposal Node
# ---------------------------------------------------------

def final_proposal_node(state: ProjectState):

    print("\n[COORDINATOR] Consolidating final proposal...")

    final_proposal = f"""
# AI Project Proposal

## Project Idea

{state.get("project_idea", "")}

## Research Report

{state.get("research_report", "")}

## Implementation Plan

{state.get("implementation_plan", "")}

## Review and Recommendations

{state.get("review_report", "")}
"""

    return {
        "final_proposal": final_proposal,
    }


# ---------------------------------------------------------
# 9. Conditional Routing
# ---------------------------------------------------------

def route_from_coordinator(state: ProjectState):

    task_type = state["task_type"]

    if task_type == "research":
        return "research"

    if task_type == "planning":
        return "research"

    if task_type == "review":
        return "review"

    if task_type == "full":
        return "research"

    return END


def route_after_research(state: ProjectState):

    task_type = state["task_type"]

    if task_type in {"planning", "full"}:
        return "planning"

    return END


def route_after_planning(state: ProjectState):

    task_type = state["task_type"]

    if task_type == "full":
        return "review"

    return END


def route_after_review(state: ProjectState):

    if state["task_type"] == "full":
        return "final_proposal"

    return END


# ---------------------------------------------------------
# 10. Build LangGraph Workflow
# ---------------------------------------------------------

def build_project_graph():

    workflow = StateGraph(ProjectState)

    # Register nodes
    workflow.add_node("coordinator", coordinator_node)
    workflow.add_node("research", research_node)
    workflow.add_node("planning", planning_node)
    workflow.add_node("review", review_node)
    workflow.add_node("final_proposal", final_proposal_node)

    # Entry point
    workflow.add_edge(START, "coordinator")

    # Coordinator selects the next agent
    workflow.add_conditional_edges(
        "coordinator",
        route_from_coordinator,
        {
            "research": "research",
            "review": "review",
            END: END,
        },
    )

    # Research decides whether planning is needed
    workflow.add_conditional_edges(
        "research",
        route_after_research,
        {
            "planning": "planning",
            END: END,
        },
    )

    # Planning decides whether review is needed
    workflow.add_conditional_edges(
        "planning",
        route_after_planning,
        {
            "review": "review",
            END: END,
        },
    )

    # Review decides whether final consolidation is needed
    workflow.add_conditional_edges(
        "review",
        route_after_review,
        {
            "final_proposal": "final_proposal",
            END: END,
        },
    )

    workflow.add_edge("final_proposal", END)

    return workflow.compile()
