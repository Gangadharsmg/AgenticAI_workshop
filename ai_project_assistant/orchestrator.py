
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
# 3. LLM-based Coordinator
# ---------------------------------------------------------

router_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are the Coordinator of an AI Project Assistant.

        Understand the student's request and select the appropriate
        workflow.

        Available task types:

        research:
        Project background, concepts, approaches, datasets,
        research directions.

        planning:
        Technology recommendations, architecture,
        implementation steps and milestones.
        Planning requires research first.

        review:
        Review an existing proposal or plan, evaluate feasibility,
        identify gaps, assess risks and suggest improvements.

        full:
        Create a complete project proposal involving research,
        planning and review.

        Return ONLY valid JSON in this format:

        {{
            "task_type": "research"
        }}

        The task_type must be one of:
        research, planning, review, full.

        Return no Markdown or additional explanation.
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

    response = router_chain.invoke({
        "project_idea": state.get("project_idea", ""),
        "user_request": state.get("user_request", ""),
        "existing_proposal": state.get("existing_proposal", ""),
    })

    print(f"[COORDINATOR] Router response: {response}")

    match = re.search(r"\{.*?\}", response, re.DOTALL)

    if match:
        try:
            result = json.loads(match.group())
            task_type = result.get("task_type", "").lower()

            if task_type in {"research", "planning", "review", "full"}:
                return task_type

        except (json.JSONDecodeError, AttributeError):
            pass

    # Fallback if LLM returns an invalid response.
    request = state.get("user_request", "").lower()

    if any(word in request for word in [
        "review", "evaluate", "critique", "feasibility",
        "identify gaps", "assess risks"
    ]):
        return "review"

    if any(word in request for word in [
        "complete proposal", "full proposal",
        "end-to-end", "complete project"
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

    # The existing review prompt expects three variables.
    # Include the supplied proposal in the review context.
    existing_proposal = state.get("existing_proposal", "")

    implementation_plan = (
        state.get("implementation_plan", "")
        or existing_proposal
        or "No implementation plan or proposal supplied."
    )

    research_report = (
        state.get("research_report", "")
        or "No research report supplied."
    )

    review_report = review_agent.invoke({
        "project_idea": state.get("project_idea", ""),
        "research_report": research_report,
        "implementation_plan": implementation_plan,
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

    if task_type in {"research", "planning", "full"}:
        return "research"

    if task_type == "review":
        return "review"

    return END


def route_after_research(state: ProjectState):

    if state["task_type"] in {"planning", "full"}:
        return "planning"

    return END


def route_after_planning(state: ProjectState):

    if state["task_type"] == "full":
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

    workflow.add_node("coordinator", coordinator_node)
    workflow.add_node("research", research_node)
    workflow.add_node("planning", planning_node)
    workflow.add_node("review", review_node)
    workflow.add_node("final_proposal", final_proposal_node)

    workflow.add_edge(START, "coordinator")

    workflow.add_conditional_edges(
        "coordinator",
        route_from_coordinator,
        {
            "research": "research",
            "review": "review",
            END: END,
        },
    )

    workflow.add_conditional_edges(
        "research",
        route_after_research,
        {
            "planning": "planning",
            END: END,
        },
    )

    workflow.add_conditional_edges(
        "planning",
        route_after_planning,
        {
            "review": "review",
            END: END,
        },
    )

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
