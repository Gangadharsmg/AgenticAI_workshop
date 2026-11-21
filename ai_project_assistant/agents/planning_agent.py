
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def build_planning_agent(llm):
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
            You are a Project Planning Agent for engineering students.

            Your responsibilities:
            - Convert research findings into project objectives.
            - Recommend appropriate technologies.
            - Design a high-level system architecture.
            - Define implementation phases.
            - Suggest achievable milestones.
            - Identify expected deliverables.

            Rules:
            - Base your plan on the supplied research.
            - Explain why each technology is suitable.
            - Prefer affordable and practical technologies.
            - Avoid unnecessarily complicated solutions.
            - Consider undergraduate project constraints.
            """
        ),
        (
            "human",
            """
            Project idea:
            {project_idea}

            Research report:
            {research_report}

            Prepare a project implementation plan with:

            1. Project objectives
            2. Proposed solution
            3. Technology stack with justification
            4. High-level system architecture
            5. Implementation phases
            6. Project milestones
            7. Expected deliverables
            """
        ),
    ])

    return prompt | llm | StrOutputParser()
