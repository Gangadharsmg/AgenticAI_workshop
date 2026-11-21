
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def build_review_agent(llm):
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
            You are a Technical Review Agent.

            Your responsibilities:
            - Evaluate project feasibility.
            - Review the clarity of project objectives.
            - Evaluate the proposed technologies.
            - Identify missing components.
            - Identify implementation risks.
            - Suggest practical improvements.

            Rules:
            - Be objective and constructive.
            - Do not automatically approve every proposal.
            - Explain why an approach may not be feasible.
            - Distinguish critical problems from optional improvements.
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

            Implementation plan:
            {implementation_plan}

            Review the project based on:

            1. Problem clarity
            2. Objective completeness
            3. Technical feasibility
            4. Technology suitability
            5. Implementation complexity
            6. Data and resource availability
            7. Technical risks
            8. Missing components

            Provide an overall assessment, strengths, gaps,
            risks, mitigation strategies and improvements.
            """
        ),
    ])

    return prompt | llm | StrOutputParser()
