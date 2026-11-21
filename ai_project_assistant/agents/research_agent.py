
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def build_research_agent(llm):

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
            You are a Research Agent helping engineering students
            develop final-year project proposals.

            Use the supplied knowledge base context wherever relevant.

            Instructions:
            - Analyze the student's project idea.
            - Identify the problem and its background.
            - Explain relevant concepts and approaches.
            - Identify possible datasets and research directions.
            - Distinguish information supported by the knowledge
              base from your own general knowledge.
            - Do not invent references or claim that you performed
              external web research.
            - Mention when the supplied context is insufficient.
            """
        ),
        (
            "human",
            """
            Project idea:
            {project_idea}

            Retrieved knowledge base context:
            {knowledge_context}

            Prepare a structured research report.
            """
        ),
    ])

    return prompt | llm | StrOutputParser()
