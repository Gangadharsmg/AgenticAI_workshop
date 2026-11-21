
from dotenv import load_dotenv

from orchestrator import build_project_graph

load_dotenv()


def main():

    graph = build_project_graph()

    print("\n===== AI PROJECT ASSISTANT =====")

    project_idea = input("\nEnter your project idea: ")

    user_request = input(
        "\nWhat would you like the assistant to do? "
    )

    result = graph.invoke({
        "project_idea": project_idea,
        "user_request": user_request,
        "existing_proposal": "",
    })

    print("\n===== RESULT =====")

    if result.get("research_report"):
        print("\n--- RESEARCH REPORT ---")
        print(result["research_report"])

    if result.get("implementation_plan"):
        print("\n--- IMPLEMENTATION PLAN ---")
        print(result["implementation_plan"])

    if result.get("review_report"):
        print("\n--- REVIEW REPORT ---")
        print(result["review_report"])

    if result.get("final_proposal"):
        print("\n--- FINAL PROPOSAL ---")
        print(result["final_proposal"])


if __name__ == "__main__":
    main()
