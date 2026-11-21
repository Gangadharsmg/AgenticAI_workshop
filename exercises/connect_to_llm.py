
import os

from dotenv import load_dotenv
from google import genai


MODEL_ID = "gemini-3.8-flash"


def main():
    # Load the API key from .env
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise SystemExit(
            "GEMINI_API_KEY not found. "
            "Add it to your .env file."
        )

    # Initialize the Gemini client
    with genai.Client(api_key=api_key) as client:

        print("=" * 55)
        print("           AI PROJECT ASSISTANT")
        print("=" * 55)

        # Get the project idea from the student
        project_idea = input(
            "\nEnter your project idea: "
        ).strip()

        if not project_idea:
            print("Please enter a project idea.")
            return

        # Create a simple prompt
        prompt = f"""
        You are an AI project mentor for engineering students.

        A student has the following project idea:

        {project_idea}

        Explain the idea in simple language.

        Include:
        1. Project title
        2. Problem being addressed
        3. Proposed solution
        4. Three key objectives
        5. Suggested technologies
        6. Expected outcome

        Keep the response practical and beginner-friendly.
        """

        try:
            print("\nGenerating response...\n")

            # Send the prompt to Gemini
            response = client.interactions.create(
                model=MODEL_ID,
                input=prompt
            )

            # Display the LLM response
            print("=" * 55)
            print("               PROJECT BRIEF")
            print("=" * 55)
            print(response.output_text or "No text response returned.")

        except Exception as error:
            print(f"\nUnable to generate response: {error}")


if __name__ == "__main__":
    main()
