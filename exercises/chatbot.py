
import os

from dotenv import load_dotenv
from google import genai


MODEL_ID = "gemini-3.8-flash"

SYSTEM_INSTRUCTION = """
You are an AI Project Assistant helping engineering students
develop their final-year project ideas.

Your responsibilities:
- Explain technical concepts in simple language.
- Suggest practical project approaches.
- Recommend suitable technologies.
- Help students improve their project ideas.
- Remember the context of the ongoing conversation.

Keep responses clear, useful and beginner-friendly.
"""


def main():
    # Load API key
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise SystemExit(
            "GEMINI_API_KEY not found. "
            "Add it to your .env file."
        )

    # Store the ID of the most recent interaction
    previous_interaction_id = None

    print("=" * 55)
    print("             AI PROJECT CHATBOT")
    print("=" * 55)
    print("Ask questions about your engineering project.")
    print("Type 'exit' to end the conversation.")
    print("=" * 55)

    try:
        with genai.Client(api_key=api_key) as client:

            while True:

                user_message = input("\nYou: ").strip()

                if not user_message:
                    continue

                if user_message.lower() in {"exit", "quit"}:
                    print("\nChatbot: Goodbye! Good luck with your project.")
                    break

                try:
                    # Prepare the request
                    request = {
                        "model": MODEL_ID,
                        "input": user_message,
                        "system_instruction": SYSTEM_INSTRUCTION,
                    }

                    # Continue the previous conversation
                    if previous_interaction_id is not None:
                        request["previous_interaction_id"] = (
                            previous_interaction_id
                        )

                    print("\nAssistant: ", end="", flush=True)

                    # Send message to Gemini
                    response = client.interactions.create(
                        **request
                    )

                    # Display the response
                    print(
                        response.output_text
                        or "No text response returned."
                    )

                    # Save interaction ID for the next message
                    previous_interaction_id = response.id

                except Exception as error:
                    print(f"\nUnable to process message: {error}")

    except Exception as error:
        print(f"\nUnable to initialize Gemini client: {error}")


if __name__ == "__main__":
    main()
