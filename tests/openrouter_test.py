import os
import traceback

from openai import OpenAI

MODEL = "openai/gpt-5.6-luna"

def main() -> None:
    client = OpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
    )

    try:
        print(f"Testing model: {MODEL}")

        response = client.responses.create(
            model=MODEL,
            input="Reply with exactly: OpenRouter connection successful.",
            max_output_tokens=100,
        )

        print("\n=== Success ===")
        print("Response ID:", response.id)
        print("Model:", response.model)
        print("Output:", response.output_text)

        if response.usage:
            print("\n=== Usage ===")
            print(response.usage)

    except Exception as error:
        print("\n=== Failed ===")
        print("Type:", type(error).__name__)
        print("Error:", str(error))
        print("\n=== Traceback ===")
        traceback.print_exc()


if __name__ == "__main__":
    main()
