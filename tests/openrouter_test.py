import os
import traceback

from openai import OpenAI

MODEL = "openai/gpt-5.6-luna"

client = OpenAI(
    api_key="",
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

except Exception as e:
    print("\n=== Failed ===")
    print("Type:", type(e).__name__)
    print("Error:", str(e))
    print("\n=== Traceback ===")
    traceback.print_exc()