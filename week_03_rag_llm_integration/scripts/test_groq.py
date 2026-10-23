
import os
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq, APIConnectionError, APIStatusError

project_root = Path(__file__).resolve().parents[2]
load_dotenv(project_root / ".env", override=True)

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    print("FAILED: API key was not loaded.")
else:
    print("API key loaded. Testing Groq API...")

    try:
        client = Groq(api_key=api_key, timeout=20.0)

        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "user",
                    "content": "Reply with exactly: Groq connection successful!"
                }
            ],
            max_completion_tokens=50,
        )

        print("SUCCESS: Groq API responded.")
        print("Model:", response.model)
        print("Reply:", response.choices[0].message.content)

    except APIStatusError as error:
        print("API ERROR:", error.status_code)
        print(error.message)

    except APIConnectionError as error:
        print("CONNECTION ERROR: Could not reach Groq.")
        print(str(error))

    except Exception as error:
        print("ERROR:", type(error).__name__)
        print(str(error))
