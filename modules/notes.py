import os
from datetime import datetime

try:
    from openai import OpenAI
    client = OpenAI()
except:
    client = None


def generate_notes_summary(notes_text, date):
    """
    Generates a daily notes summary.
    Returns (summary_text, mode)
    """

    OPENAI_KEY = os.getenv("OPENAI_API_KEY")

    if OPENAI_KEY and client:
        try:
            print("🔹 Using OpenAI API for notes summary.")
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Summarize daily technical and project notes concisely."},
                    {"role": "user", "content": f"Date: {date}\nNotes:\n{notes_text}"}
                ]
            )
            summary = response.choices[0].message["content"]
            return summary, "OpenAI"

        except Exception as e:
            print(f"⚠️ API failed, using placeholder. Error: {e}")

    print("⚠️ Using placeholder notes summary (no API key or API failed).")
    placeholder = f"Summary for {date}: This is a placeholder summary of the provided notes."
    return placeholder, "Placeholder"

