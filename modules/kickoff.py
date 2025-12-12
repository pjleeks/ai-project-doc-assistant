import os
from datetime import datetime

try:
    from openai import OpenAI
    client = OpenAI()
except:
    client = None


def generate_kickoff_summary(project_name, objectives):
    """
    Generates a kickoff summary.
    Returns (summary_text, mode)
    mode = "OpenAI" or "Placeholder"
    """

    OPENAI_KEY = os.getenv("OPENAI_API_KEY")

    if OPENAI_KEY and client:
        try:
            print("🔹 Using OpenAI API for kickoff summary.")
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You generate concise project kickoff summaries."},
                    {"role": "user", "content": f"Project name: {project_name}\nObjectives: {objectives}"}
                ]
            )
            summary = response.choices[0].message["content"]
            return summary, "OpenAI"

        except Exception as e:
            print(f"⚠️ API failed, using placeholder. Error: {e}")

    print("⚠️ Using placeholder summary (no API key or API failed).")
    placeholder = f"Kickoff summary for '{project_name}'. Objectives: {objectives}."
    return placeholder, "Placeholder"

