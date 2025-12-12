import os

try:
    from openai import OpenAI
    client = OpenAI()
except:
    client = None


def extract_tasks(notes_text):
    """
    Extracts tasks from project notes.
    Returns (tasks_markdown, mode)
    """

    OPENAI_KEY = os.getenv("OPENAI_API_KEY")

    if OPENAI_KEY and client:
        try:
            print("🔹 Using OpenAI API for task extraction.")
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Extract action items and tasks from notes. Return them as a bulleted list."},
                    {"role": "user", "content": f"Extract tasks from the following notes:\n\n{notes_text}"}
                ]
            )
            tasks = response.choices[0].message["content"]
            return tasks, "OpenAI"

        except Exception as e:
            print(f"⚠️ API failed, using placeholder. Error: {e}")

    print("⚠️ Using placeholder tasks (no API key or API failed).")
    placeholder = (
        "- Sample Task 1 from notes\n"
        "- Sample Task 2 from notes\n"
        "Original Notes:\n"
        f"{notes_text}"
    )
    return placeholder, "Placeholder"
