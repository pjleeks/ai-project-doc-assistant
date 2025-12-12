import os

try:
    from openai import OpenAI
    client = OpenAI()
except:
    client = None


def generate_final_summary(project_name, accomplishments, lessons_learned):
    """
    Generates a final project summary.
    Returns (summary_text, mode)
    """

    OPENAI_KEY = os.getenv("OPENAI_API_KEY")

    if OPENAI_KEY and client:
        try:
            print("🔹 Using OpenAI API for final summary.")
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Generate a clear final project summary."},
                    {"role": "user", "content": (
                        f"Project: {project_name}\n"
                        f"Accomplishments:\n{accomplishments}\n\n"
                        f"Lessons Learned:\n{lessons_learned}"
                    )}
                ]
            )
            summary = response.choices[0].message["content"]
            return summary, "OpenAI"

        except Exception as e:
            print(f"⚠️ API failed, using placeholder. Error: {e}")

    print("⚠️ Using placeholder final summary (no API key or API failed).")
    placeholder = (
        f"Final summary for {project_name}.\n"
        f"Accomplishments: {accomplishments}\n"
        f"Lessons learned: {lessons_learned}\n"
        f"This is placeholder-generated content."
    )
    return placeholder, "Placeholder"
