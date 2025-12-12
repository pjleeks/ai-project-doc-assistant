import os

try:
    from openai import OpenAI
    client = OpenAI()
except:
    client = None


def extract_risks(notes_text):
    """
    Extracts project risks from notes.
    Returns (risks_markdown, mode)
    """

    OPENAI_KEY = os.getenv("OPENAI_API_KEY")

    if OPENAI_KEY and client:
        try:
            print("🔹 Using OpenAI API for risk extraction.")
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Extract risks or concerns from project notes. Return as a bulleted list."},
                    {"role": "user", "content": f"Extract risks from the following notes:\n\n{notes_text}"}
                ]
            )
            risks = response.choices[0].message["content"]
            return risks, "OpenAI"

        except Exception as e:
            print(f"⚠️ API failed, using placeholder. Error: {e}")

    print("⚠️ Using placeholder risks (no API key or API failed).")
    placeholder = (
        "- Sample Risk 1 from notes\n"
        "- Sample Risk 2 from notes\n"
        "Original Notes:\n"
        f"{notes_text}"
    )
    return placeholder, "Placeholder"
