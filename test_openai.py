from openai import OpenAI
import os

# Initialize client with your API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Simple chat completion
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, can you confirm my API key works?"}
    ]
)

# Print the AI response
print(response.choices[0].message.content)
