import os
from groq import Groq

# API Configuration
GROQ_API_KEY = "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"

# Initialize Client
client = Groq(api_key=GROQ_API_KEY)

# Test Request
response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {
            "role": "user",
            "content": "Hello, test Groq API connection"
        }
    ],
    temperature=0.7,
    max_tokens=200
)

# Print Response
print(response.choices[0].message.content)
