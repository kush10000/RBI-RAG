from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()          # reads .env into environment variables
client = Anthropic()   # picks up ANTHROPIC_API_KEY automatically, so the key never appears in code

response = client.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=100,
    messages=[{"role": "user", "content": "Say hi in five words."}],
)

print(response.content[0].text)
print(response.usage)   # input/output token counts, which is what you pay for