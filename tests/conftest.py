"""Keeps the test run independent of the developer's .env.

Several modules call load_dotenv() when they are imported, so a .env that switches on M2's knowledge
base or holds real API keys would otherwise change what the tests see: the upload hook would try to
index test files, the Copilot would call the real tutor, and model calls would go out to the network.

load_dotenv() never overrides a variable that is already set (even to an empty string), so the
values below win. Tests that need one of these features turn it on themselves.
"""
import os

for _name, _value in {
    "RAFEEQ_KNOWLEDGE": "0",            # M2 off: no upload hook, no knowledge routes
    "RAFEEQ_KNOWLEDGE_TUTOR": "0",      # the Copilot uses the contract samples
    "RAFEEQ_KNOWLEDGE_ROOM_TOPICS": "0",
    "RAFEEQ_ALLOW_SAMPLES": "1",
    "LLM_API_KEY": "",                  # no router / vision calls to a real model
    "TAVILY_API_KEY": "",               # no web search credits spent by the tests
    "TUTOR_SAMPLE_FALLBACK": "1",       # the tutor's tests use the contract samples as material (in the app they are off)
    "GROQ_API_KEY": "",                 # nor the photo reader / tutor falling back to the team's Groq key
    "GEMINI_API_KEY": "",
    "GOOGLE_API_KEY": "",
}.items():
    os.environ[_name] = _value
