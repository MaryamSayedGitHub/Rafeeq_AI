"""The lecturer assistant: a LangGraph pipeline that prepares a teaching session.

    brief -> orchestrator -> research / slides / code / quiz agents -> draft
          -> human approval (pause, the teacher approves or asks for changes)
          -> final files (.pptx, code file, quiz) + long-term memory about the teacher

Ported from github.com/MaryamSayedGitHub/Lecturer_Multi_Agentic_Assistant into Rafeeq's
teacher part (Member 5). It runs inside the Rafeeq app and uses Rafeeq's Postgres.
The web layer is in rafeeq/teacher/routes.py; nothing here knows about FastAPI.
"""
