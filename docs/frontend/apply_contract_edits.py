"""Apply the M1 contract edits of the frontend branch to contracts/openapi.yaml.

Run from the repo root:   uv run python docs/frontend/apply_contract_edits.py
Each edit finds an exact anchor. If an anchor is missing (someone changed that part), the
script stops without writing anything, so nobody's edits are overwritten. Then apply that
edit by hand from docs/frontend/HANDOFF.md, section "Contract changes in this branch".
"""
import sys
from pathlib import Path

p = Path(__file__).resolve().parents[2] / "contracts" / "openapi.yaml"
t = p.read_text(encoding="utf-8")
if "/rooms/{room_id}/join-code:" in t:
    sys.exit("Already applied: contracts/openapi.yaml has /rooms/{room_id}/join-code.")
edits = []


def replace(old: str, new: str, label: str):
    global t
    if t.count(old) != 1:
        sys.exit(f"STOPPED, nothing written. {label}: anchor found {t.count(old)} times (expected 1).")
    t = t.replace(old, new)
    edits.append(label)


# 1. login 401
replace(
    """              schema: { $ref: "#/components/schemas/Login" }

  /rooms:
""",
    """              schema: { $ref: "#/components/schemas/Login" }
        "401": { description: Wrong email or password }

  /rooms:
""",
    "E1 /auth/login: document the 401",
)

# 2. register 400/409  (the register block ends right before /rooms/join)
replace(
    """          description: Sample file samples/register.json
          content:
            application/json:
              schema: { $ref: "#/components/schemas/Login" }
""",
    """          description: Sample file samples/register.json
          content:
            application/json:
              schema: { $ref: "#/components/schemas/Login" }
        "400": { description: Missing name, invalid email, short password, or unknown join code }
        "409": { description: An account with this email already exists }
""",
    "E2 /auth/register: document 400 and 409",
)

# 3. new join-code path, placed before /rooms/{room_id}/students
replace(
    """  /rooms/{room_id}/students:
""",
    """  /rooms/{room_id}/join-code:
    get:
      tags: [M1 platform]
      operationId: getJoinCode
      summary: The room's current join code. Teacher of the room only.
      parameters:
        - { $ref: "#/components/parameters/RoomId" }
      responses:
        "200":
          description: Sample file samples/join_code.json
          content:
            application/json:
              schema: { $ref: "#/components/schemas/JoinCode" }
        "403": { description: Not the teacher of this room }

  /rooms/{room_id}/students:
""",
    "E3 new path GET /rooms/{room_id}/join-code",
)

# 4. snap 422 / 503
replace(
    """          description: Sample file samples/snap_result.json
          content:
            application/json:
              schema: { $ref: "#/components/schemas/SnapResult" }
""",
    """          description: Sample file samples/snap_result.json
          content:
            application/json:
              schema: { $ref: "#/components/schemas/SnapResult" }
        "422": { description: The upload is not an image }
        "503": { description: No image reader is configured (neither M3's OCR nor a vision model) }
""",
    "E4 /snap: document 422 and 503",
)

# 5. CopilotRequest.mode
replace(
    """        file_id:
          type: string
          description: set when the student attached a photo (the Snap result id)

    CopilotReply:
""",
    """        file_id:
          type: string
          description: set when the student attached a photo (the Snap result id)
        mode:
          type: string
          enum: [auto, ask, check_work]
          default: auto
          description: >
            auto = the Copilot decides. ask / check_work = the student chose explicitly
            ("explain this photo" vs "check my solution"), so the Copilot skips classification.

    CopilotReply:
""",
    "E5 CopilotRequest: optional mode",
)

# 6. CopilotReply payload description
replace(
    """          description: >
            Depends on route. ask = a TutorAnswer, quiz = {quiz_id},
            check_work = a HandwritingCheck, plan and clarify = {}.
""",
    """          description: >
            Depends on route. ask = a TutorAnswer, quiz = {quiz_id, topic_id},
            check_work = a HandwritingCheck, plan = {topic_id}, clarify = {}.
            Extra flags the UI reads: sample (true when the result is a contract sample because
            the owning service is not connected yet), unavailable, needs_photo,
            escalated + question_id, verified_by_teacher.
""",
    "E6 CopilotReply: document payload flags",
)

# 7. RoomStudent nullable mastery/status
replace(
    """        mastery: { type: number, minimum: 0, maximum: 1 }
        status: { type: string, enum: [on_track, watch, struggling] }

    RoomStudents:
""",
    """        mastery:
          type: [number, "null"]
          minimum: 0
          maximum: 1
          description: from M4. null until M4 has data for this student
        status:
          type: [string, "null"]
          enum: [on_track, watch, struggling, null]
          description: from M4. null until M4 has data for this student

    RoomStudents:
""",
    "E7 RoomStudent: mastery and status may be null",
)

# 8. JoinCode schema after CreateRoomResult
replace(
    """    RoomFile:
      type: object
""",
    """    JoinCode:
      type: object
      additionalProperties: false
      required: [join_code]
      properties:
        join_code: { type: string, description: "e.g. K7Q-4MX" }

    RoomFile:
      type: object
""",
    "E8 new schema JoinCode",
)

p.write_text(t, encoding="utf-8")
print("Applied:", *edits, sep="\n  ")
print("Next: run uv run pytest (join_code.json and its line in tests/test_contracts.py come with the branch).")
