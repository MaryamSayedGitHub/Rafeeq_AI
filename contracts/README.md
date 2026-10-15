# contracts/

The agreement between the five parts of Rafeeq.

- `openapi.yaml`: every route, and the JSON shape each one sends and returns.
- `samples/*.json`: one real example per JSON file (18). Each one must match its schema.

Rules
1. Build against the samples. Do not wait for another member's code.
2. Adding a field: edit the schema and the sample in the same pull request.
3. Renaming or removing a field: every member who uses it must approve first.
4. Before you push: `uv run pytest tests/test_contracts.py`
