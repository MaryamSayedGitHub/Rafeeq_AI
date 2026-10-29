"""COPY THIS FILE to evals/suites/<your_feature>.py (a file whose name starts with "_" is ignored by the runner).

Each suite returns a list of Metric. Keep it fast and repeatable: build your own tiny test data, or read the
report your existing eval script already writes (see m3_report.py). If something needs a model or a key that is
missing, return Metric(name, None, note="why") and the report says SKIPPED instead of failing.
"""
from evals.core import Metric

NAME = "my_feature"
DESCRIPTION = "one line: what this suite measures"


def run(ctx: dict) -> list[Metric]:
    # ctx["live"] is True when the runner was started with --live (allowed to call paid models / the internet)
    accuracy = 0.9        # <- compute it here
    return [Metric("accuracy", accuracy, "", gate=0.8, note="what it is and how it is measured")]
