"""The order of the report = the order of the M5 presentation: the features, each with its evaluation."""

SECTIONS = {
    "A": "A · Creating materials",
    "B": "B · Sending reports",
    "C": "C · The teacher's platform",
    "Z": "Other modules",
}

# (suite NAME, section, feature title as on the slides)
FLOW = [
    ("lesson_studio", "A", "Lesson studio (+ Presentations)"),
    ("video_discovery", "A", "Video discovery"),
    ("story_video", "A", "Animated video"),
    ("parent_batch", "B", "Parent reports · sending in batches"),
    ("parent_report", "B", "Parent reports · content"),
    ("class_report", "C", "Class report"),
    ("inbox", "C", "Teacher inbox"),
    ("learning_platform", "C", "My library & learning platform"),
    ("montage", "C", "Smart montage"),
    ("voice_visual", "C", "Smart montage › Voice to Visual (future)"),
    ("m3_handwriting", "Z", "M3 handwriting report"),
]

_POS = {name: i for i, (name, _, _) in enumerate(FLOW)}


def place(name: str) -> tuple[str, str]:
    """(section label, feature title) of a suite; unknown suites go to 'Other modules'."""
    for n, sec, title in FLOW:
        if n == name:
            return SECTIONS[sec], title
    return SECTIONS["Z"], name


def order(name: str) -> int:
    return _POS.get(name, len(FLOW))
