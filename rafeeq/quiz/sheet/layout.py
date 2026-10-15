"""Where everything sits on the answer sheet (A4, units = PDF points).

The PDF maker, the grader and the test helpers all read these numbers, so a
change here changes all three together.
"""

PAGE_W, PAGE_H = 595.0, 842.0

# four black squares used to find and straighten the page in a photo
MARKER_SIZE = 26.0
MARKER_CENTERS = {
    "tl": (40.0, 40.0),
    "tr": (555.0, 40.0),
    "bl": (40.0, 802.0),
    "br": (555.0, 802.0),
}

BUBBLE_R = 10.0
MAX_ITEMS = 30
ROWS_PER_COL = 15
ROW_PITCH = 38.0
FIRST_ROW_Y = 170.0
COL_X = (115.0, 345.0)  # x of bubble "A" in each of the two columns
CHOICE_PITCH = 46.0


def bubble_center(index: int, choice: int) -> tuple[float, float]:
    """Center of bubble `choice` (0..3 = A..D) for question `index` (0-based)."""
    if not 0 <= index < MAX_ITEMS:
        raise ValueError(f"question index must be 0..{MAX_ITEMS - 1}")
    col, row = divmod(index, ROWS_PER_COL)
    return COL_X[col] + choice * CHOICE_PITCH, FIRST_ROW_Y + row * ROW_PITCH