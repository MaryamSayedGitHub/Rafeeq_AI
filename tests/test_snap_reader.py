"""/snap reading rules: never return sample text as if it were read from the photo."""
from pathlib import Path

import pytest
import yaml
from jsonschema import Draft202012Validator

from rafeeq.copilot import clients
from rafeeq.snap.reader import NotAnImage, ReaderUnavailable, read_snap

SPEC = yaml.safe_load((Path(__file__).resolve().parent.parent / "contracts" / "openapi.yaml").read_text(encoding="utf-8"))


def png_bytes() -> bytes:
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    img = np.full((120, 240, 3), 255, np.uint8)
    cv2.putText(img, "2x+3=11", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    return cv2.imencode(".png", img)[1].tobytes()


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("RAFEEQ_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    clients.disconnect("read_image")
    yield
    clients.disconnect("read_image")


def test_not_an_image_is_rejected():
    pytest.importorskip("cv2")
    with pytest.raises(NotAnImage):
        read_snap(b"definitely not a picture")


def test_no_reader_configured_is_an_error_not_a_sample():
    with pytest.raises(ReaderUnavailable):
        read_snap(png_bytes())


def test_connected_reader_result_matches_the_contract(tmp_path):
    clients.connect("read_image", lambda data: {"type": "student_work", "language": "en",
                                                "text": "2x+3=11", "confidence": 0.9})
    result = read_snap(png_bytes())
    schema = {"$ref": "#/components/schemas/SnapResult", "components": SPEC["components"]}
    assert not list(Draft202012Validator(schema).iter_errors(result))
    assert result["text"] == "2x+3=11"
    assert (tmp_path / "snaps" / f"{result['file_id']}.jpg").exists()


def test_odd_reader_output_is_normalised():
    clients.connect("read_image", lambda data: {"type": "selfie", "language": "fr", "confidence": "high"})
    result = read_snap(png_bytes())
    assert result["type"] == "other" and result["language"] == "ar" and 0 <= result["confidence"] <= 1


def test_a_tilted_page_is_found_and_flattened():
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    from rafeeq.snap.cleaner import clean, find_page

    img = np.zeros((800, 600, 3), np.uint8)
    corners = np.array([[100, 120], [500, 90], [540, 700], [80, 680]])
    cv2.fillPoly(img, [corners], (255, 255, 255))
    quad = find_page(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    assert quad is not None and np.allclose(quad, corners, atol=8)
    out, _ = clean(cv2.imencode(".png", img)[1].tobytes())
    flat = cv2.imdecode(np.frombuffer(out, np.uint8), cv2.IMREAD_GRAYSCALE)
    assert flat.shape[0] > flat.shape[1] and flat.mean() > 200   # mostly the white page, upright


def test_the_phone_rotation_tag_is_applied():
    pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    Image = pytest.importorskip("PIL.Image")
    import io

    from rafeeq.snap.cleaner import _decode_upright

    wide = Image.fromarray(np.zeros((100, 300, 3), np.uint8))   # stored landscape...
    exif = Image.Exif()
    exif[0x0112] = 6                                            # ...with "rotate 90°" (portrait photo)
    buf = io.BytesIO()
    wide.save(buf, format="JPEG", exif=exif)
    assert _decode_upright(buf.getvalue()).shape[:2] == (300, 100)
