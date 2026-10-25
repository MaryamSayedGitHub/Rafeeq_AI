"""PowerPoint -> PDF, done by the server, so a teacher can upload the .pptx as it is.

Rafeeq indexes PDFs (a slide becomes a page, so a citation says "slide 7"). This module turns a .pptx/.ppt/.odp
into a PDF with LibreOffice running headless. Nothing else is needed on the teacher's side.

Setup:
    Windows   winget install TheDocumentFoundation.LibreOffice      (or the installer from libreoffice.org)
    Linux     apt install libreoffice-impress fonts-noto-core fonts-noto-ui-core
    The program is found on PATH or in its usual folder. Set SOFFICE_PATH to point at soffice(.exe) otherwise.
    SOFFICE_TIMEOUT (seconds, default 240) limits one conversion.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

log = logging.getLogger(__name__)

CONVERTIBLE = (".pptx", ".ppt", ".pptm", ".potx", ".odp")

_COMMON = (
    r"C:\Program Files\LibreOffice\program\soffice.exe",
    r"C:\Program Files (x86)\LibreOffice\program\soffice.exe",
    "/usr/bin/soffice", "/usr/local/bin/soffice", "/opt/libreoffice/program/soffice",
    "/Applications/LibreOffice.app/Contents/MacOS/soffice",
)


class ConversionError(Exception):
    """The reason is safe to show the teacher."""


def is_convertible(path: str | Path) -> bool:
    return Path(path).suffix.lower() in CONVERTIBLE


def find_soffice() -> str | None:
    configured = os.environ.get("SOFFICE_PATH", "").strip().strip('"')
    if configured and Path(configured).exists():
        return configured
    for name in ("soffice", "libreoffice", "soffice.exe"):
        found = shutil.which(name)
        if found:
            return found
    return next((p for p in _COMMON if Path(p).exists()), None)


def to_pdf(src: str | Path, out_dir: str | Path | None = None, timeout: int | None = None) -> Path:
    """Convert one presentation. Returns the PDF path (<out_dir>/<name>.pdf). Raises ConversionError."""
    src = Path(src)
    if not src.is_file():
        raise ConversionError("The file is missing")
    soffice = find_soffice()
    if soffice is None:
        raise ConversionError("LibreOffice is not installed on the server, so PowerPoint files cannot be converted. "
                              "Install it (winget install TheDocumentFoundation.LibreOffice) or upload a PDF.")
    out = Path(out_dir) if out_dir else src.parent
    out.mkdir(parents=True, exist_ok=True)
    limit = timeout or int(os.environ.get("SOFFICE_TIMEOUT", "240") or 240)
    # one private profile per run: parallel conversions and a LibreOffice the teacher has open do not collide
    with tempfile.TemporaryDirectory(prefix="rafeeq-lo-") as profile, tempfile.TemporaryDirectory(prefix="rafeeq-pdf-") as work:
        cmd = [soffice, "--headless", "--norestore", "--nolockcheck", "--nodefault",
               f"-env:UserInstallation={Path(profile).resolve().as_uri()}",
               "--convert-to", "pdf:impress_pdf_Export", "--outdir", work, str(src)]
        try:
            run = subprocess.run(cmd, capture_output=True, timeout=limit, text=True, errors="replace")
        except subprocess.TimeoutExpired:
            raise ConversionError(f"Converting the presentation took longer than {limit} seconds")
        except OSError as e:
            raise ConversionError(f"LibreOffice could not start ({type(e).__name__})")
        produced = Path(work) / (src.stem + ".pdf")
        if run.returncode != 0 or not produced.is_file() or produced.stat().st_size == 0:
            log.warning("soffice failed (%s): %s", run.returncode, (run.stderr or run.stdout or "")[:300])
            raise ConversionError("The presentation could not be converted to PDF (is the file damaged or password-protected?)")
        target = out / (src.stem + ".pdf")
        shutil.move(str(produced), target)
    return target
