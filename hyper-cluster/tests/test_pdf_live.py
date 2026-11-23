"""The PDF path of the mobile mission control.

The phone drops a paper into the chat: the text layer is extracted (with
the standard library only), indexed and searchable.
"""

from __future__ import annotations

import zlib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from core.config import Settings
from head_node.app import create_app
from memory.pdf import MAX_PDF_BYTES, PdfError, extract_pdf_text, extract_text, is_pdf


# --------------------------------------------------------------------------- #
# The PDF builders (the test-suite writes its own PDFs)
# --------------------------------------------------------------------------- #
def build_pdf(text: str, *, compress: bool = True) -> bytes:
    """A minimal one-page PDF with a text layer."""
    escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    content = f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET".encode("latin-1")
    if compress:
        stream = zlib.compress(content)
        extra = "/Filter /FlateDecode "
    else:
        stream = content
        extra = ""
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length " + str(len(stream)).encode() + b" " + extra.encode() + b">>\nstream\n"
        + stream
        + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for index, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{index} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


def build_scanned_pdf() -> bytes:
    """A PDF without any text (a scan)."""
    return build_pdf("").replace(b"BT /F1 12 Tf 72 720 Td () Tj ET", b"0 0 100 100 re f")


# --------------------------------------------------------------------------- #
# The extraction
# --------------------------------------------------------------------------- #
def test_a_simple_pdf_is_read() -> None:
    document = extract_text(build_pdf("the quick brown fox"))
    assert "the quick brown fox" in document.text
    assert document.pages == 1
    assert document.has_text


def test_an_uncompressed_pdf_is_read() -> None:
    document = extract_text(build_pdf("plain stream", compress=False))
    assert "plain stream" in document.text


def test_the_parentheses_are_escaped() -> None:
    document = extract_text(build_pdf("a (nested) value"))
    assert "(nested)" in document.text


def test_a_broken_pdf_yields_a_warning_instead_of_an_exception() -> None:
    text, extra = extract_pdf_text(b"%PDF-1.4\nthis is not a pdf at all\n")
    assert text == ""
    assert extra.get("pdf_error") or extra.get("pdf_warnings")


def test_an_empty_payload_is_refused() -> None:
    with pytest.raises(PdfError):
        extract_text(b"")


def test_the_size_limit_is_declared() -> None:
    assert MAX_PDF_BYTES > 1024


def test_is_pdf_detects_the_magic_and_the_name() -> None:
    assert is_pdf(b"%PDF-1.4 ...")
    assert is_pdf(b"anything", "paper.pdf")
    assert not is_pdf(b"plain text", "notes.txt")


# --------------------------------------------------------------------------- #
# The upload
# --------------------------------------------------------------------------- #
@pytest.fixture
def client(settings: Settings) -> TestClient:
    app = create_app(settings)
    with TestClient(app) as test_client:
        yield test_client


def test_a_pdf_is_uploaded_and_indexed(client: TestClient, tmp_path: Path) -> None:
    payload = build_pdf("the mitochondria is the powerhouse of the cell")
    response = client.post(
        "/v1/memory/documents",
        files={"file": ("paper.pdf", payload, "application/pdf")},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["extracted"] is True
    assert body["pdf_pages"] == 1
    assert body["document"]["chunks"] >= 1
    assert "mitochondria" in body["document"].get("preview", "") or body["document"]["chunks"] >= 1


def test_the_uploaded_pdf_is_searchable(client: TestClient) -> None:
    payload = build_pdf("the mitochondria is the powerhouse of the cell")
    client.post("/v1/memory/documents", files={"file": ("paper.pdf", payload, "application/pdf")})
    hits = client.post("/v1/memory/search", json={"query": "mitochondria", "k": 3}).json()["hits"]
    assert hits
    assert "mitochondria" in hits[0]["text"]


def test_a_text_file_is_uploaded_too(client: TestClient) -> None:
    response = client.post(
        "/v1/memory/documents",
        files={"file": ("notes.txt", b"the cluster remembers this", "text/plain")},
    )
    assert response.status_code == 200
    assert response.json()["extracted"] is False


def test_an_empty_upload_is_refused(client: TestClient) -> None:
    response = client.post("/v1/memory/documents", files={"file": ("empty.txt", b"", "text/plain")})
    assert response.status_code == 422


def test_a_binary_upload_is_refused(client: TestClient) -> None:
    response = client.post(
        "/v1/memory/documents",
        files={"file": ("blob.bin", b"\xff\xfe\x00\x01", "application/octet-stream")},
    )
    assert response.status_code == 415


def test_a_scanned_pdf_is_refused_with_a_reason(client: TestClient) -> None:
    response = client.post(
        "/v1/memory/documents",
        files={"file": ("scan.pdf", build_scanned_pdf(), "application/pdf")},
    )
    assert response.status_code == 415
    assert "no readable text" in response.json()["detail"]


def test_the_documents_are_listed(client: TestClient) -> None:
    client.post(
        "/v1/memory/documents",
        files={"file": ("paper.pdf", build_pdf("listed"), "application/pdf")},
    )
    documents = client.get("/v1/memory/documents").json()["documents"]
    assert any(document["name"] == "paper.pdf" for document in documents)
