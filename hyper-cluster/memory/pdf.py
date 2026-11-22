"""A dependency-free PDF text extractor.

The dashboard accepts PDFs (the phone drops a paper into the chat), so the
memory has to be able to read them.  Pulling in ``pypdf`` would add a
dependency to the control plane for a feature that is used exactly once, so
the extractor is implemented here with nothing but the standard library:

* the objects of the file are scanned and the streams are inflated
  (``FlateDecode``, with the ``/DecodeParms`` predictor applied);
* the ``ToUnicode`` CMaps of the document are parsed (``bfchar`` / ``bfrange``),
  so subset fonts (the ones a LaTeX or a Word export uses) decode correctly;
* the content streams are tokenised and the text operators (``Tj``, ``TJ``,
  ``'``, ``"``) are decoded with the CMap of the font that is active.

A scanned PDF (an image, no text layer) has nothing to extract -- that is
reported honestly instead of returning garbage.
"""

from __future__ import annotations

import re
import zlib
from dataclasses import dataclass, field
from typing import Any

from core.logging_setup import get_logger

LOGGER = get_logger("memory.pdf")

#: The marker that starts a PDF file.
PDF_MAGIC = b"%PDF-"

#: How much of a file we are willing to read.
MAX_PDF_BYTES = 64 * 1024 * 1024


class PdfError(RuntimeError):
    """Raised when a PDF cannot be read."""


@dataclass
class PdfDocument:
    """The result of an extraction."""

    text: str = ""
    pages: int = 0
    encrypted: bool = False
    warnings: list[str] = field(default_factory=list)

    @property
    def has_text(self) -> bool:
        return bool(self.text.strip())

    def to_dict(self) -> dict[str, Any]:
        return {
            "pages": self.pages,
            "encrypted": self.encrypted,
            "characters": len(self.text),
            "warnings": list(self.warnings),
        }


# --------------------------------------------------------------------------- #
# The low level object model
# --------------------------------------------------------------------------- #
@dataclass
class _Object:
    """One indirect object of the file."""

    number: int = 0
    body: bytes = b""
    stream: bytes | None = None


_OBJECT_RE = re.compile(rb"(\d+)\s+(\d+)\s+obj\b")


def _scan_objects(data: bytes) -> dict[int, _Object]:
    """Split the file into its indirect objects."""
    objects: dict[int, _Object] = {}
    for match in _OBJECT_RE.finditer(data):
        number = int(match.group(1))
        start = match.end()
        end = data.find(b"endobj", start)
        if end < 0:
            end = len(data)
        body = data[start:end]
        stream: bytes | None = None
        marker = body.find(b"stream")
        if marker >= 0:
            header = body[:marker]
            payload_start = marker + len(b"stream")
            # the stream data starts after the EOL
            if body[payload_start : payload_start + 2] == b"\r\n":
                payload_start += 2
            elif body[payload_start : payload_start + 1] in (b"\n", b"\r"):
                payload_start += 1
            payload_end = body.rfind(b"endstream")
            if payload_end < 0:
                payload_end = len(body)
            stream = body[payload_start:payload_end]
            body = header
        objects[number] = _Object(number=number, body=body, stream=stream)
    return objects


def _inflate(stream: bytes) -> bytes | None:
    """Inflate a FlateDecode stream (with an optional PNG predictor)."""
    try:
        raw = zlib.decompress(stream)
    except zlib.error:
        try:
            raw = zlib.decompressobj().decompress(stream)
        except zlib.error:
            return None
    return raw


def _apply_predictor(data: bytes, params: bytes) -> bytes:
    """Undo the PNG predictor of a Flate stream (``/DecodeParms``)."""
    predictor = 12
    columns = 1
    match = re.search(rb"/Predictor\s+(\d+)", params)
    if match:
        predictor = int(match.group(1))
    match = re.search(rb"/Columns\s+(\d+)", params)
    if match:
        columns = int(match.group(1))
    if predictor < 10:
        return data
    row_length = columns + 1
    out = bytearray()
    previous = bytearray(columns)
    for start in range(0, len(data), row_length):
        row = data[start : start + row_length]
        if not row:
            break
        filter_type = row[0]
        current = bytearray(row[1:])
        if len(current) < columns:
            current.extend(b"\x00" * (columns - len(current)))
        if filter_type == 1:  # Sub
            for index in range(1, columns):
                current[index] = (current[index] + current[index - 1]) & 0xFF
        elif filter_type == 2:  # Up
            for index in range(columns):
                current[index] = (current[index] + previous[index]) & 0xFF
        elif filter_type == 3:  # Average
            for index in range(columns):
                left = current[index - 1] if index else 0
                current[index] = (current[index] + ((left + previous[index]) >> 1)) & 0xFF
        elif filter_type == 4:  # Paeth
            for index in range(columns):
                left = current[index - 1] if index else 0
                up = previous[index]
                up_left = previous[index - 1] if index else 0
                estimate = left + up - up_left
                da = abs(estimate - left)
                db = abs(estimate - up)
                dc = abs(estimate - up_left)
                if da <= db and da <= dc:
                    predictor_value = left
                elif db <= dc:
                    predictor_value = up
                else:
                    predictor_value = up_left
                current[index] = (current[index] + predictor_value) & 0xFF
        out.extend(current)
        previous = current
    return bytes(out)


def _decode_stream(obj: _Object) -> bytes:
    """Decode the stream of an object (identity or Flate)."""
    if obj.stream is None:
        return b""
    if b"/FlateDecode" in obj.body:
        raw = _inflate(obj.stream)
        if raw is None:
            return b""
        params = re.search(rb"/DecodeParms\s*<<(.*?)>>", obj.body, re.DOTALL)
        if params:
            raw = _apply_predictor(raw, params.group(1))
        return raw
    return obj.stream


# --------------------------------------------------------------------------- #
# The ToUnicode CMaps
# --------------------------------------------------------------------------- #
def _parse_cmap(data: bytes) -> dict[int, str]:
    """Parse a ``ToUnicode`` CMap into a ``code -> text`` table."""
    table: dict[int, str] = {}
    for block in re.findall(rb"beginbfchar(.*?)endbfchar", data, re.DOTALL):
        for src, dst in re.findall(rb"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>", block):
            table[int(src, 16)] = _hex_to_text(dst)
    for block in re.findall(rb"beginbfrange(.*?)endbfrange", data, re.DOTALL):
        for match in re.finditer(
            rb"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>\s*(?:<([0-9A-Fa-f]+)>|\[(.*?)\])",
            block,
            re.DOTALL,
        ):
            low = int(match.group(1), 16)
            high = int(match.group(2), 16)
            if high < low or high - low > 0xFFFF:
                continue
            if match.group(3):
                start = int(match.group(3), 16)
                for offset in range(high - low + 1):
                    table[low + offset] = _utf16_be(start + offset).rstrip("\x00")
            elif match.group(4):
                items = re.findall(rb"<([0-9A-Fa-f]+)>", match.group(4))
                for offset, item in enumerate(items):
                    if low + offset > high:
                        break
                    table[low + offset] = _hex_to_text(item)
    return table


def _hex_to_text(value: bytes) -> str:
    """Decode a hex string of UTF-16BE code units."""
    if len(value) % 2:
        value += b"0"
    try:
        raw = bytes.fromhex(value.decode("ascii"))
    except (ValueError, UnicodeDecodeError):
        return ""
    return raw.decode("utf-16-be", errors="replace").rstrip("\x00")


def _utf16_be(value: int) -> str:
    """A single code point as UTF-16BE text."""
    try:
        return chr(value)
    except ValueError:
        return ""


# --------------------------------------------------------------------------- #
# The content stream
# --------------------------------------------------------------------------- #
_TOKEN_RE = re.compile(
    rb"""
    (?P<string>\((?:\\.|[^\\()])*\))          # a literal string
  | (?P<hex><[0-9A-Fa-f\s]*>)                 # a hex string
  | (?P<name>/[^\s/\[\]()<>]+)                # a name
  | (?P<number>[-+]?[0-9]*\.?[0-9]+)          # a number
  | (?P<array>\[|\])                          # array delimiters
  | (?P<dict><<|>>)                           # dictionary delimiters
  | (?P<operator>[A-Za-z'"*]+[01]?)            # an operator
    """,
    re.VERBOSE | re.DOTALL,
)


def _unescape(value: bytes) -> bytes:
    """Resolve the escape sequences of a literal PDF string."""
    out = bytearray()
    index = 0
    while index < len(value):
        char = value[index]
        if char != 0x5C:  # backslash
            out.append(char)
            index += 1
            continue
        index += 1
        if index >= len(value):
            break
        escape = value[index]
        index += 1
        mapping = {
            0x6E: 0x0A,  # n
            0x72: 0x0D,  # r
            0x74: 0x09,  # t
            0x62: 0x08,  # b
            0x66: 0x0C,  # f
            0x28: 0x28,  # (
            0x29: 0x29,  # )
            0x5C: 0x5C,  # backslash
        }
        if escape in mapping:
            out.append(mapping[escape])
        elif 0x30 <= escape <= 0x37:  # octal
            digits = chr(escape)
            while len(digits) < 3 and index < len(value) and 0x30 <= value[index] <= 0x37:
                digits += chr(value[index])
                index += 1
            out.append(int(digits, 8) & 0xFF)
        elif escape in (0x0A, 0x0D):  # a line continuation
            continue
        else:
            out.append(escape)
    return bytes(out)


def _decode_text(raw: bytes, cmap: dict[int, str], two_byte: bool) -> str:
    """Decode a string with the CMap of the active font."""
    if not cmap:
        return raw.decode("latin-1", errors="replace")
    out: list[str] = []
    if two_byte:
        for index in range(0, len(raw) - 1, 2):
            code = (raw[index] << 8) | raw[index + 1]
            out.append(cmap.get(code, ""))
    else:
        for byte in raw:
            out.append(cmap.get(byte, chr(byte) if 32 <= byte < 127 else ""))
    return "".join(out)


def _cmap(font: dict[str, Any] | None) -> dict[int, str]:
    """The CMap of the active font."""
    return dict((font or {}).get("cmap") or {})


def _two_byte(font: dict[str, Any] | None) -> bool:
    """Whether the active font uses two byte codes."""
    return bool((font or {}).get("two_byte"))


def _extract_text(content: bytes, fonts: dict[str, dict[str, Any]]) -> str:
    """Extract the text of one content stream."""
    pieces: list[str] = []
    stack: list[Any] = []
    current: dict[str, Any] | None = None
    for match in _TOKEN_RE.finditer(content):
        kind = match.lastgroup
        value = match.group()
        if kind == "string":
            stack.append(("string", _unescape(value[1:-1])))
        elif kind == "hex":
            digits = re.sub(rb"[^0-9A-Fa-f]", b"", value)
            if len(digits) % 2:
                digits += b"0"
            stack.append(("string", bytes.fromhex(digits.decode("ascii"))))
        elif kind == "name":
            stack.append(("name", value[1:].decode("latin-1")))
        elif kind == "number":
            stack.append(("number", float(value)))
        elif kind in ("array", "dict"):
            stack.append((kind, None))
        else:
            operator = value.decode("latin-1")
            if operator == "Tf" and len(stack) >= 2:
                name = stack[-2][1] if stack[-2][0] == "name" else ""
                current = fonts.get(name) or {}
            elif operator in ("Tj", "'", '"'):
                if stack and stack[-1][0] == "string":
                    pieces.append(_decode_text(stack[-1][1], _cmap(current), _two_byte(current)))
                    pieces.append("\n")
            elif operator == "TJ":
                for item in reversed(stack):
                    if item[0] == "string":
                        pieces.append(_decode_text(item[1], _cmap(current), _two_byte(current)))
                    elif item[0] == "number" and item[1] < -180:
                        pieces.append(" ")
            elif operator in ("ET", "TD", "Td", "T*"):
                pieces.append("\n")
            stack.clear()
    return "".join(pieces)


def _collect_fonts(objects: dict[int, _Object]) -> dict[str, dict[str, Any]]:
    """Map the resource names of the document (``/F1``) to their font.

    A font is described by two pieces of information: the ``ToUnicode`` CMap
    (how the glyph codes map back to text) and whether it is a two byte
    (Type0 / Identity-H) font.
    """
    # 1. the font objects, by their object number
    by_number: dict[int, dict[str, Any]] = {}
    for obj in objects.values():
        body = obj.body
        if b"/Font" not in body:
            continue
        entry: dict[str, Any] = {"cmap": {}, "two_byte": b"/Type0" in body}
        match = re.search(rb"/ToUnicode\s+(\d+)\s+\d+\s+R", body)
        if match:
            target = objects.get(int(match.group(1)))
            if target is not None:
                entry["cmap"] = _parse_cmap(_decode_stream(target))
        by_number[obj.number] = entry

    # 2. the resource names -> the font objects they point at
    fonts: dict[str, dict[str, Any]] = {}
    for obj in objects.values():
        for name, number in re.findall(rb"/([A-Za-z0-9#+._-]+)\s+(\d+)\s+\d+\s+R", obj.body):
            entry = by_number.get(int(number))
            if entry is None:
                continue
            fonts.setdefault(name.decode("latin-1"), entry)
    return fonts


def extract_text(payload: bytes) -> PdfDocument:
    """Extract the text of a PDF document.

    Returns a :class:`PdfDocument`; the text is empty when the document has
    no text layer (a scan) or when it is encrypted.
    """
    if not payload:
        raise PdfError("the file is empty")
    if len(payload) > MAX_PDF_BYTES:
        raise PdfError("the PDF is too large")
    if not payload.lstrip()[:1024].find(PDF_MAGIC) == 0:
        # Some producers put junk in front of the header.
        index = payload.find(PDF_MAGIC)
        if index < 0:
            raise PdfError("this is not a PDF file")
        payload = payload[index:]

    document = PdfDocument()
    if b"/Encrypt" in payload:
        document.encrypted = True
        document.warnings.append("the PDF is encrypted; no text could be extracted")
        return document

    objects = _scan_objects(payload)
    document.pages = len(re.findall(rb"/Type\s*/Page\b", payload))
    fonts = _collect_fonts(objects)

    chunks: list[str] = []
    for obj in objects.values():
        if obj.stream is None:
            continue
        body = obj.body
        if b"/Image" in body or b"/XObject" in body and b"/Subtype" in body:
            continue
        content = _decode_stream(obj)
        if not content or (b"BT" not in content and b"Tj" not in content and b"TJ" not in content):
            continue
        text = _extract_text(content, fonts)
        if text.strip():
            chunks.append(text)

    if not chunks:
        document.warnings.append("the PDF has no text layer (it is probably a scan)")
        return document

    document.text = _tidy("\n".join(chunks))
    return document


def _tidy(text: str) -> str:
    """Collapse the whitespace of the extracted text."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    out: list[str] = []
    for line in lines:
        if not line:
            if out and out[-1] != "":
                out.append("")
            continue
        out.append(line)
    return "\n".join(out).strip()


def is_pdf(payload: bytes, name: str = "") -> bool:
    """Whether a payload looks like a PDF (by magic or by name)."""
    if payload[:1024].lstrip().startswith(PDF_MAGIC):
        return True
    if payload[:4096].find(PDF_MAGIC) >= 0:
        return True
    return name.lower().endswith(".pdf")


def extract_pdf_text(payload: bytes) -> tuple[str, dict[str, Any]]:
    """Extract the text of a PDF and the metadata that goes with it.

    A PDF that cannot be read (encrypted, a scan, broken) yields an empty
    text and a warning -- it never raises into the request path.
    """
    try:
        document = extract_text(payload)
    except PdfError as exc:
        LOGGER.warning("pdf: cannot read the document", extra={"hc_error": str(exc)})
        return "", {"pdf_error": str(exc)}
    return document.text, {
        "pdf_pages": document.pages,
        "pdf_encrypted": document.encrypted,
        "pdf_warnings": list(document.warnings),
    }


__all__ = [
    "MAX_PDF_BYTES",
    "PdfDocument",
    "PdfError",
    "extract_pdf_text",
    "extract_text",
    "is_pdf",
]
