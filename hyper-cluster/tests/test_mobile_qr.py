"""The QR encoder of the installer (the phone scans the dashboard address)."""

from __future__ import annotations

import pytest

from mobile.qr import (
    ALIGNMENT_POSITIONS,
    BLOCK_TABLE,
    QrError,
    encode,
    read_payload,
    render_terminal,
)


def finder_pattern(matrix: list[list[bool]], x: int, y: int) -> bool:
    """The 7x7 finder pattern is at the corner (x, y)."""
    for dy in range(7):
        for dx in range(7):
            distance = max(abs(dx - 3), abs(dy - 3))
            expected = distance != 2
            if matrix[y + dy][x + dx] != expected:
                return False
    return True


# --------------------------------------------------------------------------- #
# The structure
# --------------------------------------------------------------------------- #
def test_the_size_follows_the_version() -> None:
    """The version grows with the payload, the size follows the version."""
    previous = 0
    for length in (5, 20, 60, 120, 200):
        code = encode("A" * length)
        assert code.size == code.version * 4 + 17
        assert code.version >= previous
        previous = code.version
    assert previous >= 7


def test_the_finder_patterns_are_there() -> None:
    code = encode("http://192.168.1.10:8082")
    assert finder_pattern(code.modules, 0, 0)
    assert finder_pattern(code.modules, code.size - 7, 0)
    assert finder_pattern(code.modules, 0, code.size - 7)


def test_the_timing_patterns_alternate() -> None:
    code = encode("http://192.168.1.10:8082")
    for index in range(8, code.size - 8):
        assert code.modules[6][index] == (index % 2 == 0)
        assert code.modules[index][6] == (index % 2 == 0)


def test_the_dark_module_is_dark() -> None:
    code = encode("http://192.168.1.10:8082")
    assert code.modules[code.size - 8][8] is True


def test_the_alignment_patterns_are_there() -> None:
    """Version 7 and above carry the version information next to them."""
    code = encode("A" * 120)
    assert code.version >= 7
    positions = ALIGNMENT_POSITIONS[code.version]
    last = len(positions) - 1
    for i, cy in enumerate(positions):
        for j, cx in enumerate(positions):
            if (i == 0 and j == 0) or (i == 0 and j == last) or (i == last and j == 0):
                continue
            # the centre of an alignment pattern is dark
            assert code.modules[cy][cx] is True


# --------------------------------------------------------------------------- #
# The data
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "payload",
    [
        "http://192.168.1.10:8082",
        "HELLO",
        "https://omega.example.com:8082/mission?x=1",
        "A" * 40,
        "x" * 200,
        "unicode: Ω μ λ",
    ],
)
def test_the_payload_survives_the_round_trip(payload: str) -> None:
    code = encode(payload)
    assert read_payload(code) == payload.encode("utf-8")


def test_every_mask_is_valid() -> None:
    """The matrix does not depend on the mask for the payload."""
    for mask in range(8):
        code = encode("http://192.168.1.10:8082", mask=mask)
        assert code.mask == mask
        assert read_payload(code) == b"http://192.168.1.10:8082"


def test_the_mask_is_the_one_with_the_lowest_penalty() -> None:
    code = encode("http://192.168.1.10:8082")
    assert 0 <= code.mask <= 7


def test_a_too_long_payload_is_refused() -> None:
    with pytest.raises(QrError):
        encode("x" * 5000)


def test_an_unknown_error_level_is_refused() -> None:
    with pytest.raises(QrError):
        encode("hello", ecc="Z")


def test_an_invalid_mask_is_refused() -> None:
    with pytest.raises(QrError):
        encode("hello", mask=9)


def test_the_block_table_is_consistent() -> None:
    for version, (ec_count, groups) in BLOCK_TABLE.items():
        data = sum(count * size for count, size in groups)
        assert data > 0
        assert ec_count > 0
        assert version in ALIGNMENT_POSITIONS


# --------------------------------------------------------------------------- #
# The rendering
# --------------------------------------------------------------------------- #
def test_the_terminal_rendering_has_the_code_in_it() -> None:
    rendered = render_terminal("http://192.168.1.10:8082")
    assert "█" in rendered
    assert "▀" in rendered
    lines = [line for line in rendered.splitlines() if line.strip()]
    assert len(lines) > 20


def test_the_svg_is_valid() -> None:
    code = encode("http://192.168.1.10:8082")
    svg = code.to_svg()
    assert svg.startswith("<svg")
    assert svg.endswith("</svg>")
    dark = sum(1 for row in code.modules for cell in row if cell)
    assert svg.count("<rect") == dark + 1  # the dark modules and the background
