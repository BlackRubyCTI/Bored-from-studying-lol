"""A dependency-free QR encoder.

The installer prints the URL of the dashboard as a QR code so that the phone
can open it with a single scan.  ``qrencode`` is not installed everywhere
(and Termux would need another package), so the encoder lives here: byte
mode, error correction level M, versions 1 to 10 (a URL fits easily), the
eight masks with the standard penalty rules and the terminal renderer.

The matrix is verified against a reference implementation in the test-suite,
so the phone really can scan it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: The primitive polynomial of the QR Galois field.
GF_POLYNOMIAL = 0x11D

#: The error correction levels (the two bits that go into the format info).
ECC_LEVELS: dict[str, int] = {"L": 0b01, "M": 0b00, "Q": 0b11, "H": 0b10}

#: ``version -> (ec codewords per block, [(blocks, data codewords), ...])``.
#: Only the error correction level M is needed (a URL is short and the phone
#: scans it from a screen).
BLOCK_TABLE: dict[int, tuple[int, list[tuple[int, int]]]] = {
    1: (10, [(1, 16)]),
    2: (16, [(1, 28)]),
    3: (26, [(1, 44)]),
    4: (18, [(2, 32)]),
    5: (24, [(2, 43)]),
    6: (16, [(4, 27)]),
    7: (18, [(4, 31)]),
    8: (22, [(2, 38), (2, 39)]),
    9: (22, [(3, 36), (2, 37)]),
    10: (26, [(4, 43), (1, 44)]),
}

#: ``version -> total number of codewords``.
TOTAL_CODEWORDS: dict[int, int] = {
    1: 26, 2: 44, 3: 70, 4: 100, 5: 134,
    6: 172, 7: 196, 8: 242, 9: 292, 10: 346,
}

#: ``version -> the centres of the alignment patterns``.
ALIGNMENT_POSITIONS: dict[int, list[int]] = {
    1: [], 2: [6, 18], 3: [6, 22], 4: [6, 26], 5: [6, 30],
    6: [6, 34], 7: [6, 22, 38], 8: [6, 24, 42], 9: [6, 26, 46], 10: [6, 28, 50],
}


class QrError(RuntimeError):
    """Raised when a payload cannot be encoded."""


# --------------------------------------------------------------------------- #
# The Galois field
# --------------------------------------------------------------------------- #
def _gf_tables() -> tuple[list[int], list[int]]:
    """The exponent and logarithm tables of GF(256)."""
    exp = [0] * 512
    log = [0] * 256
    value = 1
    for index in range(255):
        exp[index] = value
        log[value] = index
        value <<= 1
        if value & 0x100:
            value ^= GF_POLYNOMIAL
    for index in range(255, 512):
        exp[index] = exp[index - 255]
    return exp, log


_EXP, _LOG = _gf_tables()


def _gf_mul(left: int, right: int) -> int:
    if left == 0 or right == 0:
        return 0
    return _EXP[_LOG[left] + _LOG[right]]


def _rs_generator(degree: int) -> list[int]:
    """The generator polynomial of the Reed-Solomon code."""
    poly = [1]
    for index in range(degree):
        poly = _poly_multiply(poly, [1, _EXP[index]])
    return poly


def _poly_multiply(left: list[int], right: list[int]) -> list[int]:
    """Multiply two polynomials (the highest degree first)."""
    result = [0] * (len(left) + len(right) - 1)
    for i, a in enumerate(left):
        if not a:
            continue
        for j, b in enumerate(right):
            result[i + j] ^= _gf_mul(a, b)
    return result


def _rs_remainder(data: bytes, ec_count: int) -> list[int]:
    """The error correction codewords of a block.

    This is the synthetic division of the message by the generator
    polynomial -- the remainder *is* the error correction codeword.
    """
    generator = _rs_generator(ec_count)
    buffer = list(data) + [0] * ec_count
    for index in range(len(data)):
        coefficient = buffer[index]
        if not coefficient:
            continue
        for offset in range(1, len(generator)):
            buffer[index + offset] ^= _gf_mul(generator[offset], coefficient)
    return buffer[len(data):]


# --------------------------------------------------------------------------- #
# The encoder
# --------------------------------------------------------------------------- #
@dataclass
class QrCode:
    """A rendered QR code."""

    version: int
    size: int
    modules: list[list[bool]] = field(default_factory=list)
    mask: int = 0
    ecc: str = "M"
    data: bytes = b""

    @property
    def matrix(self) -> list[list[bool]]:
        """The modules (``True`` = dark)."""
        return self.modules

    def to_text(self, *, quiet_zone: int = 4) -> str:
        """The matrix as text (one character per module)."""
        lines: list[str] = []
        blank = " " * (self.size + quiet_zone * 2)
        for _ in range(quiet_zone):
            lines.append(blank)
        for row in self.modules:
            lines.append(" " * quiet_zone + "".join("██" if cell else "  " for cell in row) + " " * quiet_zone)
        for _ in range(quiet_zone):
            lines.append(blank)
        return "\n".join(lines)

    def to_svg(self, *, quiet_zone: int = 4, size: int = 512) -> str:
        """The matrix as an SVG document."""
        total = self.size + quiet_zone * 2
        unit = size / total
        parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {size} {size}" '
            f'width="{size}" height="{size}" shape-rendering="crispEdges">',
            f'<rect width="{size}" height="{size}" fill="#ffffff"/>',
            '<g fill="#000000">',
        ]
        for y, row in enumerate(self.modules):
            for x, cell in enumerate(row):
                if cell:
                    parts.append(
                        f'<rect x="{(x + quiet_zone) * unit:.3f}" y="{(y + quiet_zone) * unit:.3f}" '
                        f'width="{unit:.3f}" height="{unit:.3f}"/>'
                    )
        parts.append("</g></svg>")
        return "\n".join(parts)


def _choose_version(length: int, ecc: str) -> int:
    """The smallest version that holds the payload."""
    for version in sorted(BLOCK_TABLE):
        ec_count, groups = BLOCK_TABLE[version]
        capacity = sum(blocks * data for blocks, data in groups)
        # the mode (4 bits) + the count (8 or 16 bits) + the payload
        header = 4 + (8 if version <= 9 else 16)
        if capacity * 8 >= header + length * 8:
            return version
    raise QrError("the payload is too long for a QR code")


def _encode_data(payload: bytes, version: int, ecc: str) -> list[int]:
    """The final codeword sequence (data + error correction, interleaved)."""
    ec_count, groups = BLOCK_TABLE[version]
    bits: list[int] = []

    # mode: byte
    bits.extend([0, 1, 0, 0])
    # the character count
    count = len(payload)
    if version <= 9:
        bits.extend([(count >> i) & 1 for i in range(7, -1, -1)])
    else:
        bits.extend([(count >> i) & 1 for i in range(15, -1, -1)])
    for byte in payload:
        bits.extend([(byte >> i) & 1 for i in range(7, -1, -1)])

    capacity = sum(blocks * data for blocks, data in groups) * 8
    # the terminator
    bits.extend([0] * min(4, capacity - len(bits)))
    # pad to a byte boundary
    while len(bits) % 8:
        bits.append(0)
    codewords = [int("".join(str(bit) for bit in bits[i : i + 8]), 2) for i in range(0, len(bits), 8)]
    # the pad bytes (0xEC, 0x11, 0xEC, ...)
    pad = [0xEC, 0x11]
    pad_index = 0
    while len(codewords) < capacity // 8:
        codewords.append(pad[pad_index % 2])
        pad_index += 1

    # split into blocks
    blocks: list[bytes] = []
    offset = 0
    for count_blocks, data_count in groups:
        for _ in range(count_blocks):
            chunk = codewords[offset : offset + data_count]
            offset += data_count
            blocks.append(bytes(chunk))
    ec_blocks = [_rs_remainder(block, ec_count) for block in blocks]

    # interleave
    result: list[int] = []
    for index in range(max(len(block) for block in blocks)):
        for block in blocks:
            if index < len(block):
                result.append(block[index])
    for index in range(ec_count):
        for block in ec_blocks:
            result.append(block[index])
    return result


def _mask_condition(mask: int, x: int, y: int) -> bool:
    """Whether a module is inverted by a mask."""
    if mask == 0:
        return (x + y) % 2 == 0
    if mask == 1:
        return y % 2 == 0
    if mask == 2:
        return x % 3 == 0
    if mask == 3:
        return (x + y) % 3 == 0
    if mask == 4:
        return (x // 3 + y // 2) % 2 == 0
    if mask == 5:
        return (x * y) % 2 + (x * y) % 3 == 0
    if mask == 6:
        return ((x * y) % 2 + (x * y) % 3) % 2 == 0
    return ((x + y) % 2 + (x * y) % 3) % 2 == 0


def _format_bits(ecc: str, mask: int) -> int:
    """The 15 bits of the format information."""
    data = (ECC_LEVELS[ecc] << 3) | mask
    # the BCH(15,5) encoding
    remainder = data << 10
    for shift in range(4, -1, -1):
        if remainder & (1 << (shift + 10)):
            remainder ^= 0b10100110111 << shift
    return ((data << 10) | remainder) ^ 0b101010000010010


def _version_bits(version: int) -> int:
    """The 18 bits of the version information (version 7 and above)."""
    remainder = version << 12
    generator = 0b1111100100101
    for shift in range(5, -1, -1):
        if remainder & (1 << (shift + 12)):
            remainder ^= generator << shift
    return (version << 12) | remainder


class _Builder:
    """Builds the module matrix (the layout follows ISO/IEC 18004:2015)."""

    def __init__(self, version: int) -> None:
        self.version = version
        self.size = version * 4 + 17
        self.modules: list[list[bool | None]] = [[None] * self.size for _ in range(self.size)]
        self.reserved: list[list[bool]] = [[False] * self.size for _ in range(self.size)]

    # ------------------------------------------------------------------ #
    def set_function(self, x: int, y: int, dark: bool) -> None:
        """Place a function module (it is never masked)."""
        if 0 <= y < self.size and 0 <= x < self.size:
            self.modules[y][x] = dark
            self.reserved[y][x] = True

    def draw_finder(self, x: int, y: int) -> None:
        """A 7x7 finder pattern (with its separator)."""
        for dy in range(-1, 8):
            for dx in range(-1, 8):
                px, py = x + dx, y + dy
                if not (0 <= px < self.size and 0 <= py < self.size):
                    continue
                dark = (
                    0 <= dx <= 6
                    and 0 <= dy <= 6
                    and (dx in (0, 6) or dy in (0, 6) or (2 <= dx <= 4 and 2 <= dy <= 4))
                )
                self.set_function(px, py, dark)

    def draw_alignment(self, x: int, y: int) -> None:
        """A 5x5 alignment pattern."""
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                self.set_function(x + dx, y + dy, max(abs(dx), abs(dy)) != 1)

    def draw_function_patterns(self) -> None:
        """The timing, the finder and the alignment patterns."""
        size = self.size
        # the timing patterns come first: the finder patterns overwrite
        # their ends.
        for index in range(size):
            self.set_function(index, 6, index % 2 == 0)
            self.set_function(6, index, index % 2 == 0)
        self.draw_finder(0, 0)
        self.draw_finder(size - 7, 0)
        self.draw_finder(0, size - 7)
        # the alignment patterns (the three that touch a finder are skipped)
        positions = ALIGNMENT_POSITIONS.get(self.version, [])
        last = len(positions) - 1
        for i, cy in enumerate(positions):
            for j, cx in enumerate(positions):
                if (i == 0 and j == 0) or (i == 0 and j == last) or (i == last and j == 0):
                    continue
                self.draw_alignment(cx, cy)

    def draw_codewords(self, codewords: list[int]) -> None:
        """Place the data in the zigzag."""
        size = self.size
        index = 0
        total = len(codewords) * 8
        right = size - 1
        while right >= 1:
            if right == 6:
                right = 5
            for vert in range(size):
                upward = ((right + 1) & 2) == 0
                y = size - 1 - vert if upward else vert
                for j in range(2):
                    x = right - j
                    if self.reserved[y][x]:
                        continue
                    if index < total:
                        self.modules[y][x] = bool((codewords[index >> 3] >> (7 - (index & 7))) & 1)
                        index += 1
                    else:
                        self.modules[y][x] = False
            right -= 2

    def draw_format(self, ecc: str, mask: int) -> None:
        """Write the format information (both copies).

        The layout follows ISO/IEC 18004:2015, 7.9: the vertical copy
        lives in the column 8 and the horizontal copy in the row 8.
        """
        size = self.size
        bits = _format_bits(ecc, mask)
        voffset = 0
        hoffset = 0
        for index in range(8):
            vbit = bool((bits >> index) & 1)
            hbit = bool((bits >> (14 - index)) & 1)
            if index == 6:
                # the timing pattern sits between the two copies
                voffset += 1
                hoffset = 1
            # the vertical copy (upper left corner)
            self.set_function(8, index + voffset, vbit)
            # the horizontal copy (upper left corner)
            self.set_function(index + hoffset, 8, hbit)
            # the horizontal copy (upper right corner)
            self.set_function(size - 1 - index, 8, vbit)
            # the vertical copy (lower left corner)
            self.set_function(8, size - 1 - index, hbit)
        # the always dark module
        self.set_function(8, size - 8, True)

    def draw_version(self) -> None:
        """Write the version information (version 7 and above)."""
        if self.version < 7:
            return
        size = self.size
        bits = _version_bits(self.version)
        for index in range(18):
            bit = bool((bits >> index) & 1)
            a = size - 11 + index % 3
            b = index // 3
            self.set_function(a, b, bit)
            self.set_function(b, a, bit)

    def apply_mask(self, mask: int) -> list[list[bool]]:
        """Apply a mask to the data modules."""
        out: list[list[bool]] = []
        for y in range(self.size):
            row: list[bool] = []
            for x in range(self.size):
                value = bool(self.modules[y][x])
                if not self.reserved[y][x] and _mask_condition(mask, x, y):
                    value = not value
                row.append(value)
            out.append(row)
        return out


def _penalty(modules: list[list[bool]]) -> int:
    """The four penalty rules of the QR specification."""
    size = len(modules)
    score = 0

    # rule 1: runs of five (or more) modules of the same colour
    for row in modules:
        run_colour = row[0]
        run = 1
        for cell in row[1:]:
            if cell == run_colour:
                run += 1
            else:
                if run >= 5:
                    score += 3 + (run - 5)
                run_colour = cell
                run = 1
        if run >= 5:
            score += 3 + (run - 5)
    for x in range(size):
        run_colour = modules[0][x]
        run = 1
        for y in range(1, size):
            cell = modules[y][x]
            if cell == run_colour:
                run += 1
            else:
                if run >= 5:
                    score += 3 + (run - 5)
                run_colour = cell
                run = 1
        if run >= 5:
            score += 3 + (run - 5)

    # rule 2: 2x2 blocks of the same colour
    for y in range(size - 1):
        for x in range(size - 1):
            cell = modules[y][x]
            if (
                cell == modules[y][x + 1]
                and cell == modules[y + 1][x]
                and cell == modules[y + 1][x + 1]
            ):
                score += 3

    # rule 3: the finder-like patterns
    pattern_a = [True, False, True, True, True, False, True, False, False, False, False]
    pattern_b = list(reversed(pattern_a))
    for row in modules:
        for index in range(size - 10):
            window = row[index : index + 11]
            if window == pattern_a or window == pattern_b:
                score += 40
    for x in range(size):
        column = [modules[y][x] for y in range(size)]
        for index in range(size - 10):
            window = column[index : index + 11]
            if window == pattern_a or window == pattern_b:
                score += 40

    # rule 4: the balance of dark and light
    dark = sum(1 for row in modules for cell in row if cell)
    ratio = dark * 100 // (size * size)
    score += 10 * (abs(ratio - 50) // 5)
    return score


def encode(payload: str | bytes, *, ecc: str = "M", mask: int | None = None) -> QrCode:
    """Encode a payload into a QR code.

    ``mask`` pins the data mask (the tests use it to compare the matrix
    with a reference implementation); ``None`` picks the mask with the
    lowest penalty score.
    """
    if isinstance(payload, str):
        payload = payload.encode("utf-8")
    if ecc not in ECC_LEVELS:
        raise QrError(f"unknown error correction level: {ecc}")
    if mask is not None and not 0 <= mask <= 7:
        raise QrError("the mask must be between 0 and 7")
    version = _choose_version(len(payload), ecc)
    codewords = _encode_data(payload, version, ecc)

    best_matrix: list[list[bool]] = []
    best_mask = 0
    best_score = None
    for candidate in range(8):
        if mask is not None and candidate != mask:
            continue
        # the function patterns, the format information and the version
        # information depend on the mask, so the matrix is built per mask.
        builder = _Builder(version)
        builder.draw_function_patterns()
        builder.draw_format(ecc, candidate)
        builder.draw_version()
        builder.draw_codewords(codewords)
        matrix = builder.apply_mask(candidate)
        score = _penalty(matrix)
        if best_score is None or score < best_score:
            best_score = score
            best_mask = candidate
            best_matrix = matrix

    return QrCode(
        version=version,
        size=len(best_matrix),
        modules=best_matrix,
        mask=best_mask,
        ecc=ecc,
        data=payload,
    )


def read_payload(code: QrCode) -> bytes:
    """Read the payload back out of a QR code (byte mode, level M).

    It is the inverse of :func:`encode`: the modules are unmasked, the
    codewords are read from the zigzag, de-interleaved and the header is
    parsed.  The test-suite uses it to prove that the encoder and the
    reader agree.
    """
    builder = _Builder(code.version)
    builder.draw_function_patterns()
    builder.draw_format(code.ecc, code.mask)
    builder.draw_version()

    # 1. read the data modules back (in the zigzag order)
    size = code.size
    bits: list[int] = []
    right = size - 1
    while right >= 1:
        if right == 6:
            right = 5
        for vert in range(size):
            upward = ((right + 1) & 2) == 0
            y = size - 1 - vert if upward else vert
            for j in range(2):
                x = right - j
                if builder.reserved[y][x]:
                    continue
                dark = code.modules[y][x]
                if not builder.reserved[y][x] and _mask_condition(code.mask, x, y):
                    dark = not dark
                bits.append(1 if dark else 0)
        right -= 2

    # 2. the codewords
    codewords = [
        int("".join(str(bit) for bit in bits[i : i + 8]), 2)
        for i in range(0, len(bits) - len(bits) % 8, 8)
    ]

    # 3. de-interleave (the blocks are interleaved column by column)
    ec_count, groups = BLOCK_TABLE[code.version]
    lengths: list[int] = []
    for count, data_count in groups:
        lengths.extend([data_count] * count)
    blocks: list[list[int]] = [[] for _ in lengths]
    position = 0
    longest = max(lengths)
    for index in range(longest):
        for number, length in enumerate(lengths):
            if index < length:
                blocks[number].append(codewords[position])
                position += 1
    # the data stream is the concatenation of the blocks
    data: list[int] = [value for block in blocks for value in block]

    # 4. the header (the mode and the character count)
    stream: list[int] = []
    for byte in data:
        for shift in range(7, -1, -1):
            stream.append((byte >> shift) & 1)
    mode = int("".join(str(bit) for bit in stream[:4]), 2)
    if mode != 0b0100:
        raise QrError(f"only the byte mode is supported (got {mode:04b})")
    if code.version <= 9:
        count = int("".join(str(bit) for bit in stream[4:12]), 2)
        offset = 12
    else:
        count = int("".join(str(bit) for bit in stream[4:20]), 2)
        offset = 20
    payload = bytearray()
    for index in range(count):
        chunk = stream[offset + index * 8 : offset + index * 8 + 8]
        if len(chunk) < 8:
            raise QrError("the payload is truncated")
        payload.append(int("".join(str(bit) for bit in chunk), 2))
    return bytes(payload)


# --------------------------------------------------------------------------- #
# The terminal renderer
# --------------------------------------------------------------------------- #
def render_terminal(payload: str, *, ecc: str = "M", quiet_zone: int = 2) -> str:
    """Render a QR code for the terminal.

    Every module is half a character wide, so the code is square on a
    terminal that uses a 1:2 cell ratio.
    """
    code = encode(payload, ecc=ecc)
    size = code.size
    lines: list[str] = []
    blank = "  " * (size + quiet_zone * 2)
    for _ in range(quiet_zone):
        lines.append(blank)
    for y in range(size):
        row = code.modules[y]
        parts: list[str] = []
        for x in range(size):
            if not row[x]:
                parts.append("██")
                continue
            # the module below decides the lower half of the cell
            below = code.modules[y + 1][x] if y + 1 < size else False
            parts.append("▀▀" if not below else "  ")
        lines.append("  " * quiet_zone + "".join(parts) + "  " * quiet_zone)
    for _ in range(quiet_zone - 1):
        lines.append(blank)
    return "\n".join(lines)


__all__ = [
    "ALIGNMENT_POSITIONS",
    "BLOCK_TABLE",
    "QrCode",
    "QrError",
    "encode",
    "read_payload",
    "render_terminal",
]
