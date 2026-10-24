"""Per-instruction facts: relocation masks, literals and memory access widths."""

from __future__ import annotations

from tools.sharc_dev import sharc_visa_tables as visa_tables
from tools.sharc_dev import sharcfn, sharcinv, sharcldr
from tools.sharc_dev.sharc_core.encoding import TYPE4B_ACCESS_WIDTHS
from tools.sharc_dev.sharc_trace import ACCESS_WIDTHS

# Field-label stems masked to build a relocation-tolerant function hash: a
# PC-relative branch offset ('reladdr'), an absolute call/jump target or
# I-register-modify amount ('addr'/'data'), so two copies of the same
# function at different addresses (or pointing at different-but-matching-
# shaped data) still hash the same.
_MASK_STEMS = ("addr", "reladdr", "data")


def mask_relocatable(insn) -> int:
    """insn.raw with every 'addr'/'reladdr'/'data' field's bits zeroed, using
    tools/sharc_dev/sharc_visa_tables.py's own (hi, lo) field layout for insn.type_name
    -- no re-derivation of field positions."""
    if insn.raw is None:
        return 0
    entry = visa_tables.get_type(insn.type_name)
    if entry is None:
        return insn.raw
    raw = insn.raw
    for label, (hi, lo) in entry["fields"].items():
        if label.split("[")[0] in _MASK_STEMS:
            width = hi - lo + 1
            raw &= ~(((1 << width) - 1) << lo)
    return raw


def extract_literal(insn_type: str, f: dict):
    """-> (value, dest_reg) for a literal-carrying form (tools/sharc_dev/sharcinv.py's
    LITERAL_FORMS), sign-extended per that table's bit width; None for any
    other form. dest_reg is the register the literal loads into for
    17a/17b (a ureg, e.g. an M-register) and 19a/19a_scaled/19a_bitrev (the
    modified I-register); None for the direct-address form 14a (the literal
    there is a memory address, not a register's contents), for 15a (the
    literal is a signed offset added to an I register the decoder never
    updates -- register-relative, not a register's contents either; see
    extract_mem_access's module note), and for 16a (which stores the
    literal to memory rather than a register)."""
    spec = sharcinv.LITERAL_FORMS.get(insn_type)
    if spec is None:
        return None
    field, bits, _is_float = spec
    raw_val = f.get(field)
    if raw_val is None:
        return None
    value = sharcfn.sign_extend(raw_val, bits)
    dest = None
    if insn_type in ("17a", "17b"):
        ureg = f.get("ureg")
        if ureg is not None:
            dest = sharcfn.ureg_name(ureg)
    elif insn_type in ("19a", "19a_scaled", "19a_bitrev"):
        # Same Is XOR Idis destination convention as sharcfn.render_modify.
        bank = 8 if f.get("g") else 0
        src_low, dis_low = f.get("is", 0), f.get("idis", 0)
        dest = "I%d" % (bank + (src_low ^ dis_low))
    elif insn_type == "18a":
        sreg = f.get("sreg", 0)
        code = sharcfn.UREG_CODES.get("USTAT1", 0) + sreg
        dest = sharcfn.ureg_name(code)
    return value, dest


def classify_literal_range(value: int, code_spans_sw, code_spans_byte, mem):
    """(in_code_range, in_data_range) for a literal value: whether it falls
    in a scanned code block's own short-word range or its loader byte-alias
    range, or in some other loader-backed (data) range. A value in both is
    reported as code only (a literal is either a code pointer or a data
    pointer, not both)."""
    in_code = any(lo <= value < hi for lo, hi in code_spans_sw)
    if not in_code:
        byte_val = sharcldr.sw_to_byte(value)
        in_code = any(
            lo <= value < hi or lo <= byte_val < hi for lo, hi in code_spans_byte
        )
    in_data = False
    if not in_code:
        for lo, hi in mem.ranges():
            if lo <= value < hi:
                in_data = True
                break
    return in_code, in_data


# Type4b/4d (decode_table.json: l/w/x at bits 18/17/16, no "ex" bit) carry
# l/x/w width tables (PRM pp.13-32/13-34; Type4b's differs from Type3b's in
# its (1, 1, 1) row, see _immoff_width): the "long if l else word" fallback used for every other
# DIRECT_MEM_FORMS/IMMOFF_MEM_FORMS member (whose lone "l" bit really does
# mean a plain long-word/Ureg-pair access, e.g. 4a/15a/14a/15b) mislabels a
# byte (bw, l=0) access as "word" and a short-word (sw, l=1) access as
# "long" for these two forms.
_LXW_WIDTH_FORMS = {"4b", "4d"}


def _immoff_width(insn_type: str, f: dict):
    if insn_type in _LXW_WIDTH_FORMS:
        fields = (f.get("l", 0), f.get("x", 0), f.get("w", 0))
        # Type4b's own table differs from Type3b's: (1, 1, 1) is the plain
        # normal word, not (lw) (PRM p.13-32; sharc_core.encoding).
        table = TYPE4B_ACCESS_WIDTHS if insn_type == "4b" else ACCESS_WIDTHS
        return table.get(fields, "unknown(l=%d,x=%d,w=%d)" % fields)
    return "long" if f.get("l") else "word"


def _type14d_width(f: dict):
    """Type14d access width from its w/ex/d/l/x bits (sharc-plus-prm
    pp.384-387, Figure 15-2), mirroring tools/sharc_dev/sharc_trace.py's "14d"
    _execute branch: it only computes a width for the w=0,ex=0 rows it
    actually runs -- BH for a store (l picks byte/short) and BHSE for a
    load (l picks byte/short, x additionally picks zero-/sign-extend) --
    which is exactly sharc_trace.ACCESS_WIDTHS' w=0 slice. w=1 (EX/LWEX)
    and a store with x=1 are cases the tracer stops on rather than
    executing, so they get the same "unknown(...)" fallback _immoff_width
    uses for an unrepresented combination."""
    w, ex, l, x = f.get("w", 0), f.get("ex", 0), f.get("l", 0), f.get("x", 0)  # noqa: E741
    if w or ex or (f.get("d") and x):
        return "unknown(w=%d,ex=%d,l=%d,x=%d)" % (w, ex, l, x)
    return ACCESS_WIDTHS[(l, x, 0)]


def _type3d_width(f: dict):
    """ACCESS uses BH/BHSE l/x widths even with ex=0 (PRM pp.13-22--13-25).

    WACCESS/exclusive widths are descriptive only; the executor still stops
    on unsupported monitor semantics or reserved combinations.
    """
    w, ex, l, x = f.get("w", 0), f.get("ex", 0), f.get("l", 0), f.get("x", 0)  # noqa: E741
    if w:
        if not ex:
            return "unknown(w=1,ex=0,l=%d,x=%d)" % (l, x)
        return "long-word" if l else "normal-word"
    return ACCESS_WIDTHS[(l, x, 0)]


def extract_mem_access(insn_type: str, f: dict):
    """-> [(space, direction, base_reg, modifier, u, form, width, abs_address), ...]
    for a memory-referencing form; [] for any other form. form is
    tools/sharc_dev/sharcinv.py's MEM_FORMS label ('direct'/'i,m-mod'/'i+imm'/
    'i,m-mod+shift') or 'dual' for the simultaneous DM+PM forms, so it stays
    in step with that table rather than reinventing addressing-mode names.

    abs_address is only ever set for a form with no I-register component at
    all (14a/14d: sharc-plus-prm pp.373-383, "DM(<addr32>)" / "PM(<addr32>)"
    direct addressing, no base register). Despite sharing DIRECT_MEM_FORMS'
    grouping with those two, Type15a is register-relative, not absolute --
    see tools/sharc_dev/sharcfn/'s render_mem_direct docstring for the manual
    citation (pp.387-390, Figure 15-3, worked example "DM(24,I5)=TCOUNT"):
    "addr" there is a signed 32-bit offset from an I register the decoder
    never updates, not a memory address. Before this fix abs_address held
    that raw 32-bit field unsigned, so a small negative offset such as -0x87
    (I7-0x87) stored as 0xffffff79 -- 429 rows in dt2-1.16, all indented
    from the true short-word addresses of any real DM/PM global. The
    base register and signed offset now go in base_reg/modifier, the same
    columns INDEXED_MEM_FORMS/IMMOFF_MEM_FORMS already use for their own
    register-relative accesses, and abs_address is left NULL."""
    rows = []
    if insn_type == "15a":
        space, direction = sharcfn._space_dir(f)
        index = (f.get("i", 0) & 0x7) + (8 if f.get("g") else 0)
        off = sharcfn.sign_extend(f.get("addr", 0) & 0xFFFFFFFF, 32)
        width = "long" if f.get("l") else "word"
        rows.append(
            (
                space,
                direction,
                "I%d" % index,
                str(off),
                None,
                sharcinv.MEM_FORMS.get(insn_type, insn_type),
                width,
                None,
            )
        )
    elif insn_type in sharcfn.DIRECT_MEM_FORMS:
        space, direction = sharcfn._space_dir(f)
        width = (
            _type14d_width(f)
            if insn_type == "14d"
            else ("long" if f.get("l") else "word")
        )
        rows.append(
            (
                space,
                direction,
                None,
                None,
                None,
                sharcinv.MEM_FORMS.get(insn_type, insn_type),
                width,
                f.get("addr"),
            )
        )
    elif insn_type in sharcfn.INDEXED_MEM_FORMS:
        space, direction = sharcfn._space_dir(f)
        if insn_type == "3b":
            width = ACCESS_WIDTHS.get((f.get("l", 0), f.get("x", 0), f.get("w", 0)))
        elif insn_type == "3d":
            width = _type3d_width(f)
        else:
            width = None
        rows.append(
            (
                space,
                direction,
                "I%d" % f.get("i", 0),
                "M%d" % f.get("m", 0),
                f.get("u"),
                sharcinv.MEM_FORMS.get(insn_type, insn_type),
                width,
                None,
            )
        )
    elif insn_type in sharcfn.IMMOFF_MEM_FORMS:
        space, direction = sharcfn._space_dir(f)
        bits = sharcfn._IMMOFF_DATA_BITS.get(insn_type, 6)
        off = sharcfn.sign_extend(f.get("data", 0), bits)
        rows.append(
            (
                space,
                direction,
                "I%d" % f.get("i", 0),
                str(off),
                None,
                sharcinv.MEM_FORMS.get(insn_type, insn_type),
                _immoff_width(insn_type, f),
                None,
            )
        )
    elif insn_type in sharcfn.DUAL_MEM_FORMS:
        rows.append(
            (
                "DM",
                "store" if f.get("dmd") else "load",
                "I%d" % f.get("dmi", 0),
                "M%d" % f.get("dmm", 0),
                None,
                "dual",
                None,
                None,
            )
        )
        rows.append(
            (
                "PM",
                "store" if f.get("pmd") else "load",
                "I%d" % f.get("pmi", 0),
                "M%d" % f.get("pmm", 0),
                None,
                "dual",
                None,
                None,
            )
        )
    return rows
