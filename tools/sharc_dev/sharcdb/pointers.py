"""Constant propagation of pointer registers to memory access addresses."""

from __future__ import annotations

import bisect
import collections
import json

import networkx as nx

from tools.sharc_dev import sharcfn, sharcinv
from tools.sharc_dev.sharc_core.memory import _access_modifier_scale

from .facts import _immoff_width, extract_literal, extract_mem_access
from .regdefs import register_effects

# --- constant-pointer propagation (`ptr` table, DB_VERSION 6) --------------
#
# A simple, deterministic forward dataflow per function, over the same
# bblocks/succ this file already built, tracking known 32-bit constants for
# every register (I0-I15, R0-R15, M/L/B, ...) plus DM(I6+k) frame slots keyed
# by their raw offset (I6 -- docs/findings/06 -- is this ABI's frame
# pointer; its own runtime value is never tracked, only slot identity, so a
# spill/reload pair resolves regardless of what I6 actually holds at
# runtime). Rounds re-seed a function's R4/R8/R12 from its call sites'
# previous-round values when every caller agrees, approximating the RTOS-
# task/RPC-argument convention _detect_roots's rtos_task detector already
# relies on for a single call site; a handful of rounds is enough for a
# short call chain to converge without chasing a true fixed point.
#
# Modelled, mirroring tools/sharc_dev/sharc_trace.py's own handling of each form (the
# chosen source of truth for scaling/addressing, since this pass is a static
# approximation of exactly what that tracer executes dynamically):
#   - 17a/17b literal loads (extract_literal already generalises the
#     destination to any ureg, R included).
#   - 19a/19a_scaled MODIFY: dest is Is XOR Idis (bank-adjusted by `g`);
#     19a_scaled's immediate is scaled x4/x2 for a normal-/short-word
#     modifier, exactly as the tracer's assume_nw32 branch does (this repo
#     always assumes 32-bit internal normal words -- tools/sharc_dev/sharc.py's
#     trace()). 19a_bitrev's destination is always invalidated: bit-reversed
#     addressing is not a plain add.
#   - 5a/5b_move ureg-to-ureg copies.
#   - A conservative add/sub/inc/dec/pass/neg subset of the 23-bit parallel
#     compute field (COMPUTE_FORMS) and the 2c short-compute field -- every
#     other opcode (multiply, shift, logic, float, dual add/subtract, ...)
#     invalidates its destination rather than guessing.
#   - Every direct (14a/14d), indexed (3a/3b/3d/6a_mem) and immediate-offset
#     (4a/4b/4d/15b) memory form's own addressing math, INCLUDING Type15a's
#     I+immediate address: tools/sharc_dev/sharcfn/'s DIRECT_MEM_FORMS groups 15a
#     with the pure-absolute 14a/14d, but PRM p.388 and tools/sharc_dev/sharc_trace.py's
#     own "15a" handler both give it an I-register-relative address (its
#     DM(I6-4)/DM(0xfffffffc,I6) example is the exact frame-slot pairing
#     this pass exists to resolve) -- treating it as pure-absolute here would
#     silently mis-locate every Type15a access whose I register isn't zero.
#     I-register post-modify (u=1) commits the new value; pre-modify/no-
#     writeback (u=0) leaves the register untouched, exactly as the tracer's
#     own post_modify branch does. Bank selection (g=1 -> physical register
#     +8, DAG2/PM) is applied uniformly here -- unlike this file's EXISTING
#     extract_mem_access, whose INDEXED_MEM_FORMS branch never adds that +8
#     (a pre-existing, out-of-scope gap confirmed directly against
#     tools/sharc_dev/sharcspec/decode_table.json's field widths and sharc_trace.py's
#     own bank math; left alone rather than risk-edited under this task).
#   - Type3c (the plain-R-register "DM(Ii,Mm) = Rd" encoding tools/sharc_dev/sharcfn/
#     also renders as the I7/M7 stack-push idiom, but dmi/dmm are ordinary
#     register fields -- FUN_1c15e3's workspace-copy loop uses it with
#     dmi=I4/dmm=M6): always DM, always mandatory post-modify (no g/u
#     fields at all), exactly like an INDEXED_MEM_FORMS u=1 access.
#
# NOT modelled -- the affected register/slot is always invalidated, never
# guessed: Type16a/16b (immediate-to-memory store), Type1a/1b dual DM+PM
# forms (tools/sharc_dev/sharc_trace.py does not execute either), any ALU/MULT/SHIFT/
# MULTIFN opcode outside the small integer subset above, and any memory LOAD
# from a non-frame-slot address (the loaded value is runtime data, not a
# static fact this pass can know from decoding alone).
#
# Merge at a block join keeps a key only when every already-visited
# predecessor's out-state agrees on it. Blocks are visited in a per-function
# topological order that drops loop back edges, reusing analyze_image()'s
# own already-computed `idom` (an edge u->v is a back edge exactly when v
# dominates u -- the same test _detect_dominators_and_loops uses); a loop
# header's in-state is therefore its value on FIRST arrival only, never
# updated by a later iteration's own store/modify -- which is exactly what
# makes a loop that walks a pointer across iterations resolve to one
# loop-invariant base address rather than "unknown" (FUN_1c15e3's per-
# iteration store, tested below). A not-yet-visited or back-edge predecessor
# simply contributes nothing to a merge, the same safe "drop rather than
# guess" default this pass uses throughout. A block with NO confident
# predecessor at all -- the function's own entry, or one only an unresolved
# indirect jump reaches (a dispatch-table case; see _build_ptr_rows) --
# falls back to the entry baseline for registers and an empty frame, and is
# itself marked unconfident so that guess can never zero out a real,
# still-tracked value at a later genuine reconvergence (see
# _build_ptr_rows's own note on FUN_1c642a, where this mattered directly).

_PTR_ROUNDS = 4

# Hardware DAG-modify reset values (docs/findings/06), the same ones
# tools/sharc_dev/sharc.py's trace() seeds by default -- held in every function's
# entry state regardless of caller, not something call-site seeding could
# ever discover on its own.
_PTR_M_RESET = {
    "M5": 0,
    "M6": 1,
    "M7": 0xFFFFFFFF,
    "M13": 0,
    "M14": 1,
    "M15": 0xFFFFFFFF,
}

_PTR_CALL_ARG_REGS = ("R4", "R8", "R12")


def _ptr_bank_reg(prefix, num, f):
    """ "I%d"/"M%d" with PGR's DAG1/DAG2 bank offset (g=1 -> +8) applied --
    see the module note above on the existing extract_mem_access gap this
    intentionally does NOT share."""
    return "%s%d" % (prefix, num + (8 if f.get("g") else 0))


def _ptr_alu_updates(t, f, state):
    """New constant register values from an instruction's own parallel ALU
    compute field (its 'compute' key) -- {} for any opcode outside the
    add/sub/inc/dec/pass/neg subset this pass models, a float op, a dual
    add/subtract, or no compute field at all. The caller invalidates the
    defined register(s) when this returns nothing for them (register_effects
    already knows a compute field always redefines its Rn)."""
    if t in sharcinv.COMPUTE_FORMS:
        field23 = f.get("compute")
        if not field23:
            return {}
        cu, d = sharcinv.classify_compute(field23)
        if cu != "ALU" or d.get("is_dual_addsub") or d.get("is_float"):
            return {}
        name = sharcinv.ALU_OPS.get(d.get("opcode", 0))
        rn, rx, ry = ((field23 >> sh) & 0xF for sh in (8, 4, 0))
        Rn, Rx, Ry = ("R%d" % rn, "R%d" % rx, "R%d" % ry)
        if name in ("add", "sub"):
            vx, vy = state.get(Rx), state.get(Ry)
            if vx is not None and vy is not None:
                return {Rn: ((vx + vy) if name == "add" else (vx - vy)) & 0xFFFFFFFF}
        elif name in ("pass", "neg", "inc", "dec"):
            vx = state.get(Rx)
            if vx is not None:
                return {
                    Rn: {"pass": vx, "neg": -vx, "inc": vx + 1, "dec": vx - 1}[name]
                    & 0xFFFFFFFF
                }
        return {}
    if t == "2c":
        field12 = f.get("compute")
        if field12 is None:
            return {}
        opcode = (field12 >> 8) & 0xF
        if opcode in sharcfn.FLOAT_SHORT_OPS:
            return {}
        rn, rx = (field12 >> 4) & 0xF, field12 & 0xF
        Rn, Rx = "R%d" % rn, "R%d" % rx
        if opcode in (0x0, 0x1):  # add, sub: RN = RN op RX
            vn, vx = state.get(Rn), state.get(Rx)
            if vn is not None and vx is not None:
                return {Rn: ((vn + vx) if opcode == 0x0 else (vn - vx)) & 0xFFFFFFFF}
        elif opcode == 0x2:  # pass: RN = RX
            vx = state.get(Rx)
            if vx is not None:
                return {Rn: vx}
        elif opcode in (0x5, 0x6):  # inc, dec: RN = RN +- 1
            vn = state.get(Rn)
            if vn is not None:
                return {Rn: ((vn + 1) if opcode == 0x5 else (vn - 1)) & 0xFFFFFFFF}
        return {}
    return {}


def _ptr_reg_updates(t, f, state):
    """New constant values from a pure register-to-register form: literal
    load (17a/17b), MODIFY (19a/19a_scaled/19a_bitrev) or move (5a/5b_move).
    {} for anything else (memory forms are _ptr_mem_form's job)."""
    if t in ("17a", "17b"):
        literal = extract_literal(t, f)
        if literal is None:
            return {}
        value, dest = literal
        return {dest: value & 0xFFFFFFFF} if dest is not None else {}

    if t in ("19a", "19a_scaled", "19a_bitrev"):
        if t == "19a_bitrev":
            return {}  # never a plain add -- see the module note.
        bank = 8 if f.get("g") else 0
        src_low, dis_low = f.get("is", 0), f.get("idis", 0)
        src, dst = "I%d" % (bank + src_low), "I%d" % (bank + (src_low ^ dis_low))
        base = state.get(src)
        if base is None:
            return {}
        delta = sharcfn.sign_extend(f.get("data", 0), 32)
        if t == "19a_scaled":
            delta *= 4 if f.get("w") else 2
        return {dst: (base + delta) & 0xFFFFFFFF}

    if t in ("5a_move", "5b_move"):
        dst = f.get("dstureg")
        if dst is None:
            return {}
        src_hi, src_lo = f.get("srcureghigh", 0), f.get("srcureglow", 0) & 3
        value = state.get(sharcfn.ureg_name((src_hi << 2) | src_lo))
        return {sharcfn.ureg_name(dst): value} if value is not None else {}

    return {}


def _ptr_access_reg(f):
    """The register on the OTHER side of a memory access (the one loaded
    from / stored to memory) -- same dreg-else-ureg fallback as
    _mem_regdef_reguse, reused here for the DM(I6+k) frame-slot case only."""
    dreg, ureg = f.get("dreg"), f.get("ureg")
    if dreg is not None:
        return "R%d" % dreg
    if ureg is not None:
        return sharcfn.ureg_name(ureg)
    return None


def _ptr_frame_access(frame, off, direction, reg, state):
    """DM(I6+off)'s own store/load of a frame slot keyed by offset alone --
    I6's own value is never consulted (see the module note). `frame` is
    mutated in place; -> the register's new value for a load, or None (a
    store, an unknown reg, or a not-yet-known slot). The caller folds a
    non-None result into its own new_values rather than writing `state`
    directly here: _ptr_exec's blanket "pop every register_effects() def"
    step runs AFTER this function returns and would otherwise immediately
    discard a direct write (confirmed against FUN_1c642a's own I6-4
    spill/reload -- see _build_ptr_rows's module note)."""
    if reg is None:
        return None
    if direction == "store":
        value = state.get(reg)
        if value is not None:
            frame[off] = value
        else:
            frame.pop(off, None)
        return None
    return frame.get(off)


def _ptr_mem_form(t, f, state, frame):
    """(new_values, ptr_row) for one memory-referencing instruction.
    new_values only ever names the base I register, for a post-modify commit
    (INDEXED/IMMOFF, u=1); ptr_row is a (base_reg, base_value, address,
    direction, width) tuple for the `ptr` table, or None when this access's
    base is I6 (frame-slot bookkeeping only -- see _ptr_frame_access) or
    nothing about it is constant at all."""
    if t in ("14a", "14d"):
        _space, direction = sharcfn._space_dir(f)
        width = "long" if f.get("l") else "word"
        address = f.get("addr", 0) & 0xFFFFFFFF
        return {}, (None, None, address, direction, width)

    if t == "15a":
        base_reg = _ptr_bank_reg("I", f.get("i", 0), f)
        _space, direction = sharcfn._space_dir(f)
        width = "long" if f.get("l") else "word"
        off = (f.get("addr", 0) & 0xFFFFFFFF) * 4 & 0xFFFFFFFF
        if base_reg == "I6":
            reg = _ptr_access_reg(f)
            value = _ptr_frame_access(frame, off, direction, reg, state)
            return ({reg: value} if value is not None else {}), None
        base_value = state.get(base_reg)
        if base_value is None:
            return {}, None
        return {}, (
            base_reg,
            base_value,
            (base_value + off) & 0xFFFFFFFF,
            direction,
            width,
        )

    if t == "3c":
        # The same indexed-post-modify addressing as INDEXED_MEM_FORMS
        # below, under a narrower encoding (always DM, always post-modify,
        # a plain R-register dreg -- no g/u/ureg fields at all; see
        # tools/sharc_dev/sharc_trace.py's own "3c" handler, which this mirrors
        # exactly, and _mem_regdef_reguse's docstring on why it isn't a
        # member of sharcfn.INDEXED_MEM_FORMS).
        base_reg, mod_reg = "I%d" % f.get("dmi", 0), "M%d" % f.get("dmm", 0)
        direction = "store" if f.get("d") else "load"
        base_value, mod_value = state.get(base_reg), state.get(mod_reg)
        if base_value is None:
            return {}, None
        new_values = {}
        if mod_value is not None:
            new_values[base_reg] = (base_value + mod_value * 4) & 0xFFFFFFFF
        return new_values, (base_reg, base_value, base_value, direction, "word")

    if t in sharcfn.INDEXED_MEM_FORMS:
        base_reg = _ptr_bank_reg("I", f.get("i", 0), f)
        mod_reg = _ptr_bank_reg("M", f.get("m", 0), f)
        u = bool(f.get("u"))
        _space, direction = sharcfn._space_dir(f)
        base_value, mod_value = state.get(base_reg), state.get(mod_reg)
        if base_value is None:
            return {}, None
        width = (
            extract_mem_access(t, f)[0][6]
            if t in ("3b", "3d")
            else ("long" if f.get("l") else "word")
        )
        if width is None or width.startswith("unknown("):
            # Unknown post-modify must invalidate the bank-adjusted I register too.
            return ({base_reg: None} if u else {}), (
                base_reg,
                base_value,
                base_value if u else None,
                direction,
                width,
            )
        access_width = {"word": "normal-word", "long": "long-word"}.get(width, width)
        # Retain this pass's byte-address/NW32 assumption; (lw) also steps by four.
        scale = _access_modifier_scale(access_width, True)
        if u:
            new_values = {base_reg: None}
            if mod_value is not None:
                new_values[base_reg] = (base_value + mod_value * scale) & 0xFFFFFFFF
            return new_values, (base_reg, base_value, base_value, direction, width)
        address = (
            (base_value + mod_value * scale) & 0xFFFFFFFF
            if mod_value is not None
            else None
        )
        return {}, (base_reg, base_value, address, direction, width)

    if t in sharcfn.IMMOFF_MEM_FORMS:
        base_reg = _ptr_bank_reg("I", f.get("i", 0), f)
        bits = sharcfn._IMMOFF_DATA_BITS.get(t, 6)
        scale = 8 if (t == "15b" and f.get("l")) else 4
        off = (sharcfn.sign_extend(f.get("data", 0), bits) * scale) & 0xFFFFFFFF
        u = bool(f.get("u")) if t != "15b" else False
        _space, direction = sharcfn._space_dir(f)
        # width is descriptive only here (address arithmetic above uses
        # `scale`/`off`, not width): use the same l/x/w -> label mapping as
        # extract_mem_access so 4b/4d aren't mislabeled "word"/"long" (see
        # _immoff_width).
        width = _immoff_width(t, f)
        if base_reg == "I6":
            reg = _ptr_access_reg(f)
            value = _ptr_frame_access(frame, off, direction, reg, state)
            return ({reg: value} if value is not None else {}), None
        base_value = state.get(base_reg)
        if base_value is None:
            return {}, None
        if u:
            return {base_reg: (base_value + off) & 0xFFFFFFFF}, (
                base_reg,
                base_value,
                base_value,
                direction,
                width,
            )
        return {}, (
            base_reg,
            base_value,
            (base_value + off) & 0xFFFFFFFF,
            direction,
            width,
        )

    return {}, None


def _ptr_exec(t, f, state, frame):
    """Advance one instruction's effect on `state` (register name -> known
    32-bit value) and `frame` (I6+offset -> known 32-bit value), both
    mutated in place; -> a `ptr` row tuple or None. Every register
    register_effects() says this instruction DEFINES is either given a
    modelled new value or invalidated -- never left holding a stale value
    from before this instruction, even for a form/opcode this pass does not
    model itself. (register_effects' own I-register naming is g-bank-
    unadjusted for a post-modify def -- see the module note; popping that
    name too, alongside applying this function's own correctly-banked one,
    is at worst an extra, harmless invalidation of an unrelated register,
    never a wrong value.)"""
    defs, _uses, _unknown = register_effects(t, f)
    new_values = dict(_ptr_alu_updates(t, f, state))
    new_values.update(_ptr_reg_updates(t, f, state))
    mem_updates, ptr_row = _ptr_mem_form(t, f, state, frame)
    new_values.update(mem_updates)
    for reg, _kind in defs:
        state.pop(reg, None)
    state.update(new_values)
    return ptr_row


def _ptr_merge(states):
    """A block's in-state (register state or frame-slot state alike): a key
    survives only when every already-visited predecessor's out-state has it
    with the SAME value."""
    if not states:
        return {}
    common = set(states[0])
    for s in states[1:]:
        common &= set(s)
    return {
        k: states[0][k] for k in common if all(s[k] == states[0][k] for s in states)
    }


def _ptr_topo_order(entry, starts, succ_by_from, idom):
    """`starts` (one function's bblock start_sw's) in a forward order: a
    plain topological sort of that function's own block graph with loop
    back edges dropped, reusing analyze_image()'s already-computed `idom`
    for the back-edge test (v dominates u -- _detect_dominators_and_loops's
    own test) instead of re-deriving dominance here. Any block the result
    doesn't order (unreachable from `entry`, or on a residual cycle an
    irreducible CFG's single-entry idom doesn't fully break) is appended in
    address order -- _build_ptr_rows's merge only ever consumes an already-
    visited predecessor, so this is a safe (if occasionally lossy)
    approximation, never a wrong one."""

    def dominates(target, node):
        n = node
        while True:
            if n == target:
                return True
            if n == entry:
                return False
            n = idom.get(n, entry)

    block_set = set(starts)
    G = nx.DiGraph()
    G.add_nodes_from(starts)
    for b in starts:
        for to_block, _kind in succ_by_from.get(b, ()):
            if to_block in block_set and not dominates(to_block, b):
                G.add_edge(b, to_block)
    try:
        order = list(nx.topological_sort(G))
    except nx.NetworkXUnfeasible:
        return sorted(starts)
    seen = set(order)
    order += sorted(x for x in starts if x not in seen)
    return order


def _build_ptr_rows(db, name, idom_rows):
    """[(sw, base_reg, base_value, address, direction, width), ...] -- the
    `ptr` table's own rows for one image. See the long module note above
    this section for what is and is not modelled."""
    insns = db.execute(
        "SELECT sw, form, fields FROM insn WHERE image=? AND aligned=1 ORDER BY sw",
        (name,),
    ).fetchall()
    decoded = {
        sw: (form, sharcinv.merge_fields(json.loads(fields_json)))
        for sw, form, fields_json in insns
        if form is not None and fields_json is not None
    }
    if not decoded:
        return []

    bblock_rows = db.execute(
        "SELECT start_sw, end_sw, function_sw FROM bblocks WHERE image=?", (name,)
    ).fetchall()
    if not bblock_rows:
        return []  # a ColdFire import: no per-instruction CFG survives it at all.

    by_func = collections.defaultdict(list)
    block_end = {}
    for start, end, fn in bblock_rows:
        block_end[start] = end
        if fn is not None:
            by_func[fn].append(start)
    for starts in by_func.values():
        starts.sort()

    succ_by_from = collections.defaultdict(list)
    preds_by_to = collections.defaultdict(list)
    for from_block, to_block, kind in db.execute(
        "SELECT from_block, to_block, kind FROM succ WHERE image=?", (name,)
    ).fetchall():
        if to_block is not None:
            succ_by_from[from_block].append((to_block, kind))
            preds_by_to[to_block].append(from_block)

    starts_sorted = sorted(block_end)
    block_members = collections.defaultdict(list)
    for sw in sorted(decoded):
        i = bisect.bisect_right(starts_sorted, sw) - 1
        if i < 0:
            continue
        start = starts_sorted[i]
        if start <= sw < block_end[start]:
            block_members[start].append(sw)

    calls_out, callers_in = collections.defaultdict(set), collections.defaultdict(list)
    for from_sw, from_fn, to_fn in db.execute(
        "SELECT from_sw, from_function, to_function FROM edges WHERE image=? AND kind='call' "
        "AND from_function IS NOT NULL AND to_function IS NOT NULL",
        (name,),
    ).fetchall():
        calls_out[from_fn].add(from_sw)
        callers_in[to_fn].append((from_sw, from_fn))

    idom_by_func = collections.defaultdict(dict)
    for fn, block, idom_block in idom_rows:
        idom_by_func[fn][block] = idom_block
    topo_by_func = {
        fn: _ptr_topo_order(fn, starts, succ_by_from, idom_by_func.get(fn, {}))
        for fn, starts in by_func.items()
    }

    seed, ptr_rows = {}, []
    for _round in range(_PTR_ROUNDS):
        call_state = {}
        block_out, block_out_frame, block_confident = {}, {}, {}
        ptr_rows = []
        for fn, starts in by_func.items():
            block_set = set(starts)
            call_sws = calls_out.get(fn, ())
            for start in topo_by_func[fn]:
                # A block with no CONFIDENT predecessor -- the true entry,
                # or one only an unresolved indirect jump reaches (a common
                # dispatch-table-case shape; succ/bblocks has no edge into
                # it at all, since a JUMP through a register carries no
                # static target -- see FUN_1c642a's own table-driven
                # dispatch) -- falls back to the SAME entry baseline
                # (M-reset + this round's call-site seed) for its register
                # state, but an EMPTY frame (a case's own stack use, if any,
                # is not this pass's business) and is itself marked NOT
                # confident: unlike a real merge input, its guessed baseline
                # must never be allowed to zero out a genuinely-tracked
                # value at some later, real reconvergence point downstream
                # (confirmed against FUN_1c642a directly: without this
                # distinction, one indirect-dispatch case's contentless
                # state wiped the whole function's frame tracking from that
                # point on).
                preds = [
                    p
                    for p in preds_by_to.get(start, ())
                    if p in block_set and p in block_out and block_confident.get(p)
                ]
                if start == fn or not preds:
                    state = dict(_PTR_M_RESET)
                    state.update(seed.get(fn, {}))
                    frame = {}
                    confident = start == fn
                else:
                    state = _ptr_merge([block_out[p] for p in preds])
                    frame = _ptr_merge([block_out_frame[p] for p in preds])
                    confident = True
                for sw in block_members.get(start, ()):
                    if sw in call_sws:
                        call_state[(fn, sw)] = {
                            r: state[r] for r in _PTR_CALL_ARG_REGS if r in state
                        }
                    form, f = decoded[sw]
                    row = _ptr_exec(form, f, state, frame)
                    if row is not None:
                        ptr_rows.append((sw,) + row)
                block_out[start], block_out_frame[start], block_confident[start] = (
                    state,
                    frame,
                    confident,
                )

        seed = {}
        for fn, calls in callers_in.items():
            snaps = [
                call_state.get((from_fn, call_sw), {}) for call_sw, from_fn in calls
            ]
            agree = {}
            for reg in _PTR_CALL_ARG_REGS:
                values = [s.get(reg) for s in snaps]
                if values and all(v is not None and v == values[0] for v in values):
                    agree[reg] = values[0]
            if agree:
                seed[fn] = agree

    return ptr_rows
