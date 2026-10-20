"""Reader of the training stream `kflash_datagen data` writes (format: the header of tools/kflash_datagen.cpp).

    for kind, rec in read_stream(fileobj): ...   # kind: "header" (dict), "seq", "anchor", "end" (dicts of numpy arrays)

and Example assembly: one anchor with the sequence's tokens and target distributions it needs, self-contained, so a
shuffle buffer can hold anchors of many sequences.
"""
import json, struct

import numpy as np

K, TOPN, NPOS = 32, 64, 7


class StreamError(Exception):
    pass


def _read(f, n):
    b = f.read(n)
    if len(b) != n:
        if not b:
            return None
        raise StreamError(f"truncated stream ({len(b)} of {n} bytes)")
    return b


def read_stream(f):
    magic = _read(f, 8)
    if magic != b"KFSTRM01":
        raise StreamError(f"not a KFlash stream: {magic!r}")
    (n,) = struct.unpack("<I", _read(f, 4))
    hdr = json.loads(_read(f, n))
    H = hdr["H"]
    yield "header", hdr
    while True:
        th = _read(f, 8)
        if th is None:
            return
        tag, ln = th[:4], struct.unpack("<I", th[4:])[0]
        p = _read(f, ln)
        if tag == b"SEQ1":
            seq, flags, n, np_, idl = struct.unpack_from("<IIiiI", p, 0)
            o = 20
            sid = p[o:o + idl].decode("utf-8", "replace")
            o += (idl + 3) // 4 * 4
            toks = np.frombuffer(p, np.int32, n, o)
            o += 4 * n
            nrows = n - 1 - np_
            top_ids = np.frombuffer(p, np.int32, nrows * TOPN, o).reshape(nrows, TOPN)
            o += 4 * nrows * TOPN
            top_p = np.frombuffer(p, np.float32, nrows * TOPN, o).reshape(nrows, TOPN)
            yield "seq", {"seq": seq, "sampled": bool(flags & 1), "n": n, "n_prompt": np_, "id": sid, "tokens": toks,
                          "top_ids": top_ids, "top_p": top_p}
        elif tag == b"ANC1":
            seq, P, npos = struct.unpack_from("<Iii", p, 0)
            o = 12
            h = np.frombuffer(p, np.float16, NPOS * H, o).reshape(NPOS, H)
            o += 2 * NPOS * H
            cand = np.frombuffer(p, np.int32, NPOS * K, o).reshape(NPOS, K)
            o += 4 * NPOS * K
            u = np.frombuffer(p, np.float32, NPOS * K, o).reshape(NPOS, K)
            o += 4 * NPOS * K
            u_tgt = np.frombuffer(p, np.float32, NPOS * TOPN, o).reshape(NPOS, TOPN)
            o += 4 * NPOS * TOPN
            u_next = np.frombuffer(p, np.float32, NPOS, o)
            yield "anchor", {"seq": seq, "P": P, "npos": npos, "h": h, "cand": cand, "u": u, "u_tgt": u_tgt,
                             "u_next": u_next}
        elif tag == b"END1":
            seq, na = struct.unpack("<Ii", p)
            yield "end", {"seq": seq, "anchors": na}
        # unknown tags: skipped (their length is known)


def example(seq, anc):
    """One anchor, self-contained: per block position p = 1..7 (index p - 1)
      h [7, H] f16, cand [7, K] i32, u [7, K] f32 (the drafter's top-32 and raw logits),
      tok_in [7] i32 = t_{p-1} (t_0 = the anchor token), tok_next [7] i32 = t_p, valid [7] bool (p <= npos),
      tgt_ids [7, 64] i32, tgt_p [7, 64] f32 (the target's top-64 of row P + p - 1), u_next [7] f32,
      u_tgt [7, 64] f32 (draft logits of the target's top-64, NaN outside the draft head)."""
    P, npos, np_ = anc["P"], anc["npos"], seq["n_prompt"]
    toks = seq["tokens"]
    tok_in = np.zeros(NPOS, np.int32)
    tok_next = np.zeros(NPOS, np.int32)
    tgt_ids = np.full((NPOS, TOPN), -1, np.int32)
    tgt_p = np.zeros((NPOS, TOPN), np.float32)
    valid = np.zeros(NPOS, bool)
    for q in range(NPOS):
        tok_in[q] = toks[P + q] if P + q < len(toks) else 0
        if q < npos:
            tok_next[q] = toks[P + q + 1]
            row = P + q - np_
            tgt_ids[q] = seq["top_ids"][row]
            tgt_p[q] = seq["top_p"][row]
            valid[q] = True
    return {"seq": anc["seq"], "P": P, "sampled": seq["sampled"], "id": seq["id"], "h": anc["h"], "cand": anc["cand"],
            "u": anc["u"], "u_tgt": anc["u_tgt"], "u_next": anc["u_next"], "tok_in": tok_in, "tok_next": tok_next,
            "valid": valid, "tgt_ids": tgt_ids, "tgt_p": tgt_p}


def examples(f):
    """(header, iterator of examples) over a stream, anchors joined with their sequence."""
    it = read_stream(f)
    kind, hdr = next(it)

    def gen():
        cur = {}
        for kind, rec in it:
            if kind == "seq":
                cur[rec["seq"]] = rec
            elif kind == "anchor":
                yield example(cur[rec["seq"]], rec)
            elif kind == "end":
                cur.pop(rec["seq"], None)
    return hdr, gen()
