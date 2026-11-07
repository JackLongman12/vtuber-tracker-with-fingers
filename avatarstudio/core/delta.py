"""Small updates: a patch that turns the previous release's AvatarStudio.exe into the new one, byte for byte.

The one-file app is a launcher with a zip of the app appended; between two releases most files in that zip (the
models, Python's runtime) are the same, and the same file compresses to the same bytes. A patch lists the new exe as
pieces: "copy these bytes from your current exe" for every file that didn't change, and the bytes themselves for the
rest. tools/release.py makes one from the previous release (make()); the updater applies it to the running exe
(apply()) and the result is checked against the full exe's checksum in SHA256SUMS.txt, so a patched update is exactly
the file a full download gives. Anything that doesn't fit (another version, a changed exe, a bad patch) falls back
to the full download.

Patch file: b"AVSDELTA1\\n", 8 bytes little-endian header length, the header (JSON: base_sha256, size, sha256,
ops = [["C", offset in the base exe, length] | ["L", offset in the literal bytes, length], ...]), then the literal
bytes.
"""
import hashlib
import json
import os
import struct
import zipfile

MAGIC = b"AVSDELTA1\n"
CHUNK = 1 << 20


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def _entries(path):
    """Each zip entry's compressed bytes in the file: {(name, crc, size, compress_size, method): (start, length)}."""
    out = {}
    with zipfile.ZipFile(path) as z, open(path, "rb") as f:
        for info in z.infolist():
            if info.compress_size == 0:
                continue
            f.seek(info.header_offset)
            head = f.read(30)
            if head[:4] != b"PK\x03\x04":
                continue
            n, m = struct.unpack("<HH", head[26:30])
            start = info.header_offset + 30 + n + m
            key = (info.filename, info.CRC, info.file_size, info.compress_size, info.compress_type)
            out[key] = (start, info.compress_size)
    return out


def _same(f_old, f_new, a, b, length):
    f_old.seek(a)
    f_new.seek(b)
    left = length
    while left:
        n = min(CHUNK, left)
        if f_old.read(n) != f_new.read(n):
            return False
        left -= n
    return True


def make(base, new, out):
    """A patch from `base` (the previous release's exe) to `new`, written to `out`. -> (patch size, bytes copied)."""
    old_entries, new_entries = _entries(base), _entries(new)
    size = os.path.getsize(new)
    copies = []  # (start in new, start in base, length), for unchanged files, checked byte for byte
    with open(base, "rb") as fo, open(new, "rb") as fn:
        for key, (start, length) in new_entries.items():
            hit = old_entries.get(key)
            if hit and length >= 4096 and _same(fo, fn, hit[0], start, length):
                copies.append((start, hit[0], length))
    copies.sort()
    ops, literal_len, pos = [], 0, 0
    tmp = out + ".partial"
    with open(new, "rb") as fn, open(tmp + ".lit", "wb") as lit:
        def literal(a, b):
            nonlocal literal_len
            if b <= a:
                return
            fn.seek(a)
            left = b - a
            while left:
                block = fn.read(min(CHUNK, left))
                lit.write(block)
                left -= len(block)
            if ops and ops[-1][0] == "L" and ops[-1][1] + ops[-1][2] == literal_len:
                ops[-1][2] += b - a
            else:
                ops.append(["L", literal_len, b - a])
            literal_len += b - a

        for start, old_start, length in copies:
            literal(pos, start)
            ops.append(["C", old_start, length])
            pos = start + length
        literal(pos, size)
    header = json.dumps({"base_sha256": sha256(base), "size": size, "sha256": sha256(new), "ops": ops}).encode()
    with open(tmp, "wb") as f, open(tmp + ".lit", "rb") as lit:
        f.write(MAGIC + struct.pack("<Q", len(header)) + header)
        for block in iter(lambda: lit.read(CHUNK), b""):
            f.write(block)
    os.remove(tmp + ".lit")
    os.replace(tmp, out)
    return os.path.getsize(out), sum(c[2] for c in copies)


def read_header(patch):
    with open(patch, "rb") as f:
        if f.read(len(MAGIC)) != MAGIC:
            raise ValueError("not an Avatar Studio patch")
        (n,) = struct.unpack("<Q", f.read(8))
        if n > 64 << 20:
            raise ValueError("damaged patch")
        header = json.loads(f.read(n))
        return header, len(MAGIC) + 8 + n


def apply(base, patch, out, expected_sha256, progress=None):
    """Rebuild the new exe from `base` (the running one) and `patch` into `out`; it must come out as
    `expected_sha256` (the full exe's line in SHA256SUMS.txt), else nothing is left behind. progress(done, total)."""
    header, lit_start = read_header(patch)
    if header.get("sha256", "").lower() != expected_sha256.lower():
        raise ValueError("the patch is for another build")
    if sha256(base) != header.get("base_sha256"):
        raise ValueError("this patch is for another version of the app")
    total, done, digest = int(header["size"]), 0, hashlib.sha256()
    try:
        with open(base, "rb") as fb, open(patch, "rb") as fp, open(out, "wb") as fo:
            for kind, offset, length in header["ops"]:
                if kind not in ("C", "L") or offset < 0 or length < 0 or done + length > total:
                    raise ValueError("damaged patch")
                src = fb if kind == "C" else fp
                src.seek(offset if kind == "C" else lit_start + offset)
                left = length
                while left:
                    block = src.read(min(CHUNK, left))
                    if not block:
                        raise ValueError("damaged patch")
                    fo.write(block)
                    digest.update(block)
                    left -= len(block)
                done += length
                if progress:
                    progress(done, total)
        if done != total or digest.hexdigest() != expected_sha256.lower():
            raise ValueError("the patched update doesn't match its checksum")
    except BaseException:
        try:
            os.remove(out)
        except OSError:
            pass
        raise
    return out
