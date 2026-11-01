"""
Dynamical Lie algebra (Pauli-string closure), Cartan split and structure constants
for any number of qubits.

Representation
--------------
A Pauli string is stored as L = ceil(n / 32) uint64 "limbs"; limb i holds qubits 32i .. 32i+31:

        limb = (x_bits << 32) | z_bits          per qubit (x, z):  I=(0,0)  X=(1,0)  Z=(0,1)  Y=(1,1)

    n <= 32 : keys have shape (N,)      one uint64 per string (8 bytes / element)
    n  > 32 : keys have shape (N, L)    2n bits per string, rounded up to whole limbs

Labels follow Qiskit's convention (left-most character = highest qubit).

    product (up to phase)  ->  XOR of the keys
    anticommute            ->  popcount(a & swap(b)) is odd   (summed over limbs; swap exchanges x / z halves)
    #Y in a string         ->  popcount((key >> 32) & key)    (summed over limbs)

The algebra is returned as one sorted key array (n <= 32: numeric order, n > 32: raw-byte order of
the rows), so it is its own index: see index_of().
"""
import numpy as np

_S = np.uint64(32)
_LO = np.uint64(0xFFFFFFFF)
_DENSE_MAX_QUBITS = 12            # dense closure uses a 4**n-byte table (<= 16 MiB)

# --------------------------------------------------------------------------- bit helpers
if hasattr(np, "bitwise_count"):                       # NumPy >= 2.0 (hardware POPCNT)
    _popcount = np.bitwise_count
else:                                                  # SWAR fallback for NumPy 1.x
    def _popcount(v):
        u = np.uint64
        v = v - ((v >> u(1)) & u(0x5555555555555555))
        v = (v & u(0x3333333333333333)) + ((v >> u(2)) & u(0x3333333333333333))
        v = (v + (v >> u(4))) & u(0x0F0F0F0F0F0F0F0F)
        return (v * u(0x0101010101010101)) >> u(56)


def _limbs(a):
    """Number of limbs of a key array."""
    return 1 if a.ndim == 1 else a.shape[1]


def _swap(k):
    """(x, z) -> (z, x) in every limb, so that x1.z2 ^ z1.x2 becomes a single AND."""
    return (k << _S) | (k >> _S)


def _parity(a, L):
    """Bool array: parity of the total popcount of each key (limb axis last when L > 1)."""
    if L > 1:
        a = np.bitwise_xor.reduce(a, axis=-1)           # parity of a sum of popcounts = popcount of the XOR
    return (_popcount(a) & 1).astype(bool)


def _rows(a):
    """1-D array with one sortable item per key: the array itself for single-limb keys, otherwise a
    zero-copy fixed-width bytes view of the rows (exact, and several times faster to sort/search than void/structured)."""
    if a.ndim == 1:
        return a
    return a.view(f"S{8 * a.shape[1]}").reshape(-1)


def _unrows(r, L):
    return r if L == 1 else r.view(np.uint64).reshape(-1, L)


def _sort_unique(a):
    """Sort the keys of `a` in place (C-contiguous) and drop duplicates.  A plain sort is much faster here
    than np.unique, whose hash-based path (NumPy >= 2.3) is slow on large, duplicate-heavy arrays."""
    r = _rows(a)
    r.sort()
    if len(r) < 2:
        return a
    keep = np.empty(len(r), bool)
    keep[0] = True
    np.not_equal(r[1:], r[:-1], out=keep[1:])
    return a[keep]


def _contains(a, q):
    """Bool mask: which keys of q occur in the sorted key array a."""
    if len(a) == 0:
        return np.zeros(len(q), bool)
    ra, rq = _rows(a), _rows(q)
    pos = np.searchsorted(ra, rq)
    np.minimum(pos, len(ra) - 1, out=pos)
    return ra[pos] == rq


def _union(a, b):
    """Union of two sorted, duplicate-free key arrays in one O(N) pass (no re-sort)."""
    if len(b) == 0:
        return a
    if len(a) == 0:
        return b
    ra, rb = _rows(a), _rows(b)
    pos = np.searchsorted(ra, rb)
    fresh = ra[np.minimum(pos, len(ra) - 1)] != rb
    return np.insert(a, pos[fresh], b[fresh], axis=0)


# --------------------------------------------------------------------------- labels <-> keys
def pack_labels(labels):
    """['XXI', 'IZZ', ...] -> (keys, n_qubits) for any n.  Phase prefixes (+, -, i) are ignored."""
    labels = [s.lstrip("+-i") for s in labels]
    if not labels:
        raise ValueError("no labels given")
    n = len(labels[0])
    if n == 0 or any(len(s) != n for s in labels):
        raise ValueError("labels must be non-empty and all have the same length")
    # column q of `ch` = qubit q  (the last character of a label is qubit 0)
    ch = np.frombuffer("".join(labels).encode("ascii", "replace"), np.uint8).reshape(len(labels), n)[:, ::-1]
    x = (ch == ord("X")) | (ch == ord("Y"))
    z = (ch == ord("Z")) | (ch == ord("Y"))
    if not (x | z | (ch == ord("I"))).all():
        raise ValueError("labels may only contain I, X, Y, Z")
    L = -(-n // 32)

    def limbs(bits):                                    # (N, n) bool -> (N, L) uint64, bit q%32 of limb q//32
        packed = np.packbits(bits, axis=1, bitorder="little")
        padded = np.zeros((len(bits), 4 * L), np.uint8)
        padded[:, :packed.shape[1]] = packed
        return padded.view("<u4").astype(np.uint64)

    keys = (limbs(x) << _S) | limbs(z)
    return (keys[:, 0].copy() if L == 1 else keys), n


def to_labels(keys, n, chunk=1 << 21):
    """keys -> array of label strings (display / debugging only; temporaries are chunked)."""
    keys = np.asarray(keys, np.uint64)
    if keys.ndim == 1:
        keys = keys[:, None]
    if keys.shape[1] != -(-n // 32):
        raise ValueError(f"{keys.shape[1]} limbs per key do not match n = {n} qubits")
    q = np.arange(n - 1, -1, -1)                        # label position -> qubit (left-most = n-1)
    limb, bit = q // 32, (q % 32).astype(np.uint64)
    table = np.frombuffer(b"IXZY", np.uint8)            # index = x + 2z
    one = np.uint64(1)
    out = np.empty(len(keys), f"U{n}")
    step = max(1, chunk // n)
    for i in range(0, len(keys), step):
        k = keys[i:i + step][:, limb]                   # (c, n): the limb that holds each qubit
        z = ((k >> bit) & one).astype(np.uint8)
        x = ((k >> (bit + _S)) & one).astype(np.uint8)
        out[i:i + step] = np.ascontiguousarray(table[x + 2 * z]).view(f"S{n}").ravel()
    return out


def index_of(basis, keys):
    """Positions of `keys` inside the sorted `basis` (-1 where absent)."""
    basis = np.ascontiguousarray(basis, np.uint64)
    L = _limbs(basis)
    keys = np.ascontiguousarray(keys, np.uint64).reshape((-1,) if L == 1 else (-1, L))
    rb, rk = _rows(basis), _rows(keys)
    pos = np.minimum(np.searchsorted(rb, rk), len(rb) - 1)
    return np.where(rb[pos] == rk, pos, -1)


# --------------------------------------------------------------------------- Lie closure
def dla_basis(gen_keys, dense=None):
    """
    Sorted key array of every Pauli string in the Lie algebra generated by `gen_keys`
    (shape (N,) for n <= 32 qubits, (N, L) otherwise; the result has the same layout).

    The algebra is spanned by left-nested commutators [g1,[g2,[...,gk]]] of generators, so a newly
    found string only has to be combined with the *generators* (BFS over Pauli strings), never with
    the whole algebra:  O(|G| * dim)  instead of O(dim^2).  Phases do not affect the span, so only
    the XOR of keys is needed.

    dense=None chooses automatically: a direct-address table when the generators fit in 12 qubits
    (fastest), sorted arrays otherwise (any number of qubits).  Both give identical output.
    """
    gens = np.ascontiguousarray(gen_keys, np.uint64)
    if gens.ndim == 2 and gens.shape[1] == 1:
        gens = gens[:, 0]
    if gens.ndim not in (1, 2) or len(gens) == 0:
        raise ValueError("need at least one generator key")
    L = _limbs(gens)
    gens = _unrows(np.unique(_rows(gens)), L)                       # sorted, duplicate-free

    if L == 1:
        used = int(np.bitwise_or.reduce(gens))
        n = max((used >> 32).bit_length(), (used & 0xFFFFFFFF).bit_length(), 1)   # qubits actually touched
        if dense is None:
            dense = n <= _DENSE_MAX_QUBITS
        if dense:
            if n > 15:
                raise ValueError("dense closure would need a table of more than 1 GiB")
            return _closure_dense(gens, n)
    elif dense:
        raise ValueError("dense closure only supports at most 32 qubits")
    return _closure_sorted(gens)


def _closure_dense(gens, n):
    """BFS with a boolean table over all 4**n strings: O(1) dedup, no sorting."""
    N, mask = np.uint64(n), np.uint64((1 << n) - 1)
    g = ((gens >> _S) << N) | (gens & _LO)                  # compact key (x << n) | z  in [0, 4**n)
    g_sw = ((g & mask) << N) | (g >> N)
    seen = np.zeros(1 << (2 * n), bool)                     # lazily allocated by the OS
    seen[g] = True
    frontier = g
    while frontier.size:
        parts = []
        for k, k_sw in zip(g, g_sw):
            hit = _parity(frontier & k_sw, 1)               # frontier terms anticommuting with k
            cand = frontier[hit]
            cand ^= k                                       # [k, f] ∝ k·f
            cand = cand[~seen[cand]]
            seen[cand] = True                               # mark now: later generators skip it
            parts.append(cand)
        frontier = np.concatenate(parts)
    d = np.flatnonzero(seen).view(np.uint64)
    hi = d >> N                                             # back to (x << 32) | z, in place; order is kept
    hi <<= _S
    d &= mask
    d |= hi
    return d


def _closure_sorted(gens, budget=1 << 21):
    """
    BFS keeping only sorted key arrays; every generator is applied to a batch of frontier strings at once.

    Multiplying by a generator is an involution on its anticommuting set, so the graph is undirected and
    BFS levels of neighbours differ by <= 1: a candidate is old iff it lies in level r-1 or r, so
    membership is tested against that small window instead of the whole algebra.
    """
    L = _limbs(gens)
    gens_sw = _swap(gens)
    step = max(1, budget // (len(gens) * L))                # frontier rows per batch (bounds temporaries)
    levels = [gens]
    prev, frontier = gens[:0], gens
    while len(frontier):
        window = _union(prev, frontier)                     # levels r-1 and r
        new = gens[:0]
        for lo in range(0, len(frontier), step):
            f = frontier[lo:lo + step]
            fi, gi = np.nonzero(_parity(f[:, None] & gens_sw[None], L))   # f[fi] anticommutes with gens[gi]
            cand = f[fi] ^ gens[gi]                                       # [g, f] ∝ g·f
            cand = _sort_unique(cand)                                     # sorted, duplicate-free
            cand = cand[~_contains(window, cand)]                         # drop strings already known
            new = _union(new, cand)                                       # also dedups across batches
        if len(new):
            levels.append(new)
        prev, frontier = frontier, new
    basis = np.concatenate(levels)
    _rows(basis).sort()
    return basis


# --------------------------------------------------------------------------- Cartan split
def cartan_split(basis):
    """
    Index arrays into `basis` for the even-#Y part (m) and the odd-#Y part (l);
    same convention and return order as the original: (m_idx, l_idx).
    """
    basis = np.ascontiguousarray(basis, np.uint64)
    odd = _parity((basis >> _S) & basis, _limbs(basis))
    return np.flatnonzero(~odd), np.flatnonzero(odd)


# --------------------------------------------------------------------------- structure constants
def structure_constants(basis, control, dtype=np.float64):
    """
    COO triplets (rows, cols, vals) of the matrix F defined by

        [ i T_j , i C ] = sum_k F[k, j] * (i T_k)       T = Pauli strings of `basis`, C = `control` key

    Only T_j that anticommute with C contribute, with F[k, j] = ±2 and T_k ∝ T_j·C.
    Sign convention identical to the original code (qiskit phase 1 -> +2, phase 3 -> -2).
    `control` is one key (a scalar for n <= 32, a row of L limbs otherwise) and must belong to the
    algebra, e.g. one of the generators.

        F = scipy.sparse.coo_matrix((vals, (rows, cols)), shape=(len(basis),) * 2)
    """
    basis = np.ascontiguousarray(basis, np.uint64)
    L = _limbs(basis)
    c = np.asarray(control, np.uint64).reshape((1,) if L == 1 else (1, L))   # one-row array: safe broadcasting
    cols = np.flatnonzero(_parity(basis & _swap(c), L))             # j with {T_j, C} = 0
    k1 = basis[cols]
    k3 = k1 ^ c                                                     # string of the product T_j·C

    rb, r3 = _rows(basis), _rows(k3)
    rows = np.searchsorted(rb, r3)
    if not np.array_equal(rb[np.minimum(rows, len(rb) - 1)], r3):
        raise ValueError("control term is not in the algebra: [T_j, C] leaves the basis")

    # T_j·C = i^e T_k  with  e = |y1| + |y2| - |y3| + 2|z1 & x2|  (mod 4);  e is odd for anticommuting pairs.
    # Only e mod 4 matters, so int16 wrap-around in the popcount sums is harmless.
    def pop(a):
        p = _popcount(a).astype(np.int16)
        return p if L == 1 else p.sum(axis=1, dtype=np.int16)

    def ycount(k):
        return pop((k >> _S) & k)

    zx = pop(k1 & (c >> _S))                                        # |z1 & x2|
    e = (ycount(k1) + ycount(c) - ycount(k3) + 2 * zx) & 3
    vals = np.where(e == 1, -2, 2).astype(dtype)                    # e = 1  <=>  T_j·C = +i T_k  ->  F = -2

    idx = np.int32 if len(basis) < 2**31 else np.int64
    return rows.astype(idx), cols.astype(idx), vals


# --------------------------------------------------------------------------- example
if __name__ == "__main__":
    import Config

    keys, n = pack_labels(Config.controlSets[Config.activeSystem])
    basis = dla_basis(keys)
    m_idx, l_idx = cartan_split(basis)

    print(f"n = {n} qubits, dim DLA = {len(basis)}   (dim m = {m_idx.size}, dim l = {l_idx.size})")
    print(to_labels(basis, n))

    # rows, cols, vals = structure_constants(basis, keys[0])