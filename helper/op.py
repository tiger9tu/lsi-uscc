class Op:
    """Sum of fermionic operator strings built from meta operators."""
    def __init__(self, terms=None):
        self.terms = []
        if terms:
            for coeff, ops in terms:
                self.add_term(coeff, ops)

    def add_term(self, coeff, ops):
        if coeff == 0:
            return
        normalized = tuple((kind, int(idx)) for kind, idx in (ops or ()))
        self.terms.append((complex(coeff), normalized))

    def dagger(self):
        conj_terms = []
        for coeff, ops in self.terms:
            rev_ops = []
            for kind, idx in reversed(ops):
                if kind == "create":
                    mapped = "annihilate"
                elif kind == "annihilate":
                    mapped = "create"
                else:
                    mapped = kind
                rev_ops.append((mapped, idx))
            conj_terms.append((np.conjugate(coeff), tuple(rev_ops)))
        return Op(conj_terms)

    def conjugate(self):
        return self.dagger()

    def __add__(self, other):
        if not isinstance(other, Op):
            return NotImplemented
        result = Op()
        for coeff, ops in self.terms + other.terms:
            result.add_term(coeff, ops)
        return result

    def __mul__(self, other):
        if isinstance(other, Op):
            result = Op()
            for coeff_a, ops_a in self.terms:
                for coeff_b, ops_b in other.terms:
                    result.add_term(coeff_a * coeff_b, ops_a + ops_b)
            return result
        elif isinstance(other, (int, float, complex)):
            result = Op()
            for coeff, ops in self.terms:
                result.add_term(coeff * other, ops)
            return result
        return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    def apply(self, ci):
        out = np.zeros_like(ci, dtype=complex)
        for coeff, ops in self.terms:
            tmp = ci
            for kind, idx in ops:
                tmp = self._apply_meta(tmp, kind, idx)
                if tmp is None:
                    break
            if tmp is None:
                continue
            out += coeff * tmp
        return out

    @staticmethod
    def _apply_meta(ci, kind, orb_idx):
        if kind in ("identity", None):
            return ci
        if kind == "number":
            return Op._apply_number(ci, orb_idx)
        if kind in ("create", "annihilate"):
            return Op._apply_ladder(ci, orb_idx, kind == "create")
        raise ValueError(f"Unknown meta operator '{kind}'")

    @staticmethod
    def _apply_number(ci, orb_idx):
        n_spin = Op._n_spin_orbitals(ci)
        if orb_idx >= n_spin:
            raise ValueError("Orbital index out of range")
        mask = 1 << orb_idx
        res = np.zeros_like(ci, dtype=complex)
        for det_idx, amp in enumerate(ci):
            if amp != 0 and det_idx & mask:
                res[det_idx] = amp
        return res

    @staticmethod
    def _apply_ladder(ci, orb_idx, is_creation):
        n_spin = Op._n_spin_orbitals(ci)
        if orb_idx >= n_spin:
            raise ValueError("Orbital index out of range")
        mask = 1 << orb_idx
        res = np.zeros_like(ci, dtype=complex)
        for det_idx, amp in enumerate(ci):
            if abs(amp) < 1e-8:
                continue
            occupied = bool(det_idx & mask)
            if is_creation and occupied:
                continue
            if (not is_creation) and (not occupied):
                continue
            phase = -1 if bin(det_idx & (mask - 1)).count("1") % 2 else 1
            new_det = (det_idx | mask) if is_creation else (det_idx & ~mask)
            res[new_det] += phase * amp
        return res

    @staticmethod
    def _n_spin_orbitals(ci):
        size = ci.size
        n_spin = int(np.log2(size))
        if (1 << n_spin) != size:
            raise ValueError("CI vector dimension must be a power of two")
        return n_spin

class UOp(Op):
    def __init__(self, theta, a_idx, i_idx):
        self.theta = float(theta)
        self.a_idx = [int(a) for a in a_idx]
        self.i_idx = [int(i) for i in i_idx]
        s = np.sin(self.theta)
        c = np.cos(self.theta) - 1.0

        terms = [
            (1.0, tuple()),
            (s, [("annihilate", i) for i in self.i_idx] + [("create", a) for a in self.a_idx[::-1]]),
            (-s,[("annihilate", a) for a in self.a_idx] + [("create", i) for i in self.i_idx[::-1]]),
        ]
        if len(self.a_idx) == 1:
            # na + ni - 2 na ni
            terms.append((c, [("number", a) for a in self.a_idx]))
            terms.append((c, [("number", i) for i in self.i_idx]))
            terms.append((-2.0 * c, [("number", a) for a in self.a_idx] + [("number", i) for i in self.i_idx]))
        elif len(self.a_idx) == 2:
            # na1 na2 + ni1 ni2 - na1 na2 ni1 - na1 na2 ni2 - na1 ni1 ni2 - na2 ni1 ni2 + 2 na1 na2 ni1 ni2 
            terms.append((c, [("number", a) for a in self.a_idx]))
            terms.append((c, [("number", i) for i in self.i_idx]))
            terms.append((-c,[("number", self.i_idx[0])] + [("number", a) for a in self.a_idx]))
            terms.append((-c, [("number", self.i_idx[1])] + [("number", a) for a in self.a_idx]))
            terms.append((-c,[("number", i) for i in self.i_idx] + [("number", self.a_idx[0])]))
            terms.append((-c,[("number", i) for i in self.i_idx]+ [("number", self.a_idx[1])]))
            terms.append((2.0 * c,  [("number", i) for i in self.i_idx] + [("number", a) for a in self.a_idx]))
        else:
            raise NotImplementedError("Only supports single and double excitations for now")     
        super().__init__(terms)

class IdentityOp(Op):
    def __init__(self):
        super().__init__([(1.0, tuple())])

class h1Op(Op):
    def __init__(self, h1):
        self.h1 = h1
        terms = []
        n_orb = h1.shape[0]
        for p in range(n_orb):
            for q in range(n_orb):
                if abs(h1[p, q]) > 1e-8:
                    # The operators are applied from front to back, so we need to reverse the order of creation and annihilation
                    terms.append((h1[p, q], [("annihilate", p), ("create", q)])) # This is not consistent with convention..
        super().__init__(terms)

class h2Op(Op):
    def __init__(self, h2):
        self.h2 = h2
        terms = []
        n_orb = h2.shape[0]
        for p in range(n_orb):
            for q in range(n_orb):
                for r in range(n_orb):
                    for s in range(n_orb):
                        if abs(h2[p, q, r, s]) > 1e-8:
                            terms.append((h2[p, q, r, s], [("annihilate", r), ("annihilate", s), ("create", q), ("create", p)]))
        super().__init__(terms)

import numpy as np
import pytest
# from las_uscc_noqe.tasks.hydro.op import Op

def test_add_term_normalizes_and_skips_zero():
    op = Op()
    op.add_term(0, (("create", 0),))
    assert op.terms == []

    op.add_term(1j, (("create", "1"), ("number", "2")))
    assert len(op.terms) == 1
    coeff, ops = op.terms[0]
    assert coeff == complex(0, 1)
    assert ops == (("create", 1), ("number", 2))


def test_dagger_conjugation_and_operator_swap():
    op = Op([(1 + 2j, (("create", 0), ("annihilate", 1), ("custom", 2)))])
    dag = op.dagger()
    assert len(dag.terms) == 1
    coeff, ops = dag.terms[0]
    assert coeff == complex(1 - 2j)
    assert ops == (("custom", 2), ("create", 1), ("annihilate", 0))


def test_apply_identity_and_number_operator():
    ci = np.array([1, 2, 3, 4], dtype=complex)
    identity = Op([(1, tuple())])
    number = Op([(1, (("number", 0),))])

    np.testing.assert_allclose(identity.apply(ci), ci)

    expected_number = np.array([0, 2, 0, 4], dtype=complex)
    np.testing.assert_allclose(number.apply(ci), expected_number)


def test_apply_creation_and_annihilation_phase():
    ci_create = np.zeros(4, dtype=complex)
    ci_create[0] = 1
    create_op = Op([(1, (("create", 0),))])
    expected_create = np.array([0, 1, 0, 0], dtype=complex)
    np.testing.assert_allclose(create_op.apply(ci_create), expected_create)

    ci_annih = np.zeros(4, dtype=complex)
    ci_annih[3] = 1
    annihilate_op = Op([(1, (("annihilate", 1),))])
    expected_annih = np.zeros(4, dtype=complex)
    expected_annih[1] = -1
    np.testing.assert_allclose(annihilate_op.apply(ci_annih), expected_annih)


def test_apply_raises_on_invalid_orbital_index():
    ci = np.ones(4, dtype=complex)
    op = Op([(1, (("number", 3),))])
    with pytest.raises(ValueError):
        op.apply(ci)


def test_n_spin_orbitals_requires_power_of_two():
    with pytest.raises(ValueError):
        Op._n_spin_orbitals(np.zeros(3, dtype=complex))


def test_simple_sum_of_ops_behaves_linearly():
    ci = np.array([1, 0, 0, 1], dtype=complex)
    op_identity = Op([(2, tuple())])
    op_number = Op([(3, (("number", 0),))])
    combined = op_identity + op_number

    expected = 2 * ci + 3 * Op([(1, (("number", 0),))]).apply(ci)
    np.testing.assert_allclose(combined.apply(ci), expected)


def test_hOp_behavior():
    h1 = np.array([[0, 1], [0, 0]])
    # h2 = np.zeros((2, 2, 2, 2))
    h1_op = h1Op(h1)
    # h2_op = h2Op(h2)

    ci = np.array([0, 0, 1, 0], dtype=complex)
    expected_h1 = np.array([0, -1, 0, 0], dtype=complex)
    expected_h2 = np.zeros(4, dtype=complex)

    np.testing.assert_allclose(h1_op.apply(ci), expected_h1)
    # np.testing.assert_allclose(h2_op.apply(ci), expected_h2)

if __name__ == "__main__":
    print("Running tests for Op class...")
    pytest.main([__file__])
