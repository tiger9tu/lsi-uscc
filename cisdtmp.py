#!/usr/bin/env python
# Copyright 2014-2021 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Solve CISD equation  H C = C e  where e = E_HF + E_CORR
'''
import numpy as np
import pyscf
from pyscf import gto, scf, lib, mcscf
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from mrh.exploratory.unitary_cc import lasuccsd
from mrh.exploratory.unitary_cc.uccsd_sym0 import get_uccsd_op
from mrh.exploratory.citools import grad, lasci_ominus1
from mrh.exploratory.citools import fockspace
from pyscf import ao2mo, fci

from typing import Iterable, Union
from typing import Dict, List, Any, TypedDict
import copy
import pickle
from enum import Enum
from scipy.linalg import eigh
from scipy import optimize


from functools import reduce
import numpy
from pyscf import gto
from pyscf import lib
from pyscf.lib import logger
from pyscf.cc import ccsd
from pyscf.cc import ccsd_rdm
from pyscf.fci import cistring
from pyscf import __config__

BLKMIN = getattr(__config__, 'ci_cisd_blkmin', 4)


def kernel(myci, eris, ci0=None, max_cycle=50, tol=1e-8, verbose=logger.INFO):
    '''
    Run CISD calculation.

    Args:
        myci : CISD (inheriting) object
        eris : ccsd._ChemistsERIs (inheriting) object (poss diff for df)
            Contains the various (pq|rs) integrals needed.

    Kwargs:
        ci0 : (List of) numpy array(s) (if None it will set)
            Initial guess for CISD coeffs.
        max_cycle : integer
            Maximum number of iterations to converge to CISD solution.
            If not converged before, calculation stops without having
            converged.
        tol : float
            Convergence tolerance.
        verbose : integer
            Level of output (roughly: the higher, the more output).

    Returns:
        conv : bool
            Is it converged?
        ecisd : List of floats or float
            The lowest :attr:`myci.nroots` eigenvalues.
        ci : List of 1D arrays or 1D array
            The lowest :attr:`myci.nroots` eigenvectors.
    '''
    log = logger.new_logger(myci, verbose)
    diag = myci.make_diagonal(eris)
    # Note that ehf is not the HF energy (see `make_diagonal`).
    ehf = diag[0]
    diag -= ehf

    if ci0 is None:
        ci0 = myci.get_init_guess(eris=eris, nroots=myci.nroots, diag=diag)[1]

    def op(xs):
        return [myci.contract(x, eris) for x in xs]

    def precond(x, e, *args):
        diagd = diag - (e-myci.level_shift)
        diagd[abs(diagd)<1e-8] = 1e-8
        return x / diagd

    if myci._dot is not None:
        nmo = myci.nmo
        nocc = myci.nocc
        def cisd_dot(x1, x2):
            return myci._dot(x1, x2, nmo, nocc)
    else:
        cisd_dot = numpy.dot

    conv, ecisd, ci = lib.davidson1(op, ci0, precond, tol=tol,
                                    max_cycle=max_cycle, max_space=myci.max_space,
                                    lindep=myci.lindep, dot=cisd_dot,
                                    nroots=myci.nroots, verbose=log)
    

    if myci.nroots == 1:
        conv = conv[0]
        ecisd = ecisd[0]
        ci = ci[0]
    return conv, ecisd, ci

def make_diagonal(myci, eris):
    '''
    Return diagonal of CISD hamiltonian in Slater determinant basis.

    Note that a constant has been substracted of all elements.
    The first element is the HF energy (minus the
    constant), the next elements are the diagonal elements with singly
    excited determinants (<D_i^a|H|D_i^a> within the constant), then
    doubly excited determinants (<D_ij^ab|H|D_ij^ab> within the
    constant).

    Args:
        myci : CISD (inheriting) object
        eris : ccsd._ChemistsERIs (inheriting) object (poss diff for df)
            Contains the various (pq|rs) integrals needed.

    Returns:
        numpy array (size: (1, 1 + #single excitations from HF det
                               + #double excitations from HF det))
            Diagonal elements of hamiltonian matrix within a constant,
            see above.
    '''
    # DO NOT use eris.mo_energy, it may differ to eris.fock.diagonal()
    mo_energy = eris.fock.diagonal()
    nmo = mo_energy.size
    jdiag = numpy.zeros((nmo,nmo))
    kdiag = numpy.zeros((nmo,nmo))
    nocc = eris.nocc
    nvir = nmo - nocc
    jdiag[:nocc,:nocc] = numpy.einsum('iijj->ij', eris.oooo)
    kdiag[:nocc,:nocc] = numpy.einsum('jiij->ij', eris.oooo)
    jdiag[:nocc,nocc:] = numpy.einsum('iijj->ij', eris.oovv)
    kdiag[:nocc,nocc:] = numpy.einsum('ijji->ij', eris.ovvo)
    if eris.vvvv is not None and len(eris.vvvv.shape) == 2:
        #:eris_vvvv = ao2mo.restore(1, eris.vvvv, nvir)
        #:jdiag1 = numpy.einsum('iijj->ij', eris_vvvv)
        diag_idx = numpy.arange(nvir)
        diag_idx = diag_idx * (diag_idx + 1) // 2 + diag_idx
        for i, ii in enumerate(diag_idx):
            jdiag[nocc+i,nocc:] = eris.vvvv[ii][diag_idx]

    jksum = (jdiag[:nocc,:nocc] * 2 - kdiag[:nocc,:nocc]).sum()
    # Note that ehf is not the HF energy.
    ehf = mo_energy[:nocc].sum() * 2 - jksum
    e_ia = lib.direct_sum('a-i->ia', mo_energy[nocc:], mo_energy[:nocc])
    e_ia -= jdiag[:nocc,nocc:] - kdiag[:nocc,nocc:]
    e1diag = ehf + e_ia
    e2diag = lib.direct_sum('ia+jb->ijab', e_ia, e_ia)
    e2diag += ehf
    e2diag += jdiag[:nocc,:nocc].reshape(nocc,nocc,1,1)
    e2diag -= jdiag[:nocc,nocc:].reshape(nocc,1,1,nvir)
    e2diag -= jdiag[:nocc,nocc:].reshape(1,nocc,nvir,1)
    e2diag += jdiag[nocc:,nocc:].reshape(1,1,nvir,nvir)
    return numpy.hstack((ehf, e1diag.reshape(-1), e2diag.reshape(-1)))

def contract(myci, civec, eris):
    '''
    Application of CISD hamiltonian onto civec.

    Args:
        myci : CISD (inheriting) object
        civec : numpy array, same length as a CI vector.
        eris : ccsd._ChemistsERIs (inheriting) object (poss diff for df)
            Contains the various (pq|rs) integrals needed.

    Returns:
        numpy array, same length as a CI vector.
    '''
    time0 = logger.process_clock(), logger.perf_counter()
    log = logger.Logger(myci.stdout, myci.verbose)
    nocc = myci.nocc
    nmo = myci.nmo
    nvir = nmo - nocc
    c0, c1, c2 = myci.cisdvec_to_amplitudes(civec, nmo, nocc, copy=False)

    t2 = myci._add_vvvv(c2, eris, t2sym='jiba')
    t2 *= .5  # due to t2+t2.transpose(1,0,3,2) in the end
    log.timer_debug1('vvvv', *time0)

    foo = eris.fock[:nocc,:nocc].copy()
    fov = eris.fock[:nocc,nocc:].copy()
    fvv = eris.fock[nocc:,nocc:].copy()

    t1  = fov * c0
    t1 += numpy.einsum('ib,ab->ia', c1, fvv)
    t1 -= numpy.einsum('ja,ji->ia', c1, foo)

    t2 += lib.einsum('kilj,klab->ijab', _cp(eris.oooo)*.5, c2)
    t2 += lib.einsum('ijac,bc->ijab', c2, fvv)
    t2 -= lib.einsum('kj,kiba->jiba', foo, c2)
    t2 += numpy.einsum('ia,jb->ijab', c1, fov)

    unit = nocc*nvir**2 + nocc**2*nvir*3 + 1
    max_memory = max(0, myci.max_memory - lib.current_memory()[0])
    blksize = min(nvir, max(BLKMIN, int(max_memory*.9e6/8/unit)))
    log.debug1('max_memory %d MB,  nocc,nvir = %d,%d  blksize = %d',
               max_memory, nocc, nvir, blksize)
    for p0, p1 in lib.prange(0, nvir, blksize):
        eris_oVoV = _cp(_cp(eris.oovv[:,:,p0:p1]).transpose(0,2,1,3))
        tmp = lib.einsum('kbjc,ikca->jiba', eris_oVoV, c2)
        t2[:,:,p0:p1] -= tmp*.5
        t2[:,:,p0:p1] -= tmp.transpose(1,0,2,3)
        tmp = None

        eris_ovvo = _cp(eris.ovvo[:,p0:p1])
        t2[:,:,p0:p1] += eris_ovvo.transpose(0,3,1,2) * (c0*.5)
        t1 += numpy.einsum('ia,iabj->jb', c1[:,p0:p1], eris_ovvo) * 2
        t1[:,p0:p1] -= numpy.einsum('ib,iajb->ja', c1, eris_oVoV)

        ovov = -.5 * eris_oVoV
        ovov += eris_ovvo.transpose(3,1,0,2)
        eris_oVoV = None
        theta = c2[:,:,p0:p1].transpose(2,0,1,3) * 2
        theta-= c2[:,:,p0:p1].transpose(2,1,0,3)
        for j in range(nocc):
            t2[:,j] += lib.einsum('ckb,ckia->iab', ovov[j], theta)
        tmp = ovov = None

        t1 += numpy.einsum('aijb,ia->jb', theta, fov[:,p0:p1])

        eris_ovoo = _cp(eris.ovoo[:,p0:p1])
        t1 -= lib.einsum('bjka,jbki->ia', theta, eris_ovoo)
        t2[:,:,p0:p1] -= lib.einsum('jbik,ka->jiba', eris_ovoo.conj(), c1)
        eris_ovoo = None

        eris_ovvv = eris.get_ovvv(slice(None), slice(p0,p1)).conj()
        t1 += lib.einsum('cjib,jcba->ia', theta, eris_ovvv)
        t2[:,:,p0:p1] += lib.einsum('iacb,jc->ijab', eris_ovvv, c1)
        tmp = eris_ovvv = None

    #:t2 + t2.transpose(1,0,3,2)
    for i in range(nocc):
        if i > 0:
            t2[i,:i]+= t2[:i,i].transpose(0,2,1)
            t2[:i,i] = t2[i,:i].transpose(0,2,1)
        t2[i,i] = t2[i,i] + t2[i,i].T

    t0  = numpy.einsum('ia,ia->', fov, c1) * 2
    t0 += numpy.einsum('iabj,ijab->', eris.ovvo, c2) * 2
    t0 -= numpy.einsum('iabj,jiab->', eris.ovvo, c2)
    cinew = myci.amplitudes_to_cisdvec(t0, t1, t2)
    return cinew

def amplitudes_to_cisdvec(c0, c1, c2):
    return numpy.hstack((c0, c1.ravel(), c2.ravel()))

def cisdvec_to_amplitudes(civec, nmo, nocc, copy=True):
    nvir = nmo - nocc
    c0 = civec[0]
    cp = lambda x: (x.copy() if copy else x)
    c1 = cp(civec[1:nocc*nvir+1].reshape(nocc,nvir))
    c2 = cp(civec[nocc*nvir+1:].reshape(nocc,nocc,nvir,nvir))
    return c0, c1, c2

def dot(v1, v2, nmo, nocc):
    nvir = nmo - nocc
    hijab = v2[1+nocc*nvir:].reshape(nocc,nocc,nvir,nvir)
    cijab = v1[1+nocc*nvir:].reshape(nocc,nocc,nvir,nvir)
    val = numpy.dot(v1, v2) * 2 - v1[0]*v2[0]
    val-= numpy.einsum('jiab,ijab->', cijab, hijab)
    return val

def t1strs(norb, nelec):
    '''Compute the FCI strings (address) for CIS single-excitation amplitudes
    and the signs of the coefficients when transferring the reference from
    physics vacuum to HF vacuum.
    '''
    addrs, signs = tn_addrs_signs(norb, nelec, 1)
    return addrs, signs

def tn_addrs_signs(norb, nelec, n_excite):
    '''Compute the FCI strings (address) for CIS n-excitation amplitudes and
    the signs of the coefficients when transferring the reference from physics
    vacuum to HF vacuum.

    If the excitation level is not compatible with the number of
    electrons and holes, empty lists are returned for the addresses and signs.
    '''
    # Not enough electrons or holes for excitation; return empty lists.
    if n_excite > min(nelec, norb-nelec):
        return [], []
    nocc = nelec

    hole_strs = cistring.gen_strings4orblist(range(nocc), nocc - n_excite)
    # For HF vacuum, hole operators are ordered from low-lying to high-lying
    # orbitals. It leads to the opposite string ordering.
    hole_strs = hole_strs[::-1]
    hole_sum = numpy.zeros(len(hole_strs), dtype=int)
    for i in range(nocc):
        hole_at_i = (hole_strs & (1 << i)) == 0
        hole_sum[hole_at_i] += i

    # The hole operators are listed from low-lying to high-lying orbitals
    # (from left to right).  For i-th (0-based) hole operator, the number of
    # orbitals which are higher than i determines the sign.  This number
    # equals to nocc-(i+1).  After removing the highest hole operator, nocc
    # becomes nocc-1, the sign for next hole operator j will be associated to
    # nocc-1-(j+1).  By iteratively calling this procedure, the overall sign
    # for annihilating three holes is (-1)**(3*nocc - 6 - sum i)
    sign = (-1) ** (n_excite * nocc - n_excite*(n_excite+1)//2 - hole_sum)

    particle_strs = cistring.gen_strings4orblist(range(nocc, norb), n_excite)
    strs = hole_strs[:,None] ^ particle_strs
    addrs = cistring.strs2addr(norb, nocc, strs.ravel())
    signs = numpy.vstack([sign] * len(particle_strs)).T.ravel()
    return addrs, signs

def to_fcivec(cisdvec, norb, nelec, frozen=None):
    '''Convert CISD coefficients to FCI coefficients'''
    if isinstance(nelec, (int, numpy.number)):
        nelecb = nelec//2
        neleca = nelec - nelecb
    else:
        neleca, nelecb = nelec
        assert (neleca == nelecb)

    frozen_mask = numpy.zeros(norb, dtype=bool)
    if frozen is None:
        nfroz = 0
    elif isinstance(frozen, (int, numpy.integer)):
        nfroz = frozen
        frozen_mask[:frozen] = True
    else:
        nfroz = len(frozen)
        frozen_mask[frozen] = True

    nocc = numpy.count_nonzero(~frozen_mask[:neleca])
    nmo = norb - nfroz
    nvir = nmo - nocc
    c0, c1, c2 = cisdvec_to_amplitudes(cisdvec, nmo, nocc, copy=False)
    t1addr, t1sign = tn_addrs_signs(nmo, nocc, 1)

    na = cistring.num_strings(nmo, nocc)
    fcivec = numpy.zeros((na,na))
    fcivec[0,0] = c0
    fcivec[0,t1addr] = fcivec[t1addr,0] = c1.ravel() * t1sign
    c2ab = c2.transpose(0,2,1,3).reshape(nocc*nvir,-1)
    c2ab = numpy.einsum('i,j,ij->ij', t1sign, t1sign, c2ab)
    fcivec[t1addr[:,None],t1addr] = c2ab


    if nocc > 1 and nvir > 1:
        c2aa = c2 - c2.transpose(1,0,2,3)
        ooidx = numpy.tril_indices(nocc, -1)
        vvidx = numpy.tril_indices(nvir, -1)
        c2aa = c2aa[ooidx][:,vvidx[0],vvidx[1]]
        t2addr, t2sign = tn_addrs_signs(nmo, nocc, 2)
        fcivec[0,t2addr] = fcivec[t2addr,0] = c2aa.ravel() * t2sign

    if nfroz == 0:
        return fcivec

    assert (norb < 63)

    strs = cistring.gen_strings4orblist(range(norb), neleca)
    na = len(strs)
    count = numpy.zeros(na, dtype=int)
    parity = numpy.zeros(na, dtype=bool)
    core_mask = numpy.ones(na, dtype=bool)
    # During the loop, count saves the number of occupied orbitals that
    # lower (with small orbital ID) than the present orbital i.
    # Moving all the frozen orbitals to the beginning of the orbital list
    # (before the occupied orbitals) leads to parity odd (= True, with
    # negative sign) or even (= False, with positive sign).
    for i in range(norb):
        if frozen_mask[i]:
            if i < neleca:
                # frozen occupied orbital should be occupied
                core_mask &= (strs & (1 << i)) != 0
                parity ^= (count & 1) == 1
            else:
                # frozen virtual orbital should not be occupied.
                # parity is not needed since it's unoccupied
                core_mask &= (strs & (1 << i)) == 0
        else:
            count += (strs & (1 << i)) != 0
    sub_strs = strs[core_mask & (count == nocc)]
    addrs = cistring.strs2addr(norb, neleca, sub_strs)
    fcivec1 = numpy.zeros((na,na))
    fcivec1[addrs[:,None],addrs] = fcivec
    fcivec1[parity,:] *= -1
    fcivec1[:,parity] *= -1
    return fcivec1

def from_fcivec(ci0, norb, nelec, frozen=None):
    '''Extract CISD coefficients from FCI coefficients'''
    if not (frozen is None or frozen == 0):
        raise NotImplementedError

    if isinstance(nelec, (int, numpy.number)):
        nelecb = nelec//2
        neleca = nelec - nelecb
    else:
        neleca, nelecb = nelec
    nocc = neleca
    nvir = norb - nocc
    t1addr, t1sign = t1strs(norb, nocc)

    c0 = ci0[0,0]
    c1 = ci0[0,t1addr] * t1sign
    c2 = numpy.einsum('i,j,ij->ij', t1sign, t1sign, ci0[t1addr[:,None],t1addr])
    c1 = c1.reshape(nocc,nvir)
    c2 = c2.reshape(nocc,nvir,nocc,nvir).transpose(0,2,1,3)
    return amplitudes_to_cisdvec(c0, c1, c2)

def overlap(cibra, ciket, nmo, nocc, s=None):
    '''Overlap between two CISD wavefunctions.

    Args:
        s : 2D array
            The overlap matrix of non-orthogonal one-particle basis
    '''
    if s is None:
        return dot(cibra, ciket, nmo, nocc)

    DEBUG = True

    nvir = nmo - nocc
    nov = nocc * nvir
    bra0, bra1, bra2 = cisdvec_to_amplitudes(cibra, nmo, nocc, copy=False)
    ket0, ket1, ket2 = cisdvec_to_amplitudes(ciket, nmo, nocc, copy=False)

# Sort the ket orbitals to make the orbitals in bra one-one mapt to orbitals
# in ket.
    if ((not DEBUG) and
        abs(numpy.linalg.det(s[:nocc,:nocc]) - 1) < 1e-2 and
        abs(numpy.linalg.det(s[nocc:,nocc:]) - 1) < 1e-2):
        ket_orb_idx = numpy.where(abs(s) > 0.9)[1]
        s = s[:,ket_orb_idx]
        oidx = ket_orb_idx[:nocc]
        vidx = ket_orb_idx[nocc:] - nocc
        ket1 = ket1[oidx[:,None],vidx]
        ket2 = ket2[oidx[:,None,None,None],oidx[:,None,None],vidx[:,None],vidx]

    ooidx = numpy.tril_indices(nocc, -1)
    vvidx = numpy.tril_indices(nvir, -1)
    bra2aa = bra2 - bra2.transpose(1,0,2,3)
    bra2aa = lib.take_2d(bra2aa.reshape(nocc**2,nvir**2),
                         ooidx[0]*nocc+ooidx[1], vvidx[0]*nvir+vvidx[1])
    ket2aa = ket2 - ket2.transpose(1,0,2,3)
    ket2aa = lib.take_2d(ket2aa.reshape(nocc**2,nvir**2),
                         ooidx[0]*nocc+ooidx[1], vvidx[0]*nvir+vvidx[1])

    occlist0 = numpy.arange(nocc).reshape(1,nocc)
    occlists = numpy.repeat(occlist0, 1+nov+bra2aa.size, axis=0)
    occlist0 = occlists[:1]
    occlist1 = occlists[1:1+nov]
    occlist2 = occlists[1+nov:]

    ia = 0
    for i in range(nocc):
        for a in range(nocc, nmo):
            occlist1[ia,i] = a
            ia += 1

    ia = 0
    for i in range(nocc):
        for j in range(i):
            for a in range(nocc, nmo):
                for b in range(nocc, a):
                    occlist2[ia,i] = a
                    occlist2[ia,j] = b
                    ia += 1

    na = len(occlists)
    if DEBUG:
        trans = numpy.empty((na,na))
        for i, idx in enumerate(occlists):
            s_sub = s[idx].T.copy()
            minors = s_sub[occlists]
            trans[i,:] = numpy.linalg.det(minors)

        # Mimic the transformation einsum('ab,ap->pb', FCI, trans).
        # The wavefunction FCI has the [excitation_alpha,excitation_beta]
        # representation.  The zero blocks like FCI[S_alpha,D_beta],
        # FCI[D_alpha,D_beta], are explicitly excluded.
        bra_mat = numpy.zeros((na,na))
        bra_mat[0,0] = bra0
        bra_mat[0,1:1+nov] = bra_mat[1:1+nov,0] = bra1.ravel()
        bra_mat[0,1+nov:] = bra_mat[1+nov:,0] = bra2aa.ravel()
        bra_mat[1:1+nov,1:1+nov] = bra2.transpose(0,2,1,3).reshape(nov,nov)
        ket_mat = numpy.zeros((na,na))
        ket_mat[0,0] = ket0
        ket_mat[0,1:1+nov] = ket_mat[1:1+nov,0] = ket1.ravel()
        ket_mat[0,1+nov:] = ket_mat[1+nov:,0] = ket2aa.ravel()
        ket_mat[1:1+nov,1:1+nov] = ket2.transpose(0,2,1,3).reshape(nov,nov)
        ovlp = lib.einsum('ab,ap,bq,pq->', bra_mat, trans, trans, ket_mat)

    else:
        nov1 = 1 + nov
        noovv = bra2aa.size
        bra_SS = numpy.zeros((nov1,nov1))
        bra_SS[0,0] = bra0
        bra_SS[0,1:] = bra_SS[1:,0] = bra1.ravel()
        bra_SS[1:,1:] = bra2.transpose(0,2,1,3).reshape(nov,nov)
        ket_SS = numpy.zeros((nov1,nov1))
        ket_SS[0,0] = ket0
        ket_SS[0,1:] = ket_SS[1:,0] = ket1.ravel()
        ket_SS[1:,1:] = ket2.transpose(0,2,1,3).reshape(nov,nov)

        trans_SS = numpy.empty((nov1,nov1))
        trans_SD = numpy.empty((nov1,noovv))
        trans_DS = numpy.empty((noovv,nov1))
        occlist01 = occlists[:nov1]
        for i, idx in enumerate(occlist01):
            s_sub = s[idx].T.copy()
            minors = s_sub[occlist01]
            trans_SS[i,:] = numpy.linalg.det(minors)

            minors = s_sub[occlist2]
            trans_SD[i,:] = numpy.linalg.det(minors)

            s_sub = s[:,idx].copy()
            minors = s_sub[occlist2]
            trans_DS[:,i] = numpy.linalg.det(minors)

        ovlp = lib.einsum('ab,ap,bq,pq->', bra_SS, trans_SS, trans_SS, ket_SS)
        ovlp+= lib.einsum('ab,a ,bq, q->', bra_SS, trans_SS[:,0], trans_SD, ket2aa.ravel())
        ovlp+= lib.einsum('ab,ap,b ,p ->', bra_SS, trans_SD, trans_SS[:,0], ket2aa.ravel())

        ovlp+= lib.einsum(' b, p,bq,pq->', bra2aa.ravel(), trans_SS[0,:], trans_DS, ket_SS)
        ovlp+= lib.einsum(' b, p,b ,p ->', bra2aa.ravel(), trans_SD[0,:], trans_DS[:,0],
                          ket2aa.ravel())

        ovlp+= lib.einsum('a ,ap, q,pq->', bra2aa.ravel(), trans_DS, trans_SS[0,:], ket_SS)
        ovlp+= lib.einsum('a ,a , q, q->', bra2aa.ravel(), trans_DS[:,0], trans_SD[0,:],
                          ket2aa.ravel())

        # FIXME: whether to approximate the overlap between double excitation coefficients
        if numpy.linalg.norm(bra2aa)*numpy.linalg.norm(ket2aa) < 1e-4:
            # Skip the overlap if coefficients of double excitation are small enough
            pass
        if (abs(numpy.linalg.det(s[:nocc,:nocc]) - 1) < 1e-2 and
            abs(numpy.linalg.det(s[nocc:,nocc:]) - 1) < 1e-2):
            # If the overlap matrix close to identity enough, use the <D|D'> overlap
            # for orthogonal single-particle basis to approximate the overlap
            # for non-orthogonal basis.
            ovlp+= numpy.dot(bra2aa.ravel(), ket2aa.ravel()) * trans_SS[0,0] * 2
        else:
            from multiprocessing import sharedctypes, Process
            buf_ctypes = sharedctypes.RawArray('d', noovv)
            trans_ket = numpy.ndarray(noovv, buffer=buf_ctypes)
            def trans_dot_ket(i0, i1):
                for i in range(i0, i1):
                    s_sub = s[occlist2[i]].T.copy()
                    minors = s_sub[occlist2]
                    trans_ket[i] = numpy.linalg.det(minors).dot(ket2aa.ravel())

            nproc = lib.num_threads()
            if nproc > 1:
                seg = (noovv+nproc-1) // nproc
                ps = []
                for i0,i1 in lib.prange(0, noovv, seg):
                    p = Process(target=trans_dot_ket, args=(i0,i1))
                    ps.append(p)
                    p.start()
                [p.join() for p in ps]
            else:
                trans_dot_ket(0, noovv)

            ovlp+= numpy.dot(bra2aa.ravel(), trans_ket) * trans_SS[0,0] * 2

    return ovlp

def make_rdm1(myci, civec=None, nmo=None, nocc=None, ao_repr=False):
    r'''
    Spin-traced one-particle density matrix in MO basis (the occupied-virtual
    blocks from the orbital response contribution are not included).

    dm1[p,q] = <q_alpha^\dagger p_alpha> + <q_beta^\dagger p_beta>

    The convention of 1-pdm is based on McWeeney's book, Eq (5.4.20).
    The contraction between 1-particle Hamiltonian and rdm1 is
    E = einsum('pq,qp', h1, rdm1)
    '''
    if civec is None: civec = myci.ci
    if nmo is None: nmo = myci.nmo
    if nocc is None: nocc = myci.nocc
    d1 = _gamma1_intermediates(myci, civec, nmo, nocc)
    return ccsd_rdm._make_rdm1(myci, d1, with_frozen=True, ao_repr=ao_repr)

def make_rdm2(myci, civec=None, nmo=None, nocc=None, ao_repr=False):
    r'''
    Spin-traced two-particle density matrix in MO basis

    dm2[p,q,r,s] = \sum_{sigma,tau} <p_sigma^\dagger r_tau^\dagger s_tau q_sigma>

    Note the contraction between ERIs (in Chemist's notation) and rdm2 is
    E = einsum('pqrs,pqrs', eri, rdm2)
    '''
    if civec is None: civec = myci.ci
    if nmo is None: nmo = myci.nmo
    if nocc is None: nocc = myci.nocc
    d1 = _gamma1_intermediates(myci, civec, nmo, nocc)
    f = lib.H5TmpFile()
    d2 = _gamma2_outcore(myci, civec, nmo, nocc, f, False)
    return ccsd_rdm._make_rdm2(myci, d1, d2, with_dm1=True, with_frozen=True,
                               ao_repr=ao_repr)

def _gamma1_intermediates(myci, civec, nmo, nocc):
    c0, c1, c2 = myci.cisdvec_to_amplitudes(civec, nmo, nocc, copy=False)
    dvo = c0.conj() * c1.T
    dvo += numpy.einsum('jb,ijab->ai', c1.conj(), c2) * 2
    dvo -= numpy.einsum('jb,ijba->ai', c1.conj(), c2)
    dov = dvo.T.conj()

    theta = c2*2 - c2.transpose(0,1,3,2)
    doo  = -numpy.einsum('ia,ka->ik', c1.conj(), c1)
    doo -= lib.einsum('ijab,ikab->jk', c2.conj(), theta)
    dvv  = numpy.einsum('ia,ic->ac', c1, c1.conj())
    dvv += lib.einsum('ijab,ijac->bc', theta, c2.conj())
    return doo, dov, dvo, dvv

def _gamma2_intermediates(myci, civec, nmo, nocc, compress_vvvv=False):
    f = lib.H5TmpFile()
    _gamma2_outcore(myci, civec, nmo, nocc, f, compress_vvvv)
    d2 = (f['dovov'][:], f['dvvvv'][:], f['doooo'][:], f['doovv'][:],
          f['dovvo'][:], None,          f['dovvv'][:], f['dooov'][:])
    return d2

def _gamma2_outcore(myci, civec, nmo, nocc, h5fobj, compress_vvvv=False):
    log = logger.Logger(myci.stdout, myci.verbose)
    nocc = myci.nocc
    nmo = myci.nmo
    nvir = nmo - nocc
    nvir_pair = nvir * (nvir+1) // 2
    c0, c1, c2 = myci.cisdvec_to_amplitudes(civec, nmo, nocc, copy=False)

    h5fobj['dovov'] = (2*c0*c2.conj().transpose(0,2,1,3) -
                       c0*c2.conj().transpose(1,2,0,3))

    doooo = lib.einsum('ijab,klab->ijkl', c2.conj(), c2)
    h5fobj['doooo'] = doooo.transpose(0,2,1,3) - doooo.transpose(1,2,0,3)*.5
    doooo = None

    dooov = -lib.einsum('ia,klac->klic', c1*2, c2.conj())
    h5fobj['dooov'] = dooov.transpose(0,2,1,3)*2 - dooov.transpose(1,2,0,3)
    dooov = None

    #:dvovv = numpy.einsum('ia,ikcd->akcd', c1, c2) * 2
    #:dvvvv = lib.einsum('ijab,ijcd->abcd', c2, c2)
    max_memory = max(0, myci.max_memory - lib.current_memory()[0])
    unit = max(nocc**2*nvir*2+nocc*nvir**2*3 + 1, nvir**3*2+nocc*nvir**2 + 1)
    blksize = min(nvir, max(BLKMIN, int(max_memory*.9e6/8/unit)))
    log.debug1('rdm intermediates: block size = %d, nvir = %d in %d blocks',
               blksize, nocc, int((nvir+blksize-1)/blksize))
    dtype = numpy.result_type(civec).char
    dovvv = h5fobj.create_dataset('dovvv', (nocc,nvir,nvir,nvir), dtype,
                                  chunks=(nocc,min(nocc,nvir),1,nvir))
    if compress_vvvv:
        dvvvv = h5fobj.create_dataset('dvvvv', (nvir_pair,nvir_pair), dtype)
    else:
        dvvvv = h5fobj.create_dataset('dvvvv', (nvir,nvir,nvir,nvir), dtype)

    for (p0, p1) in lib.prange(0, nvir, blksize):
        theta = c2[:,:,p0:p1] - c2[:,:,p0:p1].transpose(1,0,2,3) * .5
        gvvvv = lib.einsum('ijab,ijcd->abcd', theta.conj(), c2)
        if compress_vvvv:
            # symmetrize dvvvv because it does not affect the results of cisd_grad
            # dvvvv = (dvvvv+dvvvv.transpose(0,1,3,2)) * .5
            # dvvvv = (dvvvv+dvvvv.transpose(1,0,2,3)) * .5
            # now dvvvv == dvvvv.transpose(0,1,3,2) == dvvvv.transpose(1,0,3,2)
            tmp = numpy.empty((nvir,nvir,nvir))
            tmpvvvv = numpy.empty((p1-p0,nvir,nvir_pair))
            for i in range(p1-p0):
                tmp[:] = gvvvv[i].conj().transpose(1,0,2)
                lib.pack_tril(tmp+tmp.transpose(0,2,1), out=tmpvvvv[i])
            # tril of (dvvvv[p0:p1,p0:p1]+dvvvv[p0:p1,p0:p1].T)
            for i in range(p0, p1):
                for j in range(p0, i):
                    tmpvvvv[i-p0,j] += tmpvvvv[j-p0,i]
                tmpvvvv[i-p0,i] *= 2
            for i in range(p1, nvir):
                off = i * (i+1) // 2
                dvvvv[off+p0:off+p1] = tmpvvvv[:,i]
            for i in range(p0, p1):
                off = i * (i+1) // 2
                if p0 > 0:
                    tmpvvvv[i-p0,:p0] += dvvvv[off:off+p0]
                dvvvv[off:off+i+1] = tmpvvvv[i-p0,:i+1] * .25
            tmp = tmpvvvv = None
        else:
            for i in range(p0, p1):
                dvvvv[i] = gvvvv[i-p0].conj().transpose(1,0,2)

        gvovv = numpy.einsum('ia,ikcd->akcd', c1[:,p0:p1].conj()*2, c2)
        gvovv = gvovv.conj()
        dovvv[:,:,p0:p1] = gvovv.transpose(1,3,0,2)*2 - gvovv.transpose(1,2,0,3)

    theta = c2*2 - c2.transpose(1,0,2,3)
    doovv  = numpy.einsum('ia,kc->ikca', c1.conj(), -c1)
    doovv -= lib.einsum('kjcb,kica->jiab', c2.conj(), theta)
    doovv -= lib.einsum('ikcb,jkca->ijab', c2.conj(), theta)
    h5fobj['doovv'] = doovv
    doovv = None

    dovvo  = lib.einsum('ikac,jkbc->iabj', theta.conj(), theta)
    dovvo += numpy.einsum('ia,kc->iack', c1.conj(), c1) * 2
    h5fobj['dovvo'] = dovvo
    theta = dovvo = None

    dvvov = None
    return (h5fobj['dovov'], h5fobj['dvvvv'], h5fobj['doooo'], h5fobj['doovv'],
            h5fobj['dovvo'], dvvov          , h5fobj['dovvv'], h5fobj['dooov'])

def trans_rdm1(myci, cibra, ciket, nmo=None, nocc=None):
    r'''
    Spin-traced one-particle transition density matrix in MO basis.

    dm1[p,q] = <q_alpha^\dagger p_alpha> + <q_beta^\dagger p_beta>

    The convention of 1-pdm is based on McWeeney's book, Eq (5.4.20).
    The contraction between 1-particle Hamiltonian and rdm1 is
    E = einsum('pq,qp', h1, rdm1)
    '''
    if nmo is None: nmo = myci.nmo
    if nocc is None: nocc = myci.nocc
    c0bra, c1bra, c2bra = myci.cisdvec_to_amplitudes(cibra, nmo, nocc, copy=False)
    c0ket, c1ket, c2ket = myci.cisdvec_to_amplitudes(ciket, nmo, nocc, copy=False)

    dvo = c0bra.conj() * c1ket.T
    dvo += numpy.einsum('jb,ijab->ai', c1bra.conj(), c2ket) * 2
    dvo -= numpy.einsum('jb,ijba->ai', c1bra.conj(), c2ket)

    dov = c0ket * c1bra.conj()
    dov += numpy.einsum('jb,ijab->ia', c1ket, c2bra.conj()) * 2
    dov -= numpy.einsum('jb,ijba->ia', c1ket, c2bra.conj())

    theta = c2ket*2 - c2ket.transpose(0,1,3,2)
    doo  = -numpy.einsum('ia,ka->ik', c1bra.conj(), c1ket)
    doo -= lib.einsum('ijab,ikab->jk', c2bra.conj(), theta)
    dvv  = numpy.einsum('ia,ic->ac', c1ket, c1bra.conj())
    dvv += lib.einsum('ijab,ijac->bc', theta, c2bra.conj())

    dm1 = numpy.empty((nmo,nmo), dtype=doo.dtype)
    dm1[:nocc,:nocc] = doo * 2
    dm1[:nocc,nocc:] = dov * 2
    dm1[nocc:,:nocc] = dvo * 2
    dm1[nocc:,nocc:] = dvv * 2
    norm = dot(cibra, ciket, nmo, nocc)
    dm1[numpy.diag_indices(nocc)] += 2 * norm

    if myci.frozen is not None:
        nmo = myci.mo_occ.size
        nocc = numpy.count_nonzero(myci.mo_occ > 0)
        rdm1 = numpy.zeros((nmo,nmo), dtype=dm1.dtype)
        rdm1[numpy.diag_indices(nocc)] = 2 * norm
        moidx = numpy.where(myci.get_frozen_mask())[0]
        rdm1[moidx[:,None],moidx] = dm1
        dm1 = rdm1
    return dm1


def as_scanner(ci):
    '''Generating a scanner/solver for CISD PES.

    The returned solver is a function. This function requires one argument
    "mol" as input and returns total CISD energy.

    The solver will automatically use the results of last calculation as the
    initial guess of the new calculation.  All parameters assigned in the
    CISD and the underlying SCF objects (conv_tol, max_memory etc) are
    automatically applied in the solver.

    Note scanner has side effects.  It may change many underlying objects
    (_scf, with_df, with_x2c, ...) during calculation.

    Examples::

        >>> from pyscf import gto, scf, ci
        >>> mol = gto.M(atom='H 0 0 0; F 0 0 1')
        >>> ci_scanner = ci.CISD(scf.RHF(mol)).as_scanner()
        >>> e_tot = ci_scanner(gto.M(atom='H 0 0 0; F 0 0 1.1'))
        >>> e_tot = ci_scanner(gto.M(atom='H 0 0 0; F 0 0 1.5'))
    '''
    from pyscf import gto
    if isinstance(ci, lib.SinglePointScanner):
        return ci

    logger.info(ci, 'Set %s as a scanner', ci.__class__)
    name = ci.__class__.__name__ + CISD_Scanner.__name_mixin__
    return lib.set_class(CISD_Scanner(ci), (CISD_Scanner, ci.__class__), name)

class CISD_Scanner(lib.SinglePointScanner):
    def __init__(self, ci):
        self.__dict__.update(ci.__dict__)
        self._scf = ci._scf.as_scanner()

    def __call__(self, mol_or_geom, ci0=None, **kwargs):
        if isinstance(mol_or_geom, gto.Mole):
            mol = mol_or_geom
        else:
            mol = self.mol.set_geom_(mol_or_geom, inplace=False)

        self.reset(mol)

        mf_scanner = self._scf
        mf_scanner(mol)
        self.mo_coeff = mf_scanner.mo_coeff
        self.mo_occ = mf_scanner.mo_occ
        if getattr(self.ci, 'size', 0) != self.vector_size():
            self.ci = None
        if ci0 is None:
            # FIXME: Whether to use the initial guess from last step?
            # If root flips, large errors may be found in the solutions
            ci0 = self.ci
        self.kernel(ci0, **kwargs)[0]
        return self.e_tot


class CISD(lib.StreamObject):
    '''restricted CISD

    Attributes:
        verbose : int
            Print level.  Default value equals to :class:`Mole.verbose`
        max_memory : float or int
            Allowed memory in MB.  Default value equals to
            :class:`Mole.max_memory`
        conv_tol : float
            converge threshold.  Default is 1e-9.
        max_cycle : int
            max number of iterations.  Default is 50.
        max_space : int
            Davidson diagonalization space size.  Default is 12.
        direct : bool
            AO-direct CISD. Default is False.
        async_io : bool
            Allow for asynchronous function execution. Default is True.
        frozen : int or list
            If integer is given, the inner-most orbitals are frozen from CI
            amplitudes.  Given the orbital indices (0-based) in a list, both
            occupied and virtual orbitals can be frozen in CI calculation.

            >>> mol = gto.M(atom = 'H 0 0 0; F 0 0 1.1', basis = 'ccpvdz')
            >>> mf = scf.RHF(mol).run()
            >>> # freeze 2 core orbitals
            >>> myci = ci.CISD(mf).set(frozen = 2).run()
            >>> # freeze 2 core orbitals and 3 high lying unoccupied orbitals
            >>> myci.set(frozen = [0,1,16,17,18]).run()

    Saved results

        converged : bool
            CISD converged or not
        e_corr : float
            CISD correlation correction
        e_tot : float
            Total CCSD energy (HF + correlation)
        ci :
            CI wavefunction coefficients
    '''

    conv_tol = getattr(__config__, 'ci_cisd_CISD_conv_tol', 1e-9)
    max_cycle = getattr(__config__, 'ci_cisd_CISD_max_cycle', 50)
    max_space = getattr(__config__, 'ci_cisd_CISD_max_space', 12)
    lindep = getattr(__config__, 'ci_cisd_CISD_lindep', 1e-14)
    # level shift in preconditioner is helpful to avoid singularity and linear
    # dependence basis in davidson diagonalization solver
    level_shift = getattr(__config__, 'ci_cisd_CISD_level_shift', 1e-3)
    direct = getattr(__config__, 'ci_cisd_CISD_direct', False)
    async_io = getattr(__config__, 'ci_cisd_CISD_async_io', True)

    _keys = {
        'conv_tol', 'max_cycle', 'max_space', 'lindep',
        'level_shift', 'direct', 'async_io', 'mol', 'max_memory',
        'nroots', 'frozen', 'chkfile', 'converged', 'mo_coeff', 'mo_occ',
        'e_hf', 'e_corr', 'emp2', 'ci',
    }

    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None):
        from pyscf.scf import hf
        if isinstance(mf, hf.KohnShamDFT):
            raise RuntimeError(
                'CISD Warning: The first argument mf is a DFT object. '
                'CISD calculation should be initialized with HF object.\n'
                'DFT can be converted to HF object with the mf.to_hf() method\n')

        if mo_coeff is None: mo_coeff = mf.mo_coeff
        if mo_occ is None: mo_occ   = mf.mo_occ

        self.mol = mf.mol
        self._scf = mf
        self.verbose = self.mol.verbose
        self.stdout = self.mol.stdout
        self.max_memory = mf.max_memory

        self.nroots = 1
        self.frozen = frozen
        self.chkfile = mf.chkfile

##################################################
# don't modify the following attributes, they are not input options
        self.converged = False
        self.mo_coeff = mo_coeff
        self.mo_occ = mo_occ
        self.e_hf = None
        self.e_corr = None
        self.emp2 = None
        self.ci = None
        self._nocc = None
        self._nmo = None

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('')
        log.info('******** %s ********', self.__class__)
        log.info('CISD nocc = %s, nmo = %s', self.nocc, self.nmo)
        if self.frozen is not None:
            log.info('frozen orbitals %s', str(self.frozen))
        log.info('max_cycle = %d', self.max_cycle)
        log.info('direct = %d', self.direct)
        log.info('conv_tol = %g', self.conv_tol)
        log.info('max_cycle = %d', self.max_cycle)
        log.info('max_space = %d', self.max_space)
        log.info('lindep = %d', self.lindep)
        log.info('nroots = %d', self.nroots)
        log.info('max_memory %d MB (current use %d MB)',
                 self.max_memory, lib.current_memory()[0])
        return self

    @property
    def e_tot(self):
        return numpy.asarray(self.e_corr) + self.e_hf

    @property
    def nstates(self):
        return self.nroots
    @nstates.setter
    def nstates(self, x):
        self.nroots = x

    @property
    def nocc(self):
        return self.get_nocc()
    @nocc.setter
    def nocc(self, n):
        self._nocc = n

    @property
    def nmo(self):
        return self.get_nmo()
    @nmo.setter
    def nmo(self, n):
        self._nmo = n

    def vector_size(self):
        '''The size of the vector which was returned from
        :func:`amplitudes_to_cisdvec`
        '''
        nocc = self.nocc
        nvir = self.nmo - nocc
        return 1 + nocc*nvir + (nocc*nvir)**2

    def reset(self, mol=None):
        if mol is not None:
            self.mol = mol
        self._scf.reset(mol)
        return self

    get_nocc = ccsd.get_nocc
    get_nmo = ccsd.get_nmo
    get_frozen_mask = ccsd.get_frozen_mask
    get_e_hf = ccsd.get_e_hf

    def kernel(self, ci0=None, eris=None):
        return self.cisd(ci0, eris)
    def cisd(self, ci0=None, eris=None):
        self.e_hf = self.get_e_hf()
        if eris is None:
            eris = self.ao2mo(self.mo_coeff)
        if self.verbose >= logger.WARN:
            self.check_sanity()
        self.dump_flags()

        self.converged, self.e_corr, self.ci = \
                kernel(self, eris, ci0, max_cycle=self.max_cycle,
                       tol=self.conv_tol, verbose=self.verbose)
        self._finalize()
        return self.e_corr, self.ci

    def _finalize(self):
        citype = self.__class__.__name__
        if numpy.all(self.converged):
            logger.info(self, '%s converged', citype)
        else:
            logger.info(self, '%s not converged', citype)
        if self.nroots > 1:
            for i,e in enumerate(self.e_tot):
                logger.note(self, '%s root %d  E = %.16g', citype, i, e)
        else:
            logger.note(self, 'E(%s) = %.16g  E_corr = %.16g',
                        citype, self.e_tot, self.e_corr)
        return self

    def get_init_guess(self, eris=None, nroots=1, diag=None):
        '''
        MP2 energy and MP2 initial guess(es) for CISD coefficients.

        Kwargs:
            eris : ccsd._ChemistsERIs (inheriting) object (poss diff for df)
                Contains the various (pq|rs) integrals needed.
            nroots : integer
                Number of CISD solutions to be found.
            diag : numpy array (1D)
                e.g. CISD Hamiltonian diagonal in Slater determinant
                space with HF energy subtracted.

        Returns:
            Tuple of float and numpy array or
            tuple of float and list of numpy arrays (if nroots > 1)
            MP2 energy and initial guess(es) for CISD coefficients.

        '''
        if eris is None: eris = self.ao2mo(self.mo_coeff)
        nocc = self.nocc
        mo_e = eris.mo_energy
        e_ia = lib.direct_sum('i-a->ia', mo_e[:nocc], mo_e[nocc:])
        ci0 = 1
        ci1 = eris.fock[:nocc,nocc:] / e_ia
        eris_ovvo = _cp(eris.ovvo)
        ci2  = 2 * eris_ovvo.transpose(0,3,1,2)
        ci2 -= eris_ovvo.transpose(0,3,2,1)
        ci2 /= lib.direct_sum('ia,jb->ijab', e_ia, e_ia)
        self.emp2 = numpy.einsum('ijab,iabj', ci2, eris_ovvo)
        logger.info(self, 'Init t2, MP2 energy = %.15g', self.emp2)

        if abs(self.emp2) < 1e-3 and abs(ci1).sum() < 1e-3:
            # To avoid ci1 being stuck at local minimum
            ci1 = 1e-1 / e_ia

        ci_guess = amplitudes_to_cisdvec(ci0, ci1, ci2)


        if nroots > 1:
            civec_size = ci_guess.size
            dtype = ci_guess.dtype
            nroots = min(ci1.size+1, nroots)  # Consider Koopmans' theorem only

            if diag is None:
                idx = range(1, nroots)
            else:
                idx = diag[:ci1.size+1].argsort()[1:nroots]  # exclude HF determinant

            ci_guess = [ci_guess]
            for i in idx:
                g = numpy.zeros(civec_size, dtype)
                g[i] = 1.0
                ci_guess.append(g)
        return self.emp2, ci_guess

    contract = contract
    make_diagonal = make_diagonal

    def _dot(self, x1, x2, nmo=None, nocc=None):
        if nmo is None: nmo = self.nmo
        if nocc is None: nocc = self.nocc
        return dot(x1, x2, nmo, nocc)

    def ao2mo(self, mo_coeff=None):
        nmo = self.nmo
        nao = self.mo_coeff.shape[0]
        nmo_pair = nmo * (nmo+1) // 2
        nao_pair = nao * (nao+1) // 2
        mem_incore = (max(nao_pair**2, nmo**4) + nmo_pair**2) * 8/1e6
        mem_now = lib.current_memory()[0]
        if (self._scf._eri is not None and
            (mem_incore+mem_now < self.max_memory) or self.mol.incore_anyway):
            return ccsd._make_eris_incore(self, mo_coeff)

        if getattr(self._scf, 'with_df', None):
            logger.warn(self, 'CISD detected DF being used in the HF object. '
                        'MO integrals are computed based on the DF 3-index tensors.\n'
                        'It\'s recommended to use dfccsd.CCSD for the '
                        'DF-CISD calculations')
            return ccsd._make_df_eris_outcore(self, mo_coeff)

        return ccsd._make_eris_outcore(self, mo_coeff)

    def _add_vvvv(self, c2, eris, out=None, t2sym=None):
        return ccsd._add_vvvv(self, None, c2, eris, out, False, t2sym)

    def to_fcivec(self, cisdvec, norb=None, nelec=None, frozen=None):
        if norb is None: norb = self.nmo
        if nelec is None: nelec = self.nocc*2
        return to_fcivec(cisdvec, norb, nelec, frozen)

    def from_fcivec(self, fcivec, norb=None, nelec=None):
        if norb is None: norb = self.nmo
        if nelec is None: nelec = self.nocc*2
        return from_fcivec(fcivec, norb, nelec)

    make_rdm1 = make_rdm1
    make_rdm2 = make_rdm2

    trans_rdm1 = trans_rdm1

    as_scanner = as_scanner

    def dump_chk(self, ci=None, frozen=None, mo_coeff=None, mo_occ=None):
        if not self.chkfile:
            return self

        if ci is None: ci = self.ci
        if frozen is None: frozen = self.frozen
        # "None" cannot be serialized by the chkfile module
        if frozen is None:
            frozen = 0

        ci_chk = {'e_corr': self.e_corr,
                  'ci': ci,
                  'frozen': frozen}

        if mo_coeff is not None: ci_chk['mo_coeff'] = mo_coeff
        if mo_occ is not None: ci_chk['mo_occ'] = mo_occ
        if self._nmo is not None: ci_chk['_nmo'] = self._nmo
        if self._nocc is not None: ci_chk['_nocc'] = self._nocc

        lib.chkfile.save(self.chkfile, 'cisd', ci_chk)

    def amplitudes_to_cisdvec(self, c0, c1, c2):
        return amplitudes_to_cisdvec(c0, c1, c2)

    def cisdvec_to_amplitudes(self, civec, nmo=None, nocc=None, copy=True):
        if nmo is None: nmo = self.nmo
        if nocc is None: nocc = self.nocc
        return cisdvec_to_amplitudes(civec, nmo, nocc, copy=copy)

    def density_fit(self):
        raise NotImplementedError

    def nuc_grad_method(self):
        from pyscf.grad import cisd
        return cisd.Gradients(self)

    to_gpu = lib.to_gpu

class RCISD(CISD):
    pass

from pyscf import scf
scf.hf.RHF.CISD = lib.class_as_method(RCISD)
scf.rohf.ROHF.CISD = None

def _cp(a):
    return np.asarray(a, order='C')


def ci2fock(fci_vec, norb, nelec):
    """
    Transform FCI vector in (na, nb) form into full Fock-space CI vector (2^norb x 2^norb).

    Args:
        fci_vec: np.ndarray, shape (n_alpha_det, n_beta_det)
                 FCI coefficients in PySCF's string-based representation.
        norb: int, number of spatial orbitals
        nelec: tuple or int
               If tuple -> (n_alpha, n_beta)
               If int   -> total electrons (assume closed shell)

    Returns:
        fock: np.ndarray, shape (2**norb, 2**norb)
              CI coefficients in Fock basis representation.
              Row index = alpha occupation bitstring
              Col index = beta occupation bitstring
    """
    if isinstance(nelec, int):
        n_alpha = n_beta = nelec // 2
    else:
        n_alpha, n_beta = nelec

    fock = np.zeros((2**norb, 2**norb), dtype=fci_vec.dtype)

    # list of bitstring integers for alpha and beta sectors
    alpha_strs = cistring.make_strings(range(norb), n_alpha)
    beta_strs  = cistring.make_strings(range(norb), n_beta)

    for ia, a_occ in enumerate(alpha_strs):
        for ib, b_occ in enumerate(beta_strs):
            fock[a_occ, b_occ] = fci_vec[ia, ib]
    
    return fock

def cas_energy_of_ci(mc, ci):
    """
    Evaluate total energy of a CASCI/CASSCF CI vector.
    Works with RHF reference + direct_spin1 FCI solver.
    """
    mol = mc.mol
    mo  = mc.mo_coeff
    ncore, ncas = mc.ncore, mc.ncas
    nelecas = mc.nelecas  # should be (na, nb) or int

    # Ensure types are what FCI expects
    norb = int(ncas)
    if isinstance(nelecas, int):
        # If given total electrons, split assuming RHF balance
        na = nb = nelecas // 2
        nelec = (na, nb)
    else:
        nelec = tuple(nelecas)

    # Effective 1e Hamiltonian in active space + constant core energy
    h1eff, ecore = mc.h1e_for_cas(mo)  # h1eff: (ncas, ncas)
    

    # Active MO block
    mo_act = mo[:, ncore:ncore+ncas]

    # Build 2e integrals in packed 2-index (physicist's) format expected by FCI
    # Shape: (npair, npair), where npair = ncas*(ncas+1)//2
    eri_packed = ao2mo.kernel(mol, mo_act)  # compact=True by default

    # Now compute <CI|H_active|CI>
    e_active = fci.direct_spin1.energy(h1eff, eri_packed, ci, norb, nelec)

    return ecore + e_active


def list_rhf_mp2_t2(m, topk=20, thresh=1e-6):
    """
    Return a list of (i,j,a,b, I,J,A,B, t2) for RHF MP2,
    where (i,j,a,b) are occ/vir slots and (I,J,A,B) are absolute MO indices.
    only keep the topk largest amplitudes

    example:
    >>> mol = gto.M(atom="O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587",
                basis="sto-3g", spin=0, verbose=0)
    >>> mf = scf.RHF(mol).run()
    >>> m = mp.MP2(mf).run()
    >>> t2 = m.t2                      # shape (nocc, nocc, nvir, nvir)
    >>> nocc = m.nocc
    nvir = m.nmo - nocc

    >>> for i,j,a,b,I,J,A,B,t in list_rhf_mp2_t2(m, topk=15, thresh=1e-6):
    >>>     print(f"|{I},{J} -> {A},{B}|  (slots i={i},j={j},a={a},b={b})   t2 = {t:+.6e}")
    """
    # indices with |t2| > thresh
    mask_idx = np.argwhere(np.abs(t2) > thresh)
    if mask_idx.size == 0:
        return []

    vals = t2[tuple(mask_idx.T)]
    order = np.argsort(-np.abs(vals))
    if topk:
        order = order[:topk]

    rows = []
    for k in order:
        i, j, a, b = map(int, mask_idx[k])
        I, J = i, j                # occupied absolute MO indices
        A, B = nocc + a, nocc + b  # virtual absolute MO indices
        rows.append((i, j, a, b, I, J, A, B, float(vals[k])))
    return rows


from itertools import combinations
from math import comb

def extract_sd_excitations(fci_vec, norb, neleca, nelecb, thresh=1e-8):
    """
    Scan through the FCI coefficient array fci_vec (shape (C(norb,neleca), C(norb,nelecb))),
    and return all single and double excitations (particle/hole lists) relative to the
    reference determinant, along with their amplitudes.
    
    Returns
    -------
    a_idxs : List[List[int]]
        Particle (excited-to) spin-orbital lists.
    i_idxs : List[List[int]]
        Hole     (excited-from) spin-orbital lists.
    amps   : List[float]
        The non-zero CI amplitude for each excitation.
    """
    na = comb(norb, neleca)
    nb = comb(norb, nelecb)
    if fci_vec.shape != (na, nb):
        raise ValueError(f"Expected fci_vec shape {(na,nb)}, got {fci_vec.shape}")
    
    # build all alpha and beta determinants as sorted tuples of spatial orbitals
    a_dets = list(combinations(range(norb), neleca))
    b_dets = list(combinations(range(norb), nelecb))
    ref_a, ref_b = set(a_dets[0]), set(b_dets[0])
    
    a_idxs = []
    i_idxs = []
    amps   = []
    
    for ia, alpha in enumerate(a_dets):
        alpha = set(alpha)
        for ib, beta in enumerate(b_dets):
            amp = fci_vec[ia, ib]
            if abs(amp) < thresh:
                continue
            
            beta = set(beta)
            # holes = in ref but not in det
            holes_a = sorted(ref_a - alpha)
            holes_b = sorted(ref_b - beta)
            # parts = in det but not in ref
            parts_a = sorted(alpha - ref_a)
            parts_b = sorted(beta - ref_b)
            
            n_exc = len(holes_a) + len(holes_b)
            # only singles or doubles
            if not (1 <= n_exc <= 2):
                continue
            
            # spin-orbital indexing
            holes = holes_a + [h + norb for h in holes_b]
            parts = parts_a + [p + norb for p in parts_b]
            
            a_idxs.append(holes)
            i_idxs.append(parts)
            amps.append(amp)
    
    return a_idxs, i_idxs, amps

def get_energy(psi, h):
    '''The internal energy evalutation methods of psi
    involves ci0, which may cause problem, so here we
    implement a clean version'''
    c = psi.dp_ci(psi.ci_f)
    uc = psi.uop(c)
    # uc, huc = psi_to_excite.hc_x (psi_to_excite.x, h)[1:3]
    huc = psi_to_excite.contract_h2 (h, uc)
    uc, huc = uc.ravel(), huc.ravel()
    cu = uc.conj ()
    cuuc = cu.dot (uc)
    cuhuc = cu.dot (huc)
    return cuhuc / cuuc


def fock_ci_to_cas_ci(ncas, neleacas, nelebcas, uscc_ci):
    ''' Convert Fock space CI to CASCI, FOCK CI is the tensor product of
    fragment CIs, of size 2^norb x 2^norb. Many elemnts of FOCK CI
    are 0 and corresponds to invalid determinants. CASCI is a 
    c(norb,nelec) x c(norb,nelec) matrix. '''
    comb_str_a = cistring.make_strings(range(ncas), neleacas)
    comb_str_b = cistring.make_strings(range(ncas), nelebcas)
    cas_ci = np.zeros((len(comb_str_a), len(comb_str_b)), dtype=uscc_ci.dtype)
    for i, bra in enumerate(comb_str_a):
        for j, ket in enumerate(comb_str_b):
            cas_ci[i,j] = uscc_ci[bra, ket]
    return cas_ci

if __name__ == '__main__':
    from pyscf import ao2mo
    from pyscf.mp import MP2
    import numpy as np
    

    mol = gto.Mole()
    mol.verbose = 0

    mol.atom = '''H      0.000000000000   0.000000000000   0.000000000000
    H      1.000000000000   0.000000000000   0.000000000000
    H      0.273746762116   2.195450598147   0.100000000000
    H      1.232912762116   1.895450598147  -0.100000000000'''
    mol.basis = 'sto3g'
    mol.build()
    mf = scf.RHF(mol).run()
    mp2 = MP2(mf)
    mp2.kernel()

    print("mp2 energy = ", mp2.e_tot)

    nmo = mf.mo_coeff.shape[1]
    nocc = mol.nelectron // 2
    c0 = 1.0
    t1 = np.zeros((nocc, nmo-nocc))
    t2 = mp2.t2 

    civec = amplitudes_to_cisdvec(c0, t1, t2)

    norb = mf.mo_coeff.shape[1]
    nelec = mol.nelectron
    fci_vec = to_fcivec(civec, norb, nelec)

    fci_vec /= np.linalg.norm(fci_vec)


    # first let's verify that the fci energy 
    # equals to the mp2 energy
    mc = mcscf.CASCI(mf, norb, nelec).run()
    e_mc_mp2_ci = cas_energy_of_ci(mc, fci_vec)
    print("mp2 ci energy = ", e_mc_mp2_ci)

    # Then, let's assign the fci to uscc
    # and check that the energy is the same
    mc_uscc = mcscf.CASCI(mf, norb, nelec)
    mc_uscc.mo_coeff = mf.mo_coeff

    eval_fci_energy = cas_energy_of_ci(mc_uscc, fci_vec)
    
    mc_uscc.fcisolver = lasuccsd.FCISolver_USCC(mol, [], [])
    norb_f = [norb]

    fcis = mc_uscc.fcisolver
    fock_vec = ci2fock(fci_vec, norb, nelec)
    
    psi =  getattr (fcis, 'psi', fcis.build_psi ([fock_vec], norb, norb_f, nelec))
    h1eff,e_core= mc_uscc.get_h1eff(mc_uscc.mo_coeff)
    h2eff = mc_uscc.get_h2eff()
    h = [e_core, h1eff, h2eff]

    energy = psi.energy_tot(psi.x, h)

    print("psi energy = ", energy)

    # Then, let's put psi ci_0 to 0, and use ucc amplitudes to
    # excite it instead, it should give the same energy and
    # wave function as the fci initialization

    # we first create the HF gound state psi
    fci_vec0 = np.zeros_like(fci_vec)
    fci_vec0[0, 0] = 1.0
    fock_vec0 = ci2fock(fci_vec0, norb, nelec)

    psi0 =  getattr (fcis, 'psi', fcis.build_psi ([fock_vec0], norb, norb_f, nelec))

    energy0 = psi0.energy_tot(psi0.x, [e_core, h1eff, h2eff])

    # verify that the psi0 is indeed HF ground state
    print("psi0 energy = ", energy0)
    print("HF energy = ", mf.e_tot)

    
    # Then we excite psi0 using the MP2 amplitudes

    a_idxs, i_idxs, amps = extract_sd_excitations(fci_vec, norb, nocc, nocc)
    a_idx_np = [np.array(x, dtype=np.uint8) for x in a_idxs]
    i_idx_np = [np.array(x, dtype=np.uint8) for x in i_idxs]

    uscc_fsolver = lasuccsd.FCISolver_USCC(mol, a_idx_np, i_idx_np)
    uscc_fsolver.mo_coeff = mf.mo_coeff

    # print("t2 = \n", t2)
    psi_to_excite =  getattr (uscc_fsolver, 'psi', uscc_fsolver.build_psi ([fock_vec0], norb, norb_f, nelec))
    psi_to_excite.x[psi_to_excite.nconstr:psi_to_excite.nconstr+len(amps)] = amps
    print("psi energy internal mehods = ", psi_to_excite.energy_tot(psi_to_excite.x, [e_core, h1eff, h2eff])) # problem matic
    print("psi energy = ", get_energy(psi, h))

    # psi_cas_ci = fock_ci_to_cas_ci(norb, nelec // 2, nelec // 2, uc)

    # # verify that the ci created by excitataion amplitudes are 
    # # exactly the same as the mp2 ci 
    # print("fcivec = \n", fci_vec)
    # print("psi_cas_ci = \n", psi_cas_ci)
