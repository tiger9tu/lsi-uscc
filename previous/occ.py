from pyscf.fci import cistring

def decode_string(bitstring, norb):
    # return list of occupied orbitals (0-based)
    return [i for i in range(norb) if (bitstring >> i) & 1]

def list_cas_determinants(norb, nelec):
    # nelec = (nalpha, nbeta)
    ndeta = cistring.num_strings(norb, nelec[0])
    ndetb = cistring.num_strings(norb, nelec[1])
    addrs_a = list(range(ndeta))
    addrs_b = list(range(ndetb))
    strsa = cistring.addrs2str(norb, nelec[0], addrs_a)  # alpha strings as bit patterns
    strsb = cistring.addrs2str(norb, nelec[1], addrs_b)  # beta strings

    alpha_occs = [decode_string(int(x), norb) for x in strsa]
    beta_occs  = [decode_string(int(x), norb) for x in strsb]

    # returns lists so that coefficient ci[a,b] corresponds to
    # alpha_occs[a] (occupied alpha orbitals) and beta_occs[b]
    return alpha_occs, beta_occs

# Example usage for 4 orbitals, 2 alpha, 1 beta
norb = 4
nelec = (2, 1)
alpha_occs, beta_occs = list_cas_determinants(norb, nelec)
print("Alpha strings (index -> occupied):")
for idx, occ in enumerate(alpha_occs):
    print(f"{idx}: {occ}")
print("Beta strings:")
for idx, occ in enumerate(beta_occs):
    print(f"{idx}: {occ}")
