const σx = ComplexF64[0 1; 1 0]
const σy = ComplexF64[0 -im; im 0]
const σz = ComplexF64[1 0; 0 -1]
const I2 = Matrix{ComplexF64}(I, 2, 2)

function embed_one_site(op1::AbstractMatrix, n::Int, j::Int)
    ops = [I2 for _ in 1:n]
    ops[j] = op1
    out = ops[1]
    for k in 2:n
        out = kron(out, ops[k])
    end
    return out
end

function embed_two_site(op1::AbstractMatrix, op2::AbstractMatrix, n::Int, j::Int)
    @assert 1 <= j < n
    ops = [I2 for _ in 1:n]
    ops[j] = op1
    ops[j+1] = op2
    out = ops[1]
    for k in 2:n
        out = kron(out, ops[k])
    end
    return out
end

function heisenberg_dense_hamiltonian_from_gates(n, J)
    H = zeros(ComplexF64, 2^n, 2^n)
    for j in 1:n-1
        H .+= (J[j] / 2) * embed_two_site(σx, σx, n, j)
        H .+= (J[j] / 2) * embed_two_site(σy, σy, n, j)
        H .+= J[j] * embed_two_site(σz, σz, n, j)
    end
    return H
end

function L_vector_dense_from_gates(n, J, t, ks, bitstring)
    psi = basis_state(bitstring)
    H = heisenberg_dense_hamiltonian_from_gates(n, J)
    U_exact = exp(-im * t * H)

    L = zeros(Float64, length(ks))
    for j in eachindex(ks)
        Uj = trotter_dense(n, J, t, ks[j])
        amp = psi' * (Uj' * U_exact) * psi
        L[j] = abs2(amp)
    end
    return L
end

function compare_exact_vs_trotter_dense(n, J, t, ks, bitstring)
    psi = basis_state(bitstring)
    H = heisenberg_dense_hamiltonian_from_gates(n, J)
    U_exact = exp(-im * t * H)

    println("Dense exact vs dense Trotter amplitudes:")
    for k in ks
        Uj = trotter_dense(n, J, t, k)
        amp = psi' * (Uj' * U_exact) * psi
        println("k = $k, amp = $amp, |amp|^2 = $(abs2(amp))")
    end
end

function sanity_check_L_all(n, J, t, ks, k_ref_opt, k_ref_eval, sites, initial_state;
    cutoff_opt=0.0, maxdim_opt=50,
    cutoff_eval=0.0, maxdim_eval=200,
    order=1, order_ref_opt=1, order_ref_eval=1)

    bitstring = join(initial_state)

    psi = MPS(sites, initial_state)
    normalize!(psi)

    # dense exact using corrected Hamiltonian
    L_dense_corr = L_vector_dense_from_gates(n, J, t, ks, bitstring)

    # middle-out optimizer overlaps
    L_opt = L_vector_middle_out(
        n, J, t, ks, k_ref_opt, sites, psi;
        cutoff=cutoff_opt, maxdim=maxdim_opt,
        order=order, order_ref=order_ref_opt
    )

    # direct MPO reference overlaps
    L_ref = L_vector_direct_mpo(
        n, J, t, ks, k_ref_eval, sites, psi;
        cutoff=cutoff_eval, maxdim=maxdim_eval,
        order=order, order_ref=order_ref_eval
    )

    println("L_dense_corrected = ", L_dense_corr)
    println("L_middle_out      = ", L_opt)
    println("L_direct_MPO      = ", L_ref)

    return (
        L_dense_corrected = L_dense_corr,
        L_middle_out = L_opt,
        L_direct_MPO = L_ref,
    )
end

function gram_matrix_dense_from_gates(n, J, t, ks, bitstring)
    psi = basis_state(bitstring)
    r = length(ks)
    M = zeros(Float64, r, r)

    for i in 1:r
        Ui = trotter_dense(n, J, t, ks[i])
        for j in 1:r
            Uj = trotter_dense(n, J, t, ks[j])
            amp = psi' * (Ui' * Uj) * psi
            M[i, j] = abs2(amp)
        end
    end
    return M
end