include("optimization_problem.jl")
include("dense_matrix_computation.jl")

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

function observable_Z(n::Int, j::Int)
    return embed_one_site(σz, n, j)
end

function observable_ZZ(n::Int, j::Int)
    return embed_two_site(σz, σz, n, j)
end

function expect_state(O::AbstractMatrix, ψ::AbstractVector)
    return real(ψ' * O * ψ)
end

function exact_state_dense(n, J, t, bitstring)
    ψ0 = basis_state(bitstring)
    H = heisenberg_dense_hamiltonian(n, J)
    U = exp(-im * t * H)
    return U * ψ0
end

function trotter_state_dense(n, J, t, k, bitstring)
    ψ0 = basis_state(bitstring)
    U = trotter_dense(n, J, t, k)
    return U * ψ0
end

function compare_observable_dense(n, J, t, ks, coeffs, bitstring, O)
    ψ_exact = exact_state_dense(n, J, t, bitstring)
    exact_val = expect_state(O, ψ_exact)

    trotter_vals = Float64[]
    for k in ks
        ψk = trotter_state_dense(n, J, t, k, bitstring)
        push!(trotter_vals, expect_state(O, ψk))
    end

    mpf_val = sum(coeffs[j] * trotter_vals[j] for j in eachindex(ks))

    trotter_errs = abs.(trotter_vals .- exact_val)
    mpf_err = abs(mpf_val - exact_val)

    return (
        exact = exact_val,
        trotter_vals = trotter_vals,
        mpf_val = mpf_val,
        trotter_errs = trotter_errs,
        mpf_err = mpf_err,
    )
end

function print_observable_report(name, res, ks)
    println("Observable: ", name)
    println("  exact      = ", res.exact)
    for (idx, k) in enumerate(ks)
        println("  k = $(k)     = ", res.trotter_vals[idx],
                "    |error| = ", res.trotter_errs[idx])
    end
    println("  MPF        = ", res.mpf_val,
            "    |error| = ", res.mpf_err)
end