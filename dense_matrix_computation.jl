using LinearAlgebra

function RXX_dense(θ)
    c = cos(θ/2)
    s = -im * sin(θ/2)
    ComplexF64[
        c 0 0 s
        0 c s 0
        0 s c 0
        s 0 0 c
    ]
end

function RYY_dense(θ)
    c = cos(θ/2)
    s = -im * sin(θ/2)
    ComplexF64[
        c 0 0 -s
        0 c s 0
        0 s c 0
        -s 0 0 c
    ]
end

function RZZ_dense(θ)
    ComplexF64[
        exp(-im*θ/2) 0 0 0
        0 exp(im*θ/2) 0 0
        0 0 exp(im*θ/2) 0
        0 0 0 exp(-im*θ/2)
    ]
end

function embed_two_qubit_gate(U2::AbstractMatrix, n::Int, j::Int)
    @assert size(U2) == (4,4)
    @assert 1 <= j < n

    leftdim  = 2^(j-1)
    rightdim = 2^(n-j-1)

    Ileft  = Matrix{ComplexF64}(I, leftdim, leftdim)
    Iright = Matrix{ComplexF64}(I, rightdim, rightdim)

    return kron(Ileft, kron(U2, Iright))
end

function get_step_dense(n, J, dt)
    alpha = [x * dt for x in J]
    beta  = [2 * x * dt for x in J]

    U = Matrix{ComplexF64}(I, 2^n, 2^n)

    # odd bonds: 1,3,5,...
    for j in 1:2:n-1
        U = embed_two_qubit_gate(RXX_dense(alpha[j]), n, j) * U
        U = embed_two_qubit_gate(RYY_dense(alpha[j]), n, j) * U
        U = embed_two_qubit_gate(RZZ_dense(beta[j]),  n, j) * U
    end

    # even bonds: 2,4,6,...
    if n > 2
        for j in 2:2:n-1
            U = embed_two_qubit_gate(RXX_dense(alpha[j]), n, j) * U
            U = embed_two_qubit_gate(RYY_dense(alpha[j]), n, j) * U
            U = embed_two_qubit_gate(RZZ_dense(beta[j]),  n, j) * U
        end
    end

    return U
end

function trotter_dense(n, J, t, k)
    dt = t / k
    Ustep = get_step_dense(n, J, dt)
    U = Matrix{ComplexF64}(I, 2^n, 2^n)
    for _ in 1:k
        U = Ustep * U
    end
    return U
end

function basis_state(bitstring::AbstractString)
    v = ComplexF64[1.0]
    for ch in bitstring
        if ch == '0'
            v = kron(v, ComplexF64[1, 0])
        elseif ch == '1'
            v = kron(v, ComplexF64[0, 1])
        else
            error("Bitstring must contain only '0' and '1'")
        end
    end
    return v
end

function exact_F_matrix(n, J, t, ki, kj)
    Ui = trotter_dense(n, J, t, ki)
    Uj = trotter_dense(n, J, t, kj)
    return Ui' * Uj
end

function exact_F_list(n, J, t, ks)
    Fs = Matrix{ComplexF64}[]
    for ki in ks
        Ui = trotter_dense(n, J, t, ki)
        for kj in ks
            Uj = trotter_dense(n, J, t, kj)
            push!(Fs, Ui' * Uj)
        end
    end
    return Fs
end