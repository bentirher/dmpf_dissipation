using ITensors, ITensorMPS
using LinearAlgebra

ITensors.op(::OpName"RXX", ::SiteType"Qubit"; θ) = begin
    c = cos(θ/2)
    s = -im*sin(θ/2)
    ComplexF64[
        c 0 0 s
        0 c s 0
        0 s c 0
        s 0 0 c
    ]
end

ITensors.op(::OpName"RYY", ::SiteType"Qubit"; θ) = begin
    c = cos(θ/2)
    s = -im*sin(θ/2)
    ComplexF64[
        c 0 0 -s
        0 c s 0
        0 s c 0
        -s 0 0 c
    ]
end

ITensors.op(::OpName"RZZ", ::SiteType"Qubit"; θ) = begin
    ComplexF64[
        exp(-im*θ/2) 0 0 0
        0 exp(im*θ/2) 0 0
        0 0 exp(im*θ/2) 0
        0 0 0 exp(-im*θ/2)
    ]
end



function matrix_F(n, J, t, k)
    F_components = []
    for i in eachindex(k)
        S_i_dag = get_circuit_matrix(n, J, t, k[i])' #' is dagger
        for j in eachindex(k)
            F = I(2^n)
            S_j = get_circuit_matrix(n, J, t, k[j])
            k_i_used = 0
            k_j_used = 0
            while (k_i_used < k[i]) && (k_j_used < k[j])
                F = S_j * F
                F = S_i_dag * F
                k_i_used += 1
                k_j_used += 1
            end

            while k_i_used < k[i]
                F = S_i_dag * F
                k_i_used += 1
            end

            while k_j_used < k[j]
                F = S_j * F
                k_j_used += 1
            end
            push!(F_components, F)
        end
    end    
    return F_components
end


function build_F(n, J, t, k, sites, cutoff, maxdim)
    Id = MPO(sites)
    for j in 1:n
        # `op` constructs a single-site operator (identity) on site n
        Id[j] = op(sites, "Id", j)  # adjust if your ITensor requires different call
    end

    F_components = []
    for i in eachindex(k)
        S_i_dag = get_MPO(n, J, t, k[i], sites, cutoff, maxdim)
        for j in eachindex(k)
            F = Id
            S_j = get_MPO(n, J, t, k[j], sites, cutoff, maxdim)
            k_i_used = 0
            k_j_used = 0
            while (k_i_used < k[i]) && (k_j_used < k[j])
                F = apply(S_j, F; cutoff = cutoff, maxdim = maxdim)
                F = apply(S_i_dag, F; cutoff = cutoff, maxdim = maxdim)
                k_i_used += 1
                k_j_used += 1
            end

            while k_i_used < k[i]
                F = apply(S_i_dag, F; cutoff = cutoff, maxdim = maxdim)
                k_i_used += 1
            end

            while k_j_used < k[j]
                F = apply(S_j, F; cutoff = cutoff, maxdim = maxdim)
                k_j_used += 1
            end
            # normalize!(F) # If we normalize, the evs get all fucked up
            push!(F_components, F)
        end
    end    
    return F_components
end

function get_MPO(n, J, t, k, sites, cutoff, maxdim)

    dt = (t / k)
    alpha = [ x * dt for x in J ]
    beta = [ 2 * x * dt for x in J]

    Id = MPO(sites)
    for j in 1:(n)
        # `op` constructs a single-site operator (identity) on site n
        Id[j] = op(sites, "Id", j)  # adjust if your ITensor requires different call
    end

    S = Id
    gates = ITensor[]
    for _ in 1:k       
        for j in 1:2:n-1
            s1 = sites[j]
            s2 = sites[j+1]

            RXX = op("RXX", s1, s2; θ = alpha[j])
            push!(gates, RXX)
            RYY = op("RYY", s1, s2; θ = alpha[j])
            push!(gates, RYY)
            RZZ = op("RZZ", s1, s2; θ = beta[j])
            push!(gates, RZZ)
        end
        if n > 2
            for j in 2:2:n-1
                s1 = sites[j]
                s2 = sites[j+1]

                RXX = op("RXX", s1, s2; θ = alpha[j])
                push!(gates, RXX)
                RYY = op("RYY", s1, s2; θ = alpha[j])
                push!(gates, RYY)
                RZZ = op("RZZ", s1, s2; θ = beta[j])
                push!(gates, RZZ)
            end
        end
    end
    S = apply(gates, S; cutoff = cutoff, maxdim = maxdim)
    return S
end

function get_circuit_matrix(n, J, t, k)

    dt = (t / k)
    alpha = [ x * dt for x in J ]
    beta = [ 2 * x * dt for x in J]

    rxx = [[
            cos(x/2) 0 0 -im*sin(x/2);
            0 cos(x/2) -im*sin(x/2) 0;
            0 -im*sin(x/2) cos(x/2) 0;
            -im*sin(x/2) 0 0 cos(x/2) ;
            ] for x in alpha ]
    ryy = [[
            cos(x/2) 0 0 im*sin(x/2);
            0 cos(x/2) -im*sin(x/2) 0;
            0 -im*sin(x/2) cos(x/2) 0;
            im*sin(x/2) 0 0 cos(x/2) ;
            ] for x in alpha ]
    
    rzz = [[
            exp(-im*x/2) 0 0 0;
            0 exp(im*x/2) 0 0;
            0 0 exp(im*x/2) 0;
            0 0 0 exp(-im*x/2);
            ] for x in beta ]
    
    S = Matrix{ComplexF64}(I, total_dim, total_dim)
    for _ in 1:k 
        for j in 1:2:n-1
            ids = [ Matrix{ComplexF64}(I, 2, 2) for _ in 1:n]
            deleteat!(ids, j:j+1)
            insert!(ids, j, rxx)
            gate = reduce(kron, ids)
            S = gate * S

            ids = [ Matrix{ComplexF64}(I, 2, 2) for _ in 1:n]
            deleteat!(ids, j:j+1)
            insert!(ids, j, ryy)
            gate = reduce(kron, ids)
            S = gate * S

            ids = [ Matrix{ComplexF64}(I, 2, 2) for _ in 1:n]
            deleteat!(ids, j:j+1)
            insert!(ids, j, rzz)
            gate = reduce(kron, ids)
            S = gate * S       
        end
        if n > 2
            for j in 2:2:n-1
                ids = [ Matrix{ComplexF64}(I, 2, 2) for _ in 1:n]
                deleteat!(ids, j:j+1)
                insert!(ids, j, rxx)
                gate = reduce(kron, ids)
                S = gate * S

                ids = [ Matrix{ComplexF64}(I, 2, 2) for _ in 1:n]
                deleteat!(ids, j:j+1)
                insert!(ids, j, ryy)
                gate = reduce(kron, ids)
                S = gate * S

                ids = [ Matrix{ComplexF64}(I, 2, 2) for _ in 1:n]
                deleteat!(ids, j:j+1)
                insert!(ids, j, rzz)
                gate = reduce(kron, ids)
                S = gate * S           
            end
        end
    end
    return S
end