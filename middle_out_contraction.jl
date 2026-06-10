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

function identity_mpo(sites)
    n = length(sites)
    Id = MPO(sites)
    for j in 1:n
        Id[j] = op(sites, "Id", j)
    end
    return Id
end

function get_step_gates(n, J, dt, sites)
    alpha = [x * dt for x in J]
    beta  = [2 * x * dt for x in J]
    gates = ITensor[]

    for j in 1:2:n-1
        s1, s2 = sites[j], sites[j+1]
        push!(gates, op("RXX", s1, s2; θ = alpha[j]))
        push!(gates, op("RYY", s1, s2; θ = alpha[j]))
        push!(gates, op("RZZ", s1, s2; θ = beta[j]))
    end

    if n > 2
        for j in 2:2:n-1
            s1, s2 = sites[j], sites[j+1]
            push!(gates, op("RXX", s1, s2; θ = alpha[j]))
            push!(gates, op("RYY", s1, s2; θ = alpha[j]))
            push!(gates, op("RZZ", s1, s2; θ = beta[j]))
        end
    end
    return gates
end

# One backward (dagger) Trotter step:
# (g_m ... g_2 g_1)† = g_1† g_2† ... g_m†
function get_step_gates_dag(n, J, dt, sites)
    gates = get_step_gates(n, J, -dt, sites)
    return reverse(gates)
end

# Build ONE forward step as an MPO
function get_step_MPO(n, J, dt, sites, cutoff, maxdim)
    step_gates = get_step_gates(n, J, dt, sites)
    S = identity_mpo(sites)
    S = apply(step_gates, S; cutoff=cutoff, maxdim=maxdim)
    return S
end

# Build ONE backward step as an MPO
function get_step_MPO_dag(n, J, dt, sites, cutoff, maxdim)
    step_gates_dag = get_step_gates_dag(n, J, dt, sites)
    S = identity_mpo(sites)
    S = apply(step_gates_dag, S; cutoff=cutoff, maxdim=maxdim)
    return S
end

# Left multiplication: A * F
function left_multiply(A::MPO, F::MPO; cutoff, maxdim)
    return apply(A, F; cutoff=cutoff, maxdim=maxdim)
end

# Right multiplication: F * A
# Implemented as (A† * F†)† to avoid index-convention issues
function right_multiply(F::MPO, A::MPO; cutoff, maxdim)
    X = apply(dag(A), dag(F); cutoff=cutoff, maxdim=maxdim)
    return dag(X)
end

function build_F(n, J, t, k, sites, cutoff, maxdim)
    Id = identity_mpo(sites)
    F_components = MPO[]

    for i in eachindex(k)
        dt_i = t / k[i]
        step_i_dag_mpo = get_step_MPO_dag(n, J, dt_i, sites, cutoff, maxdim)

        for j in eachindex(k)
            dt_j = t / k[j]
            step_j_mpo = get_step_MPO(n, J, dt_j, sites, cutoff, maxdim)

            F = deepcopy(Id)
            time_i = 0.0
            time_j = 0.0

            while (time_i < t - 1e-12) || (time_j < t - 1e-12)
                if (time_j <= time_i) && (time_j < t - 1e-12)
                    # F <- F * S_j
                    F = right_multiply(F, step_j_mpo; cutoff=cutoff, maxdim=maxdim)
                    time_j += dt_j
                elseif time_i < t - 1e-12
                    # F <- S_i† * F
                    F = left_multiply(step_i_dag_mpo, F; cutoff=cutoff, maxdim=maxdim)
                    time_i += dt_i
                end
            end

            push!(F_components, F)
        end
    end

    return F_components
end
