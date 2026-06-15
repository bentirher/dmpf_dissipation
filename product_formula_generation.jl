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

# General interface that admits order 1, 2 and 4

function get_step_gates(n, J, dt, sites; order::Int=1)
    if order == 1
        return get_step_gates_order1(n, J, dt, sites)
    elseif order == 2
        return get_step_gates_order2(n, J, dt, sites)
    elseif order == 4
        return get_step_gates_order4(n, J, dt, sites)
    else
        error("Unsupported product-formula order = $order. Use 1, 2, or 4.")
    end
end

# Layer generation (edge-coloring)

function odd_layer_gates(n, J, dt, sites)
    alpha = [x * dt for x in J]
    beta  = [2 * x * dt for x in J]
    gates = ITensor[]
    for j in 1:2:n-1
        s1, s2 = sites[j], sites[j+1]
        push!(gates, op("RXX", s1, s2; θ=alpha[j]))
        push!(gates, op("RYY", s1, s2; θ=alpha[j]))
        push!(gates, op("RZZ", s1, s2; θ=beta[j]))
    end
    return gates
end

function even_layer_gates(n, J, dt, sites)
    alpha = [x * dt for x in J]
    beta  = [2 * x * dt for x in J]
    gates = ITensor[]
    if n > 2
        for j in 2:2:n-1
            s1, s2 = sites[j], sites[j+1]
            push!(gates, op("RXX", s1, s2; θ=alpha[j]))
            push!(gates, op("RYY", s1, s2; θ=alpha[j]))
            push!(gates, op("RZZ", s1, s2; θ=beta[j]))
        end
    end
    return gates
end

# First-order product formula

function get_step_gates_order1(n, J, dt, sites)
    return vcat(
        odd_layer_gates(n, J, dt, sites),
        even_layer_gates(n, J, dt, sites),
    )
end

# Second-order product formula in the symmetric Suzuki form

function get_step_gates_order2(n, J, dt, sites)
    return vcat(
        odd_layer_gates(n, J, dt/2, sites),
        even_layer_gates(n, J, dt,   sites),
        odd_layer_gates(n, J, dt/2, sites),
    )
end

# Fourth-order product formula using the standard Yoshida composition

function get_step_gates_order4(n, J, dt, sites)
    p1 = 1 / (4 - 4^(1/3))
    p2 = 1 - 4p1

    return vcat(
        get_step_gates_order2(n, J, p1*dt, sites),
        get_step_gates_order2(n, J, p1*dt, sites),
        get_step_gates_order2(n, J, p2*dt, sites),
        get_step_gates_order2(n, J, p1*dt, sites),
        get_step_gates_order2(n, J, p1*dt, sites),
    )
end

# Generalized dagger function that now admits order

function get_step_gates_dag(n, J, dt, sites; order::Int=1)
    gates = get_step_gates(n, J, -dt, sites; order=order)
    return reverse(gates)
end

# Building the MPO by contracting the step_gates with the identity

function get_step_MPO(n, J, dt, sites, cutoff, maxdim; order::Int=1)
    step_gates = get_step_gates(n, J, dt, sites; order=order)
    S = identity_mpo(sites)
    S = apply(step_gates, S; cutoff=cutoff, maxdim=maxdim)
    return S
end

function get_step_MPO_dag(n, J, dt, sites, cutoff, maxdim; order::Int=1)
    step_gates_dag = get_step_gates_dag(n, J, dt, sites; order=order)
    S = identity_mpo(sites)
    S = apply(step_gates_dag, S; cutoff=cutoff, maxdim=maxdim)
    return S
end