using ITensors, ITensorMPS
using LinearAlgebra
include("middle_out_contraction.jl")

function build_full_U_mpo(n, J, t, k, sites, cutoff, maxdim; order::Int=2)
    dt = t / k
    step = get_step_MPO(n, J, dt, sites, cutoff, maxdim; order=order)
    U = identity_mpo(sites)
    for _ in 1:k
        U = left_multiply(step, U; cutoff=cutoff, maxdim=maxdim)
    end
    return U
end

function build_full_Udag_mpo(n, J, t, k, sites, cutoff, maxdim; order::Int=2)
    dt = t / k
    step_dag = get_step_MPO_dag(n, J, dt, sites, cutoff, maxdim; order=order)
    Udag = identity_mpo(sites)
    for _ in 1:k
        Udag = left_multiply(step_dag, Udag; cutoff=cutoff, maxdim=maxdim)
    end
    return Udag
end

function build_F_direct_mpo(n, J, t, ki, kj, sites, cutoff, maxdim, order)
    Ui_dag = build_full_Udag_mpo(n, J, t, ki, sites, cutoff, maxdim; order)
    Uj     = build_full_U_mpo(n, J, t, kj, sites, cutoff, maxdim; order)
    Fij = left_multiply(Ui_dag, Uj; cutoff=cutoff, maxdim=maxdim)
    return Fij
end

function build_F_direct_list(n, J, t, ks, sites, cutoff, maxdim; order::Int=2)
    Fs = MPO[]
    for ki in ks
        for kj in ks
            push!(Fs, build_F_direct_mpo(n, J, t, ki, kj, sites, cutoff, maxdim, order))
        end
    end
    return Fs
end