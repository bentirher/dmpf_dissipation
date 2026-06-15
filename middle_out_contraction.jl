using ITensors, ITensorMPS
using LinearAlgebra
include("product_formula_generation.jl")

# Left multiplication: A * F
function left_multiply(A::MPO, F::MPO; cutoff, maxdim)
    return apply(A, F; cutoff=cutoff, maxdim=maxdim)
end

# # Right multiplication: F * A
# # Implemented as (A† * F†)† to avoid index-convention issues
# function right_multiply(F::MPO, A::MPO; cutoff, maxdim)
#     X = apply(dag(A), dag(F); cutoff=cutoff, maxdim=maxdim)
#     return dag(X)
# end

# Operator adjoint of an MPO: conjugate + swap bra/ket site indices
function op_dag(A::MPO)
    return swapprime(dag(A), 0 => 1)
end

# Right multiplication: F * A
function right_multiply(F::MPO, A::MPO; cutoff, maxdim)
    X = apply(op_dag(A), op_dag(F); cutoff=cutoff, maxdim=maxdim)
    return op_dag(X)
end

function build_F(n, J, t, k, sites, cutoff, maxdim; order::Int = 2)
    Id = identity_mpo(sites)
    F_components = MPO[]

    for i in eachindex(k)
        dt_i = t / k[i]
        step_i_dag_mpo = get_step_MPO_dag(n, J, dt_i, sites, cutoff, maxdim; order=order)

        for j in eachindex(k)
            dt_j = t / k[j]
            step_j_mpo = get_step_MPO(n, J, dt_j, sites, cutoff, maxdim; order=order)

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
