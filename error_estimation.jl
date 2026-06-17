include("optimization_problem.jl")

# Direct MPO calculations of M and L to use for error estimation, NOT for the optimization problem

function gram_matrix_direct_mpo(n, J, t, ks, sites, psi; cutoff=0.0, maxdim=200, order =2)
    r = length(ks)
    M = zeros(Float64, r, r)

    Fs = build_F_direct_list(n, J, t, ks, sites, cutoff, maxdim; order)

    idx = 1
    for i in 1:r
        for j in 1:r
            amp = inner(psi', Fs[idx], psi)
            M[i, j] = abs2(amp)
            idx += 1
        end
    end

    return M
end

function L_vector_direct_mpo(n, J, t, ks, k_ref, sites, psi; cutoff=0.0, maxdim=200, order::Int=2, order_ref::Int=4)
    Uref = build_full_U_mpo(n, J, t, k_ref, sites, cutoff, maxdim; order=order_ref)

    L = zeros(Float64, length(ks))
    for j in eachindex(ks)
        Uj_dag = build_full_Udag_mpo(n, J, t, ks[j], sites, cutoff, maxdim; order=order)
        F = left_multiply(Uj_dag, Uref; cutoff=cutoff, maxdim=maxdim)
        amp = inner(psi', F, psi)
        L[j] = abs2(amp)
    end
    return L
end

# Compute the MPS baseline with TEBD

function evolve_mps_tebd(n, J, t, k, sites, initial_state;
    cutoff=0.0, maxdim=100, order::Int=2)

    psi = MPS(sites, initial_state)
    normalize!(psi)

    dt = t / k
    gates = get_step_gates(n, J, dt, sites; order=order)

    for _ in 1:k
        psi = apply(gates, psi; cutoff=cutoff, maxdim=maxdim)
        normalize!(psi)
    end

    return psi
end

# Different error estimations

function dynamic_mpf_error(M::AbstractMatrix, L::AbstractVector, c::AbstractVector)
    return 1.0 + dot(c, M * c) - 2.0 * dot(L, c)
end

function single_trotter_frobenius_errors(L::AbstractVector)
    return 2.0 .- 2.0 .* L
end

function mps_baseline_frobenius_error(n, J, t, sites, initial_state;
    k_mps=100, maxdim_mps=100, cutoff_mps=0.0, order_mps::Int=4,
    k_ref=100, maxdim_ref=400, cutoff_ref=0.0, order_ref::Int=4)

    psi_mps = evolve_mps_tebd(
        n, J, t, k_mps, sites, initial_state;
        cutoff=cutoff_mps, maxdim=maxdim_mps, order=order_mps
    )

    psi_ref = evolve_mps_tebd(
        n, J, t, k_ref, sites, initial_state;
        cutoff=cutoff_ref, maxdim=maxdim_ref, order=order_ref
    )

    ov = inner(psi_mps, psi_ref)
    return 2.0 - 2.0 * abs2(ov)
end

# MAIN WRAPPER: Computes all errors

function test_dynamic_mpf_closed(
    n, J, t, ks, k_ref_opt, k_ref_eval, sites, initial_state;
    cutoff_opt=0.0, maxdim_opt=50,
    cutoff_eval=0.0, maxdim_eval=200,
    order=2, order_ref_opt=2, order_ref_eval=4,
    k_mps=100, maxdim_mps=100, cutoff_mps=0.0, order_mps=4,
    k_ref_mps=100, maxdim_ref_mps=400, cutoff_ref_mps=0.0, order_ref_mps=4
)
    psi = MPS(sites, initial_state)
    normalize!(psi)

    M_opt = gram_matrix_middle_out(n, J, t, ks, sites, psi; cutoff=cutoff_opt, maxdim=maxdim_opt, order=order)
    L_opt = L_vector_middle_out(n, J, t, ks, k_ref_opt, sites, psi; cutoff=cutoff_opt, maxdim=maxdim_opt, order=order, order_ref=order_ref_opt)

    c, λ = dynamic_mpf_coefficients(M_opt, L_opt)

    M_ref = gram_matrix_direct_mpo(n, J, t, ks, sites, psi; cutoff=cutoff_eval, maxdim=maxdim_eval, order=order)
    L_ref = L_vector_direct_mpo(n, J, t, ks, k_ref_eval, sites, psi; cutoff=cutoff_eval, maxdim=maxdim_eval, order=order, order_ref=order_ref_eval)

    E_mpf = dynamic_mpf_error(M_ref, L_ref, c)
    E_trot = single_trotter_frobenius_errors(L_ref)
    E_mps = mps_baseline_frobenius_error(n, J, t, sites, initial_state; k_mps=k_mps, maxdim_mps=maxdim_mps, cutoff_mps=cutoff_mps, order_mps=order_mps, k_ref=k_ref_mps, maxdim_ref=maxdim_ref_mps, cutoff_ref=cutoff_ref_mps, order_ref=order_ref_mps)

    return (
        M_opt = M_opt,
        L_opt = L_opt,
        coeffs = c,
        lambda = λ,
        M_ref = M_ref,
        L_ref = L_ref,
        E_mpf = E_mpf,
        E_trot = E_trot,
        E_mps = E_mps,
    )
end

using Plots

# function frobenius_data_vs_time(n, J, times, ks, k_ref_opt, k_ref_eval, sites, initial_state; cutoff=0.0)
#     bitstring = join(initial_state)
#     psi_mpo = MPS(sites, initial_state)
#     normalize!(psi_mpo)

#     r = length(ks)
#     single_errors = [Float64[] for _ in 1:r]
#     mpf_errors = Float64[]

#     for t in times
#         data = test_dynamic_mpf_closed(n, J, t, ks, k_ref_opt, k_ref_eval, sites, initial_state)

#         for j in 1:r
#             push!(single_errors[j], data.E_trot[j])
#         end
#         push!(mpf_errors, data.E_mpf)
#     end

#     return (
#         times = collect(times),
#         single_errors = single_errors,
#         mpf_errors = mpf_errors,
#     )
# end

# function plot_frobenius_errors(data, ks)
#     plt = plot(
#         xlabel = "time",
#         ylabel = "Frobenius error",
#         # title = "Frobenius errors vs time",
#         legend = :topright,
#         lw = 2,
#         size = (1600, 1000),
#         dpi = 300,
#     )

#     for (j, k) in enumerate(ks)
#         plot!(plt, data.times, data.single_errors[j], label = "Trotter k=$k")
#     end

#     plot!(plt, data.times, data.mpf_errors, label = "dynamic MPF", lw = 3, ls = :dash)

#     return plt
# end