include("error_estimation.jl")

# -------------------------
# MPS / TEBD state evolution
# -------------------------

function evolve_mps_tebd(n, J, t, k, sites, initial_state; cutoff=0.0, maxdim=100, order=2)
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


# -------------------------
# Local observable estimators
# -------------------------

function expect_Z_mps(psi, sites, i)
    psiZ = apply(op(sites, "Z", i), psi)
    return real(inner(psi', psiZ))
end

function expect_ZZ_mps(psi, sites, i, j)
    ops = ITensor[op(sites, "Z", i), op(sites, "Z", j)]
    psiZZ = apply(ops, psi)
    return real(inner(psi', psiZZ))
end

function observable_value_mps(psi, sites; obs_type=:Z, i, j=nothing)
    if obs_type == :Z
        return expect_Z_mps(psi, sites, i)
    elseif obs_type == :ZZ
        @assert j !== nothing "For obs_type = :ZZ, provide both i and j."
        return expect_ZZ_mps(psi, sites, i, j)
    else
        error("Unsupported observable type $obs_type. Use :Z or :ZZ.")
    end
end


# -------------------------
# Reference observable
# -------------------------
# This is the "exact"/reference observable for plotting and comparison.
# It is computed with a high-quality MPS/TEBD simulation, not dense matrices.

function reference_observable_mps(
    n, J, t, sites, initial_state;
    obs_type=:Z, i, j=nothing,
    cutoff_ref_obs=0.0, maxdim_ref_obs=200, order_ref_obs=4, k_ref_obs=100
)
    psi_ref = evolve_mps_tebd(
        n, J, t, k_ref_obs, sites, initial_state;
        cutoff=cutoff_ref_obs, maxdim=maxdim_ref_obs, order=order_ref_obs
    )

    return observable_value_mps(psi_ref, sites; obs_type=obs_type, i=i, j=j)
end


# -------------------------
# Single-Trotter observable values
# -------------------------

function trotter_observable_values(
    n, J, t, ks, sites, initial_state;
    obs_type=:Z, i, j=nothing,
    cutoff_trot_obs=0.0, maxdim_trot_obs=200, order=2
)
    vals = Float64[]

    for k in ks
        psi_k = evolve_mps_tebd(
            n, J, t, k, sites, initial_state;
            cutoff=cutoff_trot_obs, maxdim=maxdim_trot_obs, order=order
        )
        push!(vals, observable_value_mps(psi_k, sites; obs_type=obs_type, i=i, j=j))
    end

    return vals
end


# -------------------------
# DMPF observable value
# -------------------------

function dmpf_observable_value(coeffs, trotter_vals)
    return sum(coeffs[a] * trotter_vals[a] for a in eachindex(coeffs))
end


# -------------------------
# Baseline MPS observable value
# -------------------------

function baseline_mps_observable(
    n, J, t, sites, initial_state;
    obs_type=:Z, i, j=nothing,
    cutoff_mps=0.0, maxdim_mps=100, order_mps=4, k_mps=100
)
    psi_mps = evolve_mps_tebd(
        n, J, t, k_mps, sites, initial_state;
        cutoff=cutoff_mps, maxdim=maxdim_mps, order=order_mps
    )

    return observable_value_mps(psi_mps, sites; obs_type=obs_type, i=i, j=j)
end


# -------------------------
# Main observable report
# -------------------------
# IMPORTANT:
# `res` is the output of `test_dynamic_mpf_closed(...)`
# so we use its `coeffs` instead of recomputing DMPF data.

function observable_report(
    res,
    n, J, t, ks, sites, initial_state;
    obs_type=:Z, i, j=nothing,

    # Trotter observable simulation parameters
    cutoff_trot_obs=0.0,
    maxdim_trot_obs=200,
    order=2,

    # Reference observable parameters
    cutoff_ref_obs=0.0,
    maxdim_ref_obs=200,
    order_ref_obs=4,
    k_ref_obs=100,

    # Baseline MPS parameters
    cutoff_mps=0.0,
    maxdim_mps=100,
    order_mps=4,
    k_mps=100,
)
    exact_val = reference_observable_mps(
        n, J, t, sites, initial_state;
        obs_type=obs_type, i=i, j=j,
        cutoff_ref_obs=cutoff_ref_obs,
        maxdim_ref_obs=maxdim_ref_obs,
        order_ref_obs=order_ref_obs,
        k_ref_obs=k_ref_obs
    )

    trotter_vals = trotter_observable_values(
        n, J, t, ks, sites, initial_state;
        obs_type=obs_type, i=i, j=j,
        cutoff_trot_obs=cutoff_trot_obs,
        maxdim_trot_obs=maxdim_trot_obs,
        order=order
    )

    dmpf_val = dmpf_observable_value(res.coeffs, trotter_vals)

    mps_val = baseline_mps_observable(
        n, J, t, sites, initial_state;
        obs_type=obs_type, i=i, j=j,
        cutoff_mps=cutoff_mps,
        maxdim_mps=maxdim_mps,
        order_mps=order_mps,
        k_mps=k_mps
    )

    return (
        exact = exact_val,
        trotter = trotter_vals,
        dmpf = dmpf_val,
        mps = mps_val,
        trotter_err = abs.(trotter_vals .- exact_val),
        dmpf_err = abs(dmpf_val - exact_val),
        mps_err = abs(mps_val - exact_val),
        coeffs = res.coeffs,
    )
end


# -------------------------
# Time-series wrapper
# -------------------------
# This computes `res = test_dynamic_mpf_closed(...)` once per time,
# then passes `res` into `observable_report(...)`.

function observable_data_vs_time(
    n, J, times, ks, k_ref_opt, k_ref_eval, sites, initial_state;
    obs_type=:Z, i, j=nothing,

    # Optimizer (MOC) parameters
    cutoff_opt=0.0,
    maxdim_opt=50,
    order_ref_opt=1,
    order=1,

    # Reference values (M_ref, L_ref and reference evolution) parameters
    cutoff_eval=0.0,
    maxdim_eval=200,
    order_ref_eval=4,
    k_ref_eval_obs=100,

    # Baseline MPS parameters
    cutoff_mps=0.0,
    maxdim_mps=100,
    order_mps=4,
    k_mps=100,

    # Single-Trotter observable simulation parameters
    cutoff_trot_obs=0.0,
    maxdim_trot_obs=200,
)
    exact_vals = Float64[]
    dmpf_vals = Float64[]
    mps_vals = Float64[]
    trotter_vals = [Float64[] for _ in 1:length(ks)]

    for t in times
        res = test_dynamic_mpf_closed(
            n, J, t, ks, k_ref_opt, k_ref_eval, sites, initial_state;
            cutoff_opt=cutoff_opt,
            maxdim_opt=maxdim_opt,
            cutoff_eval=cutoff_eval,
            maxdim_eval=maxdim_eval,
            order=order,
            order_ref_opt=order_ref_opt,
            order_ref_eval=order_ref_eval,
            cutoff_mps=cutoff_mps,
            maxdim_mps=maxdim_mps,
            order_mps=order_mps,
            k_mps=k_mps,
            cutoff_ref_mps=cutoff_eval,
            maxdim_ref_mps=maxdim_eval,
            order_ref_mps=order_ref_eval,
            k_ref_mps=k_ref_eval
        )

        obs = observable_report(
            res,
            n, J, t, ks, sites, initial_state;
            obs_type=obs_type, i=i, j=j,
            cutoff_trot_obs=cutoff_trot_obs,
            maxdim_trot_obs=maxdim_trot_obs,
            order=order,
            cutoff_ref_obs=cutoff_eval,
            maxdim_ref_obs=maxdim_eval,
            order_ref_obs=order_ref_eval,
            k_ref_obs=k_ref_eval_obs,
            cutoff_mps=cutoff_mps,
            maxdim_mps=maxdim_mps,
            order_mps=order_mps,
            k_mps=k_mps
        )

        push!(exact_vals, obs.exact)
        push!(dmpf_vals, obs.dmpf)
        push!(mps_vals, obs.mps)
        for a in eachindex(ks)
            push!(trotter_vals[a], obs.trotter[a])
        end
    end

    return (
        times = collect(times),
        exact = exact_vals,
        trotter = trotter_vals,
        dmpf = dmpf_vals,
        mps = mps_vals,
    )
end