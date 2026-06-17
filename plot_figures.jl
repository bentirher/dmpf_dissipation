using Plots

function frobenius_data_vs_time(
    n, J, times, ks, k_ref_opt, k_ref_eval, sites, initial_state;
    cutoff_opt=0.0, maxdim_opt=50,
    cutoff_eval=0.0, maxdim_eval=200,
    order=2, order_ref_opt=2, order_ref_eval=4,
    k_mps=100, maxdim_mps=100, cutoff_mps=0.0, order_mps=4,
    k_ref_mps=100, maxdim_ref_mps=400, cutoff_ref_mps=0.0, order_ref_mps=4
)
    r = length(ks)
    single_errors = [Float64[] for _ in 1:r]
    mpf_errors = Float64[]
    mps_errors = Float64[]

    for t in times
        data = test_dynamic_mpf_closed(
            n, J, t, ks, k_ref_opt, k_ref_eval, sites, initial_state;
            cutoff_opt=cutoff_opt, maxdim_opt=maxdim_opt,
            cutoff_eval=cutoff_eval, maxdim_eval=maxdim_eval,
            order=order, order_ref_opt=order_ref_opt, order_ref_eval=order_ref_eval,
            k_mps=k_mps, maxdim_mps=maxdim_mps, cutoff_mps=cutoff_mps, order_mps=order_mps,
            k_ref_mps=k_ref_mps, maxdim_ref_mps=maxdim_ref_mps,
            cutoff_ref_mps=cutoff_ref_mps, order_ref_mps=order_ref_mps
        )

        for j in 1:r
            push!(single_errors[j], data.E_trot[j])
        end
        push!(mpf_errors, data.E_mpf)
        push!(mps_errors, data.E_mps)
    end

    return (
        times = collect(times),
        single_errors = single_errors,
        mpf_errors = mpf_errors,
        mps_errors = mps_errors,
    )
end

function plot_frobenius_errors(data, ks)
    plt = plot(
        xlabel = "time",
        ylabel = "Squared Frobenius error",
        legend = :topright,
        lw = 2,
        size = (1600, 1000),
        dpi = 300,
    )

    for (j, k) in enumerate(ks)
        plot!(plt, data.times, data.single_errors[j], label="Trotter k=$k")
    end

    plot!(plt, data.times, data.mpf_errors, label="dynamic MPF", lw=3, ls=:dash)
    plot!(plt, data.times, data.mps_errors, label="MPS baseline", lw=3, ls=:dot)

    return plt
end

function plot_observable_vs_time(data, ks; ylabel="Observable value", exact_label="Reference")
    plt = plot(
        xlabel = "time",
        ylabel = ylabel,
        legend = :topright,
        lw = 2,
        size = (1600, 1000),
        dpi = 300,
    )

    plot!(plt, data.times, data.exact, label=exact_label, lw=3)

    for (a, k) in enumerate(ks)
        plot!(plt, data.times, data.trotter[a], label="Trotter k=$k")
    end

    plot!(plt, data.times, data.dmpf, label="dynamic MPF", lw=3, ls=:dash)
    plot!(plt, data.times, data.mps, label="MPS baseline", lw=3, ls=:dot)

    return plt
end

function plot_observable_errors_vs_time(data, ks)
    exact = data.exact
    trotter_errs = [abs.(vals .- exact) for vals in data.trotter]
    dmpf_err = abs.(data.dmpf .- exact)
    mps_err = abs.(data.mps .- exact)

    plt = plot(
        xlabel = "time",
        ylabel = "Absolute observable error",
        legend = :topright,
        lw = 2,
        size = (1600, 1000),
        dpi = 300,
    )

    for (a, k) in enumerate(ks)
        plot!(plt, data.times, trotter_errs[a], label="Trotter k=$k")
    end

    plot!(plt, data.times, dmpf_err, label="dynamic MPF", lw=3, ls=:dash)
    plot!(plt, data.times, mps_err, label="MPS baseline", lw=3, ls=:dot)

    return plt
end