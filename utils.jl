using ITensors, ITensorMPS
using LinearAlgebra

ITensors.op(::OpName"P", ::SiteType"Qubit") =
    [1 0 0 0 
     0 -1/sqrt(2) 1/sqrt(2) 0
     0 1/sqrt(2) 1/sqrt(2) 0
     0 0 0 1
     ]

ITensors.op(::OpName"CCRy", ::SiteType"Qubit"; θ) =
    [1 0 0 0 0 0 0 0 
     0 1 0 0 0 0 0 0 
     0 0 1 0 0 0 0 0
     0 0 0 1 0 0 0 0
     0 0 0 0 1 0 0 0
     0 0 0 0 0 1 0 0
     0 0 0 0 0 0 cos(θ/2) -sin(θ/2) 
     0 0 0 0 0 0 sin(θ/2) cos(θ/2)
     ]

ITensors.op(::OpName"CRy", ::SiteType"Qubit"; θ) =
    [1 0 0 0 
     0 1 0 0
     0 0 cos(θ/2) -sin(θ/2) 
     0 0 sin(θ/2) cos(θ/2)
     ]  
     
ITensors.op(::OpName"CX", ::SiteType"Qubit") =
    [1 0 0 0 
     0 1 0 0
     0 0 0 1 
     0 0 1 0
     ]  

ITensors.op(::OpName"Reset", ::SiteType"Qubit") =
    [1 1
     0 0
     ]

function matrix_F(num_emitters, omega_m, omega_c, g, gamma, kappa, t, k)
    
    num_ancilla = Int(floor(num_emitters/2))
    total_dim = 2^(num_emitters + num_ancilla)

    F_components = []
    for i in eachindex(k)
        S_i_dag = get_circuit_matrix(num_emitters, omega_m, omega_c, g, gamma, kappa, t, k[i])' #' is dagger
        for j in eachindex(k)
            F = I(total_dim)
            S_j = get_circuit_matrix(num_emitters, omega_m, omega_c, g, gamma, kappa, t, k[j])
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


function build_F(num_emitters, omega_m, omega_c, g, gamma, kappa, t, k, sites)
    Id = MPO(sites)
    for j in 1:(num_emitters + Int(floor(num_emitters/2)))
        # `op` constructs a single-site operator (identity) on site n
        Id[j] = op(sites, "Id", j)  # adjust if your ITensor requires different call
    end

    F_components = []
    for i in eachindex(k)
        S_i_dag = dag(get_MPO(num_emitters, omega_m, omega_c, g, gamma, kappa, t, k[i], sites))
        for j in eachindex(k)
            F = Id
            S_j = get_MPO(num_emitters, omega_m, omega_c, g, gamma, kappa, t, k[j], sites)
            k_i_used = 0
            k_j_used = 0
            while (k_i_used < k[i]) && (k_j_used < k[j])
                F = apply(S_j, F; cutoff = 1e-12)
                F = apply(S_i_dag, F; cutoff = 1e-12)
                k_i_used += 1
                k_j_used += 1
            end

            while k_i_used < k[i]
                F = apply(S_i_dag, F; cutoff = 1e-12)
                k_i_used += 1
            end

            while k_j_used < k[j]
                F = apply(S_j, F; cutoff = 1e-12)
                k_j_used += 1
            end
            #normalize!(F) If we normalize, the evs get all fucked up
            push!(F_components, F)
        end
    end    
    return F_components
end

function get_MPO(num_emitters, omega_m, omega_c, g, gamma, kappa, t, k, sites)
    
    omega_eff, g_eff, gamma_g_minus, gamma_g_plus, gamma_minus_e, gamma_plus_e = compute_effective_parameters(num_emitters, omega_m, omega_c, g, gamma, kappa)

    dt = (t / k)
    alpha = omega_eff[1]*dt
    beta = g_eff[1]*dt
    theta_g_plus = 2*asin(sqrt(1 - exp(-gamma_g_plus[1] * dt)))
    theta_g_minus = 2*asin(sqrt(1 - exp(-gamma_g_minus[1] * dt)))
    theta_minus_e = 2*asin(sqrt(1 - exp(-gamma_minus_e[1] * dt)))
    theta_plus_e = 2*asin(sqrt(1 - exp(-gamma_plus_e[1] * dt)))

    Id = MPO(sites)
    for j in 1:(num_emitters + Int(floor(num_emitters/2)))
        # `op` constructs a single-site operator (identity) on site n
        Id[j] = op(sites, "Id", j)  # adjust if your ITensor requires different call
    end

    S = Id
    gates = ITensor[]
    for _ in 1:k       
        for j in 1:num_emitters
            s = sites[j]
            z = op("Z", s)
            rz = exp(-im * (0*alpha) * z)
            push!(gates, rz)
        end

        for j in 1:2:num_emitters
            s1 = sites[j]
            s2 = sites[j+1]

            yy = op("Y", s1) * op("Y", s2)
            r_yy = exp(-im * (beta/2) * yy)
            push!(gates, r_yy)


            xx = op("X", s1) * op("X", s2)
            r_xx = exp(-im * (beta/2) * xx)
            push!(gates, r_xx)
            # push!(gates, r_xx, r_yy)
        end

        # Decay layer

        # for j in 1:2:num_emitters
        #     p_gate = op("P", sites[j], sites[j+1])
        #     ccry = op("CCRy", sites[j], sites[j+1], sites[j+2]; θ = theta_plus_e - theta_g_minus)
        #     cry = op("CRy", sites[j], sites[j+2]; θ = theta_g_minus)
        #     cx = op("CX", sites[j+2], sites[j])
        #     reset = op("Reset", sites[j+2])
        #     push!(gates, p_gate, ccry, cry, cx, reset)

        #     ccry = op("CCRy", sites[j], sites[j+1], sites[j+2]; θ = theta_minus_e - theta_g_plus)
        #     cry = op("CRy", sites[j+1], sites[j+2]; θ = theta_g_plus)
        #     cx = op("CX", sites[j+2], sites[j+1])
        #     reset = op("Reset", sites[j+2])
        #     p_gate = op("P", sites[j], sites[j+1])
        #     push!(gates, ccry, cry, cx, reset, p_gate)
        # end     
    end
    S = apply(gates, S)
    return S
end

function get_circuit_matrix(num_emitters, omega_m, omega_c, g, gamma, kappa, t, k)
    omega_eff, g_eff, gamma_g_minus, gamma_g_plus, gamma_minus_e, gamma_plus_e = compute_effective_parameters(num_emitters, omega_m, omega_c, g, gamma, kappa)

    num_ancilla = Int(floor(num_emitters/2))
    system_dim = 2^num_emitters
    ancilla_dim = 2^num_ancilla
    total_dim = 2^(num_emitters + num_ancilla)

    dt = (t / k)
    alpha = omega_eff[1]*dt
    beta = g_eff[1]*dt
    theta_g_plus = 2*asin(sqrt(1 - exp(-gamma_g_plus[1] * dt)))
    theta_g_minus = 2*asin(sqrt(1 - exp(-gamma_g_minus[1] * dt)))
    theta_minus_e = 2*asin(sqrt(1 - exp(-gamma_minus_e[1] * dt)))
    theta_plus_e = 2*asin(sqrt(1 - exp(-gamma_plus_e[1] * dt)))

    rxx = [
            cos(beta/2) 0 0 -im*sin(beta/2);
            0 cos(beta/2) -im*sin(beta/2) 0;
            0 -im*sin(beta/2) cos(beta/2) 0;
            -im*sin(beta/2) 0 0 cos(beta/2) ;
            ]
    ryy = [
            cos(beta/2) 0 0 im*sin(beta/2);
            0 cos(beta/2) -im*sin(beta/2) 0;
            0 -im*sin(beta/2) cos(beta/2) 0;
            im*sin(beta/2) 0 0 cos(beta/2) ;
            ]
    
    S = Matrix{ComplexF64}(I, total_dim, total_dim)
    for _ in 1:k 
        for j in 1:3:num_emitters
            ids = [ Matrix{ComplexF64}(I, 2, 2) for _ in 1:(num_emitters + num_ancilla)]
            deleteat!(ids, j:j+1)
            insert!(ids, j, ryy)
            gate = reduce(kron, ids)
            S = gate * S

            ids = [ Matrix{ComplexF64}(I, 2, 2) for _ in 1:(num_emitters + num_ancilla)]
            deleteat!(ids, j:j+1)
            insert!(ids, j, rxx)
            gate = reduce(kron, ids)
            S = gate * S
        end
    end
    return S
end

function compute_effective_parameters(num_emitters, omega_m, omega_c, g, gamma, kappa)
    delta = [ x - omega_c for x in omega_m ]
    mean_delta = [ 0.5*(omega_m[i] + omega_m[i+1]) - omega_c for i in 1:1:(num_emitters-1) ]
    omega_eff = []
    gamma_eff = []

    for i in 1:1:(num_emitters)
        if i == 1 # This is the molecule on the very left (only coupled to one cavity)
            push!(omega_eff, omega_m[i] + (delta[i]*(g[i]^2))/((0.5*kappa[1])^2 + delta[i]^2))
            push!(gamma_eff, gamma[i] + (kappa[1]*(g[i]^2))/((0.5*kappa[1])^2 + delta[i]^2))

        elseif i == (num_emitters) # This is the molecule on the very right (only coupled to one cavity)
            j = 2*i - 1
            push!(omega_eff, omega_m[i] + (delta[i]*(g[j-1]^2))/((0.5*kappa[1])^2 + delta[i]^2))
            push!(gamma_eff, gamma[i] + (kappa[1]*(g[j-1]^2))/((0.5*kappa[1])^2 + delta[i]^2))

        else # The molecules in the middle have their frequencies and decay rates modified by two adjacent cavities
            j = 2*i - 1
            push!(omega_eff, omega_m[i] + 0.5*(delta[i]*(g[j-1]^2))/((0.5*kappa[1])^2 + delta[i]^2) + 0.5*(delta[i]*(g[j]^2))/((0.5*kappa[1])^2 + delta[i]^2) )
            push!(gamma_eff, gamma[i] + 0.5*(kappa[1]*(g[j-1]^2))/((0.5*kappa[1])^2 + delta[i]^2) + 0.5*(kappa[1]*(g[j]^2))/((0.5*kappa[1])^2 + delta[i]^2))
        end
    end

    g_eff = []
    gamma_cross = []

    for i in 1:1:(num_emitters-1)
        j = 2*i - 1
        push!(g_eff, (g[j]*g[j+1]*(mean_delta[i]))/((kappa[1]/2)^2 + (mean_delta[i])^2))
        push!(gamma_cross, (g[j]*g[j+1]*(kappa[1]))/((kappa[1]/2)^2 + mean_delta[i]^2))
    end

    det = [ omega_eff[i+1] - omega_eff[i] for i in 1:1:(num_emitters-1) ] 
    lam = [ 0.5*sqrt(4*g_eff[i]^2 + det[i]^2) for i in 1:1:(num_emitters-1) ]
    sen_theta = [ g_eff[i]/sqrt(lam[i]*(2*lam[i] + det[i])) for i in 1:1:(num_emitters-1) ]
    cos_theta = [ sqrt((2*lam[i] + det[i])/(4*lam[i])) for i in 1:1:(num_emitters-1) ]

    gamma_g_minus = [ (gamma_eff[i]*(cos_theta[i]^2) + gamma_eff[i+1]*(sen_theta[i]^2) - 2*gamma_cross[i]*sen_theta[i]*cos_theta[i]) for i in 1:1:(num_emitters-1) ]
    gamma_g_plus = [ (gamma_eff[i+1]*(cos_theta[i]^2) + gamma_eff[i]*(sen_theta[i]^2) + 2*gamma_cross[i]*sen_theta[i]*cos_theta[i]) for i in 1:1:(num_emitters-1) ]
    gamma_minus_e = [ (gamma_eff[i+1]*(cos_theta[i]^2) + gamma_eff[i]*(sen_theta[i]^2) - 2*gamma_cross[i]*sen_theta[i]*cos_theta[i]) for i in 1:1:(num_emitters-1) ]
    gamma_plus_e = [ (gamma_eff[i]*(cos_theta[i]^2) + gamma_eff[i+1]*(sen_theta[i]^2) + 2*gamma_cross[i]*sen_theta[i]*cos_theta[i]) for i in 1:1:(num_emitters-1) ]

    return omega_eff, g_eff, gamma_g_minus, gamma_g_plus, gamma_minus_e, gamma_plus_e
end