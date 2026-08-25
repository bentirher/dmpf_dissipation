# =============================================================================
# gamma_sweep.jl   --   THE RECOMMENDED NEXT EXPERIMENT
#
# Everything measured so far is a single point in parameter space: gamma = 0.05.
# Deck slide 22 swept gamma in {0.01, 0.05, 0.1, 0.2} and found chi(F_ii)
# saturating for ALL of them. The open question is whether the N-route's
# accuracy advantage survives stronger dissipation.
#
# For each gamma: compute the EXACT reference (M-route at the ceiling, where
# maxdim = 16^min(l,n-l) means no truncation), then score both routes against it
# across maxdim. Output one CSV per gamma; the driver script runs a job array so
# each gamma is its own task.
#
# THE FIGURE THIS FEEDS: max|dc| at fixed maxdim (e.g. 32) versus gamma, both
# routes, on one set of axes. Direct successor to slide 22's four-curve layout,
# and the most persuasive summary available -- it answers "does this generalise?"
# in one panel.
#
# WHAT TO WATCH:
#  - does the amplification ratio (M/N) hold at gamma = 0.2, or degrade?
#  - does the M-route's lambda_min stay negative at large gamma (it should get
#    worse: stronger dissipation -> larger Trotter errors -> but also larger N,
#    so the cancellation may actually EASE; that would be worth knowing)
#  - E_k8 grows with gamma, so N grows, so the catastrophic cancellation is
#    LESS severe at large gamma. If the advantage shrinks with gamma, the honest
#    claim becomes "the reformulation matters most in the weak-dissipation
#    regime", which is also the physically relevant one for QPU noise.
#
# Environment: N_QUBITS, GAMMA, TVAL, K0, K_REF, ORDER
# =============================================================================
import Distributions, Random
using LinearAlgebra, Printf
include("trotter_error_gram.jl")
include("open_optimization_problem.jl")
BLAS.set_num_threads(parse(Int, get(ENV,"SLURM_CPUS_PER_TASK","1")))

getenv(k,d)=get(ENV,k,string(d))
n     = parse(Int,     getenv("N_QUBITS", 4))
gamma = parse(Float64, getenv("GAMMA",    0.05))
t     = parse(Float64, getenv("TVAL",     3.0))
k0    = parse(Int,     getenv("K0",       48))
k_ref = parse(Int,     getenv("K_REF",    48))
order = parse(Int,     getenv("ORDER",    2))
ks    = [3, 8]; ct = 1e-14
chi   = theoretical_max_bond_dim(n)
grid  = filter(<=(chi), [16, 32, 64, 128, 256])
gtag  = replace(@sprintf("%.3f", gamma), "." => "p")

Random.seed!(1234)
J = rand(Distributions.Uniform(1/4,3/4), n-1); gammas = fill(gamma,n)
lsites = liouville_siteinds(n)
rho0 = vectorized_initial_state_mps(lsites, collect(0:2:(n-1)) .|> string)

@printf("gamma_sweep  n=%d GAMMA=%.3f t=%.1f k0=%d order=%d ceiling=%d\n\n",
        n, gamma, t, k0, order, chi); flush(stdout)

build_N(M,L,P) = [M[i,j]-L[i]-L[j]+P for i in 1:length(L), j in 1:length(L)]
function m_route(md)
    M,_ = open_gram_matrix(n,J,gammas,t,ks,lsites,rho0; cutoff=ct,maxdim=md,order=order,dissipation=true)
    L,_ = open_L_vector(n,J,gammas,t,ks,k_ref,lsites,rho0; cutoff=ct,maxdim=md,order=order,order_ref=order,dissipation=true)
    P   = reference_purity(n,J,gammas,t,k_ref,lsites,rho0; cutoff=ct,maxdim=md,order=order,dissipation=true)
    c,_ = dynamic_mpf_coefficients(M,L); N = build_N(M,L,P)
    (c=c, E_mpf=open_dynamic_mpf_error(M,L,c,P), N=N,
     lam=minimum(eigen(Symmetric(0.5*(N+N'))).values))
end

println("computing exact reference at maxdim=$chi ..."); flush(stdout)
ex = m_route(chi); c_exact = ex.c
@printf("  c = [%+.6f, %+.6f]  E_k3 = %.4e  E_k8 = %.4e  E_mpf = %.4e\n\n",
        c_exact[1], c_exact[2], ex.N[1,1], ex.N[2,2], ex.E_mpf); flush(stdout)

rows = ["gamma,maxdim,route,c1,dc_max,E_k3,E_k8,E_mpf,lambda_min"]
push!(rows, @sprintf("%.4f,%d,EXACT,%.8f,0.0,%.8e,%.8e,%.8e,%.8e",
                     gamma, chi, c_exact[1], ex.N[1,1], ex.N[2,2], ex.E_mpf, ex.lam))

println("maxdim | M: dc        E_k8       lam_min   | N: dc        E_k8       lam_min")
println("-"^80)
for md in grid
    m = m_route(md)
    r = test_dmpf_open_N(n, J, gammas, t, ks, k0, lsites, rho0;
                         cutoff=ct, maxdim=md, maxdim_G=md, order=order,
                         order_ref=order, dissipation=true, exact_delta=true)
    push!(rows, @sprintf("%.4f,%d,M,%.8f,%.8e,%.8e,%.8e,%.8e,%.8e", gamma, md, m.c[1],
                         maximum(abs.(m.c .- c_exact)), m.N[1,1], m.N[2,2], m.E_mpf, m.lam))
    push!(rows, @sprintf("%.4f,%d,N,%.8f,%.8e,%.8e,%.8e,%.8e,%.8e", gamma, md, r.coeffs[1],
                         maximum(abs.(r.coeffs .- c_exact)), r.E_trot[1], r.E_trot[2],
                         r.E_mpf, minimum(r.eigvals)))
    @printf("%6d | %.2e  %+.3e %+.2e | %.2e  %+.3e %+.2e\n", md,
            maximum(abs.(m.c .- c_exact)), m.N[2,2], m.lam,
            maximum(abs.(r.coeffs .- c_exact)), r.E_trot[2], minimum(r.eigvals))
    flush(stdout)
end
write("gamma_$gtag.csv", join(rows,"\n")*"\n")
println("\nwrote gamma_$gtag.csv")
