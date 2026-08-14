# ── Reactant suite ─────────────────────────────────────────────────────────────
# Reactant traces each call once, lowers it to StableHLO and lets XLA optimise the whole
# traced region before emitting device code. Compilation is therefore a one-off cost that
# must be paid *outside* the measured region — `@compile` is called during setup and only
# the resulting executable is benchmarked, so these numbers are comparable to the CPU and
# GPU suites above.
#
# The backend is pinned to "cpu" so the suite is reproducible on machines with and without
# a GPU; set "gpu" to measure the same code on an accelerator.
#
# TridiagonalDiffusionGHF is absent by design: it is a serial reference implementation
# built on scalar indexing and cannot be traced.

using Reactant

Reactant.set_default_backend("cpu")

const ghf_out_r = Reactant.to_rarray(zeros(nx, ny))
const ghf_ref_r = Reactant.to_rarray(ghf_ref)
const T_cold_r  = Reactant.to_rarray(T_cold)
const T_warm_r  = Reactant.to_rarray(T_warm)
const dT_b_r    = Reactant.to_rarray(dT_b)

const model_psc_r    = PrescribedGHF(ghf_ref_r)
const model_ss_r     = SteadyStateGHF(ghf_ref_r, T_cold_r; k = k_b, H_litho = H_b)
const ghf_target_r   = Reactant.to_rarray(ghf_target)
const model_relax_r  = ThermalRelaxationGHF(ghf_target_r, ghf_ref_r; tau = tau_b)
const model_inf_r    = InfiniteColumnGHF(ghf_ref_r, dT_b_r; k = k_b, kappa = kappa_b, H_litho = H_b)
const model_finite_r = FiniteColumnGHF(  ghf_ref_r, dT_b_r; k = k_b, kappa = kappa_b, H_litho = H_b)
const model_dif_r    = DiffusionGHF(     ghf_ref_r, T_cold_r; k = k_b, kappa = kappa_b, H_litho = H_b, nz = nz_b)

const c_psc    = @compile ghf!(ghf_out_r, T_warm_r, model_psc_r)
const c_ss     = @compile ghf!(ghf_out_r, T_warm_r, model_ss_r)
const c_relax  = @compile ghf!(ghf_out_r, T_warm_r, model_relax_r,  dt_b)
const c_inf    = @compile ghf!(ghf_out_r, T_warm_r, model_inf_r,    dt_b)
const c_finite = @compile ghf!(ghf_out_r, T_warm_r, model_finite_r, dt_b)
const c_dif    = @compile ghf!(ghf_out_r, T_warm_r, model_dif_r,    dt_b)

reactant = SUITE["Reactant"] = BenchmarkGroup()

# XLA execution is asynchronous: calling an executable enqueues work and returns. Without
# an explicit wait, a single-call benchmark measures the enqueue and not the computation
# (measured: 139 us enqueue vs 4.4 ms of actual work). `Reactant.synchronize` plays the
# same role here that `CUDA.@sync` plays in the GPU suite.
reactant["PrescribedGHF"] = @benchmarkable begin
    $c_psc($ghf_out_r, $T_warm_r, $model_psc_r)
    Reactant.synchronize($ghf_out_r)
end
reactant["SteadyStateGHF"] = @benchmarkable begin
    $c_ss($ghf_out_r, $T_warm_r, $model_ss_r)
    Reactant.synchronize($ghf_out_r)
end
reactant["ThermalRelaxationGHF"] = @benchmarkable begin
    $c_relax($ghf_out_r, $T_warm_r, $model_relax_r, $dt_b)
    Reactant.synchronize($ghf_out_r)
end
reactant["InfiniteColumnGHF"] = @benchmarkable begin
    $c_inf($ghf_out_r, $T_warm_r, $model_inf_r, $dt_b)
    Reactant.synchronize($ghf_out_r)
end
reactant["FiniteColumnGHF"] = @benchmarkable begin
    $c_finite($ghf_out_r, $T_warm_r, $model_finite_r, $dt_b)
    Reactant.synchronize($ghf_out_r)
end
reactant["DiffusionGHF"] = @benchmarkable begin
    $c_dif($ghf_out_r, $T_warm_r, $model_dif_r, $dt_b)
    Reactant.synchronize($ghf_out_r)
end

# The payoff is largest when a whole timestep is compiled rather than a single call, so
# also measure n steps as one program against n launches of a one-step program.
function nsteps!(ghf, T_ice_base, thermo, dt, n::Int)
    for _ in 1:n
        ghf!(ghf, T_ice_base, thermo, dt)
    end
    return nothing
end

const model_loop_r = DiffusionGHF(ghf_ref_r, T_cold_r; k = k_b, kappa = kappa_b,
                                  H_litho = H_b, nz = nz_b)
const c_loop = @compile nsteps!(ghf_out_r, T_warm_r, model_loop_r, dt_b, 10)

reactant["DiffusionGHF, 10 steps as one program"] = @benchmarkable begin
    $c_loop($ghf_out_r, $T_warm_r, $model_loop_r, $dt_b, 10)
    Reactant.synchronize($ghf_out_r)
end
reactant["DiffusionGHF, 10 separate launches"] = @benchmarkable begin
    for _ in 1:10
        $c_dif($ghf_out_r, $T_warm_r, $model_dif_r, $dt_b)
    end
    Reactant.synchronize($ghf_out_r)
end

# Smoothing: the default stencil is an element-wise loop, the Reactant extension swaps in
# a shifted-slice formulation whose cost scales with the number of non-zero kernel weights.
# Both kernels are measured because the top-hat is not separable and the Gaussian is.
const z_bed_r  = Reactant.to_rarray(1_000.0 .* rand(nx, ny))
const dx_r     = 1_000.0
const kern_c_r = ConstantSmoothingKernel(r = 3_000.0)
const kern_g_r = GaussianSmoothingKernel(r = 2_000.0)
const c_smooth_c = @compile smooth(z_bed_r, dx_r, dx_r, kern_c_r)
const c_smooth_g = @compile smooth(z_bed_r, dx_r, dx_r, kern_g_r)

reactant["smooth, ConstantSmoothingKernel"] = @benchmarkable Reactant.synchronize(
    $c_smooth_c($z_bed_r, $dx_r, $dx_r, $kern_c_r))
reactant["smooth, GaussianSmoothingKernel"] = @benchmarkable Reactant.synchronize(
    $c_smooth_g($z_bed_r, $dx_r, $dx_r, $kern_g_r))
