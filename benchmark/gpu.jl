# ── GPU suite ──────────────────────────────────────────────────────────────────
# All models run unchanged on CuArrays.  DiffusionGHF.ghf! dispatches through
# KernelAbstractions, so the same kernel executes on the GPU; CUDA.@sync is still
# needed for the broadcast-based models (Prescribed, SteadyState, Relaxation,
# Infinite, Finite) to flush GPU kernels before BenchmarkTools timestamps.
# DiffusionGHF.ghf! calls KernelAbstractions.synchronize internally, so CUDA.@sync
# is a no-op there but is kept for consistency.
#
# NOTE: use `CuArray`, not `cu`. `cu` converts Float64 to Float32, which would silently
# benchmark single-precision GPU kernels against the double-precision CPU suite above.

using CUDA

if CUDA.functional()
    const ghf_out_g = CUDA.zeros(Float64, nx, ny)
    const ghf_ref_g = CuArray(ghf_ref)
    const T_cold_g  = CuArray(T_cold)
    const T_warm_g  = CuArray(T_warm)
    const dT_b_g    = CuArray(dT_b)

    const model_psc_g    = PrescribedGHF(ghf_ref_g)
    const model_ss_g     = SteadyStateGHF(ghf_ref_g, T_cold_g; k = k_b, H_litho = H_b)
    const ghf_target_g   = CuArray(ghf_target)
    const model_relax_g  = ThermalRelaxationGHF(ghf_target_g, ghf_ref_g; tau = tau_b)
    const model_inf_g    = InfiniteColumnGHF(ghf_ref_g, dT_b_g; k = k_b, kappa = kappa_b, H_litho = H_b)
    const model_finite_g = FiniteColumnGHF(  ghf_ref_g, dT_b_g; k = k_b, kappa = kappa_b, H_litho = H_b)
    const model_dif_g    = DiffusionGHF(     ghf_ref_g, T_cold_g; k = k_b, kappa = kappa_b, H_litho = H_b, nz = nz_b)

    gpu = SUITE["GPU"] = BenchmarkGroup()

    gpu["PrescribedGHF"]        = @benchmarkable CUDA.@sync ghf!($ghf_out_g, $T_warm_g, $model_psc_g)
    gpu["SteadyStateGHF"]       = @benchmarkable CUDA.@sync ghf!($ghf_out_g, $T_warm_g, $model_ss_g)
    gpu["ThermalRelaxationGHF"] = @benchmarkable CUDA.@sync ghf!($ghf_out_g, $T_warm_g, $model_relax_g,  $dt_b)
    gpu["InfiniteColumnGHF"]    = @benchmarkable CUDA.@sync ghf!($ghf_out_g, $T_warm_g, $model_inf_g,    $dt_b)
    gpu["FiniteColumnGHF"]      = @benchmarkable CUDA.@sync ghf!($ghf_out_g, $T_warm_g, $model_finite_g, $dt_b)
    gpu["DiffusionGHF"]         = @benchmarkable CUDA.@sync ghf!($ghf_out_g, $T_warm_g, $model_dif_g,    $dt_b)
end
