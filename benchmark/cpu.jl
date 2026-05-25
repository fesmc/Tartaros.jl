# ── CPU arrays ─────────────────────────────────────────────────────────────────
const ghf_out = zeros(nx, ny)
const ghf_ref = fill(0.060, nx, ny)   # 60 mW m⁻² initial GHF
const T_cold  = fill(258.0, nx, ny)   # initial ice-base temperature [K]
const T_warm  = fill(273.0, nx, ny)   # post-warming ice-base temperature [K]
const dT_b    = fill(15.0,  nx, ny)   # temperature step [K]

# ── CPU models ─────────────────────────────────────────────────────────────────
const model_psc = PrescribedGHF(ghf_ref)

const model_ss  = SteadyStateGHF(ghf_ref, T_cold; k = k_b, H_litho = H_b)

const tau_b       = H_b^2 / (pi^2 * kappa_b)
const ghf_target  = fill(k_b * (model_ss.T_mantle[1, 1] - T_warm[1]) / H_b, nx, ny)
const model_relax = ThermalRelaxationGHF(ghf_target, ghf_ref; tau = tau_b)

const model_inf    = InfiniteColumnGHF(ghf_ref, dT_b; k = k_b, kappa = kappa_b, H_litho = H_b)
const model_finite = FiniteColumnGHF(  ghf_ref, dT_b; k = k_b, kappa = kappa_b, H_litho = H_b)

const model_dif = DiffusionGHF(            ghf_ref, T_cold; k = k_b, kappa = kappa_b, H_litho = H_b, nz = nz_b)
const model_tri = TridiagonalDiffusionGHF( ghf_ref, T_cold; k = k_b, kappa = kappa_b, H_litho = H_b, nz = nz_b)

# ── CPU suite ──────────────────────────────────────────────────────────────────
# Stateful models mutate internal state across samples, but per-call cost is
# independent of state value, so no reset is needed for timing accuracy.

cpu = SUITE["CPU"] = BenchmarkGroup()

cpu["PrescribedGHF"]           = @benchmarkable ghf!($ghf_out, $T_warm, $model_psc)
cpu["SteadyStateGHF"]          = @benchmarkable ghf!($ghf_out, $T_warm, $model_ss)
cpu["ThermalRelaxationGHF"]    = @benchmarkable ghf!($ghf_out, $T_warm, $model_relax,  $dt_b)
cpu["InfiniteColumnGHF"]       = @benchmarkable ghf!($ghf_out, $T_warm, $model_inf,    $dt_b)
cpu["FiniteColumnGHF"]         = @benchmarkable ghf!($ghf_out, $T_warm, $model_finite, $dt_b)
cpu["DiffusionGHF"]            = @benchmarkable ghf!($ghf_out, $T_warm, $model_dif,    $dt_b)
cpu["TridiagonalDiffusionGHF"] = @benchmarkable ghf!($ghf_out, $T_warm, $model_tri,    $dt_b)
