using Tartaros
using BenchmarkTools

# ── Grid and physical parameters ──────────────────────────────────────────────
const nx, ny  = 300, 300
const k_b     = 3.0          # thermal conductivity  [W m⁻¹ K⁻¹]
const kappa_b = 1e-6         # thermal diffusivity   [m² s⁻¹]
const H_b     = 2_000.0      # lithosphere thickness [m]
const nz_b    = 10           # vertical levels (DiffusionGHF only)
const dt_b    = 500 * 365.25 * 24 * 3600   # 500-year step [s]

SUITE = BenchmarkGroup()

include("cpu.jl")
include("gpu.jl")
