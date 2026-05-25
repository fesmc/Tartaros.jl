"""
$(TYPEDSIGNATURES)

An abstract type for the thermodynamics of the lithosphere.

# Available subtypes
 - [`PrescribedGHF`](@ref)
 - [`SteadyStateGHF`](@ref)
 - [`ThermalRelaxationGHF`](@ref)
 - [`InfiniteColumnGHF`](@ref)
 - [`FiniteColumnGHF`](@ref)
 - [`DiffusionGHF`](@ref)
"""
abstract type AbstractGHF end

"""
$(TYPEDSIGNATURES)

GHF is held constant at `ghf_ref` and excludes any feedbacks from changes in basal ice temperature.

# Fields
 - `ghf_ref`: reference GHF field [mW m⁻²]
"""
struct PrescribedGHF{M} <: AbstractGHF
    ghf_ref::M
end

"""
$(TYPEDSIGNATURES)

GHF is recomputed at every time step from Fourier's law applied to a linear
(steady-state) temperature profile through the lithosphere:

```math
\\begin{aligned}
    G = k \\cdot \\dfrac{T_\\mathrm{mantle} - T_\\mathrm{ice}}{H_\\mathrm{litho}}
\\end{aligned}
```

# Fields
 - `k`: thermal conductivity [W m⁻¹ K⁻¹]
 - `H_litho`: lithosphere thickness [m]
 - `T_mantle`: temperature at the base of the lithosphere [K]
"""
@kwdef struct SteadyStateGHF{T, M} <: AbstractGHF
    k::T
    H_litho::T
    T_mantle::M  # scalar or spatial field [K]
end

function SteadyStateGHF(ghf_ref, T_ice_base; k, H_litho)
    return SteadyStateGHF(
        k = k,
        H_litho = H_litho,
        T_mantle = (@. T_ice_base + ghf_ref * H_litho / k),
    )
end

"""
$(TYPEDSIGNATURES)

GHF relaxes exponentially toward the steady-state target `ghf_ref`
with timescale `tau` (Huybrechts/Ritz lag model):

```math
\\begin{aligned}
    \\dfrac{dG}{dt} = \\dfrac{G_\\mathrm{ref} - G}{\\tau}
\\end{aligned}
```

As shown in the example section, this model gives a relatively inaccurate transient behaviour and we therefore recommend using other models, unless the user wants to perform comparison (e.g. with legacy code).

# Fields
 - `ghf_ref`: asymptotic GHF target [W m⁻²]
 - `ghf_state`: current GHF state (updated in-place by `ghf!`) [W m⁻²]
 - `tau`: thermal relaxation timescale [s]
"""
@kwdef struct ThermalRelaxationGHF{M, T} <: AbstractGHF
    ghf_ref::M
    ghf_state::M
    tau::T
    # TODO: this could be tau(x,y), derived from H_litho(x,y) and kappa
end

ThermalRelaxationGHF(ghf_ref; tau) =
    ThermalRelaxationGHF(ghf_ref = ghf_ref, ghf_state = copy(ghf_ref), tau = tau)

ThermalRelaxationGHF(ghf_ref, ghf_init; tau) =
    ThermalRelaxationGHF(ghf_ref = ghf_ref, ghf_state = copy(ghf_init), tau = tau)

"""
$(TYPEDSIGNATURES)

GHF is computed from the semi-infinite analytical solution to the 1D heat diffusion
equation following Fourier / Carslaw & Jaeger. After a step change `delta_T` in the
ice-base temperature at t = 0, the surface GHF evolves as:

```math
G(t) = G_\\text{ref} - \\frac{k\\,\\Delta T}{\\sqrt{\\pi\\kappa\\,t}}
```

This solution is exact for a semi-infinite medium (i.e. the thermal perturbation has
not yet reached the bottom of the column). It is valid while `sqrt(kappa * t) << H_litho`.
At long times the formula returns `G → G_ref`; use `DiffusionGHF` when the finite-column
correction matters.

# Fields
 - `ghf_ref`: initial equilibrium GHF [W m⁻²]
 - `delta_T`: ice-base temperature step at t = 0, i.e. T_new − T_old [K]
 - `k`: thermal conductivity [W m⁻¹ K⁻¹]
 - `kappa`: thermal diffusivity [m² s⁻¹]
 - `H_litho`: column thickness — semi-∞ approx. valid when `sqrt(kappa * t) << H_litho` [m]
 - `t_elapsed`: total time since the temperature step (1-element mutable vector) [s]

# Convenience constructor

    InfiniteColumnGHF(ghf_ref, delta_T; k, kappa, H_litho)

Construct an `InfiniteColumnGHF` with the clock reset to t = 0 (the moment of the temperature step).
"""
struct InfiniteColumnGHF{M, T, V} <: AbstractGHF
    ghf_ref::M
    delta_T::M
    k::T
    kappa::T
    H_litho::T
    t_elapsed::V   # [1]-element Vector for in-place update within an immutable struct
end

function InfiniteColumnGHF(ghf_ref, delta_T; k, kappa, H_litho)
    return InfiniteColumnGHF(ghf_ref, delta_T, k, kappa, H_litho, [0.0])
end

"""
$(TYPEDSIGNATURES)

GHF is computed from the eigenfunction expansion (Fourier cosine series) of the 1D heat
diffusion equation in a finite lithosphere slab of thickness `H_litho`. After a step change
`delta_T` in the ice-base temperature at t = 0, the surface GHF evolves as:

```math
G(t) = G_\\text{ref} - \\frac{2 k \\Delta T}{H} \\sum_{n=0}^{N-1}
    \\exp\\!\\left(-\\lambda_n^2 \\kappa\\, t\\right), \\qquad
    \\lambda_n = \\frac{(2n+1)\\pi}{2H}
```

This results from simplifications of the analytical solution for the thermodynamics of a finite column derived by [moreno-parada_analytical_2024](@ref). The boundary conditions are Dirichlet at the ice base and Neumann (fixed mantle flux) at the
base of the column, so `G(t) → G_ref` as `t → ∞`. At early times the series reduces to the
semi-infinite approximation used by [`InfiniteColumnGHF`](@ref), but the eigenfunction expansion
converges exponentially once `t ≳ H²/(π² κ)` instead of the algebraic 1/√t tail.

`N` terms are sufficient when `exp(-((N-1/2)π/H)²κt) ≪ 1`; the default `N = 200` is accurate
for time steps as short as a few years with typical lithosphere parameters.

# Fields
 - `ghf_ref`: initial equilibrium GHF [W m⁻²]
 - `delta_T`: ice-base temperature step at t = 0, i.e. T_new − T_old [K]
 - `k`: thermal conductivity [W m⁻¹ K⁻¹]
 - `kappa`: thermal diffusivity [m² s⁻¹]
 - `H_litho`: lithosphere column thickness [m]
 - `N`: number of eigenfunction terms (default 200)
 - `t_elapsed`: total time since the temperature step (1-element mutable vector) [s]
"""
struct FiniteColumnGHF{M, T, V} <: AbstractGHF
    ghf_ref::M
    delta_T::M
    k::T
    kappa::T
    H_litho::T
    N::Int
    t_elapsed::V
end

"""
    FiniteColumnGHF(ghf_ref, delta_T; k, kappa, H_litho, N = 200)

Construct a `FiniteColumnGHF` with the clock reset to t = 0 (the moment of the temperature step).
"""
function FiniteColumnGHF(ghf_ref, delta_T; k, kappa, H_litho, N::Int = 200)
    return FiniteColumnGHF(ghf_ref, delta_T, k, kappa, H_litho, N, [0.0])
end

"""
$(TYPEDSIGNATURES)

GHF is computed by numerically integrating the 1D heat diffusion equation through the
lithosphere column. `ghf_ref` serves as the prescribed deep-mantle heat flux (bottom
boundary condition). Optionally includes radiogenic heat production `H_prod`.

# Fields
 - `ghf_ref`: deep mantle heat flux, bottom BC [W m⁻²]
 - `k`: thermal conductivity [W m⁻¹ K⁻¹]
 - `kappa`: thermal diffusivity k/(ρcₚ) [m² s⁻¹]
 - `H_litho`: lithosphere column thickness [m]
 - `H_prod`: volumetric radiogenic heat production [W m⁻³]
 - `zeta`: sigma coordinates ∈ [0, 1], nz-vector; ζ=0 base, ζ=1 surface
 - `T_col`: temperature state [K], size (nx, ny, nz)

# Convenience constructor

    DiffusionGHF(ghf_ref, T_ice_base; k, kappa, H_litho, H_prod = 0.0, nz = 5)

Construct a `DiffusionGHF` with the temperature column initialised to the analytical
steady-state profile consistent with `ghf_ref` and `T_ice_base`.
"""
@kwdef struct DiffusionGHF{
    M,      # <:AbstractArray{<:AbstractFloat, 2}
    T,      # <:AbstractFloat
    Z,      # <:AbstractVector
    C,      # <:AbstractArray{<:AbstractFloat, 3}
} <: AbstractGHF
    ghf_ref::M
    k::T
    kappa::T
    H_litho::T
    H_prod::T
    zeta::Z
    T_col::C
    d_buf::C  # Thomas-algorithm scratch: modified main diagonal
    b_buf::C  # Thomas-algorithm scratch: modified RHS
end

function DiffusionGHF(ghf_ref, T_ice_base; k, kappa, H_litho, H_prod = 0.0, nz = 5)
    ET    = eltype(T_ice_base)
    zeta  = similar(T_ice_base, ET, nz)
    copyto!(zeta, ET.(LinRange(0, 1, nz)))
    T_col = _steady_T_col(ghf_ref, T_ice_base, k, H_litho, H_prod, zeta)
    d_buf = similar(T_col)
    b_buf = similar(T_col)
    return DiffusionGHF(; ghf_ref, k, kappa, H_litho, H_prod, zeta, T_col, d_buf, b_buf)
end

"""
($TYPEDSIGNATURES)

GHF is computed by numerically integrating the 1D heat diffusion equation through the
lithosphere column using `LinearAlgebra.Tridiagonal` on a serial CPU loop.
Identical physics to [`DiffusionGHF`](@ref); use this when the KernelAbstractions
parallel dispatch is not needed or as a reference implementation.

!!! warning "TridiagonalDiffusionGHF deprecated"

    `TridiagonalDiffusionGHF` should only be used for comparison and testing purposes, since `DiffusionGHF` gives the same results with much better performance.


# Fields
 - `ghf_ref`: deep mantle heat flux, bottom BC [W m⁻²]
 - `k`: thermal conductivity [W m⁻¹ K⁻¹]
 - `kappa`: thermal diffusivity k/(ρcₚ) [m² s⁻¹]
 - `H_litho`: lithosphere column thickness [m]
 - `H_prod`: volumetric radiogenic heat production [W m⁻³]
 - `zeta`: sigma coordinates ∈ [0, 1], nz-vector; ζ=0 base, ζ=1 surface
 - `T_col`: temperature state [K], size (nx, ny, nz)

# Convenience constructor

    TridiagonalDiffusionGHF(ghf_ref, T_ice_base; k, kappa, H_litho, H_prod = 0.0, nz = 5)
"""
@kwdef struct TridiagonalDiffusionGHF{
    M,
    T,
    Z,
    C,
} <: AbstractGHF
    ghf_ref::M
    k::T
    kappa::T
    H_litho::T
    H_prod::T
    zeta::Z
    T_col::C
end

function TridiagonalDiffusionGHF(ghf_ref, T_ice_base; k, kappa, H_litho, H_prod = 0.0, nz = 5)
    zeta  = collect(LinRange(0.0, 1.0, nz))
    T_col = _steady_T_col(ghf_ref, T_ice_base, k, H_litho, H_prod, zeta)
    return TridiagonalDiffusionGHF(; ghf_ref, k, kappa, H_litho, H_prod, zeta, T_col)
end

"""
$(TYPEDSIGNATURES)

Analytical steady-state solution of the Poisson equation with heat production:

```math
    T(\\zeta) = T_\\mathrm{ice} + \\frac{ghf_\\mathrm{ref} \\cdot H \\cdot (1 - \\zeta)}{k} +
                \\frac{H_\\mathrm{prod} \\cdot H^2 \\cdot (1 - \\zeta^2)}{2k}
```
"""
function _steady_T_col(ghf_ref::AbstractArray, T_ice_base, k, H_litho, H_prod, zeta)
    ζ = reshape(zeta, ntuple(_ -> 1, ndims(ghf_ref))..., :)
    return @. T_ice_base + ghf_ref * H_litho * (1 - ζ) / k +
               H_prod * H_litho^2 * (1 - ζ^2) / (2k)
end

function _steady_T_col(ghf_ref::Number, T_ice_base, k, H_litho, H_prod, zeta)
    return @. T_ice_base + ghf_ref * H_litho * (1 - zeta) / k +
               H_prod * H_litho^2 * (1 - zeta^2) / (2k)
end

# Advance a single temperature column by dt (implicit Euler, serial CPU).
# Bottom BC: Neumann — prescribed heat flux Q_deep (ghost-node method).
# Top BC: Dirichlet — prescribed temperature T_ice.
function _solve_column!(col, Q_deep, T_ice, k, kappa, H_litho, H_prod, dt)
    nz = length(col)
    dz = H_litho / (nz - 1)
    r  = kappa * dt / dz^2
    S  = H_prod * kappa * dt / k

    n   = nz - 1
    dl  = fill(-r, n - 1)
    d   = fill(1 + 2r, n)
    du  = fill(-r, n - 1)
    rhs = col[1:n] .+ S

    du[1]   = -2r
    rhs[1] += 2r * dz * Q_deep / k
    rhs[n] += r * T_ice

    col[1:n] .= LinearAlgebra.Tridiagonal(dl, d, du) \ rhs
    col[nz]   = T_ice
    return nothing
end

function _surface_ghf(col, k, H_litho)
    nz = length(col)
    dz = H_litho / (nz - 1)
    return -k * (col[nz] - col[nz - 1]) / dz
end

@inline _getval(x::AbstractArray, i, j) = x[i, j]
@inline _getval(x::Number, _, _) = x

# One thread per (i, j) column.  Thomas algorithm: forward sweep then back substitution.
# Bottom BC: Neumann (ghost-node, prescribed heat flux Q_deep).
# Top BC:    Dirichlet (prescribed ice-base temperature T_ice).
# Requires nz ≥ 3.
@kernel function _diffusion_kernel!(T_col, ghf, ghf_ref_arr, T_ice_base,
                                     d_buf, b_buf, k, kappa, H_litho, H_prod, dt)
    i, j = @index(Global, NTuple)
    nz  = size(T_col, 3)
    n   = nz - 1
    dz  = H_litho / (nz - 1)
    r   = kappa * dt / dz^2
    S   = H_prod * kappa * dt / k
    Q   = _getval(ghf_ref_arr, i, j)
    Tbc = T_ice_base[i, j]

    # ── Forward sweep ─────────────────────────────────────────────────────────
    # d_buf[i,j,p]: modified main diagonal d'[p]
    # b_buf[i,j,p]: modified RHS           b'[p]
    #
    # System coefficients (row p):
    #   sub-diag a[p]  = -r           (p = 2..n)
    #   main-diag d[p] = 1+2r
    #   super-diag c[p]= -2r (p=1), -r (p=2..n-1); no entry at p=n
    #   rhs b[1] = T_col[1]+S+2r·dz·Q/k  (Neumann ghost-node)
    #   rhs b[p] = T_col[p]+S            (interior, 2≤p<n)
    #   rhs b[n] = T_col[n]+S+r·T_ice   (Dirichlet shift)
    d_buf[i, j, 1] = one(k) + 2r
    b_buf[i, j, 1] = T_col[i, j, 1] + S + 2r * dz * Q / k
    for p in 2:n
        c_prev         = p == 2 ? -2r : -r           # c[p-1]
        factor         = -r / d_buf[i, j, p-1]       # a[p] / d'[p-1]
        d_buf[i, j, p] = (one(k) + 2r) - factor * c_prev
        rhs_p          = T_col[i, j, p] + S + (p == n ? r * Tbc : zero(r))
        b_buf[i, j, p] = rhs_p - factor * b_buf[i, j, p-1]
    end

    # ── Back substitution ─────────────────────────────────────────────────────
    T_col[i, j, n] = b_buf[i, j, n] / d_buf[i, j, n]
    for p in n-1:-1:1
        c_p            = p == 1 ? -2r : -r            # c[p]
        T_col[i, j, p] = (b_buf[i, j, p] - c_p * T_col[i, j, p+1]) / d_buf[i, j, p]
    end
    T_col[i, j, nz] = Tbc

    # ── Surface GHF (Fourier's law, first-order upward difference) ────────────
    ghf[i, j] = -k * (T_col[i, j, nz] - T_col[i, j, nz-1]) / dz
end

# ── Enthalpy conversions ───────────────────────────────────────────────────────

"""
$(TYPEDSIGNATURES)

Convert temperature `T` and water fraction `omega` to specific enthalpy, given
specific heat `cp`, pressure melting point `T_pmp`, and latent heat of fusion `L`.
For cold material (ω = 0) this reduces to `H = cₚ T`.
"""
to_enthalpy(T, cp, T_pmp, L; omega = 0.0) =
    (1 - omega) * cp * T + omega * (cp * T_pmp + L)

"""
$(TYPEDSIGNATURES)

Recover temperature and water fraction from specific enthalpy `H`.
Returns `(T, ω)`: cold material gives ω = 0; temperate material gives T = T_pmp.
"""
function from_enthalpy(H, cp, T_pmp, L)
    H_pmp = cp * T_pmp
    return H > H_pmp ? (T_pmp, (H - H_pmp) / L) : (H / cp, zero(H / cp))
end

# ── ghf! methods ──────────────────────────────────────────────────────────────

"""
$(TYPEDSIGNATURES)

Update `ghf` in-place according to the basal ice temperature and to the specified [`AbstractGHF`](@ref).
"""
function ghf!(ghf, T_ice_base, thermo::PrescribedGHF)
    ghf .= thermo.ghf_ref
    return nothing
end

function ghf!(ghf, T_ice_base, thermo::SteadyStateGHF)
    ghf .= thermo.k .* (thermo.T_mantle .- T_ice_base) ./ thermo.H_litho
    return nothing
end

function ghf!(ghf, _, thermo::ThermalRelaxationGHF, dt)
    r = dt / thermo.tau
    @. thermo.ghf_state = (thermo.ghf_state + thermo.ghf_ref * r) / (1 + r)
    ghf .= thermo.ghf_state
    return nothing
end

function ghf!(ghf, _, thermo::InfiniteColumnGHF, dt)
    thermo.t_elapsed[1] += dt
    t = thermo.t_elapsed[1]
    @. ghf = thermo.ghf_ref - thermo.k * thermo.delta_T / sqrt(pi * thermo.kappa * t)
    return nothing
end

function ghf!(ghf, _, thermo::FiniteColumnGHF, dt)
    thermo.t_elapsed[1] += dt
    t = thermo.t_elapsed[1]
    s = sum(n -> exp(-((n + 0.5) * pi / thermo.H_litho)^2 * thermo.kappa * t), 0:thermo.N-1)
    @. ghf = thermo.ghf_ref - 2 * thermo.k * thermo.delta_T / thermo.H_litho * s
    return nothing
end

function ghf!(ghf, T_ice_base, thermo::DiffusionGHF, dt)
    backend = KernelAbstractions.get_backend(ghf)
    kernel! = _diffusion_kernel!(backend, (16, 16))
    kernel!(thermo.T_col, ghf, thermo.ghf_ref, T_ice_base,
            thermo.d_buf, thermo.b_buf,
            thermo.k, thermo.kappa, thermo.H_litho, thermo.H_prod, dt;
            ndrange = size(ghf))
    KernelAbstractions.synchronize(backend)
    return nothing
end

function ghf!(ghf, T_ice_base, thermo::TridiagonalDiffusionGHF, dt)
    for I in CartesianIndices(ghf)
        Q_deep = thermo.ghf_ref isa AbstractArray ? thermo.ghf_ref[I] : thermo.ghf_ref
        col    = @view thermo.T_col[I, :]
        _solve_column!(col, Q_deep, T_ice_base[I], thermo.k, thermo.kappa,
                       thermo.H_litho, thermo.H_prod, dt)
        ghf[I] = _surface_ghf(col, thermo.k, thermo.H_litho)
    end
    return nothing
end