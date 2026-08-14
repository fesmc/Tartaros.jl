"""
$(TYPEDEF)

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
$(TYPEDEF)

GHF is held constant at `ghf_ref` and excludes any feedbacks from changes in basal ice temperature.

# Fields
$(TYPEDFIELDS)
"""
struct PrescribedGHF{M} <: AbstractGHF
    "reference GHF field [mW m⁻²]"
    ghf_ref::M
end

"""
$(TYPEDEF)

GHF is recomputed at every time step from Fourier's law applied to a linear
(steady-state) temperature profile through the lithosphere:

```text
G = k ⋅ (T_mantle - T_ice) / H_litho
```

# Fields
$(TYPEDFIELDS)
"""
@kwdef struct SteadyStateGHF{T,M} <: AbstractGHF
    "thermal conductivity [W m⁻¹ K⁻¹]"
    k::T
    "lithosphere thickness [m]"
    H_litho::T
    "temperature at the base of the lithosphere, scalar or spatial field [K]"
    T_mantle::M
end

function SteadyStateGHF(ghf_ref, T_ice_base; k, H_litho)
    return SteadyStateGHF(
        k = k,
        H_litho = H_litho,
        T_mantle = (@. T_ice_base + ghf_ref * H_litho / k),
    )
end

"""
$(TYPEDEF)

GHF relaxes exponentially toward the steady-state target `ghf_ref`
with timescale `tau` (Huybrechts/Ritz lag model):

```text
dG/dt = (G_ref - G) / τ
```

This model has a relatively inaccurate transient behaviour (see the examples), so prefer
another one unless comparing against legacy code.

# Fields
$(TYPEDFIELDS)
"""
@kwdef struct ThermalRelaxationGHF{M,T} <: AbstractGHF
    "asymptotic GHF target [W m⁻²]"
    ghf_ref::M
    "current GHF state, updated in-place by [`ghf!`](@ref) [W m⁻²]"
    ghf_state::M
    # TODO: this could be tau(x,y), derived from H_litho(x,y) and kappa
    "thermal relaxation timescale [s]"
    tau::T
end

ThermalRelaxationGHF(ghf_ref; tau) =
    ThermalRelaxationGHF(ghf_ref = ghf_ref, ghf_state = copy(ghf_ref), tau = tau)

ThermalRelaxationGHF(ghf_ref, ghf_init; tau) =
    ThermalRelaxationGHF(ghf_ref = ghf_ref, ghf_state = copy(ghf_init), tau = tau)

"""
$(TYPEDEF)

GHF is computed from the semi-infinite analytical solution to the 1D heat diffusion
equation following Fourier / Carslaw & Jaeger. After a step change `delta_T` in the
ice-base temperature at t = 0, the surface GHF evolves as:

```text
G(t) = G_ref - k ΔT / √(π κ t)
```

Exact while the perturbation has not reached the bottom of the column, i.e. while
`sqrt(kappa * t) << H_litho`. Use [`FiniteColumnGHF`](@ref) or [`DiffusionGHF`](@ref) when
the finite-column correction matters.

# Fields
$(TYPEDFIELDS)

# Convenience constructor

    InfiniteColumnGHF(ghf_ref, delta_T; k, kappa, H_litho)

Construct an `InfiniteColumnGHF` with the clock reset to t = 0 (the moment of the temperature step).
"""
struct InfiniteColumnGHF{M,T,V} <: AbstractGHF
    "initial equilibrium GHF [W m⁻²]"
    ghf_ref::M
    "ice-base temperature step at t = 0, i.e. `T_new - T_old` [K]"
    delta_T::M
    "thermal conductivity [W m⁻¹ K⁻¹]"
    k::T
    "thermal diffusivity [m² s⁻¹]"
    kappa::T
    "column thickness — semi-∞ approx. valid when `sqrt(kappa * t) << H_litho` [m]"
    H_litho::T
    "time since the temperature step, as a 1-element array (see [`_zero_clock`](@ref)) [s]"
    t_elapsed::V
end

function InfiniteColumnGHF(ghf_ref, delta_T; k, kappa, H_litho)
    return InfiniteColumnGHF(ghf_ref, delta_T, k, kappa, H_litho, _zero_clock(ghf_ref))
end

"""
$(TYPEDSIGNATURES)

Allocate the 1-element array holding the elapsed time of a stateful GHF model.

`similar(ghf_ref, ...)` keeps the clock on the same device as the model fields: on a GPU
this avoids a host round-trip per step, and under an array compiler such as Reactant a
clock held in a host `Vector` is constant-folded and would silently never advance. For the
same reason it must only be updated and read by broadcast — `t_elapsed[1]` is scalar
indexing, which traced arrays reject.
"""
function _zero_clock(ghf_ref::AbstractArray)
    t = similar(ghf_ref, eltype(ghf_ref), 1)
    fill!(t, zero(eltype(ghf_ref)))
    return t
end

"""
$(TYPEDEF)

GHF is computed from the eigenfunction expansion (Fourier cosine series) of the 1D heat
diffusion equation in a finite lithosphere slab of thickness `H_litho`. After a step change
`delta_T` in the ice-base temperature at t = 0, the surface GHF evolves as:

```text
G(t) = G_ref - (2 k ΔT / H) ⋅ ∑ₙ exp(-λₙ² κ t)     for n = 0 … N-1
λₙ   = (2n + 1) π / (2H)
```

Simplified from the finite-column solution of [moreno-parada_analytical_2024](@ref), with
Dirichlet at the ice base and Neumann (fixed mantle flux) below, so `G(t) → G_ref`. At early
times it matches [`InfiniteColumnGHF`](@ref), but relaxes exponentially once
`t ≳ H²/(π² κ)` rather than as `1/√t`.

`N` terms suffice when `exp(-((N-1/2)π/H)²κt) ≪ 1`; the default is accurate down to
time steps of a few years for typical lithosphere parameters.

# Fields
$(TYPEDFIELDS)
"""
struct FiniteColumnGHF{M,T,V,L} <: AbstractGHF
    "initial equilibrium GHF [W m⁻²]"
    ghf_ref::M
    "ice-base temperature step at t = 0, i.e. `T_new - T_old` [K]"
    delta_T::M
    "thermal conductivity [W m⁻¹ K⁻¹]"
    k::T
    "thermal diffusivity [m² s⁻¹]"
    kappa::T
    "lithosphere column thickness [m]"
    H_litho::T
    "number of eigenfunction terms (default 200)"
    N::Int
    "time since the temperature step, as a 1-element array (see [`_zero_clock`](@ref)) [s]"
    t_elapsed::V
    "precomputed eigenvalues `λₙ² = ((2n+1)π / 2H)²`, `N`-vector [m⁻²]"
    lambda2::L
end

"""
    FiniteColumnGHF(ghf_ref, delta_T; k, kappa, H_litho, N = 200)

Construct a `FiniteColumnGHF` with the clock reset to t = 0 (the moment of the temperature step).
"""
function FiniteColumnGHF(ghf_ref, delta_T; k, kappa, H_litho, N::Int = 200)
    ET = eltype(ghf_ref)
    lambda2 = similar(ghf_ref, ET, N)
    copyto!(lambda2, ET[((n + 0.5) * pi / H_litho)^2 for n = 0:(N-1)])
    return FiniteColumnGHF(
        ghf_ref,
        delta_T,
        k,
        kappa,
        H_litho,
        N,
        _zero_clock(ghf_ref),
        lambda2,
    )
end

"""
$(TYPEDEF)

GHF is computed by numerically integrating the 1D heat diffusion equation through the
lithosphere column, with `ghf_ref` as the prescribed deep-mantle flux at the base and
optional radiogenic heat production `H_prod`.

# Fields
$(TYPEDFIELDS)

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
    "deep mantle heat flux, bottom BC [W m⁻²]"
    ghf_ref::M
    "thermal conductivity [W m⁻¹ K⁻¹]"
    k::T
    "thermal diffusivity k/(ρcₚ) [m² s⁻¹]"
    kappa::T
    "lithosphere column thickness [m]"
    H_litho::T
    "volumetric radiogenic heat production [W m⁻³]"
    H_prod::T
    "sigma coordinates ∈ [0, 1], nz-vector; ζ=0 base, ζ=1 surface"
    zeta::Z
    "temperature state [K], size (nx, ny, nz)"
    T_col::C
    "Thomas-algorithm scratch: modified main diagonal"
    d_buf::C
    "Thomas-algorithm scratch: modified RHS"
    b_buf::C
end

function DiffusionGHF(ghf_ref, T_ice_base; k, kappa, H_litho, H_prod = 0.0, nz = 5)
    ET = eltype(T_ice_base)
    zeta = similar(T_ice_base, ET, nz)
    copyto!(zeta, ET.(LinRange(0, 1, nz)))
    T_col = _steady_T_col(ghf_ref, T_ice_base, k, H_litho, H_prod, zeta)
    d_buf = similar(T_col)
    b_buf = similar(T_col)
    return DiffusionGHF(; ghf_ref, k, kappa, H_litho, H_prod, zeta, T_col, d_buf, b_buf)
end

"""
$(TYPEDSIGNATURES)

Analytical steady-state solution of the Poisson equation with heat production:

```text
T(ζ) = T_ice + ghf_ref ⋅ H (1 - ζ) / k + H_prod ⋅ H² (1 - ζ²) / (2k)
```

`ghf_ref` may be a field or a scalar; `ndims` of a scalar is 0, so the reshape leaves
`zeta` untouched in that case.
"""
function _steady_T_col(ghf_ref, T_ice_base, k, H_litho, H_prod, zeta)
    ζ = reshape(zeta, ntuple(_ -> 1, ndims(ghf_ref))..., :)
    return @. T_ice_base +
              ghf_ref * H_litho * (1 - ζ) / k +
              H_prod * H_litho^2 * (1 - ζ^2) / (2k)
end

@inline _getval(x::AbstractArray, i, j) = x[i, j]
@inline _getval(x::Number, _, _) = x

# One thread per (i, j) column.  Thomas algorithm: forward sweep then back substitution.
# Bottom BC: Neumann (ghost-node, prescribed heat flux Q_deep).
# Top BC:    Dirichlet (prescribed ice-base temperature T_ice).
# Requires nz ≥ 3.
@kernel function _diffusion_kernel!(
    T_col,
    ghf,
    ghf_ref_arr,
    T_ice_base,
    d_buf,
    b_buf,
    k,
    kappa,
    H_litho,
    H_prod,
    dt,
)
    i, j = @index(Global, NTuple)
    nz = size(T_col, 3)
    n = nz - 1
    dz = H_litho / (nz - 1)
    r = kappa * dt / dz^2
    S = H_prod * kappa * dt / k
    Q = _getval(ghf_ref_arr, i, j)
    Tbc = T_ice_base[i, j]

    # ── Forward sweep ─────────────────────────────────────────────────────────
    # d_buf holds the modified main diagonal d'[p], b_buf the modified RHS b'[p]. The
    # system is a[p] = -r, d[p] = 1+2r, c[p] = -2r at p=1 and -r elsewhere; the ghost-node
    # and Dirichlet terms enter the RHS at p = 1 and p = n respectively.
    d_buf[i, j, 1] = one(k) + 2r
    b_buf[i, j, 1] = T_col[i, j, 1] + S + 2r * dz * Q / k
    for p = 2:n
        c_prev = p == 2 ? -2r : -r           # c[p-1]
        factor = -r / d_buf[i, j, p-1]       # a[p] / d'[p-1]
        d_buf[i, j, p] = (one(k) + 2r) - factor * c_prev
        rhs_p = T_col[i, j, p] + S + (p == n ? r * Tbc : zero(r))
        b_buf[i, j, p] = rhs_p - factor * b_buf[i, j, p-1]
    end

    # ── Back substitution ─────────────────────────────────────────────────────
    T_col[i, j, n] = b_buf[i, j, n] / d_buf[i, j, n]
    for p = (n-1):-1:1
        c_p = p == 1 ? -2r : -r            # c[p]
        T_col[i, j, p] = (b_buf[i, j, p] - c_p * T_col[i, j, p+1]) / d_buf[i, j, p]
    end
    T_col[i, j, nz] = Tbc

    # ── Surface GHF (Fourier's law, first-order upward difference) ────────────
    ghf[i, j] = -k * (T_col[i, j, nz] - T_col[i, j, nz-1]) / dz
end

# ── Array-compiler friendly diffusion ─────────────────────────────────────────
# The same Thomas algorithm as `_diffusion_kernel!`, but over whole (nx, ny) slabs, which
# array compilers such as Reactant can trace where the element-indexing kernel cannot. The
# z-loop bound is static, so it unrolls and fuses; the coefficients d[p] are the same for
# every column, so no scratch arrays are needed.
#
# NOT a drop-in replacement: without a fusing compiler every slab is a separate pass over
# memory, ~14x slower than the kernel on `CuArray`s. Selected by `TartarosReactantExt`
# only. See `tartaros-roadmap/reactant.md`.
function _diffusion_slabs!(
    ghf,
    T_col,
    ghf_ref,
    T_ice_base,
    k,
    kappa,
    H_litho,
    H_prod,
    dt,
)
    nz = size(T_col, 3)
    n = nz - 1
    dz = H_litho / (nz - 1)
    r = kappa * dt / dz^2
    S = H_prod * kappa * dt / k

    # Forward sweep. d[p] is the modified main diagonal (a scalar, shared by all columns),
    # b[p] the modified RHS (one (nx, ny) slab per level).
    d = Vector{typeof(r)}(undef, n)
    b1 = @. T_col[:, :, 1] + S + 2r * dz * ghf_ref / k
    b = Vector{typeof(b1)}(undef, n)
    d[1], b[1] = one(r) + 2r, b1

    for p = 2:n
        c_prev = p == 2 ? -2r : -r            # c[p-1]
        factor = -r / d[p-1]                # a[p] / d'[p-1]
        d[p] = (one(r) + 2r) - factor * c_prev
        rhs =
            p == n ? (@. T_col[:, :, p] + S + r * T_ice_base) :  # Dirichlet shift
            (@. T_col[:, :, p] + S)
        b[p] = @. rhs - factor * b[p-1]
    end

    # Back substitution.
    T_col[:, :, n] = b[n] ./ d[n]
    for p = (n-1):-1:1
        c_p = p == 1 ? -2r : -r
        T_col[:, :, p] = (b[p] .- c_p .* T_col[:, :, p+1]) ./ d[p]
    end
    T_col[:, :, nz] = T_ice_base

    ghf .= .-k .* (T_col[:, :, nz] .- T_col[:, :, nz-1]) ./ dz
    return nothing
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

function ghf!(ghf, _, thermo::ThermalRelaxationGHF{M,T}, dt::T) where {M,T}
    r = dt / thermo.tau
    @. thermo.ghf_state = (thermo.ghf_state + thermo.ghf_ref * r) / (1 + r)
    ghf .= thermo.ghf_state
    return nothing
end

# The clock is advanced and read by broadcast only; see `_zero_clock` for why.
function ghf!(ghf, _, thermo::InfiniteColumnGHF{M,T}, dt::T) where {M,T}
    thermo.t_elapsed .+= dt
    @. ghf =
        thermo.ghf_ref -
        thermo.k * thermo.delta_T / sqrt(pi * thermo.kappa * thermo.t_elapsed)
    return nothing
end

function ghf!(ghf, _, thermo::FiniteColumnGHF{M,T}, dt::T) where {M,T}
    thermo.t_elapsed .+= dt
    s = sum(@. exp(-thermo.lambda2 * thermo.kappa * thermo.t_elapsed))
    @. ghf = thermo.ghf_ref - 2 * thermo.k * thermo.delta_T / thermo.H_litho * s
    return nothing
end

function ghf!(ghf, T_ice_base, thermo::DiffusionGHF{M,T}, dt::T) where {M,T}
    backend = KernelAbstractions.get_backend(ghf)
    kernel! = _diffusion_kernel!(backend, (16, 16))
    kernel!(
        thermo.T_col,
        ghf,
        thermo.ghf_ref,
        T_ice_base,
        thermo.d_buf,
        thermo.b_buf,
        thermo.k,
        thermo.kappa,
        thermo.H_litho,
        thermo.H_prod,
        dt;
        ndrange = size(ghf),
    )
    KernelAbstractions.synchronize(backend)
    return nothing
end
