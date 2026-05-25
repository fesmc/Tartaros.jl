"""
$(TYPEDSIGNATURES)

Abstract type for geothermal heat flux (GHF) models.

# Available subtypes
 - [`NoGHFCorrection`](@ref)
 - [`TopographicGHFCorrection`](@ref)
"""
abstract type AbstractGHFCorrection end

"""
$(TYPEDSIGNATURES)

No correction is applied to GHF.
"""
struct NoGHFCorrection <: AbstractGHFCorrection end

"""
$(TYPEDSIGNATURES)

Topographic correction of GHF based on [colgan_topographic_2021](@citet). This accounts for fact that GHF is more likely to be elevated in topographic lows (e.g. valleys) and reduced in topographic highs (e.g. ridges). The correction is applied as a linear function of the difference between the local bed elevation `z_bed` and its spatially-averaged counterpart `z_bed_mean`, scaled by a characteristic height `α` that controls the strength of the correction:

```math
\\begin{aligned}
    G_\\mathrm{corrected} = G \\cdot \\left( 1 + \\dfrac{\\bar{z}_{b} - z_{b}}{\\alpha} \\right)
\\end{aligned}
```

The spatial averaging is performed using a smoothing kernel, that can be defined via [`AbstractSmoothingKernel`](@ref).

# Fields
 - `α`: characteristic height (m), typically between 400 and 1_400 m
 - `r`: radius (m) for building average elevation, typically between 1_000 and 10_000 m
"""
@kwdef struct TopographicGHFCorrection{T} <: AbstractGHFCorrection
    α::T = 950.0
    r::T = 5_000.0
end

_correct(ghf, _, _, ::NoGHFCorrection) = ghf

function _correct(ghf, z_bed, z_bed_mean, correction::TopographicGHFCorrection)
    return ghf * (1 + (z_bed_mean - z_bed) / correction.α)
end

"""
$(TYPEDSIGNATURES)

Return the corrected GHF value (or field) given bed elevation `z_bed` and its
spatially-averaged counterpart `z_bed_mean` based on the specified [`AbstractGHFCorrection`](@ref).
"""
correct(ghf, z_bed, z_bed_mean, correction::AbstractGHFCorrection) =
    _correct(ghf, z_bed, z_bed_mean, correction)

function correct(ghf::AbstractMatrix, z_bed, z_bed_mean, correction::AbstractGHFCorrection)
    return map((g, z, z_mean) -> _correct(g, z, z_mean, correction), ghf, z_bed, z_bed_mean)
end

"""
$(TYPEDSIGNATURES)

Same as [`correct`](@ref), but modifies `ghf` in-place.
"""
function correct!(ghf::AbstractMatrix, z_bed, z_bed_mean, correction::AbstractGHFCorrection)
    map!((g, z, z_mean) -> _correct(g, z, z_mean, correction), ghf, ghf, z_bed, z_bed_mean)
    return nothing
end
