"""
$(TYPEDEF)

Abstract type for corrections applied to a reference geothermal heat flux (GHF) field.

# Available subtypes
 - [`NoGHFCorrection`](@ref)
 - [`TopographicGHFCorrection`](@ref)
"""
abstract type AbstractGHFCorrection end

"""
$(TYPEDEF)

No correction is applied to GHF.
"""
struct NoGHFCorrection <: AbstractGHFCorrection end

"""
$(TYPEDEF)

Topographic correction of GHF after [colgan_topographic_2021](@citet): GHF tends to be
elevated in topographic lows (valleys) and reduced in topographic highs (ridges). GHF is
scaled linearly with the departure of the local bed elevation `z_bed` from its spatial
mean `z_bed_mean`, over a characteristic height `α`:

```text
ghf_corrected = ghf ⋅ (1 + (z_bed_mean - z_bed) / α)
```

The spatial mean is obtained with an [`AbstractSmoothingKernel`](@ref).

# Fields
$(TYPEDFIELDS)
"""
@kwdef struct TopographicGHFCorrection{T} <: AbstractGHFCorrection
    "characteristic height, typically between `400` and `1_400` [m]"
    α::T = 950.0
    "radius for building average elevation, typically between `1_000` and `10_000` [m]"
    r::T = 5_000.0
end

_correct(ghf, _, _, ::NoGHFCorrection) = ghf

_correct(ghf, z_bed, z_bed_mean, c::TopographicGHFCorrection) =
    ghf * (1 + (z_bed_mean - z_bed) / c.α)

"""
$(TYPEDSIGNATURES)

Return the corrected GHF given bed elevation `z_bed` and its spatially-averaged
counterpart `z_bed_mean`, following the specified [`AbstractGHFCorrection`](@ref).
Scalars and fields are both accepted; the result matches the shape of the inputs.
"""
correct(ghf, z_bed, z_bed_mean, correction::AbstractGHFCorrection) =
    _correct.(ghf, z_bed, z_bed_mean, Ref(correction))

"""
$(TYPEDSIGNATURES)

Same as [`correct`](@ref), but modifies `ghf` in-place.
"""
function correct!(ghf::AbstractMatrix, z_bed, z_bed_mean, correction::AbstractGHFCorrection)
    ghf .= _correct.(ghf, z_bed, z_bed_mean, Ref(correction))
    return nothing
end
