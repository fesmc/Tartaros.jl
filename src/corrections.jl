"""
($TYPEDSIGNATURES)
"""
abstract type AbstractGHFCorrection end

"""
($TYPEDSIGNATURES)
"""
struct NoGHFCorrection <: AbstractGHFCorrection end

"""
($TYPEDSIGNATURES)
"""
@kwdef struct TopographicGHFCorrection{T} <: AbstractGHFCorrection
    α::T = 950      # characteristic height
                    # typically between 400 and 1_400 m
    r::T = 5_000    # radius for building average elevation
                    # typically between 1_000 and 10_000 m
end

_correct(ghf, _, _, ::NoGHFCorrection) = ghf

function _correct(ghf, z_bed, z_bed_mean, correction::TopographicGHFCorrection)
    return ghf * (1 + (z_bed_mean - z_bed) / correction.α)
end

"""
($TYPEDSIGNATURES)

Return the corrected GHF value (or field) given bed elevation `z_bed` and its
spatially-averaged counterpart `z_bed_mean`.
"""
correct(ghf, z_bed, z_bed_mean, correction::AbstractGHFCorrection) =
    _correct(ghf, z_bed, z_bed_mean, correction)

function correct(ghf::AbstractMatrix, z_bed, z_bed_mean, correction::AbstractGHFCorrection)
    ghf_corrected = similar(ghf)
    map!((g, z, z_mean) -> _correct(g, z, z_mean, correction), ghf_corrected, ghf, z_bed, z_bed_mean)
    return ghf_corrected
end

"""
($TYPEDSIGNATURES)

Same as [`correct`](@ref), but modifies `ghf` in-place.
"""
function correct!(ghf::AbstractMatrix, z_bed, z_bed_mean, correction::AbstractGHFCorrection)
    map!((g, z, z_mean) -> _correct(g, z, z_mean, correction), ghf, ghf, z_bed, z_bed_mean)
    return nothing
end
