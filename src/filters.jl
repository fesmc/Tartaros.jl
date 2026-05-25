# TODO: this could be part of KryosTools.jl

"""
    AbstractSmoothingKernel

Abstract type for spatial smoothing kernels. See [`ConstantSmoothingKernel`](@ref) and [`GaussianSmoothingKernel`](@ref).
"""
abstract type AbstractSmoothingKernel end

"""
    ConstantSmoothingKernel(; r = 5_000.0)

Uniform (top-hat) smoothing kernel that averages all grid cells within radius `r` (metres).
"""
@kwdef struct ConstantSmoothingKernel{T} <: AbstractSmoothingKernel
    r::T = 5_000.0   # disk radius in meters
end

"""
    GaussianSmoothingKernel(; r = 5_000.0)

Gaussian smoothing kernel with standard deviation `r` (metres), truncated at 3σ.
"""
@kwdef struct GaussianSmoothingKernel{T} <: AbstractSmoothingKernel
    r::T = 5_000.0   # standard deviation in meters
end

"""
    smooth(z, dx, dy, kernel)
    smooth(z, dx, kernel)

Spatially smooth the field `z` on a regular grid with spacing `dx` × `dy`
using `kernel`. Boundary cells are extended by nearest-neighbor repetition.
Returns a `Matrix{Float64}` of the same size as `z`.
"""
function smooth(z::AbstractMatrix, dx::Real, dy::Real, kernel::AbstractSmoothingKernel)
    K = _build_kernel(dx, dy, kernel)
    return _apply_kernel(z, K)
end

smooth(z::AbstractMatrix, dx::Real, kernel::AbstractSmoothingKernel) = smooth(z, dx, dx, kernel)

function _build_kernel(dx::Real, dy::Real, kernel::ConstantSmoothingKernel)
    r = kernel.r
    nx = 2 * ceil(Int, r / dx) + 1
    ny = 2 * ceil(Int, r / dy) + 1
    cx, cy = (nx + 1) ÷ 2, (ny + 1) ÷ 2
    K = [sqrt(((i - cx) * dx)^2 + ((j - cy) * dy)^2) <= r ? 1.0 : 0.0
         for i in 1:nx, j in 1:ny]
    return K ./ sum(K)
end

function _build_kernel(dx::Real, dy::Real, kernel::GaussianSmoothingKernel)
    r = kernel.r
    nx = 2 * ceil(Int, 3r / dx) + 1   # truncate at 3σ
    ny = 2 * ceil(Int, 3r / dy) + 1
    cx, cy = (nx + 1) ÷ 2, (ny + 1) ÷ 2
    K = [exp(-0.5 * (((i - cx) * dx)^2 + ((j - cy) * dy)^2) / r^2)
         for i in 1:nx, j in 1:ny]
    return K ./ sum(K)
end

function _apply_kernel(z::AbstractMatrix, K::AbstractMatrix)
    nz1, nz2 = size(z)
    nk1, nk2 = size(K)
    h1, h2 = nk1 ÷ 2, nk2 ÷ 2
    out = Matrix{Float64}(undef, nz1, nz2)
    for j2 in 1:nz2, j1 in 1:nz1
        acc = 0.0
        for k2 in 1:nk2, k1 in 1:nk1
            i1 = clamp(j1 - h1 + k1 - 1, 1, nz1)
            i2 = clamp(j2 - h2 + k2 - 1, 1, nz2)
            acc += K[k1, k2] * z[i1, i2]
        end
        out[j1, j2] = acc
    end
    return out
end
