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

# A Gaussian kernel is the outer product of two 1D Gaussians, so two 1D passes replace the
# full 2D stencil: O(nx·ny·(k₁+k₂)) instead of O(nx·ny·k₁·k₂). The result is identical
# because nearest-neighbour clamping is itself separable.
function smooth(z::AbstractMatrix, dx::Real, dy::Real, kernel::GaussianSmoothingKernel)
    kx, ky = _build_kernel_factors(dx, dy, kernel)
    return _apply_separable(z, kx, ky)
end

smooth(z::AbstractMatrix, dx::Real, kernel::AbstractSmoothingKernel) =
    smooth(z, dx, dx, kernel)

function _build_kernel(dx::Real, dy::Real, kernel::ConstantSmoothingKernel)
    r = kernel.r
    nx = 2 * ceil(Int, r / dx) + 1
    ny = 2 * ceil(Int, r / dy) + 1
    cx, cy = (nx + 1) ÷ 2, (ny + 1) ÷ 2
    K = [
        sqrt(((i - cx) * dx)^2 + ((j - cy) * dy)^2) <= r ? 1.0 : 0.0 for
        i = 1:nx, j = 1:ny
    ]
    return K ./ sum(K)
end

function _build_kernel(dx::Real, dy::Real, kernel::GaussianSmoothingKernel)
    r = kernel.r
    nx = 2 * ceil(Int, 3r / dx) + 1   # truncate at 3σ
    ny = 2 * ceil(Int, 3r / dy) + 1
    cx, cy = (nx + 1) ÷ 2, (ny + 1) ÷ 2
    K = [
        exp(-0.5 * (((i - cx) * dx)^2 + ((j - cy) * dy)^2) / r^2) for i = 1:nx, j = 1:ny
    ]
    return K ./ sum(K)
end

# 1D factors of the separable Gaussian: _build_kernel(dx, dy, k) == kx * ky'
function _build_kernel_factors(dx::Real, dy::Real, kernel::GaussianSmoothingKernel)
    return _gaussian_factor(dx, kernel.r), _gaussian_factor(dy, kernel.r)
end

function _gaussian_factor(d::Real, r::Real)
    n = 2 * ceil(Int, 3r / d) + 1     # truncate at 3σ
    c = (n + 1) ÷ 2
    k = [exp(-0.5 * ((i - c) * d)^2 / r^2) for i = 1:n]
    return k ./ sum(k)
end

function _apply_kernel(z::AbstractMatrix, K::AbstractMatrix)
    nz1, nz2 = size(z)
    nk1, nk2 = size(K)
    h1, h2 = nk1 ÷ 2, nk2 ÷ 2
    out = Matrix{Float64}(undef, nz1, nz2)
    for j2 = 1:nz2, j1 = 1:nz1
        acc = 0.0
        for k2 = 1:nk2, k1 = 1:nk1
            i1 = clamp(j1 - h1 + k1 - 1, 1, nz1)
            i2 = clamp(j2 - h2 + k2 - 1, 1, nz2)
            acc += K[k1, k2] * z[i1, i2]
        end
        out[j1, j2] = acc
    end
    return out
end

# Two 1D passes, used for separable (Gaussian) kernels.
function _apply_separable(z::AbstractMatrix, kx::AbstractVector, ky::AbstractVector)
    nz1, nz2 = size(z)
    h1, h2 = length(kx) ÷ 2, length(ky) ÷ 2
    tmp = Matrix{Float64}(undef, nz1, nz2)
    out = Matrix{Float64}(undef, nz1, nz2)
    for j2 = 1:nz2, j1 = 1:nz1
        acc = 0.0
        for k1 in eachindex(kx)
            acc += kx[k1] * z[clamp(j1 - h1 + k1 - 1, 1, nz1), j2]
        end
        tmp[j1, j2] = acc
    end
    for j2 = 1:nz2, j1 = 1:nz1
        acc = 0.0
        for k2 in eachindex(ky)
            acc += ky[k2] * tmp[j1, clamp(j2 - h2 + k2 - 1, 1, nz2)]
        end
        out[j1, j2] = acc
    end
    return out
end

# ── Array-compiler friendly variants ──────────────────────────────────────────
# The loops above index individual elements, which array compilers such as Reactant
# cannot trace. The formulations below express the same stencil as one shifted,
# edge-replicated slice per non-zero kernel weight. They contain no scalar indexing,
# but allocate a full field per slice and are therefore much slower on ordinary CPU
# arrays — they are selected by `TartarosReactantExt`, not by the default `smooth`.

function _pad_replicate(z::AbstractMatrix, h1::Int, h2::Int)
    zv = vcat((z[1:1, :] for _ = 1:h1)..., z, (z[end:end, :] for _ = 1:h1)...)
    return hcat((zv[:, 1:1] for _ = 1:h2)..., zv, (zv[:, end:end] for _ = 1:h2)...)
end

# Separable counterpart of `_apply_kernel_shifted`: k₁+k₂ slices instead of k₁·k₂. The
# saving is dramatic once the kernel is large — for a 13x13 Gaussian on a 300x300 field
# under Reactant this is 0.32 ms against 49 ms for the direct form.
function _apply_separable_shifted(
    z::AbstractMatrix,
    kx::AbstractVector,
    ky::AbstractVector,
)
    nz1, nz2 = size(z)
    h1, h2 = length(kx) ÷ 2, length(ky) ÷ 2

    zp = _pad_replicate(z, h1, 0)
    tmp = zero(z)
    for k1 in eachindex(kx)
        iszero(kx[k1]) && continue
        tmp = tmp .+ kx[k1] .* zp[k1:(k1+nz1-1), :]
    end

    tp = _pad_replicate(tmp, 0, h2)
    out = zero(z)
    for k2 in eachindex(ky)
        iszero(ky[k2]) && continue
        out = out .+ ky[k2] .* tp[:, k2:(k2+nz2-1)]
    end
    return out
end

function _apply_kernel_shifted(z::AbstractMatrix, K::AbstractMatrix)
    nz1, nz2 = size(z)
    nk1, nk2 = size(K)
    h1, h2 = nk1 ÷ 2, nk2 ÷ 2
    zp = _pad_replicate(z, h1, h2)
    out = zero(z)
    for k2 = 1:nk2, k1 = 1:nk1
        iszero(K[k1, k2]) && continue      # skipped at trace time
        out = out .+ K[k1, k2] .* zp[k1:(k1+nz1-1), k2:(k2+nz2-1)]
    end
    return out
end
