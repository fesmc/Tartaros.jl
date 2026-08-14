module TartarosReactantExt

# Reactant traces a Julia function once with symbolic arrays, lowers it to StableHLO and
# lets XLA fuse the whole traced region. Two Tartaros routines are written as element-wise
# loops, which cannot be traced; this extension swaps in the equivalent whole-array
# formulations. Both are slower than the defaults on ordinary arrays, which is exactly why
# they live behind this extension rather than replacing the originals.

using Tartaros: Tartaros
using Reactant: Reactant

# Any array Reactant owns: symbolic (inside a trace) or concrete (outside one).
const ReactantMatrix = Union{
    Reactant.AnyTracedRArray{<:Any,2},
    Reactant.UnionAnyConcreteRArray{<:Any,2,<:Any},
}

# ── Smoothing ─────────────────────────────────────────────────────────────────
# `Tartaros._apply_kernel` and `_apply_separable` index single elements, which cannot be
# traced. The `_shifted` variants express the same stencils as one shifted,
# edge-replicated slice per non-zero weight.
#
# The number of slices is what matters: each one is a full pass over the field. A 7x7
# top-hat (29 non-zero weights) costs 3.7 ms against 5.9 ms for the host loop, but a
# 13x13 Gaussian expressed directly (169 weights) costs 49 ms against 2.6 ms on the host.
# Hence the separable path below, which needs 13+13 slices instead of 13x13 and brings the
# same Gaussian down to 0.28 ms. All figures: 300x300 field, CPU backend, Float64.
function Tartaros.smooth(
    z::ReactantMatrix,
    dx::Real,
    dy::Real,
    kernel::Tartaros.AbstractSmoothingKernel,
)
    K = Tartaros._build_kernel(dx, dy, kernel)
    return Tartaros._apply_kernel_shifted(z, K)
end

# A Gaussian is the outer product of two 1D Gaussians, so it needs only two passes. This
# method also resolves what would otherwise be an ambiguity against Tartaros' own
# `smooth(::AbstractMatrix, ::Real, ::Real, ::GaussianSmoothingKernel)`, which is more
# specific in the kernel argument than the method above is in the array argument.
function Tartaros.smooth(
    z::ReactantMatrix,
    dx::Real,
    dy::Real,
    kernel::Tartaros.GaussianSmoothingKernel,
)
    kx, ky = Tartaros._build_kernel_factors(dx, dy, kernel)
    return Tartaros._apply_separable_shifted(z, kx, ky)
end

# ── Lithosphere diffusion ─────────────────────────────────────────────────────
# `ghf!(..., ::DiffusionGHF, dt)` launches a KernelAbstractions kernel that sweeps over z
# within each thread. Reactant can only raise such a kernel when CUDA.jl is loaded (it goes
# through the CUDA compilation path regardless of the target backend), and even then it is
# the slowest option available. `_diffusion_slabs!` runs the same Thomas algorithm over
# whole slabs instead: 4.2 ms against 13.3 ms for the kernel on a 300x300x10 CPU grid.
#
# On a GPU the kernel is faster for an isolated call (0.26 ms vs 0.36 ms); the slab form
# wins once several steps are compiled together, where fusion makes it 4.2x faster than the
# equivalent sequence of kernel launches. That is the regime Reactant is for.
function Tartaros.ghf!(
    ghf::ReactantMatrix,
    T_ice_base,
    thermo::Tartaros.DiffusionGHF{M,T},
    dt::T,
) where {M,T}
    Tartaros._diffusion_slabs!(
        ghf,
        thermo.T_col,
        thermo.ghf_ref,
        T_ice_base,
        thermo.k,
        thermo.kappa,
        thermo.H_litho,
        thermo.H_prod,
        dt,
    )
    return nothing
end

end # module
