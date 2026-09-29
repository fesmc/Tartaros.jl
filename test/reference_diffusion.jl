# Serial reference implementation of the lithosphere diffusion column.
#
# This mirrors the physics of `DiffusionGHF` using `LinearAlgebra.Tridiagonal` on a plain
# CPU loop. It exists only so the KernelAbstractions kernel in `src/thermodynamics.jl` can
# be checked against an independent, obviously-correct solve. It is deliberately not part
# of the package: it is far slower (57.6 ms vs 13.3 ms on a 300x300x10 CPU grid) and would
# otherwise be a public type whose only purpose is to test another one.

using LinearAlgebra: Tridiagonal

struct ReferenceDiffusionGHF{M,T,Z,C}
    ghf_ref::M
    k::T
    kappa::T
    H_litho::T
    H_prod::T
    zeta::Z
    T_col::C
end

function ReferenceDiffusionGHF(
    ghf_ref,
    T_ice_base;
    k,
    kappa,
    H_litho,
    H_prod = 0.0,
    nz = 5,
)
    zeta = collect(LinRange(0.0, 1.0, nz))
    T_col = Tartaros._steady_T_col(ghf_ref, T_ice_base, k, H_litho, H_prod, zeta)
    return ReferenceDiffusionGHF(ghf_ref, k, kappa, H_litho, H_prod, zeta, T_col)
end

# Advance one column by dt (implicit Euler).
# Bottom BC: Neumann — prescribed heat flux Q_deep (ghost-node method).
# Top BC:    Dirichlet — prescribed temperature T_ice.
function _reference_solve_column!(col, Q_deep, T_ice, k, kappa, H_litho, H_prod, dt)
    nz = length(col)
    dz = H_litho / (nz - 1)
    r = kappa * dt / dz^2
    S = H_prod * kappa * dt / k

    n = nz - 1
    dl = fill(-r, n - 1)
    d = fill(1 + 2r, n)
    du = fill(-r, n - 1)
    rhs = col[1:n] .+ S

    du[1] = -2r
    rhs[1] += 2r * dz * Q_deep / k
    rhs[n] += r * T_ice

    col[1:n] .= Tridiagonal(dl, d, du) \ rhs
    col[nz] = T_ice
    return nothing
end

function reference_ghf!(ghf, T_ice_base, thermo::ReferenceDiffusionGHF, dt)
    nz = size(thermo.T_col, 3)
    dz = thermo.H_litho / (nz - 1)
    for I in CartesianIndices(ghf)
        Q_deep = thermo.ghf_ref isa AbstractArray ? thermo.ghf_ref[I] : thermo.ghf_ref
        col = @view thermo.T_col[I, :]
        _reference_solve_column!(
            col,
            Q_deep,
            T_ice_base[I],
            thermo.k,
            thermo.kappa,
            thermo.H_litho,
            thermo.H_prod,
            dt,
        )
        ghf[I] = -thermo.k * (col[nz] - col[nz-1]) / dz
    end
    return nothing
end
