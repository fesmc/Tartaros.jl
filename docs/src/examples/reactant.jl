#=
# Compiling Tartaros with Reactant

[Reactant.jl](https://github.com/EnzymeAD/Reactant.jl) traces a Julia function once with
symbolic arrays, lowers it to StableHLO/MLIR and hands it to XLA, which fuses and optimises
the **whole traced region** before emitting device code. That last point is the important
one: the unit of optimisation is the function you compile, not the individual kernel, so
the payoff grows the more of a timestep you hand it at once.

Tartaros needs no Reactant-specific code from the user. Loading Reactant activates
`TartarosReactantExt`, which swaps in whole-array formulations of the two routines that are
written as element-wise loops ([`smooth`](@ref) and the [`DiffusionGHF`](@ref) column
solver). Everything else traces as written.

This page runs on the CPU backend so that it builds anywhere; the
[Running on a GPU](#Running-on-a-GPU) section shows the same code on an accelerator.
=#

using Tartaros
using Reactant

Reactant.set_default_backend("cpu")

## Grid and physical parameters
nx, ny = 100, 100
k      = 3.0       # thermal conductivity  [W m⁻¹ K⁻¹]
kappa  = 1e-6      # thermal diffusivity   [m² s⁻¹]
H      = 2_000.0   # lithosphere thickness [m]
nz     = 10        # vertical levels
dt     = 500 * 365.25 * 24 * 3600.0;   # 500-year step [s]

#=
## Moving arrays onto the device

`Reactant.to_rarray` converts an ordinary `Array` into a `ConcreteRArray`, which lives in
XLA's memory space. Models are built exactly as they would be on the CPU — Reactant traces
struct fields automatically, so no Tartaros type needs to know about it.
=#

ghf_ref_h = fill(0.060, nx, ny)    # 60 mW m⁻²
T_cold_h  = fill(258.0, nx, ny)
T_warm_h  = fill(273.0, nx, ny)

ghf_ref = Reactant.to_rarray(ghf_ref_h)
T_cold  = Reactant.to_rarray(T_cold_h)
T_warm  = Reactant.to_rarray(T_warm_h)
ghf     = Reactant.to_rarray(zeros(nx, ny))

model = DiffusionGHF(ghf_ref, T_cold; k = k, kappa = kappa, H_litho = H, nz = nz)

#=
## Compiling and running

`@compile` traces the call and returns an executable. Calling it is what actually runs the
program; the tracing cost is paid once.
=#

step! = @compile ghf!(ghf, T_warm, model, dt)
step!(ghf, T_warm, model, dt)

## Bring the result back to the host
ghf_host = Array(ghf)
extrema(ghf_host)

#=
The result is the same as the ordinary CPU path, to round-off:
=#

model_cpu = DiffusionGHF(ghf_ref_h, T_cold_h; k = k, kappa = kappa, H_litho = H, nz = nz)
ghf_cpu   = zeros(nx, ny)
ghf!(ghf_cpu, T_warm_h, model_cpu, dt)

maximum(abs, ghf_host .- ghf_cpu)

#=
## Compile the timestep, not the function

Compiling a single `ghf!` call already helps, but XLA can only fuse what it can see. In a
coupled model the natural unit is a whole timestep — GHF update plus correction plus
whatever else runs at that frequency. Handing Reactant the entire loop lets it fuse across
all of it and keep intermediates in cache instead of writing them back to memory between
calls.
=#

function nsteps!(ghf, T_ice_base, thermo, dt, n::Int)
    for _ in 1:n
        ghf!(ghf, T_ice_base, thermo, dt)
    end
    return nothing
end

model_loop = DiffusionGHF(ghf_ref, T_cold; k = k, kappa = kappa, H_litho = H, nz = nz)
run10!     = @compile nsteps!(ghf, T_warm, model_loop, dt, 10)
run10!(ghf, T_warm, model_loop, dt, 10)

Array(ghf)[1, 1]

#=
`n` is an ordinary `Int`, so the loop is unrolled while tracing and the ten steps become a
single program. This is the formulation that gives the speed-ups tabulated below.

## Measured performance

Measured on a 300×300 grid (`nz = 10` where applicable), Julia 1.12 with a single thread,
Reactant v0.2.278, against an NVIDIA RTX 2070 Super. All figures are `Float64` and are
minima over the benchmark suite in `benchmark/`. Compilation is excluded — it is a one-off
cost of roughly 25–60 s per compiled function.

!!! warning "Synchronise before timing"
    XLA execution is asynchronous: calling a compiled executable enqueues work and returns.
    Timing a single call without waiting measures the enqueue and not the computation — for
    the diffusion step below, 139 µs instead of 4.2 ms. Use `Reactant.synchronize(result)`,
    exactly as one would use `CUDA.@sync`.

### Lithosphere diffusion, 300×300×10

| Implementation                                   | CPU       | GPU     |
|:-------------------------------------------------|----------:|--------:|
| `TridiagonalDiffusionGHF` (serial reference)      | 57.6 ms   | —       |
| `DiffusionGHF`, KernelAbstractions kernel         | 13.3 ms   | 0.26 ms |
| `DiffusionGHF` under Reactant (slab formulation)  |  4.2 ms   | 0.36 ms |

On the CPU the compiled slab formulation is 3× ahead of the kernel. On the GPU the
KernelAbstractions kernel is *faster* for an isolated call — one hand-written kernel beats
a sequence of fused array operations. Reactant overtakes it only when it can fuse several
steps, as below.

### Smoothing, 300×300

| Kernel                              | Default stencil | Reactant, CPU | Reactant, GPU |
|:------------------------------------|----------------:|--------------:|--------------:|
| `ConstantSmoothingKernel(r = 3000)` |  5.9 ms         | 3.7 ms        | 0.12 ms       |
| `GaussianSmoothingKernel(r = 2000)` |  2.6 ms         | 0.28 ms       | 0.19 ms       |

The traced formulation costs one pass over the field per non-zero kernel weight, so its
cost is set by the kernel size. The 7×7 top-hat has 29 non-zero weights and gains little on
the CPU; the 13×13 Gaussian would need 169 and would be *far* slower — except that a
Gaussian is separable, so it is evaluated as 13 + 13 passes instead of 13 × 13. Expressed
directly it takes 49 ms; separably, 0.28 ms.

### Compiling ten steps as one program

| Form                            | CPU      | GPU      |
|:--------------------------------|---------:|---------:|
| 10 launches of a 1-step program | 51.3 ms  | 4.03 ms  |
| 1 launch of a 10-step program   |  9.2 ms  | 0.69 ms  |

A 5.6× gain on the CPU and 5.9× on the GPU, and on the GPU this is also 4.2× faster than
ten launches of the KernelAbstractions kernel (2.85 ms). The gain survives when every
intermediate result is consumed rather than discarded, so it is genuine fusion rather than
dead-code elimination.

### What Reactant does not accelerate

The broadcast-based models are already memory-bound and there is nothing to fuse, so
compiling them buys nothing and costs a little launch overhead: `SteadyStateGHF` runs at
39 µs on the CPU and 48 µs under Reactant, `FiniteColumnGHF` 43 µs against 51 µs. Reactant
is worth reaching for when there is a *sequence* of operations to fuse, not for a single
broadcast.

## Running on a GPU

The same code runs on an accelerator by changing the backend before any array is created.
Nothing else in the program changes:

```julia
using Tartaros, Reactant, CUDA

Reactant.set_default_backend("gpu")

ghf_ref = Reactant.to_rarray(fill(0.060, nx, ny))
T_cold  = Reactant.to_rarray(fill(258.0, nx, ny))
T_warm  = Reactant.to_rarray(fill(273.0, nx, ny))
ghf     = Reactant.to_rarray(zeros(nx, ny))

model = DiffusionGHF(ghf_ref, T_cold; k = k, kappa = kappa, H_litho = H, nz = nz)
run10! = @compile nsteps!(ghf, T_warm, model, dt, 10)
run10!(ghf, T_warm, model, dt, 10)
```

For comparison, the CUDA.jl path needs no Reactant at all and remains the right choice for
one-off kernel launches:

```julia
using Tartaros, CUDA

## NOTE: `CuArray`, not `cu` — `cu` silently converts Float64 to Float32.
model = DiffusionGHF(CuArray(fill(0.060, nx, ny)), CuArray(fill(258.0, nx, ny));
                     k = k, kappa = kappa, H_litho = H, nz = nz)
ghf   = CUDA.zeros(Float64, nx, ny)
ghf!(ghf, CuArray(fill(273.0, nx, ny)), model, dt)
```

Tartaros' KernelAbstractions kernel is *faster* than the Reactant slab formulation when
each call is launched separately on a GPU (0.26 ms against 0.36 ms) — the slab form only
wins once XLA can fuse a sequence of steps, where it is 4.2× ahead. Reach for Reactant when
compiling a whole timestep, and for CUDA.jl when calling `ghf!` in isolation.

## Things to watch out for

Two Reactant behaviours can produce wrong results without any error, and both are worth
knowing before trusting a compiled run.

**`dt` is baked into the executable.** A plain `Float64` argument becomes a compile-time
constant. An executable compiled with one `dt` and then called with another silently keeps
using the original value:

```julia
step! = @compile ghf!(ghf, T_warm, model, dt)
step!(ghf, T_warm, model, 10dt)   # still advances by dt, with no warning
```

Recompile when `dt` changes, or pass `Reactant.ConcreteRNumber(dt)` so the value becomes a
traced argument rather than a constant.

**Shapes and element types are static.** `(nx, ny, nz)` are fixed at compile time, so a new
grid size means a new compilation. Tartaros types additionally require `dt` to share the
model's floating-point type, which prevents a `Float64` `dt` from silently promoting a
`Float32` model.

Stateful models such as [`InfiniteColumnGHF`](@ref) and [`FiniteColumnGHF`](@ref) keep
their elapsed-time clock in a one-element array allocated next to the model fields, exactly
so that it survives tracing. A clock kept in an ordinary host `Vector` would be
constant-folded into the compiled program and would never advance.
=#
