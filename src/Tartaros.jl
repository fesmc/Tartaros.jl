module Tartaros

using DocStringExtensions: TYPEDEF, TYPEDFIELDS, TYPEDSIGNATURES
using Interpolations
using KernelAbstractions
using NCDatasets: NCDatasets

include("filters.jl")
include("corrections.jl")
include("loaders.jl")
include("thermodynamics.jl")

export AbstractSmoothingKernel, ConstantSmoothingKernel, GaussianSmoothingKernel
export smooth

export AbstractGHFCorrection, NoGHFCorrection, TopographicGHFCorrection
export correct, correct!

export GHFDataset, load_ghf, interpolate_ghf

export AbstractGHF, PrescribedGHF, SteadyStateGHF
export ThermalRelaxationGHF, InfiniteColumnGHF, FiniteColumnGHF, DiffusionGHF
export ghf!
export to_enthalpy, from_enthalpy

end
