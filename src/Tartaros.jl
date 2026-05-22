module Tartaros

using DocStringExtensions: TYPEDSIGNATURES
using Interpolations
using NCDatasets: NCDatasets

include("filters.jl")
include("corrections.jl")
include("loaders.jl")

export AbstractSmoothingKernel, ConstantSmoothingKernel, GaussianSmoothingKernel
export smooth

export AbstractGHFCorrection, NoGHFCorrection, TopographicGHFCorrection
export correct, correct!

export GHFDataset, load_ghf, interpolate_ghf

end
