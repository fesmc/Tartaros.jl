# Public API

## GHF models

```@docs
AbstractGHF
PrescribedGHF
SteadyStateGHF
ThermalRelaxationGHF
InfiniteColumnGHF
FiniteColumnGHF
DiffusionGHF
ghf!
```

## GHF corrections

```@docs
AbstractGHFCorrection
NoGHFCorrection
TopographicGHFCorrection
correct
correct!
```

## Filters

```@docs
AbstractSmoothingKernel
ConstantSmoothingKernel
GaussianSmoothingKernel
smooth
```

## Loaders

```@docs
GHFDataset
load_ghf
interpolate_ghf
```