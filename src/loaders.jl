"""
    GHFDataset{Tx, Ty, Tg}

A geothermal heat flux dataset on a regular grid.

# Fields
- `x`: first-axis coordinates (e.g. longitude or projected easting), length `nx`
- `y`: second-axis coordinates (e.g. latitude or projected northing), length `ny`
- `ghf`: GHF values of size `(nx, ny)`, in `units`
- `units`: physical unit string (default `"mW m⁻²"`)
- `source`: dataset name or citation
"""
struct GHFDataset{
    Tx,     # <:AbstractVector,
    Ty,     # <:AbstractVector,
    Tg,     # <:AbstractMatrix
}
    x::Tx
    y::Ty
    ghf::Tg
    units::String
    source::String
end

"""
    load_ghf(filepath; x_name, y_name, ghf_name, units, source) -> GHFDataset

Read a GHF NetCDF file and return a [`GHFDataset`](@ref).

# Keyword arguments
| Keyword    | Default       | Notes                             |
|------------|---------------|-----------------------------------|
| `x_name`   | `"x"`         | Variable name for the x-axis      |
| `y_name`   | `"y"`         | Variable name for the y-axis      |
| `ghf_name` | `"ghf"`       | Variable name for the GHF field   |
| `units`    | `"mW m⁻²"`   | Physical unit label (metadata)    |
| `source`   | `""`          | Dataset citation (metadata)       |

# Common dataset variable names
- Lucazeau (2019): `x_name="lon"`, `y_name="lat"`, `ghf_name="heat_flow"`
- Martos et al. (2017, 2018): `x_name="lon"`, `y_name="lat"`, `ghf_name="Q"`
- Shapiro & Ritzwoller (2004): `x_name="lon"`, `y_name="lat"`, `ghf_name="ghf"`
"""
function load_ghf(
    filepath::AbstractString;
    x_name::AbstractString = "x",
    y_name::AbstractString = "y",
    ghf_name::AbstractString = "ghf",
    units::AbstractString = "mW m⁻²",
    source::AbstractString = "",
    T1 = Float16,
    T2 = Float32,
)
    NCDatasets.Dataset(filepath, "r") do ds
        x   = T2.(ds[x_name][:])
        y   = T2.(ds[y_name][:])
        ghf = T1.(ds[ghf_name][:, :])
        return GHFDataset(x, y, ghf, units, source)
    end
end

"""
    interpolate_ghf(dataset, x_target, y_target) -> Array

Bilinearly interpolate `dataset.ghf` onto (`x_target`, `y_target`).
Returns `NaN` for target points outside the dataset extent.

`x_target` and `y_target` must use the same coordinate system as `dataset.x`
and `dataset.y`. They may be arrays of any shape; the output matches that shape.
"""
function interpolate_ghf(
    dataset::GHFDataset,
    x_target::AbstractArray,
    y_target::AbstractArray,
)
    x, y, ghf = dataset.x, dataset.y, dataset.ghf

    # Interpolations.jl requires strictly increasing knot vectors.
    if !issorted(x)
        p = sortperm(x)
        x, ghf = x[p], ghf[p, :]
    end
    if !issorted(y)
        p = sortperm(y)
        y, ghf = y[p], ghf[:, p]
    end

    itp = linear_interpolation((x, y), ghf, extrapolation_bc=NaN)
    return itp.(x_target, y_target)
end
