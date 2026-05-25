#=
# Topographic Correction of Greenland GHF

[colgan_topographic_2021](@citet) propose a simple correction method to account for the effect of topography on geothermal heat flux (GHF) estimates. Tartaros.jl provides a convenient implementation of this method, allowing users to apply the correction to their GHF datasets:
=#

using Tartaros
using CairoMakie
using LazyGrids
using NCDatasets

T1 = Float16
T2 = Float32
martos_path = "/home/jan/Documents/projects/esm-datasets/data/ghf/src/greenland/martos-2018/martos2018.nc"
ghf_data = load_ghf(
    martos_path;
    x_name   = "x",
    y_name   = "y",
    ghf_name = "ghf",
    source   = "Martos et al. (2018)",
)

## Load BedMachine v6 topography (polar-stereographic grid)
bm_path = "/home/jan/Documents/projects/esm-datasets/data/topography/src/BedMachineGreenland-v6.nc"
x_bm, y_bm, bed = NCDatasets.Dataset(bm_path, "r") do ds
    T2.(ds["x"][:]),
    T2.(ds["y"][:]),
    T1.(ds["bed"][:, :])
end

## Smooth bed elevation over a characteristic radius
dx = x_bm[2] - x_bm[1]
dy = y_bm[2] - y_bm[1]

x_corr = x_bm[1] - ghf_data.x[1]
x_bm .-= x_corr
y_corr = y_bm[1] - ghf_data.y[end]
y_bm .-= y_corr


kernel = ConstantSmoothingKernel(r = 5_000.0)
bed_mean = smooth(bed, dx, dy, kernel)

## Interpolate Martos GHF onto the BedMachine grid via its 2-D lat/lon arrays
X_bm, Y_bm = ndgrid(x_bm, y_bm)
ghf_on_bm = interpolate_ghf(ghf_data, X_bm, Y_bm)

## Apply the topographic correction
correction = TopographicGHFCorrection()
ghf_corrected = correct(ghf_on_bm, bed, bed_mean, correction)

## Visualise uncorrected vs corrected GHF
fig = Figure(size = (1200, 500))
ax1 = Axis(fig[1, 1]; title = "Martos (2018) – raw",                        aspect = DataAspect())
ax2 = Axis(fig[1, 2]; title = "Martos (2018) – regridded",                        aspect = DataAspect())
ax3 = Axis(fig[1, 3]; title = "Martos (2018) – topographically corrected",  aspect = DataAspect())
ax4 = Axis(fig[1, 4]; title = "BedMachine v6 – bed elevation", aspect = DataAspect())

# clims = extrema(filter(isfinite, ghf_corrected))
hm1 = heatmap!(ax1, ghf_data.x, ghf_data.y, ghf_data.ghf) #; colorrange = clims)
hm2 = heatmap!(ax2, x_bm, y_bm, ghf_on_bm) # ;    colorrange = clims)
hm3 = heatmap!(ax3, x_bm, y_bm, ghf_corrected) #; colorrange = clims)
hm4 = heatmap!(ax4, x_bm, y_bm, bed) #; colorrange = clims)
Colorbar(fig[2, 1], hm1; label = "GHF (mW m⁻²)", vertical = false)
Colorbar(fig[2, 2], hm2; label = "GHF (mW m⁻²)", vertical = false)
Colorbar(fig[2, 3], hm3; label = "GHF (mW m⁻²)", vertical = false)
Colorbar(fig[2, 4], hm4; label = "GHF (mW m⁻²)", vertical = false)
fig