using Tartaros
using Test
using Statistics: mean

@testset "Tartaros.jl" begin

    @testset "filters" begin
        z = Float64[1 2 3; 4 5 6; 7 8 9]
        dx = 1_000.0

        z_c = smooth(z, dx, ConstantSmoothingKernel(r = 1_500.0))
        @test size(z_c) == size(z)
        @test z_c[2, 2] ≈ mean(z)  # center cell averages the whole 3×3 disk

        z_g = smooth(z, dx, GaussianSmoothingKernel(r = 1_000.0))
        @test size(z_g) == size(z)
        @test z_g[2, 2] > z_g[1, 1]   # center should outweigh corner
    end

    @testset "corrections" begin
        ghf    = [50.0 60.0; 70.0 80.0]
        z_bed  = [100.0 200.0; 300.0 400.0]
        z_mean = smooth(z_bed, 1_000.0, ConstantSmoothingKernel(r = 500.0))
        corr   = TopographicGHFCorrection(α = 950.0, r = 5_000.0)

        # NoGHFCorrection is a no-op
        @test correct(42.0, 0.0, 0.0, NoGHFCorrection()) == 42.0

        # Scalar correction is consistent with the formula
        g, z, zm = 50.0, 100.0, 200.0
        @test correct(g, z, zm, corr) ≈ g * (1 + (zm - z) / corr.α)

        # Matrix dispatch agrees with scalar
        ghf_c = correct(ghf, z_bed, z_mean, corr)
        @test size(ghf_c) == size(ghf)
        @test ghf_c[1, 1] ≈ correct(ghf[1, 1], z_bed[1, 1], z_mean[1, 1], corr)

        # In-place matches allocating
        ghf2 = copy(ghf)
        correct!(ghf2, z_bed, z_mean, corr)
        @test ghf2 ≈ ghf_c
    end

    @testset "thermodynamics" begin
        nx, ny = 3, 4
        ghf = zeros(nx, ny)

        # PrescribedGHF: output is always ghf_ref regardless of T_ice_base
        ghf_ref = fill(60.0, nx, ny)
        ghf!(ghf, zeros(nx, ny), PrescribedGHF(ghf_ref))
        @test ghf ≈ ghf_ref

        # SteadyStateGHF: Fourier's law on a linear temperature profile
        k, H, Tm = 3.0, 100_000.0, 1600.0
        T_ice = fill(270.0, nx, ny)
        ghf!(ghf, T_ice, SteadyStateGHF(k = k, H_litho = H, T_mantle = Tm))
        @test ghf ≈ fill(k * (Tm - 270.0) / H, nx, ny)

        # GHF increases when basal ice temperature drops (colder base → steeper gradient)
        T_cold = fill(250.0, nx, ny)
        ghf_cold = zeros(nx, ny)
        ghf!(ghf_cold, T_cold, SteadyStateGHF(k = k, H_litho = H, T_mantle = Tm))
        @test all(ghf_cold .> ghf)

        # GHF recovers prescribed value when T_ice_base equals 0 K (degenerate check)
        ghf!(ghf, zeros(nx, ny), SteadyStateGHF(k = k, H_litho = H, T_mantle = Tm))
        @test ghf ≈ fill(k * Tm / H, nx, ny)

        # Inference constructor: round-trip ghf_ref → T_mantle → ghf should recover ghf_ref
        ghf_ref2 = fill(0.040, nx, ny)   # 40 mW/m²
        T_ice2   = fill(270.0, nx, ny)
        model_inf = SteadyStateGHF(ghf_ref2, T_ice2; k = k, H_litho = H)
        ghf!(ghf, T_ice2, model_inf)
        @test ghf ≈ ghf_ref2
    end

    @testset "InfiniteColumnGHF" begin
        nx, ny  = 2, 2
        k_a, kappa_a, H_a = 3.0, 1e-6, 2000.0
        ghf_ref_a = fill(0.060, nx, ny)
        dT        = fill(15.0,  nx, ny)

        model = InfiniteColumnGHF(ghf_ref_a, dT; k = k_a, kappa = kappa_a, H_litho = H_a)
        @test model.t_elapsed[1] == 0.0

        # One-step formula: G(dt) = G_ref − k·ΔT / √(π·κ·dt)
        ghf = zeros(nx, ny)
        dt  = 1e10
        ghf!(ghf, zeros(nx, ny), model, dt)
        expected = ghf_ref_a .- k_a .* dT ./ sqrt(pi * kappa_a * dt)
        @test ghf ≈ expected

        # Second step accumulates time: G(2·dt)
        ghf!(ghf, zeros(nx, ny), model, dt)
        expected2 = ghf_ref_a .- k_a .* dT ./ sqrt(pi * kappa_a * 2dt)
        @test ghf ≈ expected2

        # Warming (ΔT > 0) reduces GHF; cooling (ΔT < 0) increases it
        @test all(ghf .< ghf_ref_a)
        dT_neg    = fill(-15.0, nx, ny)
        model_neg = InfiniteColumnGHF(ghf_ref_a, dT_neg; k = k_a, kappa = kappa_a, H_litho = H_a)
        ghf_neg   = zeros(nx, ny)
        ghf!(ghf_neg, zeros(nx, ny), model_neg, dt)
        @test all(ghf_neg .> ghf_ref_a)
    end

    @testset "ThermalRelaxationGHF" begin
        nx, ny   = 2, 2
        ghf_tgt  = fill(0.040, nx, ny)
        ghf_init = fill(0.060, nx, ny)
        tau_s    = 1e9

        model = ThermalRelaxationGHF(ghf_tgt, ghf_init; tau = tau_s)
        @test model.ghf_state ≈ ghf_init

        # One-step formula (implicit Euler)
        ghf = zeros(nx, ny)
        dt  = 1e8
        ghf!(ghf, zeros(nx, ny), model, dt)
        r = dt / tau_s
        @test ghf ≈ (ghf_init .+ ghf_tgt .* r) ./ (1 + r)

        # Very large dt drives state to target in one step
        model2 = ThermalRelaxationGHF(ghf_tgt, ghf_init; tau = tau_s)
        ghf!(ghf, zeros(nx, ny), model2, 1e30)
        @test ghf ≈ ghf_tgt  atol = 1e-10

        # Default constructor (no ghf_init) starts at ghf_ref
        model3 = ThermalRelaxationGHF(ghf_tgt; tau = tau_s)
        @test model3.ghf_state ≈ ghf_tgt
    end

    @testset "FiniteColumnGHF" begin
        nx, ny   = 2, 2
        k_f, kappa_f, H_f = 3.0, 1e-6, 2_000.0
        ghf_ref_f = fill(0.060, nx, ny)
        dT_f      = fill(15.0,  nx, ny)

        model = FiniteColumnGHF(ghf_ref_f, dT_f; k = k_f, kappa = kappa_f, H_litho = H_f)
        @test model.t_elapsed[1] == 0.0
        @test model.N == 200

        # One-step: output equals closed-form sum
        ghf = zeros(nx, ny)
        dt  = 1e10
        ghf!(ghf, zeros(nx, ny), model, dt)
        s_expected = sum(n -> exp(-((n + 0.5) * pi / H_f)^2 * kappa_f * dt), 0:199)
        @test ghf ≈ ghf_ref_f .- 2k_f .* dT_f ./ H_f .* s_expected

        # Second step accumulates time
        ghf!(ghf, zeros(nx, ny), model, dt)
        s2 = sum(n -> exp(-((n + 0.5) * pi / H_f)^2 * kappa_f * 2dt), 0:199)
        @test ghf ≈ ghf_ref_f .- 2k_f .* dT_f ./ H_f .* s2

        # Warming (ΔT > 0) reduces GHF; cooling (ΔT < 0) increases it
        @test all(ghf .< ghf_ref_f)
        dT_neg    = fill(-15.0, nx, ny)
        model_neg = FiniteColumnGHF(ghf_ref_f, dT_neg; k = k_f, kappa = kappa_f, H_litho = H_f)
        ghf_neg   = zeros(nx, ny)
        ghf!(ghf_neg, zeros(nx, ny), model_neg, dt)
        @test all(ghf_neg .> ghf_ref_f)

        # At very large t the series vanishes and GHF recovers to ghf_ref
        model_long = FiniteColumnGHF(ghf_ref_f, dT_f; k = k_f, kappa = kappa_f, H_litho = H_f)
        ghf!(ghf, zeros(nx, ny), model_long, 1e20)
        @test ghf ≈ ghf_ref_f  atol = 1e-10
    end

    @testset "DiffusionGHF" begin
        nx, ny   = 3, 4
        k_d      = 3.0          # W m⁻¹ K⁻¹
        kappa_d  = 1e-6         # m² s⁻¹  (typical rock)
        H_d      = 2_000.0      # m
        ghf_deep = fill(0.060, nx, ny)   # 60 mW m⁻²
        T_ice    = fill(270.0,  nx, ny)

        model = DiffusionGHF(ghf_deep, T_ice; k = k_d, kappa = kappa_d, H_litho = H_d)

        # Constructor: top of column equals the Dirichlet BC
        @test model.T_col[:, :, end] ≈ T_ice

        # Constructor: bottom of column matches steady-state formula
        T_base_expected = T_ice .+ ghf_deep .* H_d ./ k_d
        @test model.T_col[:, :, 1] ≈ T_base_expected

        # Steady-state preservation: surface GHF should recover ghf_deep after one step
        ghf = zeros(nx, ny)
        dt  = 3.15e7   # 1 year in seconds
        ghf!(ghf, T_ice, model, dt)
        @test ghf ≈ ghf_deep  atol = 1e-6

        # Enthalpy round-trip for cold material (ω = 0)
        cp, T_pmp, L = 1000.0, 273.15, 334_000.0
        T_test = 250.0
        H_enth = to_enthalpy(T_test, cp, T_pmp, L)
        T_rec, omega_rec = from_enthalpy(H_enth, cp, T_pmp, L)
        @test T_rec ≈ T_test
        @test omega_rec == 0.0

        # Enthalpy round-trip for temperate material (ω > 0)
        omega_test = 0.02
        H_enth2 = to_enthalpy(T_pmp, cp, T_pmp, L; omega = omega_test)
        T_rec2, omega_rec2 = from_enthalpy(H_enth2, cp, T_pmp, L)
        @test T_rec2 ≈ T_pmp
        @test omega_rec2 ≈ omega_test
    end

    @testset "TridiagonalDiffusionGHF" begin
        nx, ny   = 3, 4
        k_d      = 3.0
        kappa_d  = 1e-6
        H_d      = 2_000.0
        ghf_deep = fill(0.060, nx, ny)
        T_ice    = fill(270.0,  nx, ny)

        model = TridiagonalDiffusionGHF(ghf_deep, T_ice; k = k_d, kappa = kappa_d, H_litho = H_d)

        @test model.T_col[:, :, end] ≈ T_ice
        @test model.T_col[:, :, 1]   ≈ T_ice .+ ghf_deep .* H_d ./ k_d

        # Steady-state preservation: surface GHF should recover ghf_deep after one step
        ghf = zeros(nx, ny)
        ghf!(ghf, T_ice, model, 3.15e7)
        @test ghf ≈ ghf_deep  atol = 1e-6

        # Results match DiffusionGHF for a warming step
        T_warm   = fill(273.0, nx, ny)
        model_ka = DiffusionGHF(ghf_deep, T_ice; k = k_d, kappa = kappa_d, H_litho = H_d)
        model_tr = TridiagonalDiffusionGHF(ghf_deep, T_ice; k = k_d, kappa = kappa_d, H_litho = H_d)
        ghf_ka   = zeros(nx, ny)
        ghf_tr   = zeros(nx, ny)
        for _ in 1:20
            ghf!(ghf_ka, T_warm, model_ka, 3.15e7)
            ghf!(ghf_tr, T_warm, model_tr, 3.15e7)
        end
        @test ghf_ka ≈ ghf_tr  atol = 1e-10
    end

end
