mod primitives;

#[cfg(feature = "cuda")]
use burn::backend::candle::CandleDevice;
use burn::backend::wgpu::WgpuDevice;
use fractal_core::{
    error::FractalError,
    registry::{
        cpu_device, initialize_metal_runtime, run_species_with_factory,
        run_species_with_factory_candle, MetalTrainBackend, PrimitiveVariantName,
        SpeciesDefinition, SpeciesId, SpeciesRunContext,
    },
};

pub use primitives::{
    b1_fractal_gated::B1FractalGated, b2_stable_hierarchical::B2StableHierarchical,
    b3_fractal_hierarchical::B3FractalHierarchical, b4_universal::B4Universal,
    generalized_mobius::GeneralizedMobius, ifs::Ifs, logistic_chaotic_map::LogisticChaoticMap,
    p1_contractive::P1Contractive, p1_fractal_hybrid::P1FractalHybrid, p2_mandelbrot::P2Mandelbrot,
    p3_hierarchical::P3Hierarchical,
};

macro_rules! define_flat_species_runner {
    ($cpu_fn:ident, $metal_fn:ident, $cuda_fn:ident, $species:ident, $rule:ident) => {
        fn $cpu_fn(
            context: SpeciesRunContext,
        ) -> Result<fractal_core::SpeciesRawMetrics, FractalError> {
            run_species_with_factory_candle::<_, _>(
                SpeciesId::$species,
                context,
                cpu_device(),
                |config, device| $rule::new(config.dim, device),
            )
        }

        fn $metal_fn(
            context: SpeciesRunContext,
            device: WgpuDevice,
        ) -> Result<fractal_core::SpeciesRawMetrics, FractalError> {
            initialize_metal_runtime(&device);
            run_species_with_factory::<MetalTrainBackend, _, _>(
                SpeciesId::$species,
                context,
                device,
                |config, device| $rule::new(config.dim, device),
            )
        }

        #[cfg(feature = "cuda")]
        fn $cuda_fn(
            context: SpeciesRunContext,
            device: CandleDevice,
        ) -> Result<fractal_core::SpeciesRawMetrics, FractalError> {
            run_species_with_factory_candle::<_, _>(
                SpeciesId::$species,
                context,
                device,
                |config, device| $rule::new(config.dim, device),
            )
        }
    };
}

macro_rules! define_hierarchical_species_runner {
    ($cpu_fn:ident, $metal_fn:ident, $cuda_fn:ident, $species:ident, $rule:ident) => {
        fn $cpu_fn(
            context: SpeciesRunContext,
        ) -> Result<fractal_core::SpeciesRawMetrics, FractalError> {
            run_species_with_factory_candle::<_, _>(
                SpeciesId::$species,
                context,
                cpu_device(),
                |config, device| $rule::new(config.dim, config.levels, device),
            )
        }

        fn $metal_fn(
            context: SpeciesRunContext,
            device: WgpuDevice,
        ) -> Result<fractal_core::SpeciesRawMetrics, FractalError> {
            initialize_metal_runtime(&device);
            run_species_with_factory::<MetalTrainBackend, _, _>(
                SpeciesId::$species,
                context,
                device,
                |config, device| $rule::new(config.dim, config.levels, device),
            )
        }

        #[cfg(feature = "cuda")]
        fn $cuda_fn(
            context: SpeciesRunContext,
            device: CandleDevice,
        ) -> Result<fractal_core::SpeciesRawMetrics, FractalError> {
            run_species_with_factory_candle::<_, _>(
                SpeciesId::$species,
                context,
                device,
                |config, device| $rule::new(config.dim, config.levels, device),
            )
        }
    };
}

define_flat_species_runner!(
    run_p1_cpu,
    run_p1_metal,
    run_p1_cuda,
    P1Contractive,
    P1Contractive
);
define_hierarchical_species_runner!(
    run_p3_cpu,
    run_p3_metal,
    run_p3_cuda,
    P3Hierarchical,
    P3Hierarchical
);
define_hierarchical_species_runner!(
    run_b2_cpu,
    run_b2_metal,
    run_b2_cuda,
    B2StableHierarchical,
    B2StableHierarchical
);
define_flat_species_runner!(
    run_b1_cpu,
    run_b1_metal,
    run_b1_cuda,
    B1FractalGated,
    B1FractalGated
);
define_flat_species_runner!(
    run_p1_hybrid_cpu,
    run_p1_hybrid_metal,
    run_p1_hybrid_cuda,
    P1FractalHybrid,
    P1FractalHybrid
);
define_flat_species_runner!(
    run_p2_cpu,
    run_p2_metal,
    run_p2_cuda,
    P2Mandelbrot,
    P2Mandelbrot
);
define_hierarchical_species_runner!(
    run_b3_cpu,
    run_b3_metal,
    run_b3_cuda,
    B3FractalHierarchical,
    B3FractalHierarchical
);
define_hierarchical_species_runner!(
    run_b4_cpu,
    run_b4_metal,
    run_b4_cuda,
    B4Universal,
    B4Universal
);
define_flat_species_runner!(run_ifs_cpu, run_ifs_metal, run_ifs_cuda, Ifs, Ifs);
define_flat_species_runner!(
    run_mobius_cpu,
    run_mobius_metal,
    run_mobius_cuda,
    GeneralizedMobius,
    GeneralizedMobius
);
define_flat_species_runner!(
    run_logistic_cpu,
    run_logistic_metal,
    run_logistic_cuda,
    LogisticChaoticMap,
    LogisticChaoticMap
);

macro_rules! species_definition {
    ($id:expr, $variant_name:expr, $cpu_fn:ident, $metal_fn:ident, $cuda_fn:ident) => {{
        #[cfg(feature = "cuda")]
        {
            SpeciesDefinition::new(
                $id,
                PrimitiveVariantName::new_unchecked($variant_name),
                $cpu_fn,
                $metal_fn,
                $cuda_fn,
            )
        }
        #[cfg(not(feature = "cuda"))]
        {
            SpeciesDefinition::new(
                $id,
                PrimitiveVariantName::new_unchecked($variant_name),
                $cpu_fn,
                $metal_fn,
            )
        }
    }};
}

pub const SPECIES_REGISTRY: [SpeciesDefinition; 11] = [
    species_definition!(
        SpeciesId::P1Contractive,
        "p1_contractive_v1",
        run_p1_cpu,
        run_p1_metal,
        run_p1_cuda
    ),
    species_definition!(
        SpeciesId::P3Hierarchical,
        "p3_hierarchical_v1",
        run_p3_cpu,
        run_p3_metal,
        run_p3_cuda
    ),
    species_definition!(
        SpeciesId::B2StableHierarchical,
        "b2_stable_hierarchical_v1",
        run_b2_cpu,
        run_b2_metal,
        run_b2_cuda
    ),
    species_definition!(
        SpeciesId::B1FractalGated,
        "b1_fractal_gated_dyn-residual-norm_v1",
        run_b1_cpu,
        run_b1_metal,
        run_b1_cuda
    ),
    species_definition!(
        SpeciesId::P1FractalHybrid,
        "p1_fractal_hybrid_v1",
        run_p1_hybrid_cpu,
        run_p1_hybrid_metal,
        run_p1_hybrid_cuda
    ),
    species_definition!(
        SpeciesId::P2Mandelbrot,
        "p2_mandelbrot_dyn-gate-norm_v1",
        run_p2_cpu,
        run_p2_metal,
        run_p2_cuda
    ),
    species_definition!(
        SpeciesId::B3FractalHierarchical,
        "b3_fractal_hierarchical_dyn-radius-depth_v1",
        run_b3_cpu,
        run_b3_metal,
        run_b3_cuda
    ),
    species_definition!(
        SpeciesId::B4Universal,
        "b4_universal_dyn-residual-norm_v1",
        run_b4_cpu,
        run_b4_metal,
        run_b4_cuda
    ),
    species_definition!(
        SpeciesId::Ifs,
        "ifs_dyn-radius-depth_v1",
        run_ifs_cpu,
        run_ifs_metal,
        run_ifs_cuda
    ),
    species_definition!(
        SpeciesId::GeneralizedMobius,
        "generalized_mobius_dyn-jitter-norm_v1",
        run_mobius_cpu,
        run_mobius_metal,
        run_mobius_cuda
    ),
    species_definition!(
        SpeciesId::LogisticChaoticMap,
        "logistic_chaotic_map_v1",
        run_logistic_cpu,
        run_logistic_metal,
        run_logistic_cuda
    ),
];

pub fn species_registry() -> &'static [SpeciesDefinition] {
    &SPECIES_REGISTRY
}

#[cfg(test)]
mod tests;
