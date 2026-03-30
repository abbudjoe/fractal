mod primitives;

#[cfg(feature = "cuda")]
use burn::backend::candle::CandleDevice;
use burn::backend::wgpu::WgpuDevice;
use fractal_core::{
    error::FractalError,
    registry::{
        cpu_device, initialize_metal_runtime, run_species_with_factory, CpuTrainBackend,
        MetalTrainBackend, SpeciesDefinition, SpeciesId, SpeciesRunContext,
    },
};

pub use primitives::{
    b2_stable_hierarchical::B2StableHierarchical, generalized_mobius::GeneralizedMobius, ifs::Ifs,
    logistic_chaotic_map::LogisticChaoticMap, p1_contractive::P1Contractive,
    p3_hierarchical::P3Hierarchical,
};

macro_rules! define_flat_species_runner {
    ($cpu_fn:ident, $metal_fn:ident, $cuda_fn:ident, $species:ident, $rule:ident) => {
        fn $cpu_fn(
            context: SpeciesRunContext,
        ) -> Result<fractal_core::SpeciesRawMetrics, FractalError> {
            run_species_with_factory::<CpuTrainBackend, _, _>(
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
            run_species_with_factory::<CpuTrainBackend, _, _>(
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
            run_species_with_factory::<CpuTrainBackend, _, _>(
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
            run_species_with_factory::<CpuTrainBackend, _, _>(
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
    ($id:expr, $cpu_fn:ident, $metal_fn:ident, $cuda_fn:ident) => {{
        #[cfg(feature = "cuda")]
        {
            SpeciesDefinition::new($id, $cpu_fn, $metal_fn, $cuda_fn)
        }
        #[cfg(not(feature = "cuda"))]
        {
            SpeciesDefinition::new($id, $cpu_fn, $metal_fn)
        }
    }};
}

pub const SPECIES_REGISTRY: [SpeciesDefinition; 6] = [
    species_definition!(
        SpeciesId::P1Contractive,
        run_p1_cpu,
        run_p1_metal,
        run_p1_cuda
    ),
    species_definition!(
        SpeciesId::P3Hierarchical,
        run_p3_cpu,
        run_p3_metal,
        run_p3_cuda
    ),
    species_definition!(
        SpeciesId::B2StableHierarchical,
        run_b2_cpu,
        run_b2_metal,
        run_b2_cuda
    ),
    species_definition!(SpeciesId::Ifs, run_ifs_cpu, run_ifs_metal, run_ifs_cuda),
    species_definition!(
        SpeciesId::GeneralizedMobius,
        run_mobius_cpu,
        run_mobius_metal,
        run_mobius_cuda
    ),
    species_definition!(
        SpeciesId::LogisticChaoticMap,
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
