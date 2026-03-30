mod primitives;

use burn::backend::wgpu::WgpuDevice;
use fractal_core::{
    error::FractalError,
    registry::{
        cpu_device, initialize_metal_runtime, run_species_with_factory, CpuTrainBackend,
        MetalTrainBackend, SpeciesDefinition, SpeciesId, SpeciesRunContext,
    },
};

pub use primitives::{
    b1_fractal_gated::B1FractalGated, b2_stable_hierarchical::B2StableHierarchical,
    b3_fractal_hierarchical::B3FractalHierarchical, b4_universal::B4Universal,
    p1_contractive::P1Contractive, p2_mandelbrot::P2Mandelbrot, p3_hierarchical::P3Hierarchical,
};

macro_rules! define_flat_species_runner {
    ($cpu_fn:ident, $metal_fn:ident, $species:ident, $rule:ident) => {
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
    };
}

macro_rules! define_hierarchical_species_runner {
    ($cpu_fn:ident, $metal_fn:ident, $species:ident, $rule:ident) => {
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
    };
}

define_flat_species_runner!(run_p1_cpu, run_p1_metal, P1Contractive, P1Contractive);
define_flat_species_runner!(run_p2_cpu, run_p2_metal, P2Mandelbrot, P2Mandelbrot);
define_hierarchical_species_runner!(run_p3_cpu, run_p3_metal, P3Hierarchical, P3Hierarchical);
define_flat_species_runner!(run_b1_cpu, run_b1_metal, B1FractalGated, B1FractalGated);
define_hierarchical_species_runner!(
    run_b2_cpu,
    run_b2_metal,
    B2StableHierarchical,
    B2StableHierarchical
);
define_hierarchical_species_runner!(
    run_b3_cpu,
    run_b3_metal,
    B3FractalHierarchical,
    B3FractalHierarchical
);
define_hierarchical_species_runner!(run_b4_cpu, run_b4_metal, B4Universal, B4Universal);

pub const SPECIES_REGISTRY: [SpeciesDefinition; 7] = [
    SpeciesDefinition::new(SpeciesId::P1Contractive, run_p1_cpu, run_p1_metal),
    SpeciesDefinition::new(SpeciesId::P2Mandelbrot, run_p2_cpu, run_p2_metal),
    SpeciesDefinition::new(SpeciesId::P3Hierarchical, run_p3_cpu, run_p3_metal),
    SpeciesDefinition::new(SpeciesId::B1FractalGated, run_b1_cpu, run_b1_metal),
    SpeciesDefinition::new(SpeciesId::B2StableHierarchical, run_b2_cpu, run_b2_metal),
    SpeciesDefinition::new(SpeciesId::B3FractalHierarchical, run_b3_cpu, run_b3_metal),
    SpeciesDefinition::new(SpeciesId::B4Universal, run_b4_cpu, run_b4_metal),
];

pub fn species_registry() -> &'static [SpeciesDefinition] {
    &SPECIES_REGISTRY
}

#[cfg(test)]
mod tests;
