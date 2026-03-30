mod primitives;

#[cfg(feature = "cuda")]
use burn::backend::candle::CandleDevice;
use burn::backend::wgpu::WgpuDevice;
use fractal_core::{
    error::FractalError,
    registry::{
        cpu_device, initialize_metal_runtime, run_species_with_factory, CpuTrainBackend,
        MetalTrainBackend, MlxDevice, MlxTrainBackend, SpeciesDefinition, SpeciesId,
        SpeciesRunContext,
    },
};

pub use primitives::{
    b1_fractal_gated::B1FractalGated, b2_stable_hierarchical::B2StableHierarchical,
    b3_fractal_hierarchical::B3FractalHierarchical, b4_universal::B4Universal,
    p1_contractive::P1Contractive, p2_mandelbrot::P2Mandelbrot, p3_hierarchical::P3Hierarchical,
};

macro_rules! define_flat_species_runner {
    ($cpu_fn:ident, $metal_fn:ident, $mlx_fn:ident, $cuda_fn:ident, $species:ident, $rule:ident) => {
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

        fn $mlx_fn(
            context: SpeciesRunContext,
            device: MlxDevice,
        ) -> Result<fractal_core::SpeciesRawMetrics, FractalError> {
            run_species_with_factory::<MlxTrainBackend, _, _>(
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
    ($cpu_fn:ident, $metal_fn:ident, $mlx_fn:ident, $cuda_fn:ident, $species:ident, $rule:ident) => {
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

        fn $mlx_fn(
            context: SpeciesRunContext,
            device: MlxDevice,
        ) -> Result<fractal_core::SpeciesRawMetrics, FractalError> {
            run_species_with_factory::<MlxTrainBackend, _, _>(
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
    run_p1_mlx,
    run_p1_cuda,
    P1Contractive,
    P1Contractive
);
define_flat_species_runner!(
    run_p2_cpu,
    run_p2_metal,
    run_p2_mlx,
    run_p2_cuda,
    P2Mandelbrot,
    P2Mandelbrot
);
define_hierarchical_species_runner!(
    run_p3_cpu,
    run_p3_metal,
    run_p3_mlx,
    run_p3_cuda,
    P3Hierarchical,
    P3Hierarchical
);
define_flat_species_runner!(
    run_b1_cpu,
    run_b1_metal,
    run_b1_mlx,
    run_b1_cuda,
    B1FractalGated,
    B1FractalGated
);
define_hierarchical_species_runner!(
    run_b2_cpu,
    run_b2_metal,
    run_b2_mlx,
    run_b2_cuda,
    B2StableHierarchical,
    B2StableHierarchical
);
define_hierarchical_species_runner!(
    run_b3_cpu,
    run_b3_metal,
    run_b3_mlx,
    run_b3_cuda,
    B3FractalHierarchical,
    B3FractalHierarchical
);
define_hierarchical_species_runner!(
    run_b4_cpu,
    run_b4_metal,
    run_b4_mlx,
    run_b4_cuda,
    B4Universal,
    B4Universal
);

macro_rules! species_definition {
    ($id:expr, $cpu_fn:ident, $metal_fn:ident, $mlx_fn:ident, $cuda_fn:ident) => {{
        #[cfg(feature = "cuda")]
        {
            SpeciesDefinition::new($id, $cpu_fn, $metal_fn, $mlx_fn, $cuda_fn)
        }
        #[cfg(not(feature = "cuda"))]
        {
            SpeciesDefinition::new($id, $cpu_fn, $metal_fn, $mlx_fn)
        }
    }};
}

pub const SPECIES_REGISTRY: [SpeciesDefinition; 7] = [
    species_definition!(
        SpeciesId::P1Contractive,
        run_p1_cpu,
        run_p1_metal,
        run_p1_mlx,
        run_p1_cuda
    ),
    species_definition!(
        SpeciesId::P2Mandelbrot,
        run_p2_cpu,
        run_p2_metal,
        run_p2_mlx,
        run_p2_cuda
    ),
    species_definition!(
        SpeciesId::P3Hierarchical,
        run_p3_cpu,
        run_p3_metal,
        run_p3_mlx,
        run_p3_cuda
    ),
    species_definition!(
        SpeciesId::B1FractalGated,
        run_b1_cpu,
        run_b1_metal,
        run_b1_mlx,
        run_b1_cuda
    ),
    species_definition!(
        SpeciesId::B2StableHierarchical,
        run_b2_cpu,
        run_b2_metal,
        run_b2_mlx,
        run_b2_cuda
    ),
    species_definition!(
        SpeciesId::B3FractalHierarchical,
        run_b3_cpu,
        run_b3_metal,
        run_b3_mlx,
        run_b3_cuda
    ),
    species_definition!(
        SpeciesId::B4Universal,
        run_b4_cpu,
        run_b4_metal,
        run_b4_mlx,
        run_b4_cuda
    ),
];

pub fn species_registry() -> &'static [SpeciesDefinition] {
    &SPECIES_REGISTRY
}

#[cfg(test)]
mod tests;
