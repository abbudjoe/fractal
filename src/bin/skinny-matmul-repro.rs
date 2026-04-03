use std::env;

use burn::{
    backend::candle::CandleDevice,
    tensor::{backend::AutodiffBackend, Tensor, TensorData},
};
#[cfg(feature = "cuda")]
use fractal_core::{CandleBf16TrainBackend, CandleF32TrainBackend};
use fractal_core::{CpuTrainBackend, ProjectionLayoutPolicy, StructuredProjectionConfig};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ReproSurface {
    RawMatmul,
    StructuredProjection,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BackendKind {
    CpuF32,
    CudaF32,
    CudaBf16,
}

#[derive(Clone, Copy, Debug)]
struct Args {
    surface: ReproSurface,
    backend: BackendKind,
    layout: ProjectionLayoutPolicy,
    batch: usize,
    d_input: usize,
    d_output: usize,
    cuda_device: usize,
}

fn usage() -> &'static str {
    "Usage: cargo run --bin skinny-matmul-repro -- [options]\n\
     \n\
     Options:\n\
       --surface <raw-matmul|structured-projection>   Default: raw-matmul\n\
       --backend <cpu-f32|cuda-f32|cuda-bf16>         Default: cpu-f32\n\
       --layout <input_by_output|output_by_input>     Default: output_by_input\n\
       --batch <N>                                    Default: 2\n\
       --d-input <N>                                  Default: 1024\n\
       --d-output <N>                                 Default: 1024\n\
       --cuda-device <N>                              Default: 0\n\
       --help                                         Show this message\n"
}

fn parse_args() -> Result<Args, String> {
    let mut args = Args {
        surface: ReproSurface::RawMatmul,
        backend: BackendKind::CpuF32,
        layout: ProjectionLayoutPolicy::OutputByInput,
        batch: 2,
        d_input: 1024,
        d_output: 1024,
        cuda_device: 0,
    };

    let mut iter = env::args().skip(1);
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--help" | "-h" => {
                print!("{}", usage());
                std::process::exit(0);
            }
            "--surface" => {
                let value = iter
                    .next()
                    .ok_or_else(|| "--surface requires a value".to_owned())?;
                args.surface = match value.as_str() {
                    "raw-matmul" => ReproSurface::RawMatmul,
                    "structured-projection" => ReproSurface::StructuredProjection,
                    _ => return Err(format!("unknown surface: {value}")),
                };
            }
            "--backend" => {
                let value = iter
                    .next()
                    .ok_or_else(|| "--backend requires a value".to_owned())?;
                args.backend = match value.as_str() {
                    "cpu-f32" => BackendKind::CpuF32,
                    "cuda-f32" => BackendKind::CudaF32,
                    "cuda-bf16" => BackendKind::CudaBf16,
                    _ => return Err(format!("unknown backend: {value}")),
                };
            }
            "--layout" => {
                let value = iter
                    .next()
                    .ok_or_else(|| "--layout requires a value".to_owned())?;
                args.layout = match value.as_str() {
                    "input_by_output" => ProjectionLayoutPolicy::InputByOutput,
                    "output_by_input" => ProjectionLayoutPolicy::OutputByInput,
                    _ => return Err(format!("unknown layout: {value}")),
                };
            }
            "--batch" => {
                args.batch = parse_usize("--batch", iter.next())?;
            }
            "--d-input" => {
                args.d_input = parse_usize("--d-input", iter.next())?;
            }
            "--d-output" => {
                args.d_output = parse_usize("--d-output", iter.next())?;
            }
            "--cuda-device" => {
                args.cuda_device = parse_usize("--cuda-device", iter.next())?;
            }
            other => return Err(format!("unknown argument: {other}")),
        }
    }

    Ok(args)
}

fn parse_usize(flag: &str, value: Option<String>) -> Result<usize, String> {
    let value = value.ok_or_else(|| format!("{flag} requires a value"))?;
    value
        .parse::<usize>()
        .map_err(|error| format!("failed to parse {flag} value {value:?}: {error}"))
}

fn build_tensor_data(rows: usize, cols: usize) -> TensorData {
    let values = (0..rows * cols)
        .map(|index| {
            let centered = (index % 97) as f32 - 48.0;
            centered / 64.0
        })
        .collect::<Vec<_>>();
    TensorData::new(values, [rows, cols])
}

fn print_shape(label: &str, shape: [usize; 2]) {
    println!("{label}=[{}, {}]", shape[0], shape[1]);
}

fn run_raw_matmul<B: AutodiffBackend>(device: &B::Device, args: &Args) {
    let input = Tensor::<B, 2>::from_data(build_tensor_data(args.batch, args.d_input), device)
        .require_grad();
    let stored_shape = args.layout.stored_weight_shape(args.d_input, args.d_output);
    let weight =
        Tensor::<B, 2>::from_data(build_tensor_data(stored_shape[0], stored_shape[1]), device)
            .require_grad();
    let rhs = match args.layout {
        ProjectionLayoutPolicy::InputByOutput => weight.clone(),
        ProjectionLayoutPolicy::OutputByInput => weight.clone().transpose(),
    };

    print_shape("input_shape", [args.batch, args.d_input]);
    print_shape("stored_weight_shape", stored_shape);
    print_shape("logical_rhs_shape", [args.d_input, args.d_output]);

    let output = input.clone().matmul(rhs);
    print_shape("forward_output_shape", output.shape().dims());

    let grads = output.sum().backward();
    let input_grad = input
        .grad(&grads)
        .expect("input grad should exist for raw matmul repro");
    let weight_grad = weight
        .grad(&grads)
        .expect("weight grad should exist for raw matmul repro");

    print_shape("input_grad_shape", input_grad.shape().dims());
    print_shape("stored_weight_grad_shape", weight_grad.shape().dims());
    println!("status=backward_ok");
}

fn run_structured_projection<B: AutodiffBackend>(device: &B::Device, args: &Args) {
    let projection = StructuredProjectionConfig::new(args.d_input, args.d_output)
        .with_layout_policy(args.layout)
        .init::<B>(device);
    let input = Tensor::<B, 2>::from_data(build_tensor_data(args.batch, args.d_input), device);
    let output = projection.forward(input);

    print_shape("structured_input_shape", [args.batch, args.d_input]);
    print_shape(
        "structured_stored_weight_shape",
        args.layout.stored_weight_shape(args.d_input, args.d_output),
    );
    print_shape("structured_output_shape", output.shape().dims());

    let _grads = output.sum().backward();
    println!("status=backward_ok");
}

fn run_with_backend<B: AutodiffBackend>(device: B::Device, args: &Args) {
    println!("surface={:?}", args.surface);
    println!("backend={:?}", args.backend);
    println!("layout={}", args.layout);
    println!("batch={}", args.batch);
    println!("d_input={}", args.d_input);
    println!("d_output={}", args.d_output);

    match args.surface {
        ReproSurface::RawMatmul => run_raw_matmul::<B>(&device, args),
        ReproSurface::StructuredProjection => run_structured_projection::<B>(&device, args),
    }
}

fn main() -> Result<(), String> {
    let args = parse_args()?;

    match args.backend {
        BackendKind::CpuF32 => run_with_backend::<CpuTrainBackend>(CandleDevice::Cpu, &args),
        BackendKind::CudaF32 => {
            #[cfg(feature = "cuda")]
            {
                run_with_backend::<CandleF32TrainBackend>(
                    CandleDevice::cuda(args.cuda_device),
                    &args,
                );
            }
            #[cfg(not(feature = "cuda"))]
            {
                return Err(
                    "cuda backend requested, but the binary was not built with --features cuda"
                        .to_owned(),
                );
            }
        }
        BackendKind::CudaBf16 => {
            #[cfg(feature = "cuda")]
            {
                run_with_backend::<CandleBf16TrainBackend>(
                    CandleDevice::cuda(args.cuda_device),
                    &args,
                );
            }
            #[cfg(not(feature = "cuda"))]
            {
                return Err(
                    "cuda backend requested, but the binary was not built with --features cuda"
                        .to_owned(),
                );
            }
        }
    }

    Ok(())
}
