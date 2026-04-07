use burn::{
    config::Config,
    module::{
        AutodiffModule, Content, Devices, DisplaySettings, Module, ModuleDisplay,
        ModuleDisplayDefault, ModuleMapper, ModuleVisitor,
    },
    nn::Initializer,
    tensor::backend::{AutodiffBackend, Backend},
};

use crate::{
    diagnostics::OutputProjectionDiagnosticSpec,
    projection::{
        ProjectionLayoutPolicy, StructuredProjection, StructuredProjectionConfig,
        StructuredProjectionRecord,
    },
};

#[derive(Config, Debug)]
pub struct LanguageModelHeadConfig {
    pub readout_width: usize,
    pub vocab_size: usize,
    #[config(default = true)]
    pub bias: bool,
    #[config(default = "Initializer::KaimingUniform{gain:1.0/3.0_f64.sqrt(), fan_out_only:false}")]
    pub initializer: Initializer,
    #[config(default = "ProjectionLayoutPolicy::OutputByInput")]
    pub layout_policy: ProjectionLayoutPolicy,
}

#[derive(Clone, Debug)]
pub struct LanguageModelHead<B: Backend> {
    projection: StructuredProjection<B>,
}

impl LanguageModelHeadConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> LanguageModelHead<B> {
        let projection = StructuredProjectionConfig::new(self.readout_width, self.vocab_size)
            .with_bias(self.bias)
            .with_initializer(self.initializer.clone())
            .with_layout_policy(self.layout_policy)
            .init(device);
        LanguageModelHead { projection }
    }
}

impl<B: Backend> LanguageModelHead<B> {
    pub fn forward<const D: usize>(
        &self,
        readout: burn::tensor::Tensor<B, D>,
    ) -> burn::tensor::Tensor<B, D> {
        self.projection.forward(readout)
    }

    pub fn layout_policy(&self) -> ProjectionLayoutPolicy {
        self.projection.layout_policy()
    }

    pub fn logical_dims(&self) -> [usize; 2] {
        self.projection.logical_dims()
    }

    pub fn diagnostic_spec(
        &self,
        model_name: impl Into<String>,
        projection_name: impl Into<String>,
        input_shape: Vec<usize>,
        output_shape: Vec<usize>,
    ) -> OutputProjectionDiagnosticSpec {
        let [readout_width, vocab_size] = self.logical_dims();
        OutputProjectionDiagnosticSpec::linear_with_layout(
            model_name,
            projection_name,
            input_shape,
            output_shape,
            self.layout_policy()
                .linear_layout_metadata(readout_width, vocab_size),
        )
    }
}

impl<B: Backend> Module<B> for LanguageModelHead<B> {
    type Record = StructuredProjectionRecord<B>;

    fn visit<V: ModuleVisitor<B>>(&self, visitor: &mut V) {
        self.projection.visit(visitor);
    }

    fn map<M: ModuleMapper<B>>(self, mapper: &mut M) -> Self {
        Self {
            projection: Module::map(self.projection, mapper),
        }
    }

    fn load_record(self, record: Self::Record) -> Self {
        Self {
            projection: self.projection.load_record(record),
        }
    }

    fn into_record(self) -> Self::Record {
        self.projection.into_record()
    }

    fn to_device(self, device: &B::Device) -> Self {
        Self {
            projection: self.projection.to_device(device),
        }
    }

    fn fork(self, device: &B::Device) -> Self {
        Self {
            projection: self.projection.fork(device),
        }
    }

    fn collect_devices(&self, devices: Devices<B>) -> Devices<B> {
        self.projection.collect_devices(devices)
    }
}

impl<B: AutodiffBackend> AutodiffModule<B> for LanguageModelHead<B> {
    type InnerModule = LanguageModelHead<B::InnerBackend>;

    fn valid(&self) -> Self::InnerModule {
        LanguageModelHead {
            projection: self.projection.valid(),
        }
    }
}

impl<B: Backend> ModuleDisplayDefault for LanguageModelHead<B> {
    fn content(&self, content: Content) -> Option<Content> {
        let [readout_width, vocab_size] = self.logical_dims();
        content
            .add("readout_width", &readout_width)
            .add("vocab_size", &vocab_size)
            .add("layout_policy", &self.layout_policy())
            .optional()
    }

    fn num_params(&self) -> usize {
        Module::num_params(self)
    }
}

impl<B: Backend> ModuleDisplay for LanguageModelHead<B> {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        DisplaySettings::new()
            .with_new_line_after_attribute(false)
            .optional()
    }
}

#[cfg(test)]
mod tests {
    use burn::{
        backend::Candle,
        nn::{Linear, LinearConfig},
        tensor::Tolerance,
    };

    use super::*;
    use crate::projection::StructuredProjectionRecord;

    type TestBackend = Candle<f32, i64>;

    fn legacy_linear_record<B: Backend>(linear: Linear<B>) -> StructuredProjectionRecord<B> {
        StructuredProjectionRecord {
            weight: linear.weight,
            bias: linear.bias,
            layout_policy: ProjectionLayoutPolicy::InputByOutput,
        }
    }

    #[test]
    fn language_model_head_matches_linear_after_loading_legacy_record() {
        let device = Default::default();
        let linear = LinearConfig::new(4, 6)
            .with_initializer(Initializer::Constant { value: 0.25 })
            .init::<TestBackend>(&device);
        let head = LanguageModelHeadConfig::new(4, 6)
            .with_layout_policy(ProjectionLayoutPolicy::OutputByInput)
            .with_initializer(Initializer::Zeros)
            .init::<TestBackend>(&device)
            .load_record(legacy_linear_record(linear.clone()));
        let input = burn::tensor::Tensor::<TestBackend, 2>::from_data(
            [[0.5, -1.0, 2.0, 1.5], [1.0, 0.25, -0.75, 0.5]],
            &device,
        );

        head.forward(input.clone())
            .into_data()
            .assert_approx_eq::<f32>(&linear.forward(input).into_data(), Tolerance::default());
        assert_eq!(head.logical_dims(), [4, 6]);
        assert_eq!(head.layout_policy(), ProjectionLayoutPolicy::OutputByInput);
    }
}
