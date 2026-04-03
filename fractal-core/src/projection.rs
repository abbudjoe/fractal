use burn::{
    config::Config,
    module::{
        AutodiffModule, Content, Devices, DisplaySettings, Module, ModuleDisplay,
        ModuleDisplayDefault, ModuleMapper, ModuleVisitor, Param,
    },
    nn::Initializer,
    record::{PrecisionSettings, Record},
    tensor::{
        backend::{AutodiffBackend, Backend},
        Tensor,
    },
};
use serde::{Deserialize, Serialize};

use crate::diagnostics::{
    LinearProjectionLayoutMetadata, RuleProjectionDiagnosticSpec, TensorLayoutMetadata,
    TensorLayoutTransform,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProjectionLayoutPolicy {
    InputByOutput,
    OutputByInput,
}

impl Default for ProjectionLayoutPolicy {
    fn default() -> Self {
        Self::InputByOutput
    }
}

impl core::fmt::Display for ProjectionLayoutPolicy {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let label = match self {
            Self::InputByOutput => "input_by_output",
            Self::OutputByInput => "output_by_input",
        };
        f.write_str(label)
    }
}

impl ProjectionLayoutPolicy {
    pub const fn stored_weight_shape(self, d_input: usize, d_output: usize) -> [usize; 2] {
        match self {
            Self::InputByOutput => [d_input, d_output],
            Self::OutputByInput => [d_output, d_input],
        }
    }

    pub const fn logical_dims_from_stored_shape(self, stored_shape: [usize; 2]) -> [usize; 2] {
        match self {
            Self::InputByOutput => stored_shape,
            Self::OutputByInput => [stored_shape[1], stored_shape[0]],
        }
    }

    pub const fn forward_rhs_transform(self) -> TensorLayoutTransform {
        match self {
            Self::InputByOutput => TensorLayoutTransform::UnsqueezedView,
            Self::OutputByInput => TensorLayoutTransform::TransposedUnsqueezedView,
        }
    }

    pub const fn backward_input_grad_rhs_transform(self) -> TensorLayoutTransform {
        match self {
            Self::InputByOutput => TensorLayoutTransform::TransposedView,
            Self::OutputByInput => TensorLayoutTransform::Identity,
        }
    }

    fn convert_weight_for_storage<B: Backend>(
        self,
        weight: Tensor<B, 2>,
        source_layout: ProjectionLayoutPolicy,
    ) -> Tensor<B, 2> {
        if self == source_layout {
            return weight;
        }

        let device = weight.device();
        let require_grad = weight.is_require_grad();
        let data = weight.transpose().into_data();
        let mut converted = Tensor::<B, 2>::from_data(data, &device);
        if require_grad {
            converted = converted.require_grad();
        }
        converted
    }

    pub fn linear_layout_metadata(
        self,
        d_input: usize,
        d_output: usize,
    ) -> LinearProjectionLayoutMetadata {
        let stored_weight_shape = self.stored_weight_shape(d_input, d_output).to_vec();
        let [logical_input, logical_output] =
            self.logical_dims_from_stored_shape(self.stored_weight_shape(d_input, d_output));
        LinearProjectionLayoutMetadata {
            stored_weight: TensorLayoutMetadata::linear_contract(
                stored_weight_shape,
                TensorLayoutTransform::Identity,
            ),
            forward_rhs: TensorLayoutMetadata::linear_contract(
                vec![1, logical_input, logical_output],
                self.forward_rhs_transform(),
            ),
            backward_input_grad_rhs: TensorLayoutMetadata::linear_contract(
                self.stored_weight_shape(d_input, d_output).to_vec(),
                self.backward_input_grad_rhs_transform(),
            ),
        }
    }
}

#[derive(Config, Debug)]
pub struct StructuredProjectionConfig {
    pub d_input: usize,
    pub d_output: usize,
    #[config(default = true)]
    pub bias: bool,
    #[config(default = "Initializer::KaimingUniform{gain:1.0/3.0_f64.sqrt(), fan_out_only:false}")]
    pub initializer: Initializer,
    #[config(default = "ProjectionLayoutPolicy::InputByOutput")]
    pub layout_policy: ProjectionLayoutPolicy,
}

#[derive(Clone, Debug)]
pub struct StructuredProjection<B: Backend> {
    weight: Param<Tensor<B, 2>>,
    bias: Option<Param<Tensor<B, 1>>>,
    layout_policy: ProjectionLayoutPolicy,
}

#[derive(Clone, Debug)]
pub struct StructuredProjectionRecord<B: Backend> {
    pub weight: Param<Tensor<B, 2>>,
    pub bias: Option<Param<Tensor<B, 1>>>,
    pub layout_policy: ProjectionLayoutPolicy,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StructuredProjectionRecordItem<W, Bi> {
    pub weight: W,
    pub bias: Bi,
    #[serde(default)]
    pub layout_policy: ProjectionLayoutPolicy,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StructuredProjectionLegacyLinearRecordItem<W, Bi> {
    pub weight: W,
    pub bias: Bi,
}

impl StructuredProjectionConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> StructuredProjection<B> {
        let weight = self.initializer.init_with(
            self.layout_policy
                .stored_weight_shape(self.d_input, self.d_output),
            Some(self.d_input),
            Some(self.d_output),
            device,
        );
        let bias = if self.bias {
            Some(self.initializer.init_with(
                [self.d_output],
                Some(self.d_input),
                Some(self.d_output),
                device,
            ))
        } else {
            None
        };

        StructuredProjection {
            weight,
            bias,
            layout_policy: self.layout_policy,
        }
    }
}

impl<B: Backend> StructuredProjection<B> {
    pub fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        let dims = input.dims();
        let [d_input, d_output] = self.logical_dims();
        assert_eq!(
            dims[D - 1],
            d_input,
            "structured projection input width mismatch: expected last dimension {d_input}, got {}",
            dims[D - 1]
        );
        let leading = dims[..D - 1].iter().copied().product::<usize>().max(1);
        let input_2d = input.reshape([leading, d_input]);
        let output_2d = self.forward_rank2(input_2d);
        let mut output_shape = dims;
        output_shape[D - 1] = d_output;
        output_2d.reshape(output_shape)
    }

    pub fn layout_policy(&self) -> ProjectionLayoutPolicy {
        self.layout_policy
    }

    pub fn logical_dims(&self) -> [usize; 2] {
        self.layout_policy
            .logical_dims_from_stored_shape(self.weight.shape().dims())
    }

    pub fn diagnostic_spec(
        &self,
        rule_name: impl Into<String>,
        projection_name: impl Into<String>,
        input_shape: Vec<usize>,
        output_shape: Vec<usize>,
    ) -> RuleProjectionDiagnosticSpec {
        let [d_input, d_output] = self.logical_dims();
        RuleProjectionDiagnosticSpec::linear_with_layout(
            rule_name,
            projection_name,
            input_shape,
            output_shape,
            self.layout_policy.linear_layout_metadata(d_input, d_output),
        )
    }

    fn forward_rank2(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let weight = match self.layout_policy {
            ProjectionLayoutPolicy::InputByOutput => self.weight.val(),
            ProjectionLayoutPolicy::OutputByInput => self.weight.val().transpose(),
        };
        let output = input.matmul(weight);
        match &self.bias {
            Some(bias) => output.add(bias.val().reshape([1, self.logical_dims()[1]])),
            None => output,
        }
    }

    fn load_weight_record(
        &self,
        record_weight: Param<Tensor<B, 2>>,
        source_layout: ProjectionLayoutPolicy,
    ) -> Param<Tensor<B, 2>> {
        let weight = self
            .weight
            .clone()
            .load_record(record_weight)
            .map(|tensor| {
                self.layout_policy
                    .convert_weight_for_storage(tensor, source_layout)
            });

        let expected_shape = self
            .layout_policy
            .stored_weight_shape(self.logical_dims()[0], self.logical_dims()[1]);
        let actual_shape = weight.shape().dims();
        assert_eq!(
            actual_shape, expected_shape,
            "structured projection weight shape mismatch: expected {expected_shape:?} for layout {} but loaded {actual_shape:?}",
            self.layout_policy
        );

        weight
    }

    fn load_bias_record(
        &self,
        record_bias: Option<Param<Tensor<B, 1>>>,
    ) -> Option<Param<Tensor<B, 1>>> {
        match (&self.bias, record_bias) {
            (Some(template), Some(record)) => Some(template.clone().load_record(record)),
            (None, None) => None,
            (Some(_), None) => {
                panic!("structured projection record is missing required bias parameter")
            }
            (None, Some(_)) => {
                panic!("structured projection record supplied an unexpected bias parameter")
            }
        }
    }
}

impl<B: Backend> Module<B> for StructuredProjection<B> {
    type Record = StructuredProjectionRecord<B>;

    fn visit<V: ModuleVisitor<B>>(&self, visitor: &mut V) {
        self.weight.visit(visitor);
        if let Some(bias) = &self.bias {
            bias.visit(visitor);
        }
    }

    fn map<M: ModuleMapper<B>>(self, mapper: &mut M) -> Self {
        Self {
            weight: Module::map(self.weight, mapper),
            bias: self.bias.map(|bias| Module::map(bias, mapper)),
            layout_policy: self.layout_policy,
        }
    }

    fn load_record(self, record: Self::Record) -> Self {
        Self {
            weight: self.load_weight_record(record.weight, record.layout_policy),
            bias: self.load_bias_record(record.bias),
            layout_policy: self.layout_policy,
        }
    }

    fn into_record(self) -> Self::Record {
        StructuredProjectionRecord {
            weight: self.weight,
            bias: self.bias,
            layout_policy: self.layout_policy,
        }
    }

    fn to_device(self, device: &B::Device) -> Self {
        Self {
            weight: self.weight.to_device(device),
            bias: self.bias.map(|bias| bias.to_device(device)),
            layout_policy: self.layout_policy,
        }
    }

    fn fork(self, device: &B::Device) -> Self {
        Self {
            weight: self.weight.fork(device),
            bias: self.bias.map(|bias| bias.fork(device)),
            layout_policy: self.layout_policy,
        }
    }

    fn collect_devices(&self, devices: Devices<B>) -> Devices<B> {
        let devices = self.weight.collect_devices(devices);
        match &self.bias {
            Some(bias) => bias.collect_devices(devices),
            None => devices,
        }
    }
}

impl<B: AutodiffBackend> AutodiffModule<B> for StructuredProjection<B> {
    type InnerModule = StructuredProjection<B::InnerBackend>;

    fn valid(&self) -> Self::InnerModule {
        StructuredProjection {
            weight: self.weight.valid(),
            bias: self.bias.as_ref().map(AutodiffModule::valid),
            layout_policy: self.layout_policy,
        }
    }
}

impl<B: Backend> ModuleDisplayDefault for StructuredProjection<B> {
    fn content(&self, content: Content) -> Option<Content> {
        let [d_input, d_output] = self.logical_dims();
        content
            .add("d_input", &d_input)
            .add("d_output", &d_output)
            .add("bias", &self.bias.is_some())
            .add("layout_policy", &self.layout_policy)
            .optional()
    }

    fn num_params(&self) -> usize {
        Module::num_params(self)
    }
}

impl ModuleDisplayDefault for ProjectionLayoutPolicy {
    fn content(&self, content: Content) -> Option<Content> {
        content.add_formatted(self).optional()
    }
}

impl ModuleDisplay for ProjectionLayoutPolicy {}

impl<B: Backend> ModuleDisplay for StructuredProjection<B> {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        DisplaySettings::new()
            .with_new_line_after_attribute(false)
            .optional()
    }
}

impl<B: Backend> Record<B> for StructuredProjectionRecord<B> {
    type Item<S: PrecisionSettings> = StructuredProjectionRecordItem<
        <Param<Tensor<B, 2>> as Record<B>>::Item<S>,
        <Option<Param<Tensor<B, 1>>> as Record<B>>::Item<S>,
    >;

    fn into_item<S: PrecisionSettings>(self) -> Self::Item<S> {
        StructuredProjectionRecordItem {
            weight: Record::<B>::into_item::<S>(self.weight),
            bias: Record::<B>::into_item::<S>(self.bias),
            layout_policy: self.layout_policy,
        }
    }

    fn from_item<S: PrecisionSettings>(item: Self::Item<S>, device: &B::Device) -> Self {
        StructuredProjectionRecord {
            weight: Record::<B>::from_item::<S>(item.weight, device),
            bias: Record::<B>::from_item::<S>(item.bias, device),
            layout_policy: item.layout_policy,
        }
    }
}

#[cfg(test)]
mod tests {
    use burn::{
        backend::Candle,
        nn::{Linear, LinearConfig},
        record::{FullPrecisionSettings, Record},
        tensor::{TensorData, Tolerance},
    };

    use super::*;

    type TestBackend = Candle<f32, i64>;

    fn legacy_linear_record<B: Backend>(linear: Linear<B>) -> StructuredProjectionRecord<B> {
        StructuredProjectionRecord {
            weight: linear.weight,
            bias: linear.bias,
            layout_policy: ProjectionLayoutPolicy::InputByOutput,
        }
    }

    #[test]
    fn structured_projection_matches_linear_after_loading_legacy_record() {
        let device = Default::default();
        let linear = LinearConfig::new(4, 3)
            .with_initializer(Initializer::Constant { value: 0.25 })
            .init::<TestBackend>(&device);
        let projection = StructuredProjectionConfig::new(4, 3)
            .with_layout_policy(ProjectionLayoutPolicy::OutputByInput)
            .with_initializer(Initializer::Zeros)
            .init::<TestBackend>(&device)
            .load_record(legacy_linear_record(linear.clone()));
        let input = burn::tensor::Tensor::<TestBackend, 2>::from_data(
            [[0.5, -1.0, 2.0, 1.5], [1.0, 0.25, -0.75, 0.5]],
            &device,
        );

        projection
            .forward(input.clone())
            .into_data()
            .assert_approx_eq::<f32>(&linear.forward(input).into_data(), Tolerance::default());
        assert_eq!(projection.weight.shape().dims(), [3, 4]);
    }

    #[test]
    fn structured_projection_native_record_roundtrips() {
        let device = Default::default();
        let projection = StructuredProjectionConfig::new(4, 3)
            .with_layout_policy(ProjectionLayoutPolicy::OutputByInput)
            .with_initializer(Initializer::Constant { value: 0.125 })
            .init::<TestBackend>(&device);
        let input =
            burn::tensor::Tensor::<TestBackend, 2>::from_data([[1.0, 2.0, 3.0, 4.0]], &device);
        let expected = projection.clone().forward(input.clone()).into_data();
        let reloaded = StructuredProjectionConfig::new(4, 3)
            .with_layout_policy(ProjectionLayoutPolicy::OutputByInput)
            .with_initializer(Initializer::Zeros)
            .init::<TestBackend>(&device)
            .load_record(projection.into_record());

        reloaded
            .forward(input)
            .into_data()
            .assert_approx_eq::<f32>(&expected, Tolerance::default());
    }

    #[test]
    fn structured_projection_record_item_deserializes_legacy_linear_shape() {
        let device = Default::default();
        let linear = LinearConfig::new(2, 2)
            .with_initializer(Initializer::Constant { value: 1.0 })
            .init::<TestBackend>(&device);
        let legacy_record =
            StructuredProjectionLegacyLinearRecordItem {
                weight:
                    <Param<burn::tensor::Tensor<TestBackend, 2>> as Record<TestBackend>>::into_item::<
                        FullPrecisionSettings,
                    >(linear.weight),
                bias: <Option<Param<burn::tensor::Tensor<TestBackend, 1>>> as Record<
                    TestBackend,
                >>::into_item::<FullPrecisionSettings>(linear.bias),
            };
        let encoded = serde_json::to_vec(&legacy_record).expect("legacy item should serialize");
        let decoded = serde_json::from_slice::<
            <StructuredProjectionRecord<TestBackend> as Record<TestBackend>>::Item<
                FullPrecisionSettings,
            >,
        >(&encoded)
        .expect("legacy item should deserialize into structured projection record");
        assert_eq!(decoded.layout_policy, ProjectionLayoutPolicy::InputByOutput);
        let weight =
            <Param<burn::tensor::Tensor<TestBackend, 2>> as Record<TestBackend>>::from_item::<
                FullPrecisionSettings,
            >(decoded.weight, &device);
        weight.to_data().assert_approx_eq::<f32>(
            &TensorData::from([[1.0, 1.0], [1.0, 1.0]]),
            Tolerance::default(),
        );
    }
}
