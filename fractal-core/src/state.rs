use burn::tensor::{backend::Backend, Bool, Tensor};

use crate::error::FractalError;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StateLayout {
    Flat,
    Complex,
    Hierarchical { levels: usize },
    HierarchicalComplex { levels: usize },
}

impl StateLayout {
    pub fn readout_width(&self, hidden_dim: usize) -> usize {
        match self {
            Self::Flat | Self::Hierarchical { .. } => hidden_dim,
            Self::Complex | Self::HierarchicalComplex { .. } => hidden_dim * 2,
        }
    }

    pub fn levels(&self) -> Option<usize> {
        match self {
            Self::Hierarchical { levels } | Self::HierarchicalComplex { levels } => Some(*levels),
            _ => None,
        }
    }
}

#[derive(Clone, Debug)]
pub enum FractalState<B: Backend> {
    Flat(Tensor<B, 2>),
    Complex(Tensor<B, 2>),
    Hierarchical(Tensor<B, 3>),
    HierarchicalComplex(Tensor<B, 3>),
}

impl<B: Backend> FractalState<B> {
    pub fn zeros(
        layout: StateLayout,
        batch_size: usize,
        hidden_dim: usize,
        device: &B::Device,
    ) -> Result<Self, FractalError> {
        let state = match layout {
            StateLayout::Flat => {
                Self::Flat(Tensor::<B, 2>::zeros([batch_size, hidden_dim], device))
            }
            StateLayout::Complex => {
                Self::Complex(Tensor::<B, 2>::zeros([batch_size, hidden_dim * 2], device))
            }
            StateLayout::Hierarchical { levels } => {
                if levels < 2 {
                    return Err(FractalError::InvalidConfig(
                        "hierarchical layouts require at least 2 levels".into(),
                    ));
                }
                Self::Hierarchical(Tensor::<B, 3>::zeros(
                    [batch_size, levels, hidden_dim],
                    device,
                ))
            }
            StateLayout::HierarchicalComplex { levels } => {
                if levels < 2 {
                    return Err(FractalError::InvalidConfig(
                        "hierarchical layouts require at least 2 levels".into(),
                    ));
                }
                Self::HierarchicalComplex(Tensor::<B, 3>::zeros(
                    [batch_size, levels, hidden_dim * 2],
                    device,
                ))
            }
        };

        Ok(state)
    }

    pub fn batch_size(&self) -> usize {
        match self {
            Self::Flat(tensor) | Self::Complex(tensor) => tensor.dims()[0],
            Self::Hierarchical(tensor) | Self::HierarchicalComplex(tensor) => tensor.dims()[0],
        }
    }

    pub fn layout(&self) -> StateLayout {
        match self {
            Self::Flat(_) => StateLayout::Flat,
            Self::Complex(_) => StateLayout::Complex,
            Self::Hierarchical(tensor) => StateLayout::Hierarchical {
                levels: tensor.dims()[1],
            },
            Self::HierarchicalComplex(tensor) => StateLayout::HierarchicalComplex {
                levels: tensor.dims()[1],
            },
        }
    }

    pub fn readout(&self) -> Tensor<B, 2> {
        match self {
            Self::Flat(tensor) | Self::Complex(tensor) => tensor.clone(),
            Self::Hierarchical(tensor) | Self::HierarchicalComplex(tensor) => {
                let [batch, levels, width] = tensor.dims();
                tensor
                    .clone()
                    .narrow(1, levels - 1, 1)
                    .reshape([batch, width])
            }
        }
    }

    pub fn batch_mask_where(
        self,
        mask: Tensor<B, 1, Bool>,
        value: Self,
    ) -> Result<Self, FractalError> {
        if self.layout() != value.layout() {
            return Err(FractalError::InvalidState(format!(
                "cannot blend states with different layouts: {:?} vs {:?}",
                self.layout(),
                value.layout()
            )));
        }

        let batch_size = self.batch_size();
        let [mask_batch] = mask.dims();
        if mask_batch != batch_size {
            return Err(FractalError::Shape(format!(
                "expected batch mask with {batch_size} entries, got {mask_batch}"
            )));
        }

        match (self, value) {
            (Self::Flat(current), Self::Flat(next)) => {
                Ok(Self::Flat(mask_batch_tensor(current, next, mask)))
            }
            (Self::Complex(current), Self::Complex(next)) => {
                Ok(Self::Complex(mask_batch_tensor(current, next, mask)))
            }
            (Self::Hierarchical(current), Self::Hierarchical(next)) => {
                Ok(Self::Hierarchical(mask_batch_tensor(current, next, mask)))
            }
            (Self::HierarchicalComplex(current), Self::HierarchicalComplex(next)) => Ok(
                Self::HierarchicalComplex(mask_batch_tensor(current, next, mask)),
            ),
            _ => Err(FractalError::InvalidState(
                "state layout changed after validation".into(),
            )),
        }
    }

    pub fn flat(&self) -> Result<Tensor<B, 2>, FractalError> {
        match self {
            Self::Flat(tensor) => Ok(tensor.clone()),
            other => Err(FractalError::InvalidState(format!(
                "expected flat state, got {:?}",
                other.layout()
            ))),
        }
    }

    pub fn complex(&self) -> Result<Tensor<B, 2>, FractalError> {
        match self {
            Self::Complex(tensor) => Ok(tensor.clone()),
            other => Err(FractalError::InvalidState(format!(
                "expected complex state, got {:?}",
                other.layout()
            ))),
        }
    }

    pub fn hierarchical(&self) -> Result<Tensor<B, 3>, FractalError> {
        match self {
            Self::Hierarchical(tensor) => Ok(tensor.clone()),
            other => Err(FractalError::InvalidState(format!(
                "expected hierarchical state, got {:?}",
                other.layout()
            ))),
        }
    }

    pub fn hierarchical_checked(
        &self,
        expected_levels: usize,
        hidden_dim: usize,
    ) -> Result<Tensor<B, 3>, FractalError> {
        let tensor = self.hierarchical()?;
        validate_hierarchical_shape(tensor, "hierarchical", expected_levels, hidden_dim)
    }

    pub fn hierarchical_complex(&self) -> Result<Tensor<B, 3>, FractalError> {
        match self {
            Self::HierarchicalComplex(tensor) => Ok(tensor.clone()),
            other => Err(FractalError::InvalidState(format!(
                "expected hierarchical complex state, got {:?}",
                other.layout()
            ))),
        }
    }

    pub fn hierarchical_complex_checked(
        &self,
        expected_levels: usize,
        hidden_dim: usize,
    ) -> Result<Tensor<B, 3>, FractalError> {
        let tensor = self.hierarchical_complex()?;
        validate_hierarchical_shape(
            tensor,
            "hierarchical complex",
            expected_levels,
            hidden_dim * 2,
        )
    }
}

fn validate_hierarchical_shape<B: Backend>(
    tensor: Tensor<B, 3>,
    layout_name: &'static str,
    expected_levels: usize,
    expected_width: usize,
) -> Result<Tensor<B, 3>, FractalError> {
    let [batch, levels, width] = tensor.dims();
    if levels != expected_levels || width != expected_width {
        return Err(FractalError::Shape(format!(
            "expected {layout_name} state [{batch}, {expected_levels}, {expected_width}], got [{batch}, {levels}, {width}]"
        )));
    }

    Ok(tensor)
}

fn mask_batch_tensor<B: Backend, const D: usize>(
    current: Tensor<B, D>,
    next: Tensor<B, D>,
    mask: Tensor<B, 1, Bool>,
) -> Tensor<B, D> {
    let dims = current.dims();
    let mut broadcast = vec![1; D];
    for (index, dim) in dims.iter().copied().enumerate().skip(1) {
        broadcast[index] = dim;
    }
    let expanded_mask = mask
        .int()
        .reshape(mask_shape::<D>(dims[0]))
        .repeat(&broadcast)
        .greater_elem(0);
    current.mask_where(expanded_mask, next)
}

fn mask_shape<const D: usize>(batch_size: usize) -> [usize; D] {
    let mut shape = [1; D];
    shape[0] = batch_size;
    shape
}
