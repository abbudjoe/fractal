use burn::tensor::{backend::Backend, Bool, Tensor, TensorData};

pub(crate) fn local_causal_mask<B: Backend>(
    batch_size: usize,
    seq_len: usize,
    local_window: usize,
    device: &B::Device,
) -> Tensor<B, 3, Bool> {
    let mut data = Vec::with_capacity(batch_size * seq_len * seq_len);
    for _ in 0..batch_size {
        for query in 0..seq_len {
            let earliest_visible = query.saturating_sub(local_window.saturating_sub(1));
            for key in 0..seq_len {
                data.push(key < earliest_visible || key > query);
            }
        }
    }
    Tensor::from_data(
        TensorData::new(data, [batch_size, seq_len, seq_len]),
        device,
    )
}
