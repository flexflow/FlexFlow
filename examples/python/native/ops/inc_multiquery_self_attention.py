import flexflow.core
import numpy as np
from flexflow.core import *


def test_inc_multiquery_self_attention(
        ffconfig,
        input_arr: np.ndarray,
        embed_dim: int,
        num_q_heads: int,
        num_kv_heads: int,
        kdim: int = 0,
        vdim: int = 0,
        dropout: float = 0.0,
        bias: bool = True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        data_type: DataType = DataType.DT_NONE,
        kernel_initializer=None,
        apply_rotary_embedding: bool = False,
        scaling_query: bool = False,
        scaling_factor: float = 1.0,
        qk_prod_scaling: bool = True,
        position_bias: bool = False,
        name=None,
):
    ffmodel = FFModel(ffconfig)

    input_tensor = ffmodel.create_tensor(input_arr.shape, data_type)

    inc_multiquery_self_attention_output = ffmodel.inc_multiquery_self_attention(
        input_tensor,
        embed_dim,
        num_q_heads,
        num_kv_heads,
        kdim=kdim,
        vdim=vdim,
        dropout=dropout,
        bias=bias,
        add_bias_kv=add_bias_kv,
        add_zero_attn=add_zero_attn,
        data_type=data_type,
        kernel_initializer=kernel_initializer,
        apply_rotary_embedding=apply_rotary_embedding,
        scaling_query=scaling_query,
        scaling_factor=scaling_factor,
        qk_prod_scaling=qk_prod_scaling,
        position_bias=position_bias,
        name="inc_multiquery_self_attention_layer",
    )

    ffoptimizer = SGDOptimizer(ffmodel, 0.001)
    ffmodel.optimizer = ffoptimizer
    ffmodel.compile(
        loss_type=LossType.LOSS_SPARSE_CATEGORICAL_CROSSENTROPY,
        metrics=[MetricsType.METRICS_ACCURACY, MetricsType.METRICS_SPARSE_CATEGORICAL_CROSSENTROPY]
    )

    dataloader_input = ffmodel.create_data_loader(input_tensor, input_arr)

    ffmodel.init_layers()

    dataloader_input.reset()
    dataloader_input.next_batch(ffmodel)

    ffmodel.forward()

    inc_multiquery_self_attention_output.inline_map(ffmodel, ffconfig)
    output_result = inc_multiquery_self_attention_output.get_array(ffmodel, ffconfig)

    return output_result


if __name__ == '__main__':
    init_flexflow_runtime()
    ffconfig = FFConfig()

    input_data = np.random.randn(ffconfig.batch_size, 10, 20).astype(np.float32)
    embed_dim_value = 64
    num_q_heads_value = 4
    num_kv_heads_value = 4

    output_result = test_inc_multiquery_self_attention(
        ffconfig,
        input_data,
        embed_dim=embed_dim_value,
        num_q_heads=num_q_heads_value,
        num_kv_heads=num_kv_heads_value,
        kdim=0,  # Example value for kdim
        vdim=0,  # Example value for vdim
        dropout=0.1,  # Example value for dropout
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        data_type=DataType.DT_FLOAT,
        kernel_initializer=None,  # Example value for kernel_initializer
        apply_rotary_embedding=False,
        scaling_query=False,
        scaling_factor=1.0,
        qk_prod_scaling=True,
        position_bias=False,
    )

    print("Input Array:")
    print(input_data)
    print("\nOutput Array after applying inc_multiquery_self_attention:")
    print(output_result)
