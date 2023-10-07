from transformers import AutoModelForCausalLM
import os
import torch

torch.set_default_tensor_type(torch.FloatTensor)
# torch.set_default_tensor_type(torch.cuda.FloatTensor)
model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-7b", trust_remote_code=True)
dst_folder = "/home/ubuntu/FlexFlow/inference/weights/tiiuae/falcon-7b/full-precision"
for name, params in model.named_parameters():
            name = (
                name.replace(".", "_")
                .replace("transformer_h_", "layers_")
                .replace("transformer_", "")
                .replace("self_attention_dense", "attention_wo")
            )
            # Split Q,K,V attention weights
            if "self_attention_query_key_value" in name:
                name_q = name.replace("self_attention_query_key_value", "attention_wq")
                name_k = name.replace("self_attention_query_key_value", "attention_wk")
                name_v = name.replace("self_attention_query_key_value", "attention_wv")
                q, k, v = torch.split(
                    params,
                    [
                        model.config.hidden_size,
                        model.config.hidden_size // model.config.n_head,
                        model.config.hidden_size // model.config.n_head,
                    ],
                    0,
                )
                q.detach().cpu().numpy().tofile(os.path.join(dst_folder, name_q))
                k.detach().cpu().numpy().tofile(os.path.join(dst_folder, name_k))
                v.detach().cpu().numpy().tofile(os.path.join(dst_folder, name_v))
            else:
                params.detach().cpu().numpy().tofile(os.path.join(dst_folder, name))
            # LM head weight
model.lm_head.weight.detach().cpu().numpy().tofile(
    os.path.join(dst_folder, "lm_head_weight")
)