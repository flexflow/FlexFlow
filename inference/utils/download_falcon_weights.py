# from transformer import RWForCausalLM
# from configuration_RW import RWConfig
from transformers import AutoModel
import torch
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-7b", trust_remote_code=True)
# model = AutoModel.from_pretrained("tiiuae/falcon-7b", trust_remote_code=True)


# model = RWForCausalLM.from_pretrained("tiiuae/falcon-7b")
# print(model.config)


for name, params in model.named_parameters():
    name = (
        name.replace("h.", "layers_")
        .replace(".", "_").replace("word_embeddings", "tok_embeddings")
        .replace("self_attn", "attention").replace("transformer_", "").replace("self_attention_dense", "attention_wo"))
    # name = (
    #     name.replace("h.", "layers_")
    #     .replace(".", "_").replace("word_embeddings", "tok_embeddings")
    #     .replace("self_attn", "attention").replace("transformer_", ""))

    print(name)
    print(params.shape)
    
    #split q, k, v
    if "self_attention_query_key_value" in name:
        name_q = name.replace("self_attention_query_key_value", "attention_wq")
        name_k = name.replace("self_attention_query_key_value", "attention_wk")
        name_v = name.replace("self_attention_query_key_value", "attention_wv")
        q, k, v = torch.split(params, [4544, 64, 64], 0)
        print(q.shape)
        print(k.shape)
        print(v.shape)
        q.detach().cpu().numpy().tofile('/home/ubuntu/FlexFlow/inference/weights/falcon_7B_weights_new/' + name_q)
        k.detach().cpu().numpy().tofile('/home/ubuntu/FlexFlow/inference/weights/falcon_7B_weights_new/' + name_k)
        v.detach().cpu().numpy().tofile('/home/ubuntu/FlexFlow/inference/weights/falcon_7B_weights_new/' + name_v)
    
    else:
       params.detach().cpu().numpy().tofile('/home/ubuntu/FlexFlow/inference/weights/falcon_7B_weights_new/' + name)