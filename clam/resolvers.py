from omegaconf import OmegaConf


def resolve_context_len(data_type: str, context_len: int, k_step_preds: int = 1) -> int:
    if data_type == "lapo":
        # context_len, o_t, o_t+1
        return context_len + 2
    elif data_type == "n_step":
        # context_len, o_t, o_t+1, ..., o_t+k
        return context_len + 1 + k_step_preds

    return 1


def resolve_lam_name(
    context_len: int,
    encoder_name: str,
    vq_hp_name: str,
    apply_quantization: bool = False,
    code_dim: int = 64,
    latent_action_dim: int = 8,
    separate_categorical_la: bool = False,
) -> str:
    lam_hp_name = f"cl-{context_len}_e-{encoder_name}"
    if apply_quantization:
        lam_hp_name += f"_vq-{vq_hp_name}"
    elif separate_categorical_la:
        lam_hp_name += f"_lad-{latent_action_dim}_vq-{vq_hp_name}"
    else:
        lam_hp_name += f"_lad-{latent_action_dim}"
    return lam_hp_name


OmegaConf.register_new_resolver("multiply", lambda a, b: a * b)
OmegaConf.register_new_resolver("concat", lambda l: ",".join(l[:2]))
OmegaConf.register_new_resolver("resolve_context_len", resolve_context_len)
OmegaConf.register_new_resolver("resolve_lam_name", resolve_lam_name)
