from dataclasses import replace

from transformers import DeepseekV2Config, PretrainedConfig
from vllm.config.speculative import SpeculativeConfig

_orig_post_init = SpeculativeConfig.__post_init__
_orig_hf_config_override = SpeculativeConfig.hf_config_override


# Transformers 5.14 inherited a hidden_size % num_heads check from Llama in
# DeepseekV2Config. K3 MLA has independent projection/head dimensions (e.g.
# hidden_size=7168, num_heads=96), so that MHA constraint does not apply.
# strict stores unbound validators; patch that entry, not all config validation.
if hasattr(DeepseekV2Config, "__class_validators__"):
    _orig_validate_architecture = DeepseekV2Config.validate_architecture

    def _validate_dspark_architecture(config):
        if config.model_type != "k3_dspark":
            _orig_validate_architecture(config)

    DeepseekV2Config.__class_validators__ = [
        _validate_dspark_architecture if validator is _orig_validate_architecture else validator
        for validator in DeepseekV2Config.__class_validators__
    ]


def _normalize_legacy_qwen3_dspark_config(hf_config: PretrainedConfig) -> PretrainedConfig:
    hf_config = _orig_hf_config_override(hf_config)
    architectures = hf_config.architectures or ()
    if hf_config.model_type == "qwen3" and "DSparkDraftModel" in architectures:
        dflash_config = hf_config.dflash_config
        hf_config.update(
            {
                "architectures": ["Qwen3DSparkModel"],
                "mask_token_id": dflash_config["mask_token_id"],
                "target_layer_ids": dflash_config["target_layer_ids"],
            }
        )
    return hf_config


def _normalize_deepseek_v4_dspark_draft(draft_model_config) -> None:
    """Restore the DSpark draft architecture after VL config conversion.

    DeepSeek-V4-Vision uses the same checkpoint for the target and DSpark
    drafter.  vLLM first rewrites that checkpoint to ``DSparkDraftModel``, but
    rebuilding ``model_arch_config`` with multimodal detection can restore the
    top-level ``*ForConditionalGeneration`` architecture.  The drafter would
    then instantiate a second full VL target and register duplicate attention
    layer names.  Update both config representations without re-running the
    multimodal architecture conversion.
    """
    hf_config = getattr(draft_model_config, "hf_config", None)
    text_config = getattr(hf_config, "text_config", None)
    draft_hf_config = text_config if text_config is not None else hf_config
    root_model_type = getattr(hf_config, "model_type", None)
    text_model_type = getattr(draft_hf_config, "model_type", None)
    is_v41 = root_model_type == "deepseek_v4.1" or text_model_type == "deepseek_v4.1_text"
    if (
        hf_config is None
        or root_model_type not in ("deepseek_v4", "deepseek_v4.1")
        or getattr(draft_hf_config, "dspark_target_layer_ids", None) is None
    ):
        return

    architecture = "DeepseekV41DSparkDraftModel" if is_v41 else "DSparkDraftModel"
    if is_v41:
        # The Aurora target and draft experts intentionally have different
        # widths.  SpeculativeConfig owns a private config copy, so adapting
        # these fields cannot alter the target model.
        draft_hf_config.update(
            {
                "n_routed_experts": draft_hf_config.dspark_n_routed_experts,
                "num_experts_per_tok": draft_hf_config.dspark_n_activated_experts,
                "n_mtp_layers": getattr(draft_hf_config, "num_nextn_predict_layers", 3),
            }
        )
    normalized_model_type = "deepseek_v4.1" if is_v41 else root_model_type
    hf_config.update(
        {
            "architectures": [architecture],
            "model_type": normalized_model_type,
        }
    )
    arch_updates = dict(
        architectures=[architecture],
        model_type=normalized_model_type,
        is_mm_prefix_lm=False,
    )
    if is_v41:
        arch_updates.update(
            num_experts=draft_hf_config.n_routed_experts,
            num_experts_per_token=draft_hf_config.num_experts_per_tok,
        )
    draft_model_config.model_arch_config = replace(
        draft_model_config.model_arch_config,
        **arch_updates,
    )
    architectures = draft_model_config.model_arch_config.architectures
    model_info, architecture = draft_model_config.registry.inspect_model_cls(
        architectures,
        draft_model_config,
    )
    draft_model_config._model_info = model_info
    draft_model_config._architecture = architecture


def _dspark_post_init(self):
    _orig_post_init(self)
    if self.use_dspark():
        draft_model_config = getattr(self, "draft_model_config", None)
        draft_hf_config = getattr(draft_model_config, "hf_config", None)
        _normalize_deepseek_v4_dspark_draft(draft_model_config)
        # deepseek v4 dspark
        if getattr(draft_hf_config, "ptd_token_id", None) is None:  # type: ignore
            draft_hf_config.ptd_token_id = getattr(draft_hf_config, "dspark_noise_token_id", None)  # type: ignore
        # gqa backend dspark
        if getattr(draft_hf_config, "ptd_token_id", None) is None:  # type: ignore
            draft_hf_config.ptd_token_id = getattr(draft_hf_config, "mask_token_id", None)  # type: ignore


SpeculativeConfig.hf_config_override = staticmethod(_normalize_legacy_qwen3_dspark_config)
SpeculativeConfig.__post_init__ = _dspark_post_init
