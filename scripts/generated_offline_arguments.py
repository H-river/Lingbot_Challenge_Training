# Extracted verbatim from pinned LingBot argument dataclasses; no training entrypoint imported.
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Literal
from lingbotvla.utils.arguments import TrainingArguments, DataArguments, ModelArguments, EvalArguments

@dataclass
class MyTrainingArguments(TrainingArguments):
    freeze_vit: bool = field(
        default=False,
        metadata={"help": "Whether or not to freeze the vit parameters."},
    )
    vit_lr: float = field(
        default=1e-6,
        metadata={"help": "Maximum learning rate for vit parameters."},
    )
    freeze_vision_encoder: bool = field(
        default=False,
        metadata={"help": "Whether or not to freeze the vision encoder in PI0 model."},
    )
    train_expert_only: bool = field(
        default=False,
        metadata={"help": "Train action expert only or not."},
    )
    train_state_proj: bool = field(
        default=True,
        metadata={"help": "Train state proj only or not."},
    )
    tokenizer_max_length: int = field(
        default=48,
        metadata={"help": "Maximum length of the tokenizer."},
    )
    action_dim: int = field(
        default=7,
        metadata={"help": "Action dimension."},
    )
    max_action_dim: int = field(
        default=32,
        metadata={"help": "Action dimension after padding."},
    )
    max_state_dim: int = field(
        default=32,
        metadata={"help": "State dimension after padding."},
    )
    chunk_size: int = field(
        default=50,
        metadata={"help": "Chunk size of action."},
    )
    vlm_causal: bool = field(
        default=False,
        metadata={"help": "Whether to use causal atten for img anb lang tokens in vlm."},
    )
    loss_type: str = field(
        default='fm',
        metadata={"help": "Which loss to use."},
    )
    align_params: Optional[Dict[str, Any]] = field(
        default_factory=dict,
        metadata={"help": "The config of vaco"},
    )
    attention_implementation: str = field(
        default="flex",
        metadata={"help": "Attention implementation: flex, flex_cached, or eager."},
    )
    my_tokenizer_max_length: int = field(
        default=72,
        metadata={"help": ""},
    )
    decayed_max_grad_norm: float = field(
        default=1.0,
        metadata={"help": "Maximum norm for the decayed gradients."},
    )
    stable_train_steps: int = field(
        default=100000,
        metadata={"help": "Training steps for stable training, after this step, the decayed_max_grad_norm will be applied."},
    )
    resume_dataloader_state: bool = field(
        default=True,
        metadata={"help": "Whether to resume dataloader."},
    )
    use_moe: bool = field(
        default=False,
        metadata={"help": "Whether to use MoE."},
    )
    token_moe_layers: Optional[List[int]] = field(
        default=None,
        metadata={"help": "Layer indices for token-level routing MoE, e.g. [1,2,...,34]."},
    )
    token_num_experts: int = field(
        default=32,
        metadata={"help": "Number of experts per token-level MoE layer."},
    )
    token_top_k: int = field(
        default=1,
        metadata={"help": "Top-k for token-level routing."},
    )
    token_moe_intermediate_size: int = field(
        default=256,
        metadata={"help": "Intermediate size for token-level MoE expert FFN."},
    )
    token_shared_intermediate_size: int = field(
        default=256,
        metadata={"help": "Intermediate size for token-level shared expert FFN."},
    )
    bias_update_speed: float = field(
        default=0.001,
        metadata={"help": "Bias update speed for loss-free MoE load balancing."},
    )
    bias_centering: bool = field(
        default=False,
        metadata={"help": "If True, center the loss-free correction bias each step (subtract per-layer mean) to prevent cumulative drift. Routing-invariant; pure numerical hygiene."},
    )
    bias_update_interval: int = field(
        default=1,
        metadata={"help": "Apply the loss-free bias update once every N optimizer steps, accumulating tokens_per_expert in between. >1 averages load over more tokens so sign(load-mean) is reliable when the global batch is small (per-step load too noisy -> bias random-walks). Also makes the load monitor (maxvio/min_load/has_dead) windowed/less noisy."},
    )
    sequence_wise_loss_coeff: float = field(
        default=0.0,
        metadata={"help": "Coefficient for sequence-wise MoE balance loss (DeepSeek-V3 style, complementary to loss-free bias). 0 = disabled."},
    )
    sequence_wise_mode: str = field(
        default="per_sequence",
        metadata={"help": "Granularity of the sequence-wise balance loss: 'per_sequence' (balance experts within each sample's token sequence, DeepSeek-V3 intent) or 'global' (treat the whole batch as one sequence). Ablation knob."},
    )
    router_z_loss_coeff: float = field(
        default=0.0,
        metadata={"help": "Coefficient for router z-loss computed on raw router logits. 0 = disabled."},
    )
    moe_monitor_interval: int = field(
        default=50,
        metadata={"help": "Step interval for writing per-layer MoE monitor scalars (moe_maxvio/minvio/minload/entropy/topksigmoid) and expert-selection histograms to TensorBoard. moe_summary/* is always written every step. Set small (e.g. 1/10) for close debugging."},
    )
    router_activation: str = field(
        default="softmax",
        metadata={"help": "Router activation function: 'softmax' or 'sigmoid'. Default 'softmax' for backward compat."},
    )
    routed_scaling_factor: float = field(
        default=1.0,
        metadata={"help": "Scaling factor applied to routing weights after norm (DeepSeek-V3 style). 1.0 = disabled."},
    )
    use_shared_expert_gate: bool = field(
        default=True,
        metadata={"help": "Whether to use sigmoid gate on shared expert output. True = current behavior, False = direct add (DS-V3 style)."},
    )
    # Action expert architecture params
    expert_hidden_size: int = field(
        default=768,
        metadata={"help": "Hidden size for action expert."},
    )
    expert_intermediate_size: int = field(
        default=2752,
        metadata={"help": "FFN intermediate size for action expert."},
    )
    action_fp32: bool = field(
        default=False,
        metadata={"help": "Whether to use fp32 action and state."},
    )
    precompute_grid_thw: bool = field(
        default=False,
        metadata={"help": "Whether to precompute and cache grid_thw-derived tensors (rotary_pos_emb, window_index, etc.) for fixed-resolution training."},
    )
    use_moe_expert_lr: bool = field(
        default=False,
        metadata={"help": "Whether to apply scaled LR to MoE routed experts."},
    )
    split_fused_experts_from_decoder_fsdp: bool = field(
        default=False,
        metadata={"help": "Whether to exclude Qwen2FusedExperts params from Qwen2DecoderLayer FSDP2 units without wrapping the experts in FSDP2."},
    )
    vlm_fsdp: bool = field(
        default=False,
        metadata={"help": "Whether to apply FSDP2 for VLM."},
    )

@dataclass
class MyDataArguments(DataArguments):
    source_name: str = field(
        default=None,
        metadata={"help": "Source name of dataset."},
    )
    robot_config_root: str = field(
        default=None,
        metadata={"help": "Path to get all robot configs."},
    )
    joints: Optional[List[str]] = field(
        default=None,
        metadata={"help": "The order of joints and their dim"},
    )
    cameras:Optional[List[str]] = field(
        default=None,
        metadata={"help": "The order of used images"},
    )
    norm_type: Optional[List[str]] = field(default=None, metadata={"help": "Normalization type."})
    img_size: int = field(
        default=256,
        metadata={"help": "Size of the image."},
    )
    norm_stats_file: str = field(
        default=None,
        metadata={"help": "Path to the normalization stats file."},
    )
    prompt_type: Literal["global", "subtask"] = field(
        default="global",
        metadata={"help": "Type of the prompt."},
    )
    use_future_image: bool = field(
        default=False,
        metadata={"help": "Whether to use future image."},
    )

@dataclass
class Arguments:

    model: "ModelArguments" = field(default_factory=ModelArguments)
    data: "MyDataArguments" = field(default_factory=MyDataArguments)
    train: "MyTrainingArguments" = field(default_factory=MyTrainingArguments)
    eval: "EvalArguments" = field(default_factory=EvalArguments)
