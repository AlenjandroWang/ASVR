from dataclasses import dataclass
from transformers import PretrainedConfig
from dataclasses import field


@dataclass
class AttentionBlockConfig:
    embed_dim: int = 2560
    n_head: int = 40
    mlp_bias: bool = True
    attn_bias: bool = True
    attn_pdrop: float = 0.0
    resid_pdrop: float = 0.1


@dataclass
class AttentionStackConfig:
    n_layer: int = 6
    # block: AttentionBlockConfig = AttentionBlockConfig()
    block: AttentionBlockConfig = field(default_factory=AttentionBlockConfig)


class RQTransformerConfig(PretrainedConfig):
    model_type = "rqtransformer_model"
    def __init__(
        self,
        block_size=None,
        input_embed_dim_1=None,
        input_embed_dim_2=None,
        embed_dim=None,
        vocab_size=None,
        head=None,
        architectures=None,
        **kwargs,
    ):
        super().__init__()

        self.block_size = block_size
        self.input_embed_dim_1 = input_embed_dim_1
        self.input_embed_dim_2 = input_embed_dim_2
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.head = head
        self.architectures = architectures  