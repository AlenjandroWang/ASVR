from transformers import PretrainedConfig


class RQVAESIGLIPTransformerConfig(PretrainedConfig):
    model_type = "rqvaesigliptransformer_model"
    def __init__(
        self,
        rqvaesiglip=None,
        rqtransformer=None,
        hidden_size=None,
        architectures=None,
        **kwargs,
    ):
        super().__init__()

        self.rqvaesiglip = rqvaesiglip
        self.rqtransformer = rqtransformer
        self.hidden_size = hidden_size
        self.architectures = architectures