from transformers import PreTrainedModel, AutoConfig, AutoModel

from .configuration_rqvaesigliptransformer import RQVAESIGLIPTransformerConfig
from .rqvaesiglip import RQVAESiglipModel
from .rqtransformer import RQTransformer


# class RQVAESIGLIPTransformer(PreTrainedModel):
#     config_class = RQVAESIGLIPTransformerConfig
#     def __init__(self, config: RQVAESIGLIPTransformerConfig):
#         super().__init__(config)

#         rqvaesiglip_config = RQVAESiglipModel.config_class.from_dict(config.rqvaesiglip)
#         self.rqvaesiglip = RQVAESiglipModel._from_config(rqvaesiglip_config)

#         # rqtransformer_config = RQTransformer.config_class.from_dict(config.rqtransformer)
#         # self.rqtransformer = RQTransformer._from_config(rqtransformer_config)

class RQVAESIGLIPTransformer(PreTrainedModel):
    config_class = RQVAESIGLIPTransformerConfig
    def __init__(self, config: RQVAESIGLIPTransformerConfig):
        super().__init__(config)

        rqvaesiglip_config = RQVAESiglipModel.config_class.from_dict(config.rqvaesiglip)
        rqtransformer_config = RQTransformer.config_class.from_dict(config.rqtransformer)

        self.rqvaesiglip = RQVAESiglipModel._from_config(rqvaesiglip_config)
        # self.rqtransformer = RQTransformer._from_config(rqtransformer_config)
        self.rqtransformer_visual = RQTransformer._from_config(rqtransformer_config)
        self.rqtransformer_semantic = RQTransformer._from_config(rqtransformer_config)

AutoConfig.register("rqvaesigliptransformer_model", RQVAESIGLIPTransformerConfig)
AutoModel.register(RQVAESIGLIPTransformerConfig, RQVAESIGLIPTransformer)