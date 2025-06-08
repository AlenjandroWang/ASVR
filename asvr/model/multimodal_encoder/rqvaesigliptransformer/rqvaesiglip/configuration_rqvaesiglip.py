from transformers import PretrainedConfig


class RQVAESiglipConfig(PretrainedConfig):
    model_type = "rqvaesiglip_model"
    def __init__(
        self,
        embed_dim=None,
        hidden_size=None,
        n_embed=None,
        latent_shape=None,
        code_shape=None,
        shared_codebook=None,
        restart_unused_codes=None,
        ddconfig=None,
        decay=0.99,
        latent_loss_weight=0.25,
        architectures=None,
        decoder_latent_shape=None,
        pretrained_model="google/siglip-large-patch16-256",
        **kwargs,
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.n_embed = n_embed
        self.latent_shape = latent_shape
        self.code_shape = code_shape
        self.shared_codebook = shared_codebook
        self.restart_unused_codes = restart_unused_codes
        self.ddconfig = ddconfig
        self.decay = decay
        self.latent_loss_weight = latent_loss_weight
        self.architectures = architectures
        self.decoder_latent_shape = decoder_latent_shape
        self.pretrained_model = pretrained_model


# from transformers import PretrainedConfig
# import yaml


# class RQVAESiglipConfig(PretrainedConfig):
#     model_type = "rqvaesiglip_model"
#     def __init__(
#         self,
#         embed_dim=None,
#         n_embed=None,
#         latent_shape=None,
#         code_shape=None,
#         shared_codebook=None,
#         restart_unused_codes=None,
#         ddconfig=None,
#         decay=0.99,
#         latent_loss_weight=0.25,
#         architectures=None,
#         decoder_latent_shape=None,
#         pretrained_model="google/siglip-so400m-patch14-384",
#         # pretrained_model="google/siglip-large-patch16-256",
#         **kwargs,
#     ):
#         super().__init__()
        
#         self.embed_dim = embed_dim
#         self.n_embed = n_embed
#         self.latent_shape = latent_shape
#         self.code_shape = code_shape
#         self.shared_codebook = shared_codebook
#         self.restart_unused_codes = restart_unused_codes
#         # self.ddconfig = ddconfig
#         self.decay = decay
#         self.latent_loss_weight = latent_loss_weight
#         self.architectures = architectures
#         self.decoder_latent_shape = decoder_latent_shape
#         self.pretrained_model = pretrained_model
        
#         with open("../../model_zoo/rqvae/imagenet_3.8B_rqvae_50e/stage1/config.yaml", "r") as f:
#             config = yaml.safe_load(f)
#         ddconfig = config["arch"]["ddconfig"]
#         self.ddconfig = ddconfig
#         print("================= decoder ddconfig =================")
#         print(self.ddconfig)
#         print("====================================================")
