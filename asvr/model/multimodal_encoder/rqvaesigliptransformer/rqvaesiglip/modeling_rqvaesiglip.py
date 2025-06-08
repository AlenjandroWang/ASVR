import re
import torch

from torch import nn
from torch.nn import functional as F
from transformers import PreTrainedModel, AutoConfig, AutoModel, AutoProcessor

from .configuration_rqvaesiglip import RQVAESiglipConfig
from .modules import Decoder, PostQuantResnetBlock, ProjectResnetBlock
from .quantizations import RQBottleneck
from .siglip import SiglipModel

from typing import Any, Optional, Tuple, Union
from dataclasses import dataclass
from transformers.utils import ModelOutput
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling


def build_projector(dim_in, dim_out, projector_type='mlp2x_gelu'):
    if projector_type == 'linear':
        linear = nn.Linear(dim_in, dim_out)
        linear.reset_parameters()
        return linear
    elif projector_type == 'nonlinear':
        linear = nn.Linear(dim_in, dim_out)
        linear.reset_parameters()
        modules = [linear, nn.GELU()]
        return nn.Sequential(*modules)
    elif projector_type == 'conv':
        return nn.Conv2d(dim_in, dim_out, 1)
    else:  # mlp2x_gelu
        linear_1 = nn.Linear(dim_in, dim_in)
        linear_1.reset_parameters()
        modules = [linear_1]
        modules.append(nn.GELU())
        linear_2 = nn.Linear(dim_in, dim_out)
        linear_2.reset_parameters()
        modules.append(linear_2)

        return nn.Sequential(*modules)


@dataclass
# Copied from transformers.models.clip.modeling_clip.CLIPOutput with CLIP->Siglip
class SiglipOutput(ModelOutput):
    """
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`):
            Contrastive loss for image-text similarity.
        logits_per_image:(`torch.FloatTensor` of shape `(image_batch_size, text_batch_size)`):
            The scaled dot product scores between `image_embeds` and `text_embeds`. This represents the image-text
            similarity scores.
        logits_per_text:(`torch.FloatTensor` of shape `(text_batch_size, image_batch_size)`):
            The scaled dot product scores between `text_embeds` and `image_embeds`. This represents the text-image
            similarity scores.
        text_embeds(`torch.FloatTensor` of shape `(batch_size, output_dim`):
            The text embeddings obtained by applying the projection layer to the pooled output of [`SiglipTextModel`].
        image_embeds(`torch.FloatTensor` of shape `(batch_size, output_dim`):
            The image embeddings obtained by applying the projection layer to the pooled output of [`SiglipVisionModel`].
        text_model_output(`BaseModelOutputWithPooling`):
            The output of the [`SiglipTextModel`].
        vision_model_output(`BaseModelOutputWithPooling`):
            The output of the [`SiglipVisionModel`].
    """

    loss: Optional[torch.FloatTensor] = None
    logits_per_image: torch.FloatTensor = None
    logits_per_text: torch.FloatTensor = None
    text_embeds: torch.FloatTensor = None
    image_embeds: torch.FloatTensor = None
    text_model_output: BaseModelOutputWithPooling = None
    vision_model_output: BaseModelOutputWithPooling = None

    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            self[k] if k not in ["text_model_output", "vision_model_output"] else getattr(self, k).to_tuple()
            for k in self.keys()
        )

# new encoder
class RQVAESiglipModel(PreTrainedModel):
    config_class = RQVAESiglipConfig
    def __init__(self, config: RQVAESiglipConfig):
        super().__init__(config)

        # self.siglip_model = SiglipModel.from_pretrained(config.pretrained_model)
        siglip_config = SiglipModel.config_class.from_pretrained(config.pretrained_model)
        self.siglip_model = SiglipModel._from_config(siglip_config)

        # self.prequant_semantic = ProjectResnetBlock(in_channels=config.hidden_size,
        #                                             out_channels=config.embed_dim,
        #                                             dropout=0.0)
        self.prequant_visual = ProjectResnetBlock(in_channels=config.hidden_size,
                                                  out_channels=config.embed_dim,
                                                  dropout=0.0)
        self.quantizer_semantic = RQBottleneck(
            latent_shape=config.latent_shape,
            code_shape=config.code_shape,
            n_embed=config.n_embed,
            decay=config.decay,
            shared_codebook=config.shared_codebook,
            restart_unused_codes=config.restart_unused_codes,
        )
        self.quantizer = RQBottleneck(
            latent_shape=config.latent_shape,
            code_shape=config.code_shape,
            n_embed=config.n_embed,
            decay=config.decay,
            shared_codebook=config.shared_codebook,
            restart_unused_codes=config.restart_unused_codes,
        )
        # self.postquant_semantic = ProjectResnetBlock(in_channels=config.embed_dim,
        #                                              out_channels=config.hidden_size,
        #                                              dropout=0.0)
        self.postquant_visual = ProjectResnetBlock(in_channels=config.embed_dim,
                                                    out_channels=config.hidden_size,
                                                    dropout=0.0)
        self.post_quant_conv = PostQuantResnetBlock(in_channels=config.hidden_size,
                                                    out_channels=config.ddconfig["decoder_in_channels"],
                                                    dropout=0.0)
        self.decoder = Decoder(**config.ddconfig)

        try:
            self.decoder_latent_shape = config.decoder_latent_shape
        except:
            self.decoder_latent_shape = None


    def encode_text(self, text):
        # Use SigLIP model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions, output_hidden_states, return_dict = None, None, None
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        text_model = self.siglip_model.text_model
        text_outputs = text_model(
            input_ids=text,
            attention_mask=None,
            position_ids=None,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        text_embeds = text_outputs[1]
        return text_embeds

    def encode_image(self, image):
        vision_model = self.siglip_model.vision_model
        hidden_states = vision_model.embeddings(image)

        attention_mask = None
        output_attentions = None
        visual_n, semantic_n = 22, 2  # 取到倒数第n层的特征
        for i, encoder_layer in enumerate(vision_model.encoder.layers):
            if vision_model.encoder.gradient_checkpointing and vision_model.encoder.training:
                layer_outputs = vision_model.encoder._gradient_checkpointing_func(
                    encoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    output_attentions,
                )
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    output_attentions=output_attentions,
                )
            hidden_states = layer_outputs[0]
            
            if i == len(vision_model.encoder.layers) - visual_n:
                B, L, C = hidden_states.shape
                hidden_states_visual = hidden_states.reshape(B, int(L**0.5), int(L**0.5), C)
                
                hidden_states_visual = hidden_states_visual.permute(0, 3, 1, 2).contiguous()
                hidden_states_visual = self.prequant_visual(hidden_states_visual)
                hidden_states_visual = hidden_states_visual.permute(0, 2, 3, 1).contiguous()

                z_q_visual, quant_loss_visual, code_visual = self.quantizer(hidden_states_visual)
            
            if i == len(vision_model.encoder.layers) - semantic_n:
                hidden_state_26 = hidden_states
                B, L, C = hidden_states.shape
                hidden_states_semantic = hidden_states.reshape(B, int(L**0.5), int(L**0.5), C)
                
                # hidden_states_semantic = hidden_states_semantic.permute(0, 3, 1, 2).contiguous()
                # hidden_states_semantic = self.prequant_semantic(hidden_states_semantic)
                # hidden_states_semantic = hidden_states_semantic.permute(0, 2, 3, 1).contiguous()
                
                z_q_semantic, quant_loss_semantic, code_semantic = self.quantizer_semantic(hidden_states_semantic)

                return z_q_visual, code_visual, z_q_semantic, code_semantic
    
    def decode(self, z_q):
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        z_q = self.postquant_visual(z_q)
        
        if self.decoder_latent_shape is not None:
            z_q = F.interpolate(z_q.to(torch.float32), size=tuple(self.decoder_latent_shape), mode='bilinear').to(torch.bfloat16)
        z_q = self.post_quant_conv(z_q)
        out = self.decoder(z_q)

        return out 

    @torch.no_grad()
    def get_code_emb_with_depth(self, code, mode=None):
        # 分 visual codebook 取 image embedding, mode = "visual" or "semantic"
        if mode == "visual":
            visual_embedding = self.quantizer.embed_code_with_depth(code)
            return visual_embedding
        elif mode == "semantic":
            semantic_embedding = self.quantizer_semantic.embed_code_with_depth(code)
            return semantic_embedding    
# old encoder
# class RQVAESiglipModel(PreTrainedModel):
#     config_class = RQVAESiglipConfig
#     def __init__(self, config: RQVAESiglipConfig):
#         super().__init__(config)

#         # siglip_config = SiglipModel.config_class.from_pretrained(config.pretrained_model)
#         # self.siglip_model = SiglipModel._from_config(siglip_config)
#         self.siglip_model = SiglipModel.from_pretrained(config.pretrained_model)

#         self.prequant_semantic = ProjectResnetBlock(in_channels=config.hidden_size,
#                                                     out_channels=config.embed_dim,
#                                                     dropout=0.0)
#         self.prequant_visual = ProjectResnetBlock(in_channels=config.hidden_size,
#                                                   out_channels=config.embed_dim,
#                                                   dropout=0.0)

#         self.quantizer_semantic = RQBottleneck(
#             latent_shape=config.latent_shape,
#             code_shape=config.code_shape,
#             n_embed=config.n_embed,
#             decay=config.decay,
#             shared_codebook=config.shared_codebook,
#             restart_unused_codes=config.restart_unused_codes,
#         )

#         self.quantizer = RQBottleneck(
#             latent_shape=config.latent_shape,
#             code_shape=config.code_shape,
#             n_embed=config.n_embed,
#             decay=config.decay,
#             shared_codebook=config.shared_codebook,
#             restart_unused_codes=config.restart_unused_codes,
#         )

#         self.postquant_semantic = ProjectResnetBlock(in_channels=config.embed_dim,
#                                                      out_channels=config.hidden_size,
#                                                      dropout=0.0)
#         self.postquant_visual = ProjectResnetBlock(in_channels=config.embed_dim,
#                                                     out_channels=config.hidden_size,
#                                                     dropout=0.0)

#         self.post_quant_conv = PostQuantResnetBlock(in_channels=config.hidden_size,
#                                                     out_channels=config.ddconfig["decoder_in_channels"],
#                                                     dropout=0.0)

#         self.decoder = Decoder(**config.ddconfig)

#         try:
#             self.decoder_latent_shape = config.decoder_latent_shape
#         except:
#             self.decoder_latent_shape = None
        
#         # self.logit_scale = self.siglip_model.logit_scale
#         # self.logit_bias = self.siglip_model.logit_bias
        

#     def encode_text(self, text):
#         # Use SigLIP model's config for some fields (if specified) instead of those of vision & text components.
#         output_attentions, output_hidden_states, return_dict = None, None, None
#         output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
#         output_hidden_states = (
#             output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
#         )
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         text_model = self.siglip_model.text_model
#         text_outputs = text_model(
#             input_ids=text,
#             attention_mask=None,
#             position_ids=None,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )
#         text_embeds = text_outputs[1]
#         return text_embeds


#     def encode_image(self, image):
#         vision_model = self.siglip_model.vision_model
#         hidden_states = vision_model.embeddings(image)

#         attention_mask = None
#         output_attentions = None
#         visual_n, semantic_n = 22, 2  # 取到倒数第n层的特征
#         for i, encoder_layer in enumerate(vision_model.encoder.layers):
#             if vision_model.encoder.gradient_checkpointing and vision_model.encoder.training:
#                 layer_outputs = vision_model.encoder._gradient_checkpointing_func(
#                     encoder_layer.__call__,
#                     hidden_states,
#                     attention_mask,
#                     output_attentions,
#                 )
#             else:
#                 layer_outputs = encoder_layer(
#                     hidden_states,
#                     attention_mask,
#                     output_attentions=output_attentions,
#                 )
#             hidden_states = layer_outputs[0]
            
#             if i == len(vision_model.encoder.layers) - visual_n:
#                 B, L, C = hidden_states.shape
#                 hidden_states_visual = hidden_states.reshape(B, int(L**0.5), int(L**0.5), C)
                
#                 hidden_states_visual = hidden_states_visual.permute(0, 3, 1, 2).contiguous()
#                 hidden_states_visual = self.prequant_visual(hidden_states_visual)
#                 hidden_states_visual = hidden_states_visual.permute(0, 2, 3, 1).contiguous()

#                 z_q_visual, quant_loss_visual, code_visual = self.quantizer(hidden_states_visual)

#                 z_q_visual = z_q_visual.permute(0, 3, 1, 2).contiguous()
#                 z_q_visual = self.postquant_visual(z_q_visual)
#                 z_q_visual = z_q_visual.permute(0, 2, 3, 1).contiguous()
            
#             if i == len(vision_model.encoder.layers) - semantic_n:
#                 hidden_state_26 = hidden_states
#                 B, L, C = hidden_states.shape
#                 hidden_states_semantic = hidden_states.reshape(B, int(L**0.5), int(L**0.5), C)
                
#                 hidden_states_semantic = hidden_states_semantic.permute(0, 3, 1, 2).contiguous()
#                 hidden_states_semantic = self.prequant_semantic(hidden_states_semantic)
#                 hidden_states_semantic = hidden_states_semantic.permute(0, 2, 3, 1).contiguous()
                
#                 z_q_semantic, quant_loss_semantic, code_semantic = self.quantizer_semantic(hidden_states_semantic)
                
#                 z_q_semantic = z_q_semantic.permute(0, 3, 1, 2).contiguous()
#                 z_q_semantic = self.postquant_semantic(z_q_semantic)
#                 z_q_semantic = z_q_semantic.permute(0, 2, 3, 1).contiguous()
#                 hidden_states = z_q_semantic.reshape(B, L, C)

#         last_hidden_state = hidden_states
#         last_hidden_state = vision_model.post_layernorm(last_hidden_state)
#         pooler_output = vision_model.head(last_hidden_state) if vision_model.use_head else None
                
#         return z_q_visual, code_visual, z_q_semantic, code_semantic, pooler_output

    
#     def decode(self, z_q):
#         z_q = z_q.permute(0, 3, 1, 2).contiguous()
#         if self.decoder_latent_shape is not None:
#             z_q = F.interpolate(z_q.to(torch.float32), size=tuple(self.decoder_latent_shape), mode='bilinear').to(torch.bfloat16)
#         z_q = self.post_quant_conv(z_q)
#         out = self.decoder(z_q)

#         return out
    

#     @torch.no_grad()
#     def get_code_emb_with_depth(self, code, mode=None):
#         # 分 visual codebook 取 image embedding, mode = "visual" or "semantic"
#         if mode == "visual":
#             visual_embedding = self.quantizer.embed_code_with_depth(code)
#             return visual_embedding
#         elif mode == "semantic":
#             semantic_embedding = self.quantizer_semantic.embed_code_with_depth(code)
#             return semantic_embedding
    

#     def forward(self, image: Optional[torch.Tensor] = None, text: Optional[torch.Tensor] = None):
#         vision_output = self.encode_image(image) if image is not None else None
        
#         zq_visual, code_visual = vision_output[0], vision_output[1]
#         zq_semantic, code_semantic = vision_output[2], vision_output[3]
#         pooler_output = vision_output[-1]

#         # normalized features
#         image_embeds = pooler_output / pooler_output.norm(p=2, dim=-1, keepdim=True)
#         if text is not None:
#             text_embeds = self.encode_text(text)  # tokenized tokens
#             text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
#         else:
#             text_embeds = None

#         images_recon = self.decode(z_q=zq_visual)

#         clip_loss_dict = {
#             "image_features": image_embeds,
#             "text_features": text_embeds,
#             "logit_scale": self.siglip_model.logit_scale
#         }
#         if self.siglip_model.logit_bias is not None:
#             clip_loss_dict['logit_bias'] = self.siglip_model.logit_bias

#         return clip_loss_dict, zq_semantic, pooler_output, images_recon, code_visual, code_semantic
    

AutoConfig.register("rqvaesiglip_model", RQVAESiglipConfig)
AutoModel.register(RQVAESiglipConfig, RQVAESiglipModel)