#    Copyright 2024 Hao Zhang
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union, Dict
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

import transformers
from transformers import AutoConfig, AutoModelForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput
from transformers.modeling_outputs import ModelOutput
from dataclasses import dataclass
from..asvr_arch import ASVRMetaModel, ASVRMetaForCausalLM
from transformers import Qwen2Config, Qwen2Model, Qwen2ForCausalLM



@dataclass
class CloudCausalLMOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    loss_vision: Optional[torch.FloatTensor] = None
    loss_text: Optional[torch.FloatTensor] = None

class ASVRQwenConfig(Qwen2Config):
    model_type = "asvr_qwen"


class ASVRQwenModel(ASVRMetaModel, Qwen2Model):
    config_class = ASVRQwenConfig

    def __init__(self, config: Qwen2Config):
        super(ASVRQwenModel, self).__init__(config)


class ASVRQwenForCausalLM(Qwen2ForCausalLM, ASVRMetaForCausalLM):
    config_class = ASVRQwenConfig

    def __init__(self, config):
        super(Qwen2ForCausalLM, self).__init__(config)
        self.model = ASVRQwenModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        config.model_type = "asvr_qwen"
        config.rope_scaling = None
        # Initialize weights and apply final processing
        self.projector = nn.Sequential(nn.Linear(config.hidden_size, 2048), 
                                        nn.GELU(), 
                                        nn.Linear(2048, 2048)) #new

        # self.projector = nn.Sequential(nn.Linear(config.hidden_size, 2560), 
        #                                 nn.GELU(), 
        #                                 nn.Linear(2560, 2560)) #old
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        cache_position=None,
    ) -> Union[Tuple, CloudCausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                tokens_visual, 
                tokens_semantic,
                image_indices
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes
            )
        
        
       
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position, # only for transformers version 4.38.2
        )
  
        hidden_states = outputs[0]
        if tokens_visual is not None and tokens_semantic is not None:
            
            B,SEQ_LEN,depth = tokens_semantic.shape
            img_embedding= self.extract_image_features(hidden_states, image_indices).to(hidden_states.dtype)
            img_out= self.projector(img_embedding)
            # outs_visual = self.get_vision_tokenizer().vision_tower.rqtransformer_visual(img_out,tokens_visual,self.get_vision_tokenizer().vision_tower.rqvaesiglip, mode="visual")
            outs_semantic = self.get_vision_tokenizer().vision_tower.rqtransformer_semantic(img_out,tokens_semantic,self.get_vision_tokenizer().vision_tower.rqvaesiglip, mode="semantic")
            
            tokens_semantic = tokens_semantic.reshape(B*SEQ_LEN,depth).contiguous()
            # B, SEQ_LEN, depth, codebook_size = outs_visual.shape
            # visual_logits_visual = outs_visual.reshape(B*SEQ_LEN*depth, codebook_size).contiguous()
            # B, SEQ_LEN, depth, codebook_size = outs_semantic.shape
            # visual_logits_semantic = outs_semantic.reshape(B*SEQ_LEN*depth, codebook_size).contiguous()

            visual_loss_list_visual = [torch.tensor(0, device=hidden_states.device, dtype=hidden_states.dtype) for _ in range(8)]  # modify
            visual_loss_list_semantic = [torch.tensor(0, device=hidden_states.device, dtype=hidden_states.dtype) for _ in range(8)]


     
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        


        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss_text = loss_fct(shift_logits, shift_labels)

            if tokens_visual is not None and tokens_semantic is not None:
                # visual_labels_visual = tokens_visual.view(-1)  
                # visual_labels_semantic = tokens_semantic.view(-1)  
                # visual_labels_visual = visual_labels_visual.to(visual_logits_visual.device)
                # visual_labels_semantic = visual_labels_semantic.to(visual_logits_semantic.device)
                # # visual_flatten_loss_visual = loss_fct(visual_logits_visual, visual_labels_visual)
                # visual_flatten_loss_semantic = loss_fct(visual_logits_semantic, visual_labels_semantic)

                for i in range(len(outs_semantic)):  
                    # print("outs_semantic[i]:",outs_semantic[i].shape)
                    # print("tokens_semantic[..., i]:",tokens_semantic[..., i].shape)
              
                    
                    visual_loss_list_semantic[i] = loss_fct(outs_semantic[i].to(tokens_semantic[..., i].device), tokens_semantic[..., i])  # visual_idx.shape = B, 729,8
                
                total_weight = sum([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
                visual_flatten_loss_semantic = sum(w * loss for w, loss in zip([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], visual_loss_list_semantic)) / total_weight
                # loss_vision = visual_flatten_loss_visual + visual_flatten_loss_semantic
                loss_vision = visual_flatten_loss_semantic.to(loss_text.dtype).to(loss_text.device)
                loss = loss_vision + loss_text
            else:
                loss = loss_text
                loss_vision = 0. * loss
        # print(f"loss_vision:{loss_vision},loss_text:{loss_text}")
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output + (loss_vision, loss_text) if loss is not None else output

        return CloudCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            loss_vision=loss_vision,
            loss_text=loss_text
        )



        # return super().forward(
        #     input_ids=input_ids,
        #     attention_mask=attention_mask,
        #     position_ids=position_ids,
        #     past_key_values=past_key_values,
        #     inputs_embeds=inputs_embeds,
        #     labels=labels,
        #     use_cache=use_cache,
        #     output_attentions=output_attentions,
        #     output_hidden_states=output_hidden_states,
        #     return_dict=return_dict
        # )

    # @torch.no_grad()
    # def generate(
    #     self,
    #     inputs: Optional[torch.Tensor] = None,
    #     images: Optional[torch.Tensor] = None,
    #     image_sizes: Optional[torch.Tensor] = None,
    #     modalities: Optional[List[str]] = ["image"],
    #     **kwargs,
    # ) -> Union[GenerateOutput, torch.LongTensor]:
    #     position_ids = kwargs.pop("position_ids", None)
    #     attention_mask = kwargs.pop("attention_mask", None)
    #     if "inputs_embeds" in kwargs:
    #         raise NotImplementedError("`inputs_embeds` is not supported")

    #     if images is not None:
    #         (inputs, position_ids, attention_mask, _, inputs_embeds, _) = self.prepare_inputs_labels_for_multimodal(inputs, position_ids, attention_mask, None, None, images, modalities, image_sizes=image_sizes)
    #     else:
    #         inputs_embeds = self.get_model().embed_tokens(inputs)

    #     return super().generate(position_ids=position_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs)
        if images is not None:
            inputs["images"] = images
        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes
        return inputs


AutoConfig.register("asvr_qwen", ASVRQwenConfig)
AutoModelForCausalLM.register(ASVRQwenConfig, ASVRQwenForCausalLM)