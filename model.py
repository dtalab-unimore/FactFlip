import torch

from transformers import AutoModel, AutoModelForCausalLM
from torch import nn
import torch.nn.functional as F
from utils import get_device

class CustomModel(nn.Module):
  def __init__(self):
    super(CustomModel, self).__init__()

  def compute_average_layers(self, embs):
    if not isinstance(embs, tuple):
      return embs
    value = None
    for emb in embs:
      if value is None:
        value = emb
      else:
        value += emb
    value /= len(embs)
    return value

class GenerativeModel(CustomModel):
  def __init__(self, config):
    super(GenerativeModel, self).__init__()

    self.num_classes = config["num_classes"]
    self.embed_size = config["embed_size"]

    # using Qwen
    self.plm = AutoModelForCausalLM.from_pretrained(
      config["model_name"],
      output_hidden_states=True,
      torch_dtype=torch.float16,
      device_map="auto"
    )

  #@torch.autocast(device_type="cuda")
  def forward(self, ids_sent1, segs_sent1, att_mask_sent1):
    """
    Return full logits from Qwen’s lm_head.
    ids_sent1: [batch, seq_len]
    att_mask_sent1: [batch, seq_len]
    """
    outputs = self.plm(
      input_ids=ids_sent1,
      attention_mask=att_mask_sent1,
    )

    logits = outputs.logits[torch.arange(len(outputs.logits), device=outputs.logits.device), -1, :]

    # outputs.logits → [batch_size, seq_len, vocab_size]
    return logits

  @torch.autocast(device_type="cuda")
  def compute_concept_vector(self, ids_sent1, segs_sent1, att_mask_sent1, ids_sent2, segs_sent2, att_mask_sent2):
    # Forward pass
    out_concept = self.plm(
      input_ids=ids_sent1,
      attention_mask=att_mask_sent1,
      output_hidden_states=True
    ).hidden_states[-1]  # shape: [batch, seq_len, hidden_size]

    out_random = self.plm(
      input_ids=ids_sent2,
      attention_mask=att_mask_sent2,
      output_hidden_states=True
    ).hidden_states[-1]  # shape: [batch, seq_len, hidden_size]

    out_concept = self.compute_average_layers(out_concept) # with only one layer, this is redundant
    out_random = self.compute_average_layers(out_random) # with only one layer, this is redundant

    batch_size = out_concept.size(0)
    batch_idx = torch.arange(batch_size, device=out_concept.device)

    out_concept = out_concept[batch_idx, -1, :]
    out_random = out_random[batch_idx, -1, :]

    return None, out_concept, out_random

class RobertaModel(CustomModel):
  def __init__(self, config):
    super(RobertaModel, self).__init__()

    self.num_classes = config["num_classes"]
    self.embed_size = config["embed_size"]

    if config["backbone"] is not None:
      self.plm = AutoModel.from_pretrained(config["backbone"]).to(get_device())
    else:
      self.plm = AutoModel.from_pretrained(config["model_name"]).to(get_device())

    config = self.plm.config
    config.type_vocab_size = 2
    self.plm.embeddings.token_type_embeddings = nn.Embedding(
      config.type_vocab_size, config.hidden_size
    )
    self.plm._init_weights(self.plm.embeddings.token_type_embeddings) # re-initialize token_type_embeddings to possibly accept more than 2 segment ids
    self.linear_layer = torch.nn.Linear(in_features=self.embed_size, out_features=self.num_classes)
    self._init_weights(self.linear_layer)

  def _init_weights(self, module):
    """Initialize the weights"""
    if isinstance(module, (nn.Linear, nn.Embedding)):
      module.weight.data.normal_(mean=0.0, std=self.plm.config.initializer_range)
    elif isinstance(module, nn.LayerNorm):
      module.bias.data.zero_()
      module.weight.data.fill_(1.0)
    if isinstance(module, nn.Linear) and module.bias is not None:
      module.bias.data.zero_()

  @torch.autocast(device_type="cuda")
  def forward(self, ids_sent1, segs_sent1, att_mask_sent1):
    out_sent1 = self.plm(ids_sent1, token_type_ids=segs_sent1, attention_mask=att_mask_sent1, output_hidden_states=True)
    embed_sent1 = out_sent1.hidden_states[-1]

    H_sent = embed_sent1[:,0,:]
    predictions = self.linear_layer(H_sent)
    return predictions

  @torch.autocast(device_type="cuda")
  def compute_concept_vector(self, ids_sent1, segs_sent1, att_mask_sent1, ids_sent2, segs_sent2, att_mask_sent2):
    out_concept = self.plm(
      ids_sent1,
      token_type_ids=segs_sent1,
      attention_mask=att_mask_sent1,
      output_hidden_states=True
    )
    out_random = self.plm(
      ids_sent2,
      token_type_ids=segs_sent2,
      attention_mask=att_mask_sent2,
      output_hidden_states=True
    )
    out_concept, out_random = out_concept.hidden_states[-1], out_random.hidden_states[-1]
    out_concept, out_random = self.compute_average_layers(out_concept), self.compute_average_layers(out_random)
    out_concept = out_concept[:,0,:]
    out_random = out_random[:,0,:]

    concept_vector = out_concept - out_random
    return concept_vector, out_concept, out_random
