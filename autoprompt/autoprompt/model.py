import torch

from transformers import AutoModel, AutoModelForCausalLM
from torch import nn
import torch.nn.functional as F

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

class LlamaModel(CustomModel):
  def __init__(self, config):
    super(LlamaModel, self).__init__()

    self.num_classes = config["num_classes"]
    self.embed_size = config["embed_size"]

    # Use LLaMA (decoder-only)
    self.plm = AutoModelForCausalLM.from_pretrained(
      config["model_name"],
      output_hidden_states=True,
      torch_dtype=torch.float16,
      device_map="auto"
    )

  #@torch.autocast(device_type="cuda")
  def forward(self, ids_sent1, segs_sent1, att_mask_sent1):
    """
    Return full logits from LLaMA’s lm_head.
    ids_sent1: [batch, seq_len]
    att_mask_sent1: [batch, seq_len]
    """
    outputs = self.plm(
      input_ids=ids_sent1,
      attention_mask=att_mask_sent1,
    )

    #last_token_idx = att_mask_sent1.sum(dim=1) - 1
    logits = outputs.logits[torch.arange(len(outputs.logits), device=outputs.logits.device), -1, :] #last_token_idx, :]

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

    # Average across layers if needed (you can also skip if using last_hidden_state only)
    out_concept = self.compute_average_layers(out_concept)
    out_random = self.compute_average_layers(out_random)

    batch_size = out_concept.size(0)
    batch_idx = torch.arange(batch_size, device=out_concept.device)
    #last_token_idx_concept = att_mask_sent1.sum(dim=1) - 1
    #last_token_idx_random = att_mask_sent2.sum(dim=1) - 1

    out_concept = out_concept[batch_idx, -1, :] #last_token_idx_concept, :]
    out_random = out_random[batch_idx, -1, :] #last_token_idx_random, :]

    """# Compute concept vector (optional: pooling to single vector)
    concept_vector = (out_concept * att_mask_sent1.unsqueeze(-1)).sum(1) / att_mask_sent1.sum(1, keepdim=True) \
                     - (out_random * att_mask_sent2.unsqueeze(-1)).sum(1) / att_mask_sent2.sum(1, keepdim=True)"""

    # Return the last hidden states (sequence embeddings) before lm_head
    return None, out_concept, out_random

class RobertaModel(CustomModel):
  def __init__(self, config):
    super(RobertaModel, self).__init__()

    self.num_classes = config["num_classes"]
    self.embed_size = config["embed_size"]

    if config["backbone"] is not None:
      self.plm = AutoModel.from_pretrained(config["backbone"])
    else:
      self.plm = AutoModel.from_pretrained(config["model_name"])

    config = self.plm.config
    config.type_vocab_size = 2
    self.plm.embeddings.token_type_embeddings = nn.Embedding(
      config.type_vocab_size, config.hidden_size
    )
    self.plm._init_weights(self.plm.embeddings.token_type_embeddings)
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
    # concept_vector = torch.mean(concept_vector, dim=0)
    return concept_vector, out_concept, out_random #F.normalize(concept_vector, p=2, dim=1), F.normalize((out_concept+out_random) / 2, p=2, dim=1)

  @torch.autocast(device_type="cuda")
  def forward_without_classifier(self, ids_sent1, segs_sent1, att_mask_sent1, norm=True):
    out_emb = self.plm(
      ids_sent1,
      token_type_ids=segs_sent1,
      attention_mask=att_mask_sent1,
      output_hidden_states=True
    )
    out_emb = out_emb.hidden_states[-1]
    out_emb = self.compute_average_layers(out_emb)
    out_emb = out_emb[:,0,:]

    if norm:
      return F.normalize(out_emb, p=2, dim=1)
    else:
      return out_emb