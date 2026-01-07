import torch

from transformers import AutoModel
from torch import nn
import torch.nn.functional as F

class RobertaModel(torch.nn.Module):
  def __init__(self, config):
    super(RobertaModel, self).__init__()

    self.num_classes = config["num_classes"]
    self.embed_size = config["embed_size"]
    self.avg_tokens = config["avg_tokens"]
    freeze = config["freeze"]

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

    """for param in self.plm.parameters():
      if freeze:
        param.requires_grad = False
      else:
        param.requires_grad = True"""

    if freeze:
      for param in self.plm.parameters():
        param.requires_grad = False

        # Unfreeze the last two transformer blocks
      if hasattr(self.plm, 'encoder'):  # works for BERT-like models
        blocks = self.plm.encoder.layer
      elif hasattr(self.plm, 'transformer'):  # e.g., GPT-2
        blocks = self.plm.transformer.h
      else:
        raise ValueError("Unexpected model structure")

      for layer in blocks[-2:]:
        for param in layer.parameters():
          param.requires_grad = True

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

  #@torch.autocast(device_type="cuda")
  def forward(self, ids_sent1, segs_sent1, att_mask_sent1):
    out_sent1 = self.plm(ids_sent1, token_type_ids=segs_sent1, attention_mask=att_mask_sent1, output_hidden_states=True)
    embed_sent1 = out_sent1.hidden_states[-1]

    if self.avg_tokens:
      H_sent = torch.sum(embed_sent1, dim=1) / torch.sum(att_mask_sent1, dim=1).unsqueeze(-1)
    else:
      H_sent = embed_sent1[:,0,:]
    predictions = self.linear_layer(H_sent)
    return predictions

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

  #@torch.autocast(device_type="cuda")
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
    if self.avg_tokens:
      out_concept = torch.sum(out_concept, dim=1) / torch.sum(att_mask_sent1, dim=1).unsqueeze(-1)
      out_random = torch.sum(out_random, dim=1) / torch.sum(att_mask_sent2, dim=1).unsqueeze(-1)
    else:
      out_concept = out_concept[:,0,:]
      out_random = out_random[:,0,:]

    concept_vector = out_concept - out_random
    # concept_vector = torch.mean(concept_vector, dim=0)
    return concept_vector, out_concept, out_random #F.normalize(concept_vector, p=2, dim=1), F.normalize((out_concept+out_random) / 2, p=2, dim=1)

  #@torch.autocast(device_type="cuda")
  def forward_without_classifier(self, ids_sent1, segs_sent1, att_mask_sent1, norm=True):
    out_emb = self.plm(
      ids_sent1,
      token_type_ids=segs_sent1,
      attention_mask=att_mask_sent1,
      output_hidden_states=True
    )
    out_emb = out_emb.hidden_states[-1]
    out_emb = self.compute_average_layers(out_emb)
    if self.avg_tokens:
      out_emb = torch.sum(out_emb, dim=1) / torch.sum(att_mask_sent1, dim=1).unsqueeze(-1) #out_emb[torch.arange(out_emb.shape[0]),torch.argmin(att_mask_sent1, dim=-1)-1,:]
    else:
      out_emb = out_emb[:,0,:]

    if norm:
      return F.normalize(out_emb, p=2, dim=1)
    else:
      return out_emb