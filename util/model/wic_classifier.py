from transformers import RobertaConfig, RobertaModel, ElectraConfig, ElectraModel
from transformers.modeling_outputs import SequenceClassifierOutput
import torch.nn as nn
import torch

class wic_classifier(nn.Module):
    def __init__(self, resize_token_embd_len, model_name="roberta"):
        super().__init__()
        if model_name == 'roberta':
            model_config = RobertaConfig.from_pretrained(pretrained_model_name_or_path="klue/roberta-large",
                                                         hidden_dropout_prob=0.1)
            self.model = RobertaModel.from_pretrained("klue/roberta-large", config=model_config)
            self.model.resize_token_embeddings(resize_token_embd_len)
            # self.Layernorm = nn.LayerNorm(1024)
            self.dropout = nn.Dropout(0.1)
            self.out_proj = nn.Linear(1024 * 3, 2)
        elif model_name == 'electra':
            model_config = ElectraConfig.from_pretrained(
                pretrained_model_name_or_path='monologg/koelectra-base-v3-discriminator', num_labels=2)
            self.model = ElectraModel.from_pretrained('monologg/koelectra-base-v3-discriminator', config=model_config)
            self.model.resize_token_embeddings(resize_token_embd_len)
            self.dropout = nn.Dropout(0.1)
            self.out_proj = nn.Linear(768 * 3, 2)
        elif model_name == 'electra_tunib':
            model_config = ElectraConfig.from_pretrained(
                pretrained_model_name_or_path='tunib/electra-ko-base', num_labels=2)
            self.model = ElectraModel.from_pretrained('tunib/electra-ko-base', config=model_config)
            self.model.resize_token_embeddings(resize_token_embd_len)
            self.dropout = nn.Dropout(0.1)
            self.out_proj = nn.Linear(768 * 3, 2)

        self.criterion = nn.CrossEntropyLoss()


    def get_span_representation(self, output, span):
        repre = torch.stack([
            torch.sum(output[i, l[0]:l[1], :], dim=0) / (l[1]-l[0]) for i, l in enumerate(span)
        ])
        return repre
    def forward(self, input_ids, attention_mask,  span_1, span_2, labels=None):

        outputs = self.model(input_ids, attention_mask)
        output = outputs.last_hidden_state

        span_1 = self.get_span_representation(output, span_1) #  bsz, hidden
        span_2 = self.get_span_representation(output, span_2) #  bsz, hidden
        cls = output[:,0,:].squeeze(1)

        x = self.dropout(torch.cat([span_1,span_2,cls],dim=1))

        if labels is None:
            return self.out_proj(x)

        logits = self.out_proj(x)

        loss = self.criterion(logits, labels)

        return SequenceClassifierOutput(logits= logits, loss = loss)


