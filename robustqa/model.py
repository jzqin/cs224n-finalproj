import copy
import math
import random

import numpy as np
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, GELU

from transformers import DistilBertPreTrainedModel, DistilBertModel

MASK_TOKEN = -100 # is this best way of initializing this?
CLS_TOKEN = 101
SEP_TOKEN = 102

class AuxMLMModel(DistilBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.distilbert = DistilBertModel(config)

        self.qa_outputs = nn.Linear(config.dim, config.num_labels)
        assert config.num_labels == 2
        self.dropout = nn.Dropout(config.qa_dropout)

        self.vocab_transform = nn.Linear(config.dim, config.dim)
        self.vocab_layer_norm = nn.LayerNorm(config.dim, eps=1e-12)
        self.vocab_projector = nn.Linear(config.dim, config.vocab_size)

        self.init_weights()

        self.mlm_probability = 0.15 # this is default for BERT and RoBERTa
        self.mlm_loss_fct = nn.CrossEntropyLoss()

        self.vocab_size = None
        self.mask_token = MASK_TOKEN # maybe should come up with a better way of initializing this, in case we want to change it?

    # from MLM
    def get_output_embeddings(self):
        return self.vocab_projector

    # from MLM
    def set_output_embeddings(self, new_embeddings):
        self.vocab_projector = new_embeddings

    # add vocabulary size to model for MLM
    def add_vocab_size(self, vocab_size):
        # vocab should be a list of strings
        self.vocab_size = vocab_size
        
    # Synchronous masking for MLM task
    def mlm_mask(self, inputs):
        # random.seed(0)
        import pdb; pdb.set_trace()
        if self.vocab_size is None:
            raise AttributeError('AuxMLMModel must have vocabulary size added via add_vocab_size() before training occurs')

        # from RoBERTa paper: https://github.com/huggingface/transformers/blob/master/src/transformers/data/data_collator.py#L356
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training
        probability_matrix = torch.full(labels.shape, self.mlm_probability, device=inputs.device)

        special_tokens_mask = torch.zeros_like(inputs)
        special_tokens_mask[inputs == CLS_TOKEN] = 1 # which tokens can't be masked? [CLS] and [SEP]
        special_tokens_mask[inputs == SEP_TOKEN] = 1
        special_tokens_mask = special_tokens_mask.bool() 

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8, device=inputs.device)).bool() & masked_indices
        inputs[indices_replaced] = self.mask_token

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5, device=inputs.device)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(self.vocab_size, labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

        """
        #15% of input tokens changed to something else.
        #80% of these tokens are changed to [MASK] (focus on this first)
        for i in range(len(input_ids)): # TODO input_ids with batched shape
            rand1 = random.random()
            if (rand1 > 0.85):
                rand2 = random.random()
                if (rand2 > 0.2):
                    input_ids[i] = mask_token
                elif (rand2 > 0.1):
                    # 10% of tokens changed to random other word
                    input_ids[i] = random.randint(0, self.vocab_size - 1)
                else:
                    # 10% of tokens remain the same
                    pass
        
        return input_ids
        """
        
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        output_attentions=None,
        output_hidden_states=None,
        mask_token=None,
        gamma=0 # the proportion of MLM loss [0,1] 
    ):
        r"""
        start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        """

        input_ids, mlm_labels = self.mlm_mask(input_ids) # mask inputs to both losses

        distilbert_output = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict = None
        )        

        hidden_states = distilbert_output[0]  # (bs, max_query_len, dim)

        # Compute logits from QA
        hidden_states = self.dropout(hidden_states)  # (bs, max_query_len, dim)
        logits = self.qa_outputs(hidden_states)  # (bs, max_query_len, 2)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)  # (bs, max_query_len)
        end_logits = end_logits.squeeze(-1)  # (bs, max_query_len)

        # Compute logits from MLM
        prediction_logits = self.vocab_transform(hidden_states)  # (bs, max_query_length, dim)
        prediction_logits = GELU(prediction_logits)  # (bs, max_query_length, dim)
        prediction_logits = self.vocab_layer_norm(prediction_logits)  # (bs, max_query_length, dim)
        prediction_logits = self.vocab_projector(prediction_logits)  # (bs, max_query_length, vocab_size)

        # Compute Cross-Entropy Loss from QA
        qa_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            qa_start_loss = loss_fct(start_logits, start_positions)
            qa_end_loss = loss_fct(end_logits, end_positions)
            qa_loss = (qa_start_loss + qa_end_loss) / 2

        # Compute Cross-Entropy Loss from MLM
        # note: CrossEntropyLoss automatically ignores all positions with value -100
        mlm_loss = None
        if mlm_labels is not None:
            mlm_loss = self.mlm_loss_fct(prediction_logits.view(-1, prediction_logits.size(-1)), mlm_labels.view(-1))

        # compute total loss
        total_loss = qa_loss + gamma *  mlm_loss

        output = (start_logits, end_logits, prediction_logits) + distilbert_output[1:]

        return ((total_loss,) + output) if mlm_loss is not None else output


