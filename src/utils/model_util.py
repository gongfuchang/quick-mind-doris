import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoModel, AutoTokenizer
import os
from src.logger import logger

# EMBEDDING_MODEL_NAME = 'intfloat/e5-small-v2'
EMBEDDING_MODEL_NAME = 'intfloat/multilingual-e5-small'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """Average pool last hidden states, ignoring padding tokens"""
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def get_device():
    return DEVICE


def get_model_tuple() -> tuple[AutoTokenizer, AutoModel]:
    """
    Return tokenizer and model for chunks embedding
    @return: a tuple of AutoTokenizer, AutoModel
    """
    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)  # Max token length is 512
    model = AutoModel.from_pretrained(EMBEDDING_MODEL_NAME)
    model.to(DEVICE)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    return tokenizer, model


if __name__ == '__main__':
    pass


