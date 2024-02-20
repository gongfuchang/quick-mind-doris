import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoModel, AutoTokenizer
import os
import threading

from openai import OpenAI
from zhipuai import ZhipuAI

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
from abc import ABC, abstractmethod


class Client(ABC):
    @abstractmethod
    def __init__(self, config: dict):
        pass

    @abstractmethod
    def create_completion(self, messages: [str], temperature: float = 0.5, max_tokens: int = 512, stream: bool = False,
                          tools: [str] = None, stop_words: [str] = None):
        pass


class OpenaiClient(Client):
    def __init__(self, config: dict):
        self.config = config
        # Initialize your OpenAI client here with the config
        self.openai = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    def create_completion(self, messages: [str], temperature: float = 0.5, max_tokens: int = 512, stream: bool = False,
                          tools: [str] = None, stop_words: [str] = None):
        # Implement the method to create a completion with OpenAI
        return self.openai.chat.completions.create(model=self.config.get("model"), messages=messages,
                                                   temperature=temperature, max_tokens=max_tokens, stream=stream,
                                                   tools=tools, stop=stop_words)


class GlmClient(Client):
    def __init__(self, config: dict):
        self.config = config
        # Initialize your GLM client here with the config
        self.glm = ZhipuAI(api_key=os.getenv('ZHIPU_API_KEY'))

    def create_completion(self, messages: [str], temperature: float = 0.5, max_tokens: int = 512, stream: bool = False,
                          tools: [str] = None, stop_words: [str] = None):
        # Implement the method to create a completion with GLM
        return self.glm.chat.completions.create(model="glm-4", messages=messages,
                                                temperature=temperature, max_tokens=max_tokens, stream=stream,
                                                tools=tools, stop=stop_words)


class ClientFactory:
    def get_client(self, type: str, config: dict = None) -> Client:
        if type == 'GPT':
            return OpenaiClient(config)
        elif type == 'GLM':
            return GlmClient(config)
        else:
            raise ValueError(f"Unsupported client type: {type}")


factory = ClientFactory()


def get_client(llm_type: str, config: dict = None) -> Client:
    thread_local = threading.local()
    # Check if the client already exists in the current thread
    if hasattr(thread_local, 'client'):
        return thread_local.client

    # If not, create a new client and store it in the thread-local variable
    thread_local.client = factory.get_client(llm_type, config)

    return thread_local.client
