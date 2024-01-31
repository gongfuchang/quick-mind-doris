"""
Simple FastAPI app that queries opensearch and a semantic index for retrieval-augmented generation.
"""
import os
import pickle
from typing import Dict, List
import re
import numpy as np
import pandas as pd
import tiktoken
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.logger import logger
from src.prep.build_opensearch_index import (INDEX_NAME, get_opensearch,
                                             query_opensearch)
from src.prep.build_semantic_index import query_semantic, get_embeddings_index, get_embeddings_array
from src.prep.build_vault_dict import get_vault
from src.utils.model_util import get_model_tuple, get_device

# Load vault dictionary
vault = get_vault()
logger.info(f'Vault loaded with {len(vault)} documents')

os_client = None
# Create opensearch client
# try:
#     os_client = get_opensearch('localhost')
# except ConnectionRefusedError:
#     os_client = get_opensearch('localhost')  # Change to 'localhost' if running locally
# logger.info(f'OS client initialized: {os_client.info()}')

# Load semantic index
doc_embeddings_array = get_embeddings_array()
embedding_index = get_embeddings_index()
tokenizer, model = get_model_tuple()

logger.info(f'Semantic index loaded with {len(embedding_index)} documents')


# Create app
app = FastAPI()

# List of allowed origins. You can also allow all by using ["*"]
origins = [
    "http://localhost",  # or whatever hosts you want to allow
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# If the user did not provide a query, we will use this default query.
_default_query = "如何快速开启 Apache Doris 之旅?"

_rag_query_text = """
You are a large language AI assistant built by QuickMind AI. You are given a user question, and please write clean, concise and accurate answer to the question. You will be given a set of related contexts to the question, each starting with a reference number like [[citation:x]], where x is a number. Please use the context and cite the context at the end of each sentence if applicable.

Your answer must be correct, accurate and written by an expert using an unbiased and professional tone. Please limit to 1024 tokens. Do not give any information that is not related to the question, and do not repeat. Say "information is missing on" followed by the related topic, if the given context do not provide sufficient information.

Please cite the contexts with the reference numbers, in the format [citation:x]. If a sentence comes from multiple contexts, please list all applicable citations, like [citation:3][citation:5]. Other than code and specific names and citations, your answer must be written in the same language as the question.

Use chinese if the question contains chinese characters otherwise use english instead.

Here are the set of contexts:

{context}

Remember, don't blindly repeat the contexts verbatim. And here is the user question:
"""

# A set of stop words to use - this is not a complete set, and you may want to
# add more given your observation.
stop_words = [
    "<|im_end|>",
    "[End]",
    "[end]",
    "\nReferences:\n",
    "\nSources:\n",
    "End.",
]

_more_questions_prompt = """
You are a helpful assistant that helps the user to ask Apache Doris related questions, based on user's original question and the related contexts. Please identify worthwhile topics that can be follow-ups, and write questions no longer than 20 words each. Please make sure that specifics, like events, names, locations, are included in follow up questions so they can be asked standalone. For example, if the original question asks about "the Manhattan project", in the follow up question, do not just say "the project", but use the full name "the Manhattan project". Your related questions must be in the same language as the original question.

Here are the contexts of the question:

{context}

Do NOT make up any question not related with Apache Doris.

Remember, based on the original question and related contexts, suggest three such further questions. Do NOT repeat the original question. Each related question should be no longer than 20 words. Here is the original question:
"""

def parse_os_response(response: dict) -> List[dict]:
    """Parse response from opensearch index.

    Args:
        response: Response from opensearch query.

    Returns:
        List of hits with chunkID and rank
    """
    hits = []

    for rank, hit in enumerate(response['hits']['hits']):
        hits.append({'id': hit['_id'], 'rank': rank})

    return hits


def parse_semantic_response(indices: np.ndarray, embedding_index: Dict[int, str]) -> List[dict]:
    """Parse response from semantic index.

    Args:
        indices: Response from semantic query, an array of ints.

    Returns:
        List of hits with chunkID and rank
    """
    hits = []

    for rank, idx in enumerate(indices):
        hits.append({'id': embedding_index[idx], 'rank': rank})

    return hits


def num_tokens_from_string(string: str, model_name: str) -> int:
    """Returns the number of tokens in a string based on tiktoken encoding.

    Args:
        string: String to count tokens for
        model_name: Tokenizer model type

    Returns:
        Number of tokens in the string
    """
    encoding = tiktoken.encoding_for_model(model_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def get_chunks_from_hits(hits: List[dict], model_name: str = 'gpt-3.5-turbo', max_tokens: int = 3200) -> List[dict]:
    """Deduplicates and scores a list of chunks. (There may be duplicate chunks as we query multiple indices.)

    Args:
        hits: List of hits from opensearch, semantic index, etc.
        model_name: Downstream model for retrieval-augmented generation. Used to tokenize chunks and limit the size of
            input based on LLM context window size.
        max_tokens: Maximum tokens to allow in chunks. Defaults to 3,200.

    Returns:
        List of chunks for retrieval-augmented generation.
    """
    # Combine os and semantic hits and rank them
    df = pd.DataFrame(hits)
    df['score'] = df['rank'].apply(lambda x: 10 - x)
    ranked = df.groupby('id').agg({'score': 'sum'}).sort_values('score', ascending=False).reset_index()

    # Get context based on ranked IDs
    chunks = []
    token_count = 0

    for chunk_id in ranked['id'].tolist():
        content = ''.join(vault[chunk_id]['chunks'])
        title = vault[chunk_id]['title']

        # Check if token count exceeds max_tokens
        token_count += num_tokens_from_string(content, model_name)
        if token_count > max_tokens:
            break

        chunks.append({'title': title, 'content': content})

    return chunks


@app.get('/get_chunks')
def get_chunks(query: str):
    # Get hits from opensearch
    os_response = query_opensearch(query, os_client, INDEX_NAME)
    os_hits = parse_os_response(os_response)
    logger.debug(f'OS hits: {os_hits}')

    # Get hits from semantic index
    semantic_response = query_semantic(query, tokenizer, model, doc_embeddings_array)
    related_context = get_related_chunks_from_hits(semantic_response, embedding_index, vault)
    return related_context

@app.get('/query')
def query(query: str):
    # invoke get_chunks to retrieve related context
    context = get_chunks(query)
    # use the context to generate via openai api


def get_related_chunks_from_hits(hits: List[dict], embedding_index: Dict[int, str], vault: dict, max_lines: int = 10) -> List[dict]:
    """
    Deduplicates and scores a list of chunks. (There may be duplicate chunks as we query multiple indices.)

    For each hit in semantic_hits, it starts from the current position of the hit in the embedding_index,
    and looks up and down for the start or end of the related paragraph. The end is marked by the start of the next paragraph.
    For example, if the hit is 1210 and the corresponding content is 'This is the paragraph text',
    looking up is to find 1209, 1208 until 1205 '## This is the paragraph title'.
    Looking down is to find 1211, 1212 until 1218 '## This is another paragraph title'.
    If the results obtained from looking up and down exceed 10 lines, the search is exited in advance.

    Args:
        semantic_hits: List of hits from semantic index.
        embedding_index: Mapping of document embedding row index to document doc-id.
        vault: Dictionary of vault documents.
        max_lines: Maximum lines to allow in chunks. Defaults to 10.

    Returns:
        List of chunks for retrieval-augmented generation.
    """
    chunks = []
    for hit_index in hits:
        chunk_id = embedding_index[hit_index]
        up_index = hit_index - 1
        down_index = hit_index + 1
        lines = 0
        chunk = {'title': vault[chunk_id]['title'], 'content': ''.join(vault[chunk_id]['chunks'])}
        content_list = [chunk['content']]
        # Look up
        while up_index in embedding_index and lines < max_lines:
            chunk_id = embedding_index[up_index]
            chunk_content = ' '.join(vault[chunk_id]['chunks'])
            if re.match(r'^#+ ', chunk_content):  # Paragraph start
                break
            content_list.insert(0, chunk_content)
            lines += 1
            up_index -= 1
        # Look down
        while down_index in embedding_index and lines < max_lines:
            chunk_id = embedding_index[down_index]
            chunk_content = ' '.join(vault[chunk_id]['chunks'])
            if re.match(r'^#+ ', chunk_content):  # Paragraph start
                break
            content_list.append(chunk_content)
            lines += 1
            down_index += 1
        chunk['content'] = '\n'.join(content_list)
        chunks.append(chunk)
    return chunks

if __name__ == '__main__':
    test_query = '怎么设置严格模式'
    # os_response = query_opensearch(test_query, os_client, INDEX_NAME)
    # os_hits = parse_os_response(os_response)
    # logger.debug(f'OS hits: {os_hits}')
    # semantic_response is array of <chunk index>: [123, 223, 345 ..]
    semantic_response = query_semantic(f'query: {test_query}', tokenizer, model, doc_embeddings_array)
    # semantic_hits is array of {chunk_id: 123, rank: 0}
    semantic_hits = parse_semantic_response(semantic_response, embedding_index)
    logger.debug(f'Semantic hits: {semantic_hits}')

    # context is array of {title: 'title', content: 'content'}
    context = get_chunks_from_hits(semantic_hits)
    logger.info(f'Context: {context}')

    # Combine os and semantic hits and rank them
    related_context = get_related_chunks_from_hits(semantic_response, embedding_index, vault)
    logger.debug(f'Related Context: {related_context}')

