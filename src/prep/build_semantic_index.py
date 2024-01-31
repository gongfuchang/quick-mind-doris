"""
Reads vault dictionary, creates embeddings for each chunk, and creates a semantic index.
"""
import os
import pickle
import time
from typing import List
import torch
import gc
import numpy as np
import torch.nn.functional as F

from src.logger import logger
from src.utils.model_util import get_model_tuple, get_device, average_pool
from src.prep2.build_vault_dict import get_vault
from src.utils.file_util import get_file_path

EMBEDDINGS_ARRAY_NPY = get_file_path('data/embedding/doc_embeddings_array2.npy')

EMBEDDING_INDEX_PICKLE = get_file_path('data/embedding/embedding_index2.pickle')


def get_batch_embeddings(document_batch: List[str], tokenizer, model) -> List[np.ndarray]:
    """Embed a batch of documents

    Args:
        document_batch: List of documents to embed
        tokenizer: Tokenizer to tokenize documents; should be compatible with model
        model: Model to embed documents

    Returns:
        List of document embeddings
    """

    docs_tokenized = tokenizer(document_batch, max_length=512, padding=True, truncation=True, return_tensors='pt')
    docs_tokenized = {key: val.to(get_device()) for key, val in docs_tokenized.items()}
    outputs = model(**docs_tokenized)
    embeddings = average_pool(outputs.last_hidden_state, docs_tokenized['attention_mask'])
    embeddings_normed = F.normalize(embeddings, p=2, dim=1)  # Normalize embeddings for downstream cosine similarity

    return embeddings_normed.detach().cpu().numpy()


def build_embedding_index(vault: dict) -> dict[int, str]:
    """Build an index that maps document embedding row index to document chunk-id. 
    Used to retrieve document id after ANN on document embeddings.

    Args:
        vault: Dictionary of vault documents

    Returns:
        Mapping of document embedding row index to document doc-id
    """
    embedding_index = dict()
    embedding_idx = 0

    for chunk_id, chunk_entry in vault.items():
        embedding_index[embedding_idx] = chunk_id
        embedding_idx += 1

    return embedding_index


def build_embedding_array(vault_sub: dict, tokenizer, model, batch_size=20) -> np.ndarray:
    """Embedding all document chunks and return embedding array

    Args:
        vault: Dictionary of vault documents: chunk_id: {doc_id: doc_id, chunks: chunks}
        tokenizer: Tokenizer to tokenize documents; should be compatible with model
        model: Model to embed documents
        batch_size: Size of document batch to embed each time. Defaults to 4.

    Returns:
        Numpy array of n_chunks x embedding-dim document embeddings
    """
    chunk_embedded = 0
    chunk_batch = []
    chunks_batched = 0
    embedding_list = []

    for chunk_id, chunk_entry in vault_sub.items():
        # Get path and chunks
        if chunk_embedded % 100 == 0:
            logger.info(f'Embedding document chunk: {chunk_id} ({chunk_embedded:,})')
        chunk_embedded += 1
        chunks = chunk_entry['chunks'] # chunk_folder is a list of chunks
        processed_chunk = 'passage: ' + ' '.join(chunks)  # Remove extra whitespace and add prefix

        # logger.info(f'Chunk: {processed_chunk}')
        chunk_batch.append(processed_chunk)  # Add chunk to batch
        chunks_batched += 1

        if chunks_batched % batch_size == 0:
            # Compute embeddings in batch and append to list of embeddings
            start_time = time.time()
            chunk_embeddings = get_batch_embeddings(chunk_batch, tokenizer, model)
            if chunk_embedded % 10000 == 0:
                logger.info(
                    f'Time taken for get_batch_embeddings<{chunk_embedded} - {chunk_batch}>: {time.time() - start_time} seconds')
                logger.info(f'gc collected and GPU cache cleared at {chunk_embedded}')
                # Force garbage collection
                gc.collect()
                # Clear GPU cache
                torch.cuda.empty_cache()
            embedding_list.append(chunk_embeddings)

            # Reset batch
            chunks_batched = 0
            chunk_batch = []

    # Add any remaining chunks to batch
    if chunks_batched > 0:
        chunk_embeddings = get_batch_embeddings(chunk_batch, tokenizer, model)
        embedding_list.append(chunk_embeddings)

    doc_embeddings_array = np.concatenate(embedding_list, axis=0)

    doc_embeddings_array = np.reshape(doc_embeddings_array, (-1, doc_embeddings_array.shape[-1]))

    # doc_embeddings_array = np.concatenate(embedding_list, axis=0)
    # # Reshape to 2D array where embedding dim is 2nd dim
    # doc_embeddings_array = np.reshape(doc_embeddings_array, (-1, doc_embeddings_array.shape[-1]))


    # Delete large objects
    del chunk_batch
    del embedding_list

    # Force garbage collection
    gc.collect()

    # Clear GPU cache
    torch.cuda.empty_cache()

    return doc_embeddings_array


def query_semantic(query, tokenizer, model, doc_embeddings_array, n_results=3):
    query_tokenized = tokenizer(f'query: {query}', max_length=512, padding=False, truncation=True, return_tensors='pt').to(get_device())
    outputs = model(**query_tokenized)
    query_embedding = average_pool(outputs.last_hidden_state, query_tokenized['attention_mask'])
    query_embedding = F.normalize(query_embedding, p=2, dim=1).detach().cpu().numpy()

    cos_sims = np.dot(doc_embeddings_array, query_embedding.T)
    cos_sims = cos_sims.flatten()

    top_indices = np.argsort(cos_sims)[-n_results:][::-1]

    return top_indices


def get_embeddings_array():
    doc_embeddings_array = np.load(EMBEDDINGS_ARRAY_NPY)
    return doc_embeddings_array


def get_embeddings_index():
    with open(EMBEDDING_INDEX_PICKLE, 'rb') as f:
        embedding_index = pickle.load(f)
        return embedding_index


if __name__ == '__main__':
    # Load docs
    vault = get_vault()
    logger.info(f'Vault length: {len(vault):,}')

    # Load tokenizer and model
    tokenizer, model = get_model_tuple()

    # Build and save embedding index and array
    embedding_index = build_embedding_index(vault)
    logger.info(f'Embedding index length: {len(embedding_index):,}')

    start_time = time.time()
    doc_embeddings_array = build_embedding_array(vault, tokenizer, model)
    logger.info(f'Time taken for embedding all chunks: {time.time() - start_time} seconds')

    assert len(embedding_index) == doc_embeddings_array.shape[0], 'Length of embedding index != embedding count'

    os.makedirs(os.path.dirname(EMBEDDING_INDEX_PICKLE), exist_ok=True)
    with open(EMBEDDING_INDEX_PICKLE, 'wb') as f:
        pickle.dump(embedding_index, f, protocol=pickle.HIGHEST_PROTOCOL)

    np.save(EMBEDDINGS_ARRAY_NPY, doc_embeddings_array)

    assert len(embedding_index) == doc_embeddings_array.shape[0], 'Length of embedding index != number of embeddings'

