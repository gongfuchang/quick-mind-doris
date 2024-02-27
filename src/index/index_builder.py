"""
Reads vault dictionary, creates embeddings for each chunk, and creates a semantic index.
"""

import time
from typing import List
import torch
import gc
import os
import numpy as np
import torch.nn.functional as F

from src.logger import logger
from src.utils.model_util import get_model_tuple, get_device, average_pool

from abc import ABC, abstractmethod
from src.utils.file_util import get_file_path
import pickle
import re

EMBEDDING_INDEX_PICKLE = get_file_path('data/embedding/embedding_index.pickle')
class IndexBuilder(ABC):
    def __init__(self, vault: dict, doc: dict, tokenizer, model):
        """Initialize index builder
        Args:
            vault: Dictionary of vault documents: chunk_id: {doc_id: doc_id, chunks: chunks}
            tokenizer: Tokenizer to tokenize documents; should be compatible with model
            model: Model to embed documents
        """
        self.index_mapping = None
        self.embeddings_array = None

        self.vault = vault
        self.doc = doc
        self.tokenizer = tokenizer
        self.model = model

    def _get_query_prefix(self):
        return 'query: ' if self.model is not None and 'e5-small' in self.model.name_or_path else ''

    def _get_passage_prefix(self):
        return 'passage: ' if self.model is not None and 'e5-small' in self.model.name_or_path else ''

    def _clean_text(self, text):
        # Replace \n, \t, \r with space
        text = re.sub(r'[\n\t\r]', ' ', text.strip())
        # Replace multiple spaces with one space
        text = re.sub(r' +', ' ', text)
        return text
    def _vectorize(self, document_batch: List[str]) -> List[np.ndarray]:
        """Embed a batch of documents

        Args:
            document_batch: List of documents to embed
            tokenizer: Tokenizer to tokenize documents; should be compatible with model
            model: Model to embed documents

        Returns:
            List of document embeddings
        """

        docs_tokenized = self.tokenizer(document_batch, max_length=512, padding=True, truncation=True,
                                        return_tensors='pt')
        docs_tokenized = {key: val.to(get_device()) for key, val in docs_tokenized.items()}
        outputs = self.model(**docs_tokenized)
        embeddings = average_pool(outputs.last_hidden_state, docs_tokenized['attention_mask'])
        embeddings_normed = F.normalize(embeddings, p=2, dim=1)  # Normalize embeddings for downstream cosine similarity

        return embeddings_normed.detach().cpu().numpy()

    def build_index_mapping(self, vault: dict) -> dict[int, str]:
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

        os.makedirs(os.path.dirname(EMBEDDING_INDEX_PICKLE), exist_ok=True)
        with open(EMBEDDING_INDEX_PICKLE, 'wb') as f:
            pickle.dump(embedding_index, f, protocol=pickle.HIGHEST_PROTOCOL)

    def get_index_mapping(self):
        if self.index_mapping is None:
            if not os.path.exists(EMBEDDING_INDEX_PICKLE):
                return None
            with open(EMBEDDING_INDEX_PICKLE, 'rb') as f:
                self.index_mapping = pickle.load(f)
        return self.index_mapping

    def clean_up_before_build(self):
        """Clean up before building index"""
        pass

    def build(self, batch_size: int = 20, bulk_size: int = 200) -> np.ndarray:
        """Embedding all document chunks and return embedding array

        Args:
            batch_size: Size of document batch to embed each time. Defaults to 20.
            bulk_size: Size of document to bulk to index. Defaults to 200.

        Returns:
            Numpy array of n_chunks x embedding-dim document embeddings
        """
        self.clean_up_before_build()
        chunk_embedded = 0
        chunk_batch = []
        chunks_batched = 0
        embedding_list = []  # [{id: 'chunk_id', embeddings: []}]

        for chunk_id, chunk_entry in self.vault.items():
            # Get path and chunks
            if chunk_embedded > 0 and chunk_embedded % bulk_size == 0:
                logger.info(f'Embedding document chunk: {chunk_id} ({chunk_embedded:,})')
                self._bulk(list(self.vault.items())[chunk_embedded - len(embedding_list): chunk_embedded], embedding_list)
                embedding_list = []

            chunk_embedded += 1
            chunks = chunk_entry['chunks']  # chunk_folder is a list of chunks
            processed_chunk = self._get_passage_prefix() + ' '.join([self._clean_text(c) for c in chunks])  # Remove extra whitespace and add prefix

            # logger.info(f'Chunk: {processed_chunk}')
            chunk_batch.append(processed_chunk)  # Add chunk to batch
            chunks_batched += 1

            if chunks_batched % batch_size == 0:
                # Compute embeddings in batch and append to list of embeddings
                start_time = time.time()
                chunk_embeddings = self._vectorize(chunk_batch)
                if chunk_embedded % 1000 == 0:
                    logger.info(
                        f'Time taken for get_batch_embeddings<{chunk_embedded} - {chunk_batch}>: {time.time() - start_time} seconds')
                    logger.info(f'gc collected and GPU cache cleared at {chunk_embedded}')
                    # Force garbage collection
                    gc.collect()
                    # Clear GPU cache
                    torch.cuda.empty_cache()
                embedding_list.extend(chunk_embeddings)

                # Reset batch
                chunks_batched = 0
                chunk_batch = []

        # Add any remaining chunks to batch
        if chunks_batched > 0:
            chunk_embeddings = self._vectorize(chunk_batch)
            embedding_list.extend(chunk_embeddings)

        if len(embedding_list) > 0:
            logger.info(f'Finally process embedding document chunk in size of: {len(embedding_list)}')
            self._bulk(list(self.vault.items())[chunk_embedded - len(embedding_list): chunk_embedded], embedding_list)

        # Delete large objects
        del chunk_batch
        del embedding_list

        # Force garbage collection
        gc.collect()

        # Clear GPU/CPU cache
        torch.cuda.empty_cache()



    @abstractmethod
    def _bulk(self, chunk_array: list, doc_embeddings_array: np.ndarray):
        """
        Bulk insert document embeddings to index
        @param chunk_array: in form of tuple list [(chunk_id, {dock_id: doc_id, chunks: chunks, title: title})]
        @param doc_embeddings_array:
        @return:
        """
        pass
    @abstractmethod
    def query_semantic(self, query: str, n_results=3):
        pass

    @abstractmethod
    def get_embeddings_array(self):
        pass

    def _get_related_chunks_from_hits(self, hits: List[dict], max_lines: int = 30) -> List[dict]:
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
            max_lines: Maximum lines to allow in chunks. Defaults to 10.

        Returns:
            List of chunks for retrieval-augmented generation, in form of [{
                content: ''
                title: ''
            }].
        """
        index_mapping = self.get_index_mapping()

        chunks = []
        docs = {}
        for hit_index in hits:
            chunk_id = index_mapping[hit_index]
            # 如果所在的 doc 内容篇幅较小，直接返回 doc 全篇内容
            doc_id = self.vault[chunk_id]['doc_id']
            doc_content = self.doc[doc_id]['content']
            if doc_content is not None and len(doc_content) < 1024:
                docs.update({doc_id: {'content': doc_content, 'title': ''}})
                continue

            # 按顺序查找上下文：获取到 chunk 然后遍历 chunk 里的每一行，直到找到下一个标题，或者达到最大行数
            up_index = hit_index - 1
            down_index = hit_index + 1
            lines = 0
            chunk = {'title': self.vault[chunk_id]['title'], 'content': ''.join(self.vault[chunk_id]['chunks'])}
            content_list = [chunk['content']]
            # Look up
            paragraph_level = None
            while up_index in index_mapping and lines < max_lines:
                chunk_id = index_mapping[up_index]
                upper_lines = self.vault[chunk_id]['chunks']
                for line in reversed(upper_lines):
                    lines += 1
                    content_list.insert(0, line)
                    match_result = re.match(r'^#+ ', line)
                    if not match_result:
                        continue
                    curr_para_level = len(match_result.group(0).strip())
                    if match_result and curr_para_level <= 1:
                        # 如果找到了标题，就退出
                        paragraph_level = curr_para_level
                        break

                # 检查哨兵
                if paragraph_level is not None:
                    break
                up_index -= 1

            # Look down
            while down_index in index_mapping and lines < max_lines:
                chunk_id = index_mapping[down_index]
                down_lines = self.vault[chunk_id]['chunks']
                for line in down_lines:
                    lines += 1
                    content_list.append(line) # 可能会多加一行，暂时不处理
                    match_result = re.match(r'^#+ ', line)
                    if not match_result:
                        continue
                    curr_para_level = len(match_result.group(0).strip())
                    if paragraph_level is not None and curr_para_level >= paragraph_level:
                        break

                down_index += 1
            chunk['content'] = '\n'.join(content_list)
            chunks.append(chunk)
        chunks.extend(list(docs.values()))
        return chunks