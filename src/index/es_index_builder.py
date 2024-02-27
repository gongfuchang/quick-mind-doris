import random

from src.index.index_builder import IndexBuilder
from src.utils.model_util import get_model_tuple, get_device, average_pool
import numpy as np

from src.prep.build_vault_dict import get_vault_dict, get_doc_dict
import os
import pickle
import time
from src.logger import logger
from sentence_transformers import SentenceTransformer

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from src.utils.model_util import get_device
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
import pprint
pp = pprint.PrettyPrinter(indent=4)
class ElasticIndexBuilder(IndexBuilder):
    def __init__(self, vault: dict, doc: dict, tokenizer, model):
        super().__init__(vault, doc, tokenizer, model)
        self.client = Elasticsearch('https://150.158.133.10:9200',
            http_auth=('', ''),
            verify_certs=False
        )
        self.index_name = "qm-index"

        self.model = SentenceTransformer('BAAI/bge-large-zh-v1.5')


    def _vectorize(self, document_batch: list[str]) -> list[np.ndarray]:
        return self.model.encode(document_batch, device=get_device(), show_progress_bar=False, normalize_embeddings=True)

    def _get_query_prefix(self):
        return '为这个句子生成表示以用于检索相关文章：'

    def _get_passage_prefix(self):
        return ''
    def clean_up_before_build(self):
        if self.client.indices.exists(index=self.index_name):
            self.client.indices.delete(index=self.index_name, ignore_unavailable=True)

        index_body = {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 1
            },
            "mappings": {
                "properties": {
                    "chunk_id": {"type": "text"},
                    "doc_id": {"type": "text"},
                    "title": {"type": "text"},
                    "content": {"type": "text"},
                    "chunk_emb": {
                        "type": "dense_vector",
                        "dims": 1024,
                        "index": True,
                        "similarity": "dot_product"
                    }
                }
            }
        }
        self.client.indices.create(index=self.index_name, body=index_body)

    def query_semantic(self, query, n_results=3):
        instruction = f"{self._get_query_prefix()}{query}"
        query_emb = self.model.encode(instruction + query, normalize_embeddings=True)

        payload = {
            # "query": {
            #     "match": {
            #         "title": {
            #             "query": query,
            #             "boost": 0.2
            #         }
            #     }
            # },
            "knn": [{
                "field": "chunk_emb",
                "query_vector": query_emb,
                "k": 10,
                "num_candidates": 20,
                "boost": 0.8
            }],
            "size": 10,
            "_source": ["chunk_id", "doc_id", "title", "content"]
        }

        top_matches = self.client.search(index=self.index_name, body=payload)['hits']['hits']
        pp.pprint(top_matches)
        top_ids = [m['_source']['chunk_id'] for m in top_matches]
        reverse_mapping = {v: k for k, v in self.get_index_mapping().items()}
        top_indices = [reverse_mapping[i] for i in top_ids if i in reverse_mapping]

        return self._get_related_chunks_from_hits(top_indices)

    def get_embeddings_array(self):
        pass

    def _bulk(self, chunk_array: list, doc_embeddings_array: np.ndarray):
        docs = [
            {
                "_index": self.index_name,
                "_source": {
                    "chunk_id": chunk_array[index][0],
                    "doc_id": chunk_array[index][1]['doc_id'],
                    "title": chunk_array[index][1]['title'],
                    "content": ' '.join(chunk_array[index][1]['chunks']),
                    "chunk_emb": doc_embeddings_array[index].tolist()
                },
            }for index in range(len(chunk_array))]
        bulk(client=self.client, actions=docs)


if __name__ == '__main__':
    # Load docs
    vault = None
    doc = None
    vault = get_vault_dict()
    doc = get_doc_dict()
    # logger.info(f'Vault length: {len(vault):,}')

    # Load tokenizer and model
    tokenizer, model = get_model_tuple()
    # Build and save embedding index and array

    builder = ElasticIndexBuilder(vault, doc, tokenizer, model)
    start_time = time.time()
    # builder.build()
    logger.info(f'Time taken for embedding all chunks: {time.time() - start_time} seconds')

    query_list = [
            'Apache Doris的OUTFILE支持导出阿里的OSS吗？',
            # '如何在Apache Doris中实现节点缩容？',
            # '想请教一下大佬们，如果我想新增一列时间列，这张表里任一字段的数据更新，这个时间列都会自动更新时间，Doris支持这种操作么？还是说，得在数据导入前给每列数据加一个最新时间数据再进行导入Doris,'
    ]
    for query in query_list:
        result = builder.query_semantic(query)



    # result = builder.query_semantic('bitor')

