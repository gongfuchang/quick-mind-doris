import random

from src.index.index_builder import IndexBuilder
from src.utils.model_util import get_model_tuple, get_device, average_pool
import numpy as np
import pandas as pd

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
from src.utils.file_util import get_file_path
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
import pprint
from langchain.text_splitter import MarkdownTextSplitter
from src.utils.display_util import print_dataframe

pp = pprint.PrettyPrinter(indent=4)
import random
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class ElasticIndexBuilder2(IndexBuilder):
    def __init__(self, vault: dict, doc: dict, tokenizer, model):
        super().__init__(vault, doc, tokenizer, model)
        self.client = Elasticsearch('https://150.158.133.10:9200',
            http_auth=('', ''),
            verify_certs=False
        )
        self.index_name = "qm2-index"

        self.model = SentenceTransformer('BAAI/bge-large-zh-v1.5')

        self.vault = self._build_vault()

        self.rerank_tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-reranker-large')
        self.rerank_model = AutoModelForSequenceClassification.from_pretrained('BAAI/bge-reranker-large')

    def _build_vault(self):
        vault_path = get_file_path('assets/doris-udf8.txt')
        with open(vault_path, 'r', encoding='utf-8', errors='replace') as f:
            markdown_text = f.read()
            markdown_splitter = MarkdownTextSplitter(chunk_size=150, chunk_overlap=0)
            docs = markdown_splitter.create_documents([markdown_text])
            vault = {}
            for doc in docs:
                content = self._clean_text(doc.dict()['page_content'])
                vault[f'id_{random.random()}'] = {'chunks': [content]}
                # vault['doc_id'] = ''
                # vault['title'] = ''

            print(f'vault length: {len(vault.keys())}')
            return vault

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
                    # "doc_id": {"type": "text"},
                    # "title": {"type": "text"},
                    "content": {"type": "text", "analyzer": "ik_max_word", "search_analyzer": "ik_smart"},
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

    def query_semantic(self, query: str, n_results=3):
        query_emb = self.model.encode(f"{self._get_query_prefix()}{query}", normalize_embeddings=True)

        payload = {
            "query": {
                "match": {
                    "content": {
                        "query": query,
                        "boost": 0.3
                    }
                }
            },
            "knn": [{
                "field": "chunk_emb",
                "query_vector": query_emb,
                "k": 10,
                "num_candidates": 15,
                "boost": 0.8
            }],
            "size": 10,
            "_source": ["chunk_id", "content"]
        }

        top_matches = self.client.search(index=self.index_name, body=payload)['hits']['hits']
        top_matches_df = pd.DataFrame(top_matches)
        top_matches_df['_rerank_score'] = top_matches_df['_source'].apply(lambda x: self.rerank(query, x['content']))


        # Define the new order of columns
        columns_order = ['_rerank_score', '_score', 'content']

        print(f'query: {query}')

        print('before rerank')
        top_matches_df['content'] = top_matches_df['_source'].apply(lambda x: x['content'])
        top_matches_df = top_matches_df.reindex(columns=columns_order)
        print_dataframe(top_matches_df)

        # Reindex the DataFrame
        sorted_df = top_matches_df.sort_values(by='_rerank_score', ascending=False)
        sorted_df = sorted_df.reindex(columns=columns_order)
        print('after rerank')
        print_dataframe(sorted_df)


    def get_embeddings_array(self):
        pass

    def _bulk(self, chunk_array: list, doc_embeddings_array: np.ndarray):
        docs = [
            {
                "_index": self.index_name,
                "_source": {
                    "chunk_id": chunk_array[index][0],
                    # "doc_id": chunk_array[index][1]['doc_id'],
                    # "title": chunk_array[index][1]['title'],
                    "content": ' '.join(chunk_array[index][1]['chunks']),
                    "chunk_emb": doc_embeddings_array[index].tolist()
                },
            }for index in range(len(chunk_array))]
        bulk(client=self.client, actions=docs)

    def rerank(self, query, answers):

        # model.eval()

        pairs = [[query, answers]]
        with torch.no_grad():
            inputs = self.rerank_tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
            scores = self.rerank_model(**inputs, return_dict=True).logits.view(-1, ).float()
            return scores[0]


if __name__ == '__main__':
    # Load docs
    vault = None
    doc = None
    # vault = get_vault_dict()
    # doc = get_doc_dict()
    # logger.info(f'Vault length: {len(vault):,}')

    # Load tokenizer and model
    tokenizer, model = get_model_tuple()
    # Build and save embedding index and array

    builder = ElasticIndexBuilder2(vault, doc, tokenizer, model)
    start_time = time.time()
    # builder.build()
    logger.info(f'Time taken for embedding all chunks: {time.time() - start_time} seconds')

    query_list = [
        '想请教一下大佬们，如果我想新增一列时间列，这张表里任一字段的数据更新，这个时间列都会自动更新时间，Doris支持这种操作么？还是说，得在数据导入前给每列数据加一个最新时间数据再进行导入Doris,',
        '如何在Apache Doris中实现节点缩容？',
        '在Doris中新建了一张表，在MySQL中可以查到，但在Doris中查不到，怎么办？',
        '在使用mysql -doris整库同步时，如果在mysql中增加字段，增加表，doris会自动同步过去吗？还是需要手动在doris中操作？',
        '测试了一下，增加字段可以同步，但增加表不行，修改字段也不同步，需要做什么操作吗？',
        '部分列更新 set enable_unique_key_partial_update=true 这个参数可以设置全局吗,',
        '大佬，doris可以把map结构的数据，里面的k和v转化为 列吗,',
        '我之前json数据我创建为text了，现在想要修改为json类型的，发现报错，不知道怎么回事,',
        'doris可以修改列的数据类型嘛,',
        '@Petrichor  大佬您好， unuqie模型可以修改列名称吗？ doris 1.2,',
        'Doris是否支持RAID？ ',
        'Doris支持哪些云存储服务？',
        '如何在Apache Doris中创建外部表访问SqlServer？有参考样例可以看吗？',
        'Apache Doris的OUTFILE支持导出阿里的OSS吗？',
        '在执行BACKUP SNAPSHOT后，通过show backup查看备份状态时，报错ERROR 2020 (HY000): Got packet bigger than bytes，有遇到过这个问题的吗？',
        'Doris中对分区数有没有限制？一张表最大能添加多少个分区？',
        '对历史数据如果要搞分区有什么好的解决方案吗？',
        'Doris 2.x版本的FE支持切换catalog吗？',
        '我使用insert into values语句插入了8819条数据，但在Apache Doris库里只查询到了7400多条数据，日志只有内存发生gc的告警。有遇到过这种情况吗？',
        '各位大佬，uniq key 或者那几个模型，对key的要求是不是不能用text之类的啊,我们要是主键包含text类型的应该怎么做啊,,',
        'Doris目前可以热挂载磁盘吗？',
        'routine load 作业配置了max_filter_ratio属性，暂停后能将这个属性给删除吗',
        '使用delete删除数据会有什么影响吗？ ',
        'Doris的BE挂的时候如果有数据通过streamload方式入库，这些数据丢失了，Doris有没有机制还原的？',
        # '新增 时间列  表 数据更新 导入',
        '时间列 数据 导入',
    ]
    for query in query_list:
        result = builder.query_semantic(query)



    # result = builder.query_semantic('bitor')

