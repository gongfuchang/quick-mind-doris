import random

from src.utils.file_util import get_file_path
from src.index.index_builder import IndexBuilder
from src.utils.model_util import get_model_tuple, get_device, average_pool
import numpy as np

from src.prep.build_vault_dict import get_vault_dict, get_doc_dict
import os
import pickle
import time
from src.logger import logger

from pinecone import Pinecone, PodSpec


class PineconeIndexBuilder(IndexBuilder):
    def __init__(self, vault: dict, doc: dict, tokenizer, model):
        super().__init__(vault, doc, tokenizer, model)
        self.client = Pinecone(api_key="")
        self.index_name = "quick-mind"
        self.index = None
        if self.index_name in self.client.list_indexes().names():
            self.index = self.client.Index(self.index_name)

        # self.vault = dict(list(self.vault.items())[:1])

    def clean_up_before_build(self):
        if self.index_name in self.client.list_indexes().names():
            self.client.delete_index(self.index_name)

        self.client.create_index(
            name=self.index_name,
            dimension=768,
            metric="cosine",
            spec=PodSpec(
                environment="gcp-starter"
            )
        )

    def query_semantic(self, query, n_results=3):
        query_embedding = self._vectorize([f'{self._get_query_prefix()}{query}'])

        top_matches = self.index.query(
            vector=query_embedding.tolist(),
            top_k=3,
            include_values=False,
            include_metadata=True
        )
        # TODO Read score filter value from configuration
        score_threshhold = 0.5
        filter_out = list(filter(lambda m: m['score'] <= score_threshhold, top_matches.get('matches')))
        logger.debug(f'query【{query}】')
        for out in filter_out:
            logger.debug(f'filter-out<score={out["score"]}>: {out}')
        top_matches = list(filter(lambda m: m['score'] > score_threshhold, top_matches.get('matches')))
        for mt in top_matches:
            logger.debug(f'filter-in<score={mt["score"]}>: {mt}')

        top_ids = [match['id'] for match in top_matches]
        reverse_mapping = {v: k for k, v in self.get_index_mapping().items()}
        top_indices = [reverse_mapping[i] for i in top_ids if i in reverse_mapping]

        return self._get_related_chunks_from_hits(top_indices)

    def get_embeddings_array(self):
        pass

    def _bulk(self, chunk_array: list, doc_embeddings_array: np.ndarray):
        vectors = [{
            "id": chunk_array[index][0],
            "values": doc_embeddings_array[index].tolist(),
            "metadata": chunk_array[index][1]
        } for index in range(len(chunk_array))]
        self.index.upsert(
            vectors=vectors
        )
        pass

    # def build(self):
    #     self.clean_up_before_build()
    #     import random
    #     arr = ['you shouldn\'t drink wine', 'you should not drink anything', 'you should clean up the guy',
    #            'I know the guy named mary', 'I know the guy named tom',
    #            '回家的路上看到一只小狗', '快乐的小狗从来不知道回家', '他今晚回家吗？', '爱情想要有一个家',
    #            '刘德华为啥不演坏人？', '牙膏只用中华为啥不行？', '华为2024年净盈利300亿']
    #     vectors = [{
    #         "id": str(random.random()),
    #         "values": self._vectorize(f'passage: {name}').tolist()[0],
    #         "metadata": {"content": f'{name}'}
    #     } for name in arr]
    #     self.index.upsert(vectors=vectors)


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

    builder = PineconeIndexBuilder(vault, doc, tokenizer, model)
    start_time = time.time()
    # builder.build()
    logger.info(f'Time taken for embedding all chunks: {time.time() - start_time} seconds')

    query_list = ['Apache Doris的OUTFILE支持导出阿里的OSS吗？',
                  '如何在Apache Doris中实现节点缩容？',
                  '想请教一下大佬们，如果我想新增一列时间列，这张表里任一字段的数据更新，这个时间列都会自动更新时间，Doris支持这种操作么？还是说，得在数据导入前给每列数据加一个最新时间数据再进行导入Doris,']
    for query in query_list:
        result = builder.query_semantic(query)
        # print(result)

    # result = builder.query_semantic('bitor')

