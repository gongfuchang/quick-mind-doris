from src.index.index_builder import IndexBuilder
from src.utils.model_util import get_model_tuple, get_device, average_pool
import numpy as np
import pandas as pd
import re
import uuid
import time
from src.logger import logger
from sentence_transformers import SentenceTransformer
import jieba
import jieba.analyse
import pickle
import os
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from src.utils.model_util import get_device
import urllib3
from src.utils.file_util import get_file_path
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
import pprint
from langchain.text_splitter import MarkdownTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

from src.utils.display_util import print_dataframe

pp = pprint.PrettyPrinter(indent=4)
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

idf_path = get_file_path('data/idf/doris-idf.txt')
dict_path = get_file_path('data/idf/doris-dict.txt')
stopw_path = get_file_path('data/idf/hit_stopwords.txt')
DATA_VAULT_DICT_PICKLE = get_file_path('data/vault_dict.pickle')

class ElasticIndexBuilder(IndexBuilder):
    def __init__(self, vault: dict, doc: dict, tokenizer, model):
        super().__init__(vault, doc, tokenizer, model)
        self.client = Elasticsearch('https://150.158.133.10:9200',
            basic_auth=('elastic', ''),
            verify_certs=False
        )
        self.index_name = "qm2-index"

        self.model = SentenceTransformer('BAAI/bge-large-zh-v1.5')
        self.model.eval()
        self.model.encode("test encoding", device=get_device(), normalize_embeddings=True)


        self.rerank_tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-reranker-large')
        self.rerank_model = AutoModelForSequenceClassification.from_pretrained('BAAI/bge-reranker-large')
        self.rerank_model.to(get_device())
        self.rerank_model.eval()

        jieba.load_userdict(dict_path)
        jieba.analyse.set_stop_words(stopw_path)
        # jieba.analyse.set_idf_path(idf_path)

    def build_vault(self):
        vault_path = get_file_path('assets/doris-udf8.txt')
        with open(vault_path, 'r', encoding='utf-8', errors='replace') as f:
            texts = f.read()
            # splitter = MarkdownTextSplitter(chunk_size=150, chunk_overlap=0)
            # docs = splitter.create_documents([texts])
            splitter = RecursiveCharacterTextSplitter(
                # Set a really small chunk size, just to show.
                chunk_size=256,
                chunk_overlap=20,
                length_function=len,
            )
            docs = splitter.create_documents([texts])
            vault = {}
            for doc in docs:
                # content = self._clean_text(doc.dict()['page_content'])
                unique_id = str(uuid.uuid4()).replace('-', '')
                vault[unique_id] = {'chunks': [doc.dict()['page_content']]}
                # vault['doc_id'] = ''
                # vault['title'] = ''

            print(f'vault length: {len(vault.keys())}')
        os.makedirs(os.path.dirname(DATA_VAULT_DICT_PICKLE), exist_ok=True)
        with open(DATA_VAULT_DICT_PICKLE, 'wb+') as f:
            pickle.dump(vault, f, protocol=pickle.HIGHEST_PROTOCOL)

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
        # Define the regex pattern for 'Apache Doris' or 'Doris'
        pattern = r'Apache\s*Doris|Doris'
        # Use re.sub to replace the pattern with an empty string
        bm25_query = re.sub(pattern, '', query, flags= re.IGNORECASE)

        # for long query, extract top 5 keywords
        if len(bm25_query) > 50:
            start_time = time.time()
            query_tags = jieba.analyse.extract_tags(bm25_query, withWeight=True, topK=5)
            # query_tags = jieba.analyse.extract_tags(bm25_query,
            #   topK=10,
            #   withWeight=True,
            #   # allowPOS=('ns', 'n', 'vn', 'v')
            # )
            print("Tags:", query_tags)
            bm25_query = ' '.join([tag[0] for tag in query_tags])
            logger.info(f'[query extract tags cost]: {time.time() - start_time} seconds')


        start_time = time.time()
        query_emb = self.model.encode(f"{self._get_query_prefix()}{query}", device=get_device(), batch_size=20, normalize_embeddings=True)
        # query_emb = self.model.encode(f"{self._get_query_prefix()}{bm25_query}", device=get_device(), batch_size=20, normalize_embeddings=True)
        logger.info(f'[query embedding cost]: {time.time() - start_time} seconds')
        start_time = time.time()
        payload = {
            "query": {
                "match": {
                    "content": {
                        "query": bm25_query,
                        "boost": 0.3
                    }
                }
            },
            "knn": [{
                "field": "chunk_emb",
                "query_vector": query_emb,
                "k": 10,
                "num_candidates": 20,
                "boost": 0.8
            }],
            "size": 15,
            "_source": ["chunk_id", "content"]
        }

        start_time = time.time()
        top_matches = self.client.search(index=self.index_name, body=payload)['hits']['hits']
        logger.info(f'[index search cost]: {time.time() - start_time} seconds')
        top_matches_df = pd.DataFrame(top_matches)

        # Define the new order of columns
        columns_order = ['_rerank_score', '_score', 'content', 'chunk_id']

        print(f'query: {query}')

        print('before rerank')
        top_matches_df['content'] = top_matches_df['_source'].apply(lambda x: x['content'])
        top_matches_df['chunk_id'] = top_matches_df['_source'].apply(lambda x: x['chunk_id'])

        top_matches_df = top_matches_df.reindex(columns=columns_order)
        print_dataframe(top_matches_df)
        # Drop duplicates based on 'content' column
        top_matches_df = top_matches_df.drop_duplicates(subset='content')

        # 默认第一个top1是最相关的，重排剩下的 2-10，然后按照重排的结果排序
        top1_df = pd.DataFrame(top_matches_df.iloc[0]).transpose()  # Get the top 1 item
        remaining_df = top_matches_df.iloc[1:]  # Get the remaining items
        # remaining_df['_rerank_score'] = remaining_df['content'].apply(lambda x: self.rerank(query, x))
        # Step 1: Extract 'content' column and convert it into a list
        content_list = remaining_df['content'].tolist()
        # Step 2: Pass this list to the 'self.rerank(query, array)' function
        scores = self.rerank(query, content_list)
        # Step 3: Add this list as a new column to the 'remaining_df' DataFrame

        remaining_df.loc[:, '_rerank_score'] = scores
        start_time = time.time()
        logger.info(f'[rerank cost]: {time.time() - start_time} seconds')

        # Reindex the DataFrame
        remaining_sorted_df = remaining_df.sort_values(by='_rerank_score', ascending=False)
        remaining_sorted_df = remaining_sorted_df.reindex(columns=columns_order)
        final_sorted_df = pd.concat([top1_df, remaining_sorted_df])
        print('after rerank')
        print_dataframe(final_sorted_df)
        # read the top n chunks with chunk_id
        top_hits = final_sorted_df['chunk_id'].tolist()[:n_results]

        contexts = self._get_related_chunks_from_hits(top_hits, 5, True)
        print('Top hits contexts:\n' + '-' * 50)
        for index, context in enumerate(contexts):
            lines_num = context['content'].count('\n')
            print(f"{'*' * 30}[{index+1}]-lines[{lines_num}]{'*' * 30}\n")
            print(context['content'])
        return contexts

    def _get_related_chunks_from_hits(self, hits: list[str], max_lines: int = 3, highlighted: bool = False) -> list[dict]:
        chunks = []
        chunk_id_list = list(self.vault.keys())
        chunks_count = len(chunk_id_list)
        for chunk_id in hits:
            index = chunk_id_list.index(chunk_id)
            up_index = index - 1
            down_index = index + 1
            lines = 0
            chunk = {'content': ''.join(self.vault[chunk_id]['chunks'])}
            if highlighted:
                chunk['content'] = f"【{ chunk['content'] }】"

            content_list = [chunk['content']]
            # Look up
            paragraph_level = None
            while up_index > 0 and lines < max_lines:
                chunk_id = chunk_id_list[up_index]
                chunk_content = ' '.join(self.vault[chunk_id]['chunks'])
                match_result = re.match(r'^#+ ', chunk_content)
                if match_result:  # Paragraph start
                    paragraph_level = len(match_result.group(0).strip()) # 比如 “##” 或者 “###” 等
                    if paragraph_level <= 2:  # 2 is the level of the title
                        break
                content_list.insert(0, chunk_content)
                lines += 1
                up_index -= 1
            # Look down
            lines = 0
            while down_index < chunks_count and lines < max_lines:
                chunk_id = chunk_id_list[down_index]
                chunk_content = ' '.join(self.vault[chunk_id]['chunks'])
                match_result = re.match(r'^#+ ', chunk_content)
                if match_result:  # Paragraph start
                    curr_para_level = len(match_result.group(0).strip())
                    if curr_para_level <= 2:
                        break
                    if paragraph_level is not None and curr_para_level < paragraph_level:
                        break
                content_list.append(chunk_content)
                lines += 1
                down_index += 1
            chunk['content'] = '\n'.join(content_list)
            chunks.append(chunk)

        return chunks

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
                    "content": ' '.join([self._clean_text(txt) for txt in chunk_array[index][1]['chunks']]),
                    "chunk_emb": doc_embeddings_array[index].tolist(),
                },
            }for index in range(len(chunk_array))]
        bulk(client=self.client, actions=docs)

    def rerank(self, query: str, answers: list[str]):

        # model.eval()

        pairs = [[query, answer] for answer in answers]
        with torch.no_grad():
            start_time = time.time()
            inputs = self.rerank_tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
            inputs = {key: val.to(get_device()) for key, val in inputs.items()}
            logger.info(f'[rerank tokenization cost]: {time.time() - start_time} seconds')
            start_time = time.time()
            scores = self.rerank_model(**inputs, return_dict=True).logits.view(-1, ).float()
            logger.info(f'[rerank model cost]: {time.time() - start_time} seconds')
            return scores[0].item()


if __name__ == '__main__':
    # Load docs
    vault = None
    doc = None

    # doc = get_doc_dict()
    # logger.info(f'Vault length: {len(vault):,}')

    # Load tokenizer and model
    tokenizer, model = get_model_tuple()
    # Build and save embedding index and array
    builder = ElasticIndexBuilder(vault, doc, tokenizer, model)
    start_time = time.time()
    # builder.build_vault()
    builder.vault = pickle.load(open(DATA_VAULT_DICT_PICKLE, 'rb'))
    # builder.build()
    logger.info(f'Time taken for embedding all chunks: {time.time() - start_time} seconds')

    query_list = [
        '想请教一下大佬们，如果我想新增一列时间列，这张表里任一字段的数据更新，这个时间列都会自动更新时间，Doris支持这种操作么？还是说，得在数据导入前给每列数据加一个最新时间数据再进行导入Doris,',
        '如何在Apache Doris中实现节点缩容？',
        # '在Doris中新建了一张表，在MySQL中可以查到，但在Doris中查不到，怎么办？',
        # '在使用mysql -doris整库同步时，如果在mysql中增加字段，增加表，doris会自动同步过去吗？还是需要手动在doris中操作？',
        # '测试了一下，增加字段可以同步，但增加表不行，修改字段也不同步，需要做什么操作吗？',
        # '部分列更新 set enable_unique_key_partial_update=true 这个参数可以设置全局吗,',
        # '大佬，doris可以把map结构的数据，里面的k和v转化为 列吗,',
        # '我之前json数据我创建为text了，现在想要修改为json类型的，发现报错，不知道怎么回事,',
        # 'doris可以修改列的数据类型嘛,',
        # '@Petrichor  大佬您好， unuqie模型可以修改列名称吗？ doris 1.2,',
        'Doris是否支持RAID？ ',
        'Doris支持哪些云存储服务？',
        # '如何在Apache Doris中创建外部表访问SqlServer？有参考样例可以看吗？',
        # 'Apache Doris的OUTFILE支持导出阿里的OSS吗？',
        # '在执行BACKUP SNAPSHOT后，通过show backup查看备份状态时，报错ERROR 2020 (HY000): Got packet bigger than bytes，有遇到过这个问题的吗？',
        # 'Doris中对分区数有没有限制？一张表最大能添加多少个分区？',
        # '对历史数据如果要搞分区有什么好的解决方案吗？',
        # 'Doris 2.x版本的FE支持切换catalog吗？',
        # '我使用insert into values语句插入了8819条数据，但在Apache Doris库里只查询到了7400多条数据，日志只有内存发生gc的告警。有遇到过这种情况吗？',
        # '各位大佬，uniq key 或者那几个模型，对key的要求是不是不能用text之类的啊,我们要是主键包含text类型的应该怎么做啊,,',
        'Doris目前可以热挂载磁盘吗？',
        # 'routine load 作业配置了max_filter_ratio属性，暂停后能将这个属性给删除吗',
        # '使用delete删除数据会有什么影响吗？ ',
        # 'Doris的BE挂的时候如果有数据通过streamload方式入库，这些数据丢失了，Doris有没有机制还原的？',
        # "BITOR"
    ]
    for query in query_list:
        result = builder.query_semantic(query, 5)



    # result = builder.query_semantic('bitor')

