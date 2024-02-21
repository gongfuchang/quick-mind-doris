"""
Reads vault dictionary and indexes documents into an opensearch index.
"""
import pickle

from opensearchpy import OpenSearch, RequestsHttpConnection
from opensearchpy.helpers import bulk
from build_vault_dict import get_doc_dict
from src.logger import logger
from datetime import datetime
INDEX_NAME = 'doris-vault'


def get_opensearch(host: str = 'localhost') -> OpenSearch:
    """Create an opensearch client.

    Args:
        host: Name of opensearch host. Defaults to 'localhost'.

    Returns:
        Opensearch client
    """
    port = 9200
    auth = ('admin', 'admin')

    client = OpenSearch(
        hosts=[{'host': host, 'port': port}],
        connection_class=RequestsHttpConnection,
        http_compress=True,  # enables gzip compression for request bodies
        http_auth=auth,
        use_ssl=True,
        verify_certs=False,
        ssl_assert_hostname=False,
        ssl_show_warn=False
    )
    return client


def create_index(client: OpenSearch, index_name: str) -> None:
    """Create opensearch index with custom analyzer. The document structure is as follows:
doc_id: {
    title: doc title,
    title_display:
    language: zh-CN or en,
    version: none if no version provided, otherwise the version number,
    deprecated: none if not deprecated, otherwise the deprecated version number,
    content: doc content, the full text of the doc
}
    Args:
        client: Opensearch client
        index_name: Name of opensearch index
    """
    index_body = {
        'settings': {
            'analysis': {
                'char_filter': {
                    'html_strip_filter': {
                        'type': 'html_strip'
                    }
                }
            },
            'index': {
                'query': {
                    'default_field': 'content'
                }
            }
        },
        'mappings': {
            'properties': {
                'type': {'type': 'keyword'},
                'title': {'type': 'text', 'analyzer': 'ik_max_word'},
                'title_display': {'type': 'text', 'analyzer': 'ik_max_word'},
                'content': {'type': 'text', 'analyzer': 'ik_max_word'},
                'language': {'type': 'keyword'},
                'version_from': {'type': 'text'},
                'deprecated_from': {'type': 'text'},
                'website': {'type': 'keyword'},
                'create_time': {'type': 'date'},
                'update_time': {'type': 'date'},
            }
        }
    }

    # Iterate over all rows in df and add to index
    client.indices.delete(index=index_name, ignore_unavailable=True)
    client.indices.create(index=index_name, body=index_body)


def index_vault(vault: dict[str, dict], client: OpenSearch, index_name: str) -> None:
    """Index vault into opensearch index.The document structure is as follows:
doc_id: {
    title: doc title,
    title_display:
    language: zh-CN or en,
    version: none if no version provided, otherwise the version number,
    deprecated: none if not deprecated, otherwise the deprecated version number,
    content: doc content, the full text of the doc
}

    Args:
        vault: Doris helper doc vault dictionary
        client: Opensearch client
        index_name: Name of opensearh index
    """
    docs_indexed = 0
    chunks_indexed = 0
    docs = []

    for doc_id, doc in vault.items():
        title = doc['title']
        doc_type = 'chunk'  # support various types of documents
        title_display = doc['title_display']
        content = doc['content']
        language = doc['language']
        version_from = doc['version']
        deprecated_from = doc['deprecated']
        docs_indexed += 1
        if docs_indexed % 100 == 0:
            logger.info(f'Indexing {docs_indexed:,} documents')
        docs.append(
            {'_index': index_name, '_id': doc_id, 'title': title, 'title_display': title_display, 'type': doc_type,
             'content': content, 'version_from': version_from, 'deprecated_from': deprecated_from, 'language': language, 'create_time': datetime.now(), 'update_time': 0})

        chunks_indexed += 1

        if chunks_indexed % 200 == 0:
            try:
                bulk(client, docs)
                logger.info(f"Indexed chunk bulk from: {chunks_indexed}")
            except Exception as e:
                logger.error(f'Error in indexing({chunks_indexed}): {e}')
            finally:
                docs = []

    if chunks_indexed % 200 > 0:
        bulk(client, docs)

    logger.info(f'Totally Indexed {chunks_indexed:,} documents')


def query_opensearch(query: str, client: OpenSearch, index_name: str, version_from: str, n_results: int = 3) -> dict:
    """Custom query for opensearch index.

    Args:
        query: Query string
        client: OpenSearch client
        index_name: Name of opensearch index
        type: Type of chunk to query ('chunk', 'doc'). Defaults to 'chunk'.
        n_results: Number of results to return Defaults to 10.

    Returns:
        Result of opensearch query
    """
    query = {
        'size': n_results,
        'query': {
            'bool': {
                'should': [
                    {
                        'match': {
                            'title': {
                                'query': query,
                                'boost': 5.0  # Give matches in the 'title' field a higher score
                            }
                        }
                    },
                    {
                        'match': {
                            'title_display': {
                                'query': query,
                                'boost': 2.0  # Give matches in the 'chunk_header' field a higher score
                            }
                        }
                    },
                    {
                        'match': {
                            'content': {
                                'query': query
                            }
                        }
                    },
                ],
                # 'filter': [ TODO filter by version comparison
                #     {
                #         'term': {
                #             'version_from': version_from
                #         }
                #     }
                # ]
            }
        }
    }

    response = client.search(index=index_name, body=query)

    return response


if __name__ == '__main__':
    # Load vault dictionary
    doc_dict = get_doc_dict()
    logger.info(f'Doc length: {len(doc_dict):,}')

    # Create client
    client = get_opensearch("150.158.133.10")
    logger.info(f'Client: {client.info()}')

    # Create index
    create_index(client, INDEX_NAME)

    # Index vault
    index_vault(doc_dict, client, INDEX_NAME)

    # Count the number of documents in the index
    logger.info(f'Client cat count: {client.cat.count(index=INDEX_NAME, params={"format": "json"})}')

    # Test query
    test_query = '扩容节点'
    response = query_opensearch(test_query, client, INDEX_NAME, version_from='1.2', n_results=3)
    logger.info(f'Test query: {test_query}')
    logger.info(f'Response: {response}')
