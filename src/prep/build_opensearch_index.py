"""
Reads vault dictionary and indexes documents into an opensearch index.
"""
import pickle

from opensearchpy import OpenSearch, RequestsHttpConnection
from opensearchpy.helpers import bulk

from src.logger import logger

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
    """Create opensearch index with custom analyzer

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
                    'default_field': 'chunk'
                }
            }
        },
        'mappings': {
            'properties': {
                'title': {'type': 'text', 'analyzer': 'ik_max_word'},
                'type': {'type': 'keyword'},
                'path': {'type': 'keyword'},
                'chunk_header': {'type': 'text', 'analyzer': 'ik_max_word'},
                'chunk': {'type': 'text', 'analyzer': 'ik_max_word'},
                'author': {'type': 'keyword'},
                'website': {'type': 'keyword'},
                'create_time': {'type': 'date'},
                'update_time': {'type': 'date'},
                'copyright_type': {'type': 'integer'},
            }
        }
    }

    # Iterate over all rows in df and add to index
    client.indices.delete(index=index_name, ignore_unavailable=True)
    client.indices.create(index=index_name, body=index_body)


def index_vault(vault: dict[str, dict], client: OpenSearch, index_name: str) -> None:
    """Index vault into opensearch index

    Args:
        vault: Doris helper doc vault dictionary
        client: Opensearch client
        index_name: Name of opensearh index
    """
    docs_indexed = 0
    chunks_indexed = 0
    docs = []

    for chunk_id, doc in vault.items():
        path = doc['path']
        title = doc['title']
        doc_type = doc['type']
        chunk = doc['chunk']
        chunk_header = doc['chunk_header']
        docs_indexed += 1
        if docs_indexed % 100 == 0:
            logger.info(f'Indexing {chunk_id} - Path: {path} ({docs_indexed:,} docs)')

        docs.append({'_index': index_name, '_id': chunk_id, 'title': title, 'type': doc_type,
                     'path': path, 'chunk_header': chunk_header, 'chunk': chunk,
                     'author': '', 'website': '', 'create_time': 0, 'update_time': 0,
                     'copyright_type': 1})  # TODO
        chunks_indexed += 1

        if chunks_indexed % 2000 == 0:
            bulk(client, docs)
            docs = []

    if chunks_indexed % 200 > 0:
        bulk(client, docs)

    logger.info(f'Indexed {chunks_indexed:,} chunks (including full documents)')


def query_opensearch(query: str, client: OpenSearch, index_name: str, type: str = 'chunk', n_results: int = 3) -> dict:
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
                            'chunk_header': {
                                'query': query,
                                'boost': 2.0  # Give matches in the 'chunk_header' field a higher score
                            }
                        }
                    },
                    {
                        'match': {
                            'chunk': {
                                'query': query
                            }
                        }
                    },
                    {
                        'match': {
                            'author': {
                                'query': query
                            }
                        }
                    },
                    # {
                    #     'match': {
                    #         'author': {
                    #             'website': query
                    #         }
                    #     }
                    # }
                ],
                'filter': [
                    {
                        'term': {
                            'type': 'chunk'
                        }
                    }
                ]
            }
        }
    }  # type: ignore

    response = client.search(index=index_name, body=query)

    return response


if __name__ == '__main__':
    # Load vault dictionary
    vault = pickle.load(open('data/vault_dict.pickle', 'rb'))
    logger.info(f'Vault length: {len(vault):,}')

    # Create client
    client = get_opensearch()
    logger.info(f'Client: {client.info()}')

    # Create index
    create_index(client, INDEX_NAME)

    # Index vault
    index_vault(vault, client, INDEX_NAME)

    # Count the number of documents in the index
    logger.info(f'Client cat count: {client.cat.count(index=INDEX_NAME, params={"format": "json"})}')

    # Test query
    test_query = '居文君冠军, 侯逸凡亚军'
    response = query_opensearch(test_query, client, INDEX_NAME, type='doc', n_results=3)
    logger.info(f'Test query: {test_query}')
    logger.info(f'Response: {response}')
