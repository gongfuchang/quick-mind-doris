import unittest
import numpy as np
from unittest.mock import patch, MagicMock
from src.index.numpy_index_builder import NumpyIndexBuilder

class TestNumpyIndexBuilder(unittest.TestCase):
    @patch.object(NumpyIndexBuilder, '_vectorize')
    def should_return_top_indices_when_query_is_given(self, mock_vectorize):
        mock_vectorize.return_value = np.array([0.5, 0.5, 0.5])
        builder = NumpyIndexBuilder()
        doc_embeddings_array = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
        result = builder.query_semantic('test query', doc_embeddings_array, 2)
        self.assertEqual(result.tolist(), [1, 2])

    def should_return_empty_array_when_no_query_is_given(self):
        builder = NumpyIndexBuilder()
        doc_embeddings_array = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
        result = builder.query_semantic('', doc_embeddings_array, 2)
        self.assertEqual(result.tolist(), [])

    def should_return_empty_array_when_doc_embeddings_array_is_empty(self):
        builder = NumpyIndexBuilder()
        doc_embeddings_array = np.array([])
        result = builder.query_semantic('test query', doc_embeddings_array, 2)
        self.assertEqual(result.tolist(), [])

if __name__ == '__main__':
    unittest.main()