"""
Simple FastAPI app that queries opensearch and a semantic index for retrieval-augmented generation.
"""
import concurrent.futures
import json
import re
import traceback
from typing import Annotated, List, Generator, Optional
from typing import Dict

import numpy as np
import tiktoken
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, HTMLResponse, StreamingResponse
from pydantic import BaseModel

from src.logger import logger
# from src.prep.build_opensearch_index import (INDEX_NAME, get_opensearch,
#                                              query_opensearch)
from src.index.numpy_index_builder import NumpyIndexBuilder
from src.index.pinecone_index_builder import PineconeIndexBuilder
from src.index.es_index_builder import ElasticIndexBuilder
from src.prep.build_vault_dict import get_vault_dict, get_doc_dict
from src.utils.model_util import get_model_tuple, get_client
from src.utils.type_util import to_bool
from src.utils.prompt_util import get_more_questions_prompt, get_query_prompt
os_client = None
# Create opensearch client
# try:
#     os_client = get_opensearch('localhost')
# except ConnectionRefusedError:
#     os_client = get_opensearch('localhost')  # Change to 'localhost' if running locally
# logger.info(f'OS client initialized: {os_client.info()}')

# Build and save embedding index and array
tokenizer, model = get_model_tuple()
vault = get_vault_dict()
doc = get_doc_dict()
builder = ElasticIndexBuilder(vault, doc, tokenizer, model)
embedding_index = builder.get_index_mapping()

# If the user did not provide a query, we will use this default query.
_default_query = "如何快速开启 Apache Doris 之旅?"


class QueryModel(BaseModel):
    query: str
    generate_related_questions: Optional[bool] = True


class RAG(uvicorn.Server):
    """
    Retrieval-Augmented Generation for Apache Doris Helper QA.

    It uses the helper documents to obtain results based on user queries, and then uses
    LLM models to generate the answer as well as related questions. The results
    are then stored in a KV so that it can be retrieved later (TODO).
    """

    def __init__(self, config, llm_type: str):
        super().__init__(config)

        self.llm_type = llm_type

        # Load vault dictionary
        self.vault = get_vault_dict()
        logger.info(f'Vault loaded with {len(self.vault)} documents')

        # An executor to carry out async tasks, such as uploading to KV.
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.handler_max_concurrency * 2
        )
        # whether we should generate related questions.
        self.should_do_related_questions = to_bool(self.deployment_template["config"]["related_question"])

        self.app = FastAPI()
        self.app.get("/query")(self.query_function)
        self.app.get("/answer")(self.answer_function)

        # List of allowed origins. You can also allow all by using ["*"]
        origins = [
            "http://localhost",  # or whatever hosts you want to allow
        ]

        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    async def run(self, sockets=None):
        await super().run(sockets=sockets)

    def launch(self):
        uvicorn.run(app=self.app, port=self.config.port, host=self.config.host)

    # TODO using rdb to store
    deployment_template = {
        "llm_type": "GLM",  # default to GPT
        "config": {
            "search_engin": "",  # TODO support search engin
            "model": "gpt-4-0125-preview", #"gpt-3.5-turbo",
            # For all the search queries and results, we will use the KV store to
            # store them so that we can retrieve them later. Specify the name of the
            # KV here.
            "kv_name": "",  # TODO support kv restore
            # If set to true, will generate related questions. Otherwise, will not.
            "related_question": "true",
            "base_url": "https://api.openai-proxy.com"
        }
    }

    # It's just a bunch of api calls, so our own deployment can be made massively
    # concurrent.
    handler_max_concurrency = 16

    def get_related_questions(self, query, contexts):
        """
        Gets related questions based on the query and context.
        """

        def ask_related_questions(
                questions: Annotated[
                    List[str],
                    "related question to the original question and context."
                ]
        ):
            """
            ask further questions that are related to the input and output.
            """
            pass

        try:
            dt = self.deployment_template
            response = get_client(self.llm_type or dt["llm_type"], dt["config"]).create_completion(
                messages=[
                    {
                        "role": "system",
                        "content": get_more_questions_prompt().format(
                            context="\n\n".join([c["content"] for c in contexts])
                        ),
                    },
                    {
                        "role": "user",
                        "content": query,
                    },
                ],
                max_tokens = 512
                # tools=[{
                #     "type": "function",
                #     "function": get_tools_spec(ask_related_questions),
                # }]
            )
            related = response.choices[0].message.content
            logger.info(f"Related questions: {related}")
            # return [{'question': related}]
            return [{'question': q} for q in related.split('\n')]
            # related = response.choices[0].message.tool_calls[0].function.arguments
            # if isinstance(related, str):
            #     related = json.loads(related)

            # return related["questions"][:5]
        except Exception as e:
            # For any exceptions, we will just return an empty list.
            logger.error(
                "encountered error while generating related questions:"
                f" {e}\n{traceback.format_exc()}"
            )
            logger.exception("Exception in get_related_questions")
            return []

    def _raw_stream_response(
            self, contexts, llm_response, related_questions_future
    ) -> Generator[str, None, None]:
        """
        A generator that yields the raw stream response. You do not need to call
        this directly. Instead, use the stream_and_upload_to_kv which will also
        upload the response to KV.
        """
        # First, yield the contexts.
        yield json.dumps(contexts)
        yield "\n\n__LLM_RESPONSE__\n\n"
        # Second, yield the llm response.
        if not contexts:
            # Prepend a warning to the user
            yield (
                "(The search engine returned nothing for this query. Please take the"
                " answer with a grain of salt.)\n\n"
            )
        for chunk in llm_response:
            if chunk.choices:
                yield chunk.choices[0].delta.content or ""
        # Third, yield the related questions. If any error happens, we will just
        # return an empty list.
        if related_questions_future is not None:
            try:
                related_questions = related_questions_future.result()
                result = json.dumps(related_questions)
            except Exception as e:
                logger.error(f"encountered error: {e}\n{traceback.format_exc()}")
                result = "[]"
            yield "\n\n__RELATED_QUESTIONS__\n\n"
            yield result

    async def streamify(
            self, contexts, llm_response, related_questions_future
    ) -> Generator[str, None, None]:
        import asyncio
        """
        Streams the results of the query to the user.
        """
        all_yielded_results = []
        for result in self._raw_stream_response(
                contexts, llm_response, related_questions_future
        ):
            all_yielded_results.append(result)
            yield result
            await asyncio.sleep(0.05)

    def answer_function(self, query: str) -> HTMLResponse:
        return self.query_function(query, 'False', 'False', 'False')

    def query_function(
            self,
            query: str,
            generate_related_questions: str = 'True',
            stream: str = 'True',
            with_cite: str = 'True',
    ) -> Response:
        """
        Query the search engine and returns the response.

        The query can have the following fields:
            - query: the user query.
            - generate_related_questions: if set to false, will not generate related
                questions. Otherwise, will depend on the environment variable
                RELATED_QUESTIONS. Default: true.
        """

        # First, do a search query.
        query = query or _default_query
        generate_related_questions = to_bool(generate_related_questions)
        stream = to_bool(stream)
        with_cite = to_bool(with_cite)
        # Basic attack protection: remove "[INST]" or "[/INST]" from the query
        query = re.sub(r"\[/?INST\]", "", query)
        contexts = self._get_chunks(query)

        system_prompt = get_query_prompt(with_cite).format(
            context="\n\n".join(
                [(f"[[citation:{i + 1}]] {c['content']}" if with_cite else c['content']) for i, c in enumerate(contexts)]
            )
        )
        try:
            dt = self.deployment_template
            client = get_client(self.llm_type or dt["llm_type"], dt["config"])
            llm_response = client.create_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query},
                ],
                max_tokens=1024,
                temperature=0.65,
                stream=stream,
            )

            if self.should_do_related_questions and generate_related_questions:
                # While the answer is being generated, we can start generating
                # related questions as a future.
                related_questions_future = self.executor.submit(
                    self.get_related_questions, query, contexts
                )
            else:
                related_questions_future = None
        except Exception as e:
            logger.error(f"encountered error: {e}\n{traceback.format_exc()}")
            return HTMLResponse("Internal server error.", 503)

        if not stream:
            # If we are not streaming, we will just return the response as a JSON.
            # return HTMLResponse(llm_response.choices[0].message.content, 200)
            return {
                "question": query,
                "contexts": [ctx["content"] for ctx in contexts],
                "answer": llm_response.choices[0].message.content,
            }
        return StreamingResponse(
            self.streamify(
                contexts, llm_response, related_questions_future
            ),
            media_type="text/event-stream",
        )

    def _get_chunks(self, query: str):
        # Get hits from opensearch
        # os_response = query_opensearch(query, os_client, INDEX_NAME)
        # os_hits = parse_os_response(os_response)
        # logger.debug(f'OS hits: {os_hits}')

        # Get hits from semantic index
        related_context = builder.query_semantic(query)
        return related_context

    def _parse_os_response(self, response: dict) -> List[dict]:
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

    def _parse_semantic_response(self, indices: np.ndarray, embedding_index: Dict[int, str]) -> List[dict]:
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

    def _num_tokens_from_string(self, string: str, model_name: str) -> int:
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


import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument('--port', type=int, default=8000, help='Port number')
    parser.add_argument('--llm_type', type=str, default='GPT', help='LLM type, GPT or GLM')
    args = parser.parse_args()

    config = uvicorn.Config(app='main:app', host="0.0.0.0", port=args.port)
    rag = RAG(config, llm_type=args.llm_type)
    rag.launch()
