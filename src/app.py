"""
Simple FastAPI app that queries opensearch and a semantic index for retrieval-augmented generation.
"""
import concurrent.futures
import json
import os
import re
import threading
import traceback
from typing import Annotated, List, Generator, Optional
from typing import Dict

import numpy as np
import tiktoken
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from src.logger import logger
# from src.prep.build_opensearch_index import (INDEX_NAME, get_opensearch,
#                                              query_opensearch)
from src.prep.build_semantic_index import query_semantic, get_embeddings_index, get_embeddings_array
from src.prep.build_vault_dict import get_vault
from src.utils.model_util import get_model_tuple
from src.utils.type_util import to_bool

os_client = None
# Create opensearch client
# try:
#     os_client = get_opensearch('localhost')
# except ConnectionRefusedError:
#     os_client = get_opensearch('localhost')  # Change to 'localhost' if running locally
# logger.info(f'OS client initialized: {os_client.info()}')

# Load semantic index
doc_embeddings_array = get_embeddings_array()
embedding_index = get_embeddings_index()
tokenizer, model = get_model_tuple()

logger.info(f'Semantic index loaded with {len(embedding_index)} documents')

# If the user did not provide a query, we will use this default query.
_default_query = "如何快速开启 Apache Doris 之旅?"

is_test = False
_rag_query_text = """
You are a large language AI assistant built by QuickMind AI. You are given a user question, and please write clean, concise and accurate answer to the question. You will be given a set of related contexts to the question, each starting with a reference number like [[citation:x]], where x is a number. Please use the context and cite the context at the end of each sentence if applicable.

Your answer must be correct, accurate and written by an expert using an unbiased and professional tone. Please limit to 1024 tokens. Do not give any information that is not related to the question, and do not repeat. Say "information is missing on" followed by the related topic, if the given context do not provide sufficient information.

Please cite the contexts with the reference numbers, in the format [citation:x]. If a sentence comes from multiple contexts, please list all applicable citations, like [citation:3][citation:5]. Other than code and specific names and citations, your answer must be written in the same language as the question.

Use chinese if the question contains chinese characters otherwise use english instead.

Here are the set of contexts:

{context}

Remember, don't blindly repeat the contexts verbatim. And here is the user question:
"""

# A set of stop words to use - this is not a complete set, and you may want to
# add more given your observation.
stop_words = [
    "<|im_end|>",
    "[End]",
    "[end]",
    "\nReferences:\n",
    "\nSources:\n",
    "End.",
]



_more_questions_prompt = """
You are a helpful assistant that helps the user to ask Apache Doris related questions, based on user's original question and the related contexts. Please identify worthwhile topics that can be follow-ups. Please make sure that specifics, like attributes, usage, operations, are included in follow up questions so they can be asked standalone. For example, if the original question asks about "如何删除 BACKEND 节点？", in the follow up question, do not just say "这个节点", but use the full name "BACKEND  节点". Your related questions must be in the same language as the original question.

Here are the contexts of the question:

{context}

If the generated question is not related with Apache Doris, just ignore it and give empty answer.
Each question should not be longer than 20 words and should not contain carriage return, line feed or tab characters.
Remember, based on the original question and related contexts, suggest three such further questions. Do NOT repeat the original question. Each related question should be no longer than 20 words. Here is the original question:
"""

if is_test:
    _rag_query_text = "Just say: 'I am a test.'"
    _more_questions_prompt = "Just say twice following in two lines: 'I am a test for related question.'"
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

    def __init__(self, config):
        super().__init__(config)
        # Load vault dictionary
        # self.vault = None
        self.vault = get_vault()
        logger.info(f'Vault loaded with {len(self.vault)} documents')

        self.model = self.deployment_template["env"]["LLM_MODEL"]
        # An executor to carry out async tasks, such as uploading to KV.
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.handler_max_concurrency * 2
        )
        # whether we should generate related questions.
        self.should_do_related_questions = to_bool(self.deployment_template["env"]["RELATED_QUESTIONS"])

        self.app = FastAPI()
        self.app.get("/query")(self.query_function)
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
        uvicorn.run(self.app)

    deployment_template = {
        "env": {
            "SEARCH_ENGIN": "",  # TODO support search engin
            "LLM_MODEL": "gpt-3.5-turbo",
            # For all the search queries and results, we will use the KV store to
            # store them so that we can retrieve them later. Specify the name of the
            # KV here.
            "KV_NAME": "",  # TODO support kv restore
            # If set to true, will generate related questions. Otherwise, will not.
            "RELATED_QUESTIONS": "true",
            "BASE_URL": "https://api.openai-proxy.com"
        },
        # Secrets you need to have: search api subscription key, and lepton
        # workspace token to query lepton's llama models.
        "secret": {
            # TODO support search engin api key
            "SERPER_SEARCH_API_KEY": "",
            "OPENAI_API_KEY": "",
        },
    }

    # It's just a bunch of api calls, so our own deployment can be made massively
    # concurrent.
    handler_max_concurrency = 16

    def local_client(self):
        """
        Gets a thread-local client, so in case openai clients are not thread safe,
        each thread will have its own client.
        """

        thread_local = threading.local()
        try:
            return thread_local.client
        except AttributeError:
            dt = self.deployment_template
            from openai import OpenAI
            thread_local.client = OpenAI(
                # This is the default and can be omitted
                api_key = os.getenv("OPENAI_API_KEY") or dt["secret"]["OPENAI_API_KEY"],
            )
            return thread_local.client

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
            response = self.local_client().chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": _more_questions_prompt.format(
                            context="\n\n".join([c["content"] for c in contexts])
                        ),
                    },
                    {
                        "role": "user",
                        "content": query,
                    },
                ],
                # tools=[{
                #     "type": "function",
                #     "function": get_tools_spec(ask_related_questions),
                # }],
                max_tokens=512,
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
            await asyncio.sleep(0.1)

    def query_function(
            self,
            query: str,
            generate_related_questions
    ) -> StreamingResponse:
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
        # Basic attack protection: remove "[INST]" or "[/INST]" from the query
        query = re.sub(r"\[/?INST\]", "", query)
        contexts = self._get_chunks(query)

        system_prompt = _rag_query_text.format(
            context="\n\n".join(
                [f"[[citation:{i + 1}]] {c['content']}" for i, c in enumerate(contexts)]
            )
        )
        try:
            client = self.local_client()
            #
            # llm_response = client.chat.completions.create(
            #     messages=[
            #         {
            #             "role": "user",
            #             "content": "Say this is a test",
            #         }
            #     ],
            #     model="gpt-3.5-turbo",
            # )
            llm_response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query},
                ],
                max_tokens=1024,
                # stop=stop_words, # TODO support stop words
                stream=True,
                temperature=0.65,
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
        semantic_response = query_semantic(query, tokenizer, model, doc_embeddings_array)
        related_context = self._get_related_chunks_from_hits(semantic_response, embedding_index, self.vault)
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

    def _get_related_chunks_from_hits(self, hits: List[dict], embedding_index: Dict[int, str], vault: dict,
                                      max_lines: int = 10) -> List[dict]:
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
            embedding_index: Mapping of document embedding row index to document doc-id.
            vault: Dictionary of vault documents.
            max_lines: Maximum lines to allow in chunks. Defaults to 10.

        Returns:
            List of chunks for retrieval-augmented generation, in form of [{
                content: ''
                title: ''
            }].
        """
        chunks = []
        for hit_index in hits:
            chunk_id = embedding_index[hit_index]
            up_index = hit_index - 1
            down_index = hit_index + 1
            lines = 0
            chunk = {'title': vault[chunk_id]['title'], 'content': ''.join(vault[chunk_id]['chunks'])}
            content_list = [chunk['content']]
            # Look up
            while up_index in embedding_index and lines < max_lines:
                chunk_id = embedding_index[up_index]
                chunk_content = ' '.join(vault[chunk_id]['chunks'])
                if re.match(r'^#+ ', chunk_content):  # Paragraph start
                    break
                content_list.insert(0, chunk_content)
                lines += 1
                up_index -= 1
            # Look down
            while down_index in embedding_index and lines < max_lines:
                chunk_id = embedding_index[down_index]
                chunk_content = ' '.join(vault[chunk_id]['chunks'])
                if re.match(r'^#+ ', chunk_content):  # Paragraph start
                    break
                content_list.append(chunk_content)
                lines += 1
                down_index += 1
            chunk['content'] = '\n'.join(content_list)
            chunks.append(chunk)
        return chunks


if __name__ == '__main__':
    config = uvicorn.Config(app='main:app', host="0.0.0.0", port=8000)
    rag = RAG(config)
    rag.launch()
