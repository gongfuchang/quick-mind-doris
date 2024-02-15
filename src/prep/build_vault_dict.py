"""
Reads markdown files and returns a vault dictionary in the format below.

# Note: chunk_id is either <title-id> or <title>. The former is a chunk while the latter is the entire doc.
dock_id: {
    title: doc title,
    title_display:
    language: zh-CN or en,
    version: none if no version provided, otherwise the version number,
    content: doc content, the full text of the doc
}

chunk_id: { doc: doc, chunks: [chunk1, chunk2, ...] }
"""
import argparse
import json
import os
import pickle
from collections import defaultdict
from pathlib import Path
from typing import List

from src.logger import logger
import re
import base64
from src.utils.file_util import get_file_path
DATA_VAULT_DICT_PICKLE = get_file_path('data/vault_dict2.pickle')

def folder_chunks(content: str, max_chunk_lines=5, max_token_num=300) -> List[str]:
    """Folder up the text into chunks, where each new paragraph / top-level bullet in a new chunk.

    Args:
        lines: Lines in a document
		chunks: Chunks to folder up
        max_chunk_lines: Maximum number of lines in a chunk before being discarded. Defaults to 5.
        max_token_num: Maximum number of characters in a chunk before being discarded. Defaults to 300.

    """
    # TODO 先按照一行一行的方式切分，后续再考虑按照段落切分
    if content is None:
        return []
    try:
        chunks = [process_string(line.strip()) for line in content.split('\n')]
        return [[item] for item in (filter(lambda x: filter_line_valid(x), chunks))] # [[1], [2]...]
    except Exception as e:
        logger.error(f'【skip】Error in foldering chunks: {e}')
        return []


def filter_line_valid(line: str) -> bool:
    """
    Judge if needs to filter out invalid lines
    @param line: line to filter
    @return: True if valid, otherwise False
    """
    if line is None:
        return False
    # 正则表达式匹配任何有效的中英文字符
    # TODO 其他情景待补充
    pattern = r'[\u4e00-\u9fa5a-zA-Z]'
    match = re.search(pattern, line)
    # 如果没有匹配到任何字符，返回False，否则返回True
    return match is not None

def process_string(str: str):
    # Check if the string is a markdown IMG or Link
    if re.match(r'!\[.*\]\(.*\)|\[.*\]\(.*\)', str):
        return None

    # Check if the string is a tag and return its content
    # Use re.sub() to replace all HTML tags with an empty string
    output_string = re.sub(r'<[^>]*>', '', str)
    if output_string is not None:
        return output_string

    return str
def encode_doc_id(title_display: str, title: str, version: str) -> str:
    """Encode the doc id in the format of base32 code of <title_display>_<title>_<version>"""
    input_string = f"{title_display}_{title}_{version}" if version is not None else f"{title_display}_{title}"
    encoded_string = base64.b32encode(input_string.encode())
    return encoded_string.decode()

def create_vault_dict(filename: str) -> dict[str, dict[str, str]]:
    """ Parse the helper doc in form of markdown to create a chunks of documents in form of json list like

    following is the sample for doc
        ---
        {
        "title": "BITOR",
        "language": "zh-CN"
        }
        ---

        <!--split-->

        ## bitor
        ### description

        ---
        {
            "title": "Release 1.2.6",
            "language": "zh-CN"
        }
        ---

        <!--split-->


        # Behavior Changed

        - 新增 BE 配置项 `allow_invalid_decimalv2_triteral` 以控制是否可以导入超过小数精度的 Decimal 类型数据，用于兼容之前的逻辑。

        # Bug Fixes

        ## 查询

    Args:
        filename: Path to helper doc
    Returns:
        Dictionary of full docs and chunks in a vault
    """
    vault = dict()
    doc_dict = dict()

    with open(filename, 'r', encoding='utf-8', errors='replace') as f:
        # split the doc into chunks with a regex pattern
        parts = re.split(r'---\s*\n*\s*\{\s*["\']title["\']\s*:\s*["\']([^"\']*)["\']\s*,\s*["\']language["\']\s*:\s*["\']([^"\']*)["\'](?:,\n\s*["\']toc_min_heading_level["\']: (\d+),\n\s*["\']toc_max_heading_level["\']: (\d+))?\s*\n*\s*\}\s*\n*\s*---\s*\n*\s*<!--split-->', f.read())

        # trim empty string in the parts
        parts = list(filter(lambda x: x != '', parts))
        # print(parts)
        # devide the parts to docs, which doc contains 3 attributes: title, language, and content
        docs = list(zip(*[iter(parts)] * 5))
        version_count = 0
        deprecated_count = 0
        # parse each doc to find the version info in content attribute: <version since="2.0.0"></version>
        for doc_index, doc in enumerate(docs):
            # print(chunk)
            title = doc[0].strip()
            language = doc[1].strip()
            content = doc[4].strip()
            # print(title, language, content)
            # parse the version info
            version = None
            deprecated = None


            # parse title display from content, which is in the format of '## title display' or '# title display'
            title_display_match = re.search(r'#{1,2} (.+)', content)
            title_display = title_display_match.group(1) if title_display_match is not None else None
            if title_display is None:
                title_display = title

            # TODO 只针对于 function 处理 version 信息
            if content.startswith('##'):
                version_match = re.search(r'<version(?:\s+[a-z]+=["\'][^"\']+["\'])? since=["\']([^"\']+)["\']', content)
                deprecated_match = re.search(r'<version(?:\s+[a-z]+=["\'][^"\']+["\'])? deprecated=["\']([^"\']+)["\']', content)
                if version_match:
                    version = version_match.group(1)
                if deprecated_match:
                    deprecated = deprecated_match.group(1)

            doc_id = encode_doc_id(title_display, title, version)
            # check the title_display is unique, otherwise log error and continue the loop
            if doc_id in doc_dict:
                # logger.warn(f'Display title is not unique: {title_display}')
                logger.error(f'【skip】Title and display title are both duplicated: {title} ==> {title_display} ==>{version}')
                continue

            # print(version)
            # add the chunk to vault
            doc_dict[doc_id] = {
                'title': title,
                'title_display': title_display,
                'language': language,
                'version': version,
                'deprecated': deprecated,
                'content': content
            }
            # print vault in json style
            if version is not None:
                # print(json.dumps(vault[doc_id], indent=4, ensure_ascii=False))
                version_count += 1
            if deprecated is not None:
                print(json.dumps(doc_dict[doc_id], indent=4, ensure_ascii=False))
                deprecated_count += 1
    print(f'version count: {version_count}, {deprecated_count}')
    # TODO 保存 doc 到 db
    # 将 doc_dict 按 chunks 打散 TODO  folder chunks 替代单个 chunk
    for doc_id, doc in doc_dict.items():
        chunks_folder = folder_chunks(doc['content'])
        for index, chunks in enumerate(chunks_folder):
            chunk_id = f'{doc_id}_{index}'
            vault[chunk_id] = {
                'doc_id': doc_id,
                'title': doc['title'],
                'chunks': chunks
            }
    return vault

def get_vault():
    with open(DATA_VAULT_DICT_PICKLE, 'rb') as f:
        return pickle.load(f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create vault dictionary')
    parser.add_argument('--vault_path', type=str, help='Path to helper doc vault')
    args = parser.parse_args()

    valt_path = get_file_path('assets/doris-udf8.txt')

    vault = create_vault_dict(valt_path)
    logger.info(f'Number of docs in vault: {len(vault):,}')

    os.makedirs(os.path.dirname(DATA_VAULT_DICT_PICKLE), exist_ok=True)

    with open(DATA_VAULT_DICT_PICKLE, 'wb+') as f:
        pickle.dump(vault, f, protocol=pickle.HIGHEST_PROTOCOL)


