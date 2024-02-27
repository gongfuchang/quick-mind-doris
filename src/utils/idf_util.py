import math
import os
import jieba
import jieba.analyse

from src.utils.file_util import get_file_path
import re

if __name__ == '__main__':
    corpus_path = get_file_path('assets/doris-udf8.txt')  # 存储语料库的路径，按照类别分
    seg_path = get_file_path('data/idf/doris-idf.txt')  # 拼出分词后语料的目录

    with open(corpus_path, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()

    new_text = " ".join(re.findall('[\u4e00-\u9fa5]+', content, re.S))

    jieba.load_userdict(get_file_path('assets/doris_dict.txt'))  # 应用自定义词典
    # jieba.analyse.set_stop_words('停用词库.txt')  # 去除自定义停用词
    tags = jieba.analyse.extract_tags(new_text, withWeight=True, topK=5000)

    with open(seg_path, 'w') as f:
        content_2_write = '\n'.join([f'{tag[0]} {tag[1]}' for tag in tags])
        f.write(content_2_write)


    text = "想请教一下大佬们，如果我想新增一列时间列，这张表里任一字段的数据更新，这个时间列都会自动更新时间，Doris支持这种操作么？还是说，得在数据导入前给每列数据加一个最新时间数据再进行导入Doris,"

    # Extract keywords using TextRank algorithm
    keywords = jieba.analyse.textrank(text, topK=10)
    print("Keywords:", keywords)
    jieba.analyse.set_idf_path(seg_path)

    tags = jieba.analyse.extract_tags(text,
        topK=20,
        withWeight=True,
        allowPOS=('ns', 'n', 'vn', 'v')
    )

    print("Tags:", tags)
    print(' '.join([tag[0] for tag in tags]))
