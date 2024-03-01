import math
import os
import jieba
import jieba.analyse

from src.utils.file_util import get_file_path
import re
corpus_path = get_file_path('assets/doris-udf8.txt')
idf_path = get_file_path('data/idf/doris-idf.txt')
dict_path = get_file_path('data/idf/doris-dict.txt')
stopw_path = get_file_path('data/idf/hit_stopwords.txt')
def make_idf():
    with open(corpus_path, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()

    lines = re.findall('[\u4e00-\u9fa5]+', content, re.S)
    arr = list(filter(lambda x: '时间列' in x, lines))
    print('\n'.join(arr))
    new_text = "".join(lines)
    # new_text = " ".join(content.split('\n'))

    for text in arr:
        cuts = jieba.cut(text,use_paddle=True)
        # print('/'.join(cuts))

    jieba.load_userdict(dict_path)  # 应用自定义词典
    jieba.analyse.set_stop_words(stopw_path)  # 去除自定义停用词
    tags = jieba.analyse.extract_tags(new_text, withWeight=True, topK=10000)

    with open(idf_path, 'w', encoding='utf-8') as f:
        content_2_write = '\n'.join([f'{tag[0]} {tag[1]}' for tag in tags])
        f.write(content_2_write)

def test_idf():
    # Extract keywords using TextRank algorithm
    for text in text_arr:
        keywords = jieba.analyse.textrank(text, topK=10)
        print("Keywords:", keywords)
        # jieba.load_userdict(dict_path)
        jieba.analyse.set_idf_path(idf_path)
        cuts = jieba.cut(text)
        print('/'.join(cuts))
        tags = jieba.analyse.extract_tags(text,
            topK=20,
            withWeight=True,
            allowPOS=('ns', 'n', 'vn', 'v')
        )

        print("Tags:", tags)
        print(' '.join([tag[0] for tag in tags]))

        print()

if __name__ == '__main__':
    # make_idf()
    text_arr = [
        "想请教一下大佬们，如果我想新增一列时间列，这张表里任一字段的数据更新，这个时间列都会自动更新时间，Doris支持这种操作么？还是说，得在数据导入前给每列数据加一个最新时间数据再进行导入Doris,"
        # '时间列包含数据更新',
    ]
    for str in text_arr:
        jieba.load_userdict(dict_path)
        # jieba.analyse.set_idf_path(idf_path)
        seg_list = jieba.analyse.extract_tags(str,
            topK=10,
            withWeight=True,
            allowPOS=('ns', 'n', 'vn', 'v')
        )
        print(seg_list)
        print(' '.join([tag[0] for tag in seg_list]))
        jieba.cut_for_search(str)
        seg_list = jieba.cut_for_search(str)
        print("Full Mode: " + "/ ".join(seg_list))  # 全模式

    # seg_list = jieba.cut("我来到北京清华大学", cut_all=True)
    # print("Full Mode: " + "/ ".join(seg_list))  # 全模式
    #
    # seg_list = jieba.cut("我来到北京清华大学", cut_all=False)
    # print("Default Mode: " + "/ ".join(seg_list))  # 精确模式
    #
    # seg_list = jieba.cut("他来到了网易杭研大厦")  # 默认是精确模式
    # print(", ".join(seg_list))
    #
    # seg_list = jieba.cut_for_search("小明硕士毕业于中国科学院计算所，后在日本京都大学深造")  # 搜索引擎模式
    # print(", ".join(seg_list))
