from LAC import LAC

# # 装载分词模型
# lac = LAC(mode='seg')
#
# # 单个样本输入，输入为Unicode编码的字符串
# text = u"LAC是个优秀的分词工具"
# seg_result = lac.run(text)
#
# # 批量样本输入, 输入为多个句子组成的list，平均速率会更快
# texts = [u"LAC是个优秀的分词工具", u"百度是一家高科技公司"]
# seg_result = lac.run(texts)
#
# print(seg_result)


# 装载LAC模型
lac = LAC(mode='lac')

# 单个样本输入，输入为Unicode编码的字符串
text = u"LAC是个优秀的分词工具"
lac_result = lac.run(text)

# 批量样本输入, 输入为多个句子组成的list，平均速率更快
texts = [
    # u"LAC是个优秀的分词工具助手",
    # u"百度是一家高科技公司的楷模和典范",
    '我有三个列，时间列，姓名列，年龄列',
    "想请教一下大佬们，如果我想新增一列时间列",
    "这个时间列都会自动更新时间，Doris支持这种操作么？",
    "还是说，得在数据导入前给每列数据加一个最新时间数据再进行导入Doris,"
    '时间列包含数据更新'
]

import re
from src.utils.file_util import get_file_path
from tqdm import tqdm #tqdm是一个非常易用的用来显示进度的库

corpus_path = get_file_path('assets/doris-udf8.txt')
stopw_path = get_file_path('data/idf/hit_stopwords.txt')
dict_path = get_file_path('data/idf/doris-dict.txt')

load_stopwords = lambda path: set([line.strip() for line in open(path, 'r', encoding='utf-8').readlines()])
stopwords = load_stopwords(stopw_path)

with open(corpus_path, 'r', encoding='utf-8', errors='replace') as f:
    content = f.read()


lines = re.findall('[\u4e00-\u9fa5]+', content, re.S)
# lines = lines[2000:2080]
lines.extend(['想请教一下大佬们，如果我想新增一列时间列，这张表里任一字段的数据更新'])
# lac_result = lac.run(lines)
# for r in lac_result:
#     print(r)

def find_combinations(words, tags):
    target_tags = ['ORG', 'n', 'nz', 'nw']
    combinations = []
    for i in range(len(tags)):
        word = words[i]
        if word in stopwords:
            continue
        if tags[i] in target_tags:
            # Check for combinations of 2
            if i < len(tags) - 1 and (tags[i + 1] in target_tags and words[i + 1] not in stopwords):
                combinations.append((words[i], words[i + 1]))
            # Check for combinations of 3
            if i < len(tags) - 2 and (tags[i + 1] in target_tags  and words[i + 1] not in stopwords) \
                    and (tags[i + 2] in target_tags and words[i + 2] not in stopwords):
                combinations.append((words[i], words[i + 1], words[i + 2]))

            if i < len(tags) - 1 and (tags[i + 1] in ['v', 'vn'] and words[i + 1] not in stopwords):
                combinations.append((words[i], words[i + 1]))

    return [''.join(c) for c in combinations]

combs_words = {}
for line in tqdm(lines):
    # print(line)
    lac_result = lac.run([line])

    # Create a dictionary to store all combinations

    # Iterate over lac_result and find combinations
    for words, tags in lac_result:
        # 将 lac_result 过滤其中所有 ORG, n, nz, nw, v, vn，
        filter_words = [word for word, tag in zip(words, tags) if
                        tag in ['ORG', 'n', 'nz', 'nw', 'v', 'vn']
                        and word not in stopwords and len(word) > 1]
        for word in filter_words:
            combs_words[word] = True

        # 把名词组合、名动词组合放进去
        combinations = find_combinations(words, tags)
        # if len(combinations) > 0:
        #     print(f'{lac_result}=>{combinations}')
        for combination in combinations:
            # Convert the combination to a string and add it to the dictionary
            combination_str = ''.join(combination)
            combs_words[combination_str] = True

# Convert the dictionary to a list
dict_words = list(combs_words.keys())


# Write the list to a file
with open(dict_path, 'w', encoding='UTF-8') as f:
    for combination in dict_words:
        f.write(combination + '\n')