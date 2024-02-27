import pandas as pd
import requests
import time
import logging
from src.utils.file_util import get_file_path
from urllib.parse import quote
import shutil
import os
from src.eval.torchv_generator import request_torchv

# 设置日志级别
logging.basicConfig(level=logging.INFO)


source_file_path = get_file_path('assets/eval/doris_qa_0206.csv')
dst_file_path = get_file_path('data/eval/torchv_fetched_answers.csv')
# 读取 csv 文件
df = pd.read_csv(source_file_path if not(os.path.exists(dst_file_path)) else dst_file_path)
# df = df.head(1)

# 创建两个新的列来存储获取的答案和请求时间
if 'fetched_answer' not in df.columns:
    df['fetched_contexts'] = ''
    df['fetched_answer'] = ''
    df['request_time'] = 0

# Initialize counter and total time
counter = 0
total_time = 0


# 备份文件
if os.path.exists(dst_file_path):
    backup_file_path = f'{dst_file_path}_{time.strftime("%y%m%d%H%M") }.backup'
    shutil.copy2(dst_file_path, backup_file_path)

# 遍历每一行
for index, row in df.iterrows():
    # 检查 fetched_answer 是否不为空或者为 'ERROR'，说明处理过了，跳过
    if pd.notna(row['fetched_answer']) and row['fetched_answer'] != "" and row['fetched_answer'] != 'ERROR':
        continue

    # 尝试两次
    for i in range(2):
        try:
            # 记录开始时间
            start_time = time.time()

            # 发送 GET 请求
            response, result = request_torchv(query=row['question'])

            # 记录结束时间并计算耗时
            end_time = time.time()
            elapsed_time = round(end_time - start_time, 2)

            total_time += elapsed_time  # Add elapsed time to total time

            # 检查响应状态码，如果状态码不是200，将引发异常
            response.raise_for_status()

            # 提取上下文
            df.at[index, 'fetched_contexts'] = result['contexts']

            # 提取答案
            answer = result['answer']

            # 存储答案和请求时间
            df.at[index, 'fetched_answer'] = answer
            df.at[index, 'request_time'] = elapsed_time
            # 如果请求成功，跳出循环
            logging.info(f"Request succeeded for [{index}]-question: {row['question'][:20]}")
            counter += 1  # Increment counter
            # If counter reaches 20, print total time and reset counter and total time
            if counter % 5 == 0:
                minutes, seconds = divmod(total_time, 60)
                logging.info(f"Total time for {counter} requests: {round(minutes)}m{round(seconds)}s")
                df.to_csv(dst_file_path, index=False)
            break
        except requests.exceptions.RequestException as err:
            # 如果请求失败，记录 'ERROR' 和请求时间
            df.at[index, 'fetched_answer'] = 'ERROR'
            df.at[index, 'request_time'] = elapsed_time
            # 如果是第一次失败，记录一个警告日志
            if i == 0:
                logging.warning(f"Request failed for question: {row['question'][:20]}, error: {err}. Retrying...")
            # 如果重试也失败，记录一个错误日志
            else:
                logging.error(f"Retry failed for question: {row['question'][:20]}, error: {err}")


# 将新的 DataFrame 写入 csv 文件
df.to_csv(dst_file_path, index=False)