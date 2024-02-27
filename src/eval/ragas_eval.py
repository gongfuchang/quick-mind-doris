from ragas import evaluate
from datasets import load_dataset, concatenate_datasets
from ragas.metrics import (
    context_precision,
    answer_relevancy,
)
from src.utils.file_util import get_file_path
import pandas as pd
"""
分别eval 三个 result 文件：glm_result， gpt_result， torchv_result
然后将 eval 结果 join 写入同一个数据集，join 的依据是 question. 
Head 是： question,   answer, contexts,   glm_context_precision,  glm_answer_relevancy,   gpt_context_precision,  gpt_answer_relevancy,   torchv_context_precision,   torchv_answer_relevancy 
最后写入 csv: ragas_eval_result.csv
"""
# Load and evaluate the three result files
result_files = ['glm_result.csv', 'gpt_result.csv', 'torchv_result.csv']
results = []

for i, file in enumerate(result_files):
    dataset = load_dataset('csv', data_files=get_file_path(file))
    result = evaluate(
        dataset["eval"].select([0]),
        metrics=[
            context_precision,
            answer_relevancy
        ],
    )
    result_df = result.to_pandas()
    result_df.columns = ['question', 'answer', 'contexts', f'{file[:-10]}_context_precision', f'{file[:-4]}_answer_relevancy']
    if i == 0:
        df = result_df
    else:
        df = pd.merge(df, result_df, on='question')

# Write the dataset to a CSV file
df.to_csv(get_file_path('ragas_eval_result.csv'), index=False)