
from src.utils.file_util import get_file_path
import pandas as pd
"""
分别eval 三个 result 文件：glm_result， gpt_result， torchv_result
然后将 eval 结果 join 写入同一个数据集，join 的依据是 question. 
Head 是： question,   answer, contexts,   glm_context_precision,  glm_answer_relevancy,   gpt_context_precision,  gpt_answer_relevancy,   torchv_context_precision,   torchv_answer_relevancy 
最后写入 csv: ragas_eval_result.csv
"""
# Load and evaluate the three result files
providers = ['glm', 'gpt', 'torchv']
result_files = [get_file_path(f'data/eval/{provider}_fetched_answers.csv') for provider in providers]
results = []
def ragas_eval():
    from ragas import evaluate
    from datasets import load_dataset, concatenate_datasets
    from ragas.metrics import (
        context_precision,
        answer_relevancy,
    )
    for i, file in enumerate(result_files):
        dataset = load_dataset('csv', data_files=file)
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

def assemble_result():
    import os
    import ast
    import xlsxwriter

    """
        分别读取三个 result 文件，然后将他们 join 到同一个 csv 中，join 的依据是第一列 dlg_id.
        join 后，列 question、answer 任取其中一个即可因为他们都是相同的；原先的列名 feteched_contexts 和 feteched_answer 分别加上前缀 glm, gpt, torchv.
    """
    df = pd.read_csv(result_files[0])
    # copy dlg_id, question, answer columns to a new dataframe
    df = df[['dlg_id', 'question', 'answer']]

    for file in result_files:
        # only keep dlg_id and fetched_contexts, fetched_answer
        target_df = pd.read_csv(file)
        target_df = target_df[['dlg_id', 'fetched_contexts', 'fetched_answer', 'request_time']]

        # join others with dlg_id, and then rename joined columns
        file_name = os.path.basename(file)
        provider = file_name.split('_')[0]
        target_df.rename(columns={
            'fetched_contexts': f'{provider}_fetched_contexts',
            'fetched_answer': f'{provider}_fetched_answer',
            'request_time': f'{provider}_request_time'
        }, inplace=True)
        # contexts is a string like "['', '', '']", so we need to parse it as array, and then join them into a string with '\n'
        target_df[f'{provider}_fetched_contexts'] = target_df[f'{provider}_fetched_contexts'].apply(
            lambda x: ('\n\n' + '*' * 50 + '\n').join(ast.literal_eval(x)) # parse x as array
            if x is not None and type(x) is str
            else []
        )

        df = pd.merge(df, target_df, on='dlg_id')

    csv_path = get_file_path('data/eval/merged_eval_result.csv')
    df.to_csv(csv_path, index=False, header=True, escapechar="\\")
    # Convert the CSV to an Excel file
    excel_path = csv_path.replace('.csv', '.xlsx')
    data_csv = pd.read_csv(csv_path, header=0, escapechar="\\", delimiter=",")
    data_csv.to_excel(excel_path, header=True, sheet_name='data', engine='xlsxwriter')

if __name__ == '__main__':
    assemble_result()