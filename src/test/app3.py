import json

from fastapi import FastAPI
from sse_starlette.sse import EventSourceResponse
from fastapi.responses import StreamingResponse

import time


app = FastAPI()


def data_streamer():
    # yield "hello"
    yield json.dumps({"created":1708599292313,"id":"1760619156603158530","model":"Baichuan2-Turbo","costTime":3245,"conversationId":"1760619156603158528","references":[],"paragraphs":[],"multiParts":[],"choices":[{"message":{"role":"assistant","content":"y"}}]})
    time.sleep(5)

    for i in range(10):
        # yield f"_{i}_".encode("utf-8")
        yield json.dumps({"created":1708599627956,"id":"1760620564391936002","model":"Baichuan2-Turbo","costTime":6538,"conversationId":"1760620564391936000","references":[],"paragraphs":[],"multiParts":[],"choices":[{"message":{"role":"assistant","content":"关信"}}]})

        time.sleep(2)

    yield json.dumps({"created":1708599627956,"id":"1760620564391936002","model":"Baichuan2-Turbo","costTime":6538,"conversationId":"1760620564391936000","references":[],"paragraphs":[],"multiParts":[],"choices":[{"message":{"role":"chat_status","content":"回复完成"}}]})
    time.sleep(5)
    yield json.dumps({"created":1708599627956,"id":"1760620564391936002","model":"Baichuan2-Turbo","costTime":6538,"conversationId":"1760620564391936000","references":[{"containerId":"1750510855777497088","containerPkId":"9","containerName":"helper-merged","dataSetId":"1750510967358566400","dataSetUrl":"https://chat2kl.oss-cn-beijing.aliyuncs.com/1750488470515744768/2024/0125/9e50adb3-7314-4665-9272-2ea65331ef18.txt?OSSAccessKeyId\u003dLTAI5t6CsmJ2DSBfqFAnjYv5\u0026Expires\u003d2021549304\u0026Signature\u003d1HjLNdjobDFxt%2BILeU1B48Mv6iM%3D","dataSetPkId":"167","dataSetName":"doris-udf8.txt","suffix":"txt"}],"paragraphs":[{"containerId":"1750510855777497088","containerPkId":"9","containerName":"helper-merged","dataSetId":"1750510967358566400","dataSetPkId":"167","dataSetName":"doris-udf8.txt","page":1742,"content":"注：1.2.0 版本之后支持根据 State 取消作业。\n\n### Example\n\n1. 撤销数据库 example_db 上， label 为 `example_db_test_load_label` 的导入作业\n\n   ```sql\n   CANCEL LOAD\n   FROM example_db\n   WHERE LABEL \u003d \"example_db_test_load_label\";\n   ```\n\n2. 撤销数据库 example*db 上， 所有包含 example* 的导入作业。\n\n   ```sql\n   CANCEL LOAD\n   FROM example_db\n   WHERE LABEL like \"example_\";\n   ```\n\n\u003cversion since\u003d\"1.2.0\"\u003e\n\n3. 取消状态为 LOADING 的导入作业。\n\n   ```sql\n   CANCEL LOAD\n   FROM example_db\n   WHERE STATE \u003d \"loading\";\n   ```\n\n\u003c/version\u003e\n\n### Keywords\n\n    CANCEL, LOAD\n\n### Best Practice\n\n1. 只能取消处于 PENDING、ETL、LOADING 状态的未完成的导入作业。\n2. 当执行批量撤销时，Doris 不会保证所有对应的导入作业原子的撤销。即有可能仅有部分导入作业撤销成功。用户可以通过 SHOW LOAD 语句查看作业状态，并尝试重复执行 CANCEL LOAD 语句。\n---\n{\n    \"title\": \"RESUME-ROUTINE-LOAD\",\n    \"language\": \"zh-CN\"\n}\n---\n\n\u003c!--split--\u003e\n\n## RESUME-ROUTINE-LOAD\n\n### Name\n\nRESUME ROUTINE LOAD\n\n### Description\n\n用于重启一个被暂停的 Routine Load 作业。重启的作业，将继续从之前已消费的 offset 继续消费。\n\n```sql\nRESUME [ALL] ROUTINE LOAD FOR job_name\n```\n\n### Example\n\n1. 重启名称为 test1 的例行导入作业。","suffix":"txt"},{"containerId":"1750510855777497088","containerPkId":"9","containerName":"helper-merged","dataSetId":"1750510967358566400","dataSetPkId":"167","dataSetName":"doris-udf8.txt","page":2209,"content":"### 动态分区表与手动分区表相互转换\n\n  对于一个表来说，动态分区和手动分区可以自由转换，但二者不能同时存在，有且只有一种状态。\n\n  #### 手动分区转换为动态分区\n\n  如果一个表在创建时未指定动态分区，可以通过 `ALTER TABLE` 在运行时修改动态分区相关属性来转化为动态分区，具体示例可以通过 `HELP ALTER TABLE` 查看。\n\n  开启动态分区功能后，Doris 将不再允许用户手动管理分区，会根据动态分区属性来自动管理分区。\n\n  **注意**：如果已设定 `dynamic_partition.start`，分区范围在动态分区起始偏移之前的历史分区将会被删除。\n\n  #### 动态分区转换为手动分区\n\n  通过执行 `ALTER TABLE tbl_name SET (\"dynamic_partition.enable\" \u003d \"false\")` 即可关闭动态分区功能，将其转换为手动分区表。\n\n  关闭动态分区功能后，Doris 将不再自动管理分区，需要用户手动通过 `ALTER TABLE` 的方式创建或删除分区。\n\n\n## 常见问题\n\n1. 创建动态分区表后提示 `Could not create table with dynamic partition when fe config dynamic_partition_enable is false`\n\n   由于动态分区的总开关，也就是 FE 的配置 `dynamic_partition_enable` 为 false，导致无法创建动态分区表。\n\n   这时候请修改 FE 的配置文件，增加一行 `dynamic_partition_enable\u003dtrue`，并重启 FE。或者执行命令 ADMIN SET FRONTEND CONFIG (\"dynamic_partition_enable\" \u003d \"true\") 将动态分区开关打开即可。\n\n2. 关于动态分区的副本设置\n\n    动态分区是由系统内部的调度逻辑自动创建的。在自动创建分区时，所使用的分区属性（包括分区的副本数等），都是单独使用 `dynamic_partition` 前缀的属性，而不是使用表的默认属性。举例说明：","suffix":"txt"},{"containerId":"1750510855777497088","containerPkId":"9","containerName":"helper-merged","dataSetId":"1750510967358566400","dataSetPkId":"167","dataSetName":"doris-udf8.txt","page":3108,"content":"## 注意事项\n\n* 如果不开启并发导出，查询结果是由单个 BE 节点，单线程导出的。因此导出时间和导出结果集大小正相关。开启并发导出可以降低导出的时间。\n* 导出命令不会检查文件及文件路径是否存在。是否会自动创建路径、或是否会覆盖已存在文件，完全由远端存储系统的语义决定。\n* 如果在导出过程中出现错误，可能会有导出文件残留在远端存储系统上。Doris 不会清理这些文件。需要用户手动清理。\n* 导出命令的超时时间同查询的超时时间。可以通过 `SET query_timeout\u003dxxx` 进行设置。\n* 对于结果集为空的查询，依然会产生一个文件。\n* 文件切分会保证一行数据完整的存储在单一文件中。因此文件的大小并不严格等于 `max_file_size`。\n* 对于部分输出为非可见字符的函数，如 BITMAP、HLL 类型，输出为 `\\N`，即 NULL。\n* 目前部分地理信息函数，如 `ST_Point` 的输出类型为 VARCHAR，但实际输出值为经过编码的二进制字符。当前这些函数会输出乱码。对于地理函数，请使用 `ST_AsText` 进行输出。\n\n## 更多帮助\n\n关于 OUTFILE 使用的更多详细语法及最佳实践，请参阅 [OUTFILE](../../sql-manual/sql-reference/Data-Manipulation-Statements/OUTFILE.md) 命令手册，你也可以在 MySql 客户端命令行下输入 `HELP OUTFILE` 获取更多帮助信息。\n---\n{\n    \"title\": \"Runtime Filter\",\n    \"language\": \"zh-CN\"\n}\n---\n\n\u003c!--split--\u003e\n\n# Runtime Filter\n\nRuntime Filter 是在 Doris 0.15 版本中正式加入的新功能。旨在为某些 Join 查询在运行时动态生成过滤条件，来减少扫描的数据量，避免不必要的I/O和网络传输，从而加速查询。\n\n它的设计、实现和效果可以参阅 [ISSUE 6116](https://github.com/apache/incubator-doris/issues/6116)。\n\n## 名词解释","suffix":"txt"}],"multiParts":[],"choices":[{"finish_reason":"stop","message":{"role":"assistant","content":""}}]})


@app.post('/query')
async def main():
    return EventSourceResponse(data_streamer(), media_type='text/event-stream')
    # return StreamingResponse(data_streamer(), media_type='text/event-stream')



if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8002);