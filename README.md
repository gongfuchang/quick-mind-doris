# quick-mind-doris
<img src="assets/snapshot.png" alt="icon"/>

## 准备工作
- build vault dictionary：根据文档（assets/doris-udf8.txt）创建实体对象文件（data/vault_dict.pickle）
```shell
python build_vault_dict.py
```
- build index：根据实体对象文件（data/vault_dict.pickle）创建索引文件
```shell
python build_index.py
```
- build opensearch index：根据文档（data/vault_dict.pickle）创建 opensearch 索引
```shell
python build_opensearch_index.py
```

## 启动 server
- 填写 OpenAI API Key 后启动：
```shell
export LLM_TYPE=GLM or GPT
export OPENAI_API_KEY=sk-xxxxxx
python app.py
```
## 启动 web-client
- 创建 .env.local 文件，填写上面已启动的 sever URL，和 多轮对话 URL，例如：
```shell
BOT_URL="http://qmbot.kittygpt.cn:8088"
SEARCH_SERVER_URL="http://qms.kittygpt.cn:8088"
```
- 启动 web-client (dev模式)
```shell
cd web
npm install
npm run dev
```

- 编译启动 web-client
```shell
npm run build
npm run start
```
## Features

- Co-reference resolution(指代消解), rolling up 3 chat histories with LLM
- Double hits, both index match(vector search) and symantic search(elasticsearch)
- Metadata filter，e.g. version
- 目前支持 Self-host Model：chatglm-6b

## Roadmap

- [ ] Management backend
- [ ] Desktop App with tauri
- [ ] User login, accounts
...


## Get Started
[TODO]


## LICENSE

[Anti 996 License](https://github.com/kattgu7/Anti-996-License/blob/master/LICENSE_CN_EN)