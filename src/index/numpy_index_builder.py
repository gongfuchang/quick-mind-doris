from src.utils.file_util import get_file_path
from src.index.index_builder import IndexBuilder
from src.utils.model_util import get_model_tuple, get_device, average_pool
import numpy as np

from src.prep.build_vault_dict import get_vault_dict, get_doc_dict
import os
import pickle
import time
from src.logger import logger

EMBEDDINGS_ARRAY_NPY = get_file_path('data/embedding/doc_embeddings_array.npy')


class NumpyIndexBuilder(IndexBuilder):
    def query_semantic(self, query, n_results=3):
        query_embedding = self._vectorize([f'{self._get_query_prefix()}{query}'])

        cos_sims = np.dot(self.get_embeddings_array(), query_embedding.T)
        cos_sims = cos_sims.flatten()

        # # Create a boolean array where True means the corresponding value in cos_sims is greater than or equal to 0.6
        # mask = cos_sims >= 0.6
        #
        # # Use the mask to filter cos_sims
        # filtered_cos_sims = cos_sims[mask]
        #
        # # Get the indices that would sort the filtered array
        # sorted_indices = np.argsort(filtered_cos_sims)

        # Get the last n indices (the indices of the top n values)
        top_indices = np.argsort(cos_sims)[-n_results:][::-1]
        return self._get_related_chunks_from_hits(top_indices)

    def get_embeddings_array(self, reload: bool = False):
        if self.embeddings_array is None or reload:
            if not os.path.exists(EMBEDDINGS_ARRAY_NPY):
                return None
            self.embeddings_array = np.load(EMBEDDINGS_ARRAY_NPY)

        return self.embeddings_array

    def _reshape(self, doc_embeddings_array):
        return doc_embeddings_array
        # # Reshape to 2D array where embedding dim is 2nd dim
        # array = np.concatenate(doc_embeddings_array, axis=0)
        # return np.reshape(array, (-1, array.shape[-1]))

    def _bulk(self, chunk_array: list, doc_embeddings_array: np.ndarray):
        array = self._reshape(doc_embeddings_array)
        existing_array = self.get_embeddings_array(True)
        # if existing_array is existing, concatenate with existing array
        if existing_array is not None:
            array = np.concatenate([existing_array, array], axis=0)

        np.save(EMBEDDINGS_ARRAY_NPY, self._reshape(array))
        self.get_embeddings_array(True)

    def build_list(self, passages):
        parts = self.split_list(passages, 20)
        for p in parts:
            array = self._vectorize(p)
            np.save(EMBEDDINGS_ARRAY_NPY, array)

    def split_list(self, alist, wanted_parts=1):
        length = len(alist)
        return [alist[i * length // wanted_parts: (i + 1) * length // wanted_parts]
                for i in range(wanted_parts)]

passages = [["","# Doris BE存储层Benchmark工具","","## 用途","","    可以用来测试BE存储层的一些部分(例如segment、page)的性能。根据输入数据构造出指定对象,利用google benchmark进行性能测试。","","## 编译","","1. 确保环境已经能顺利编译Doris本体,可以参考[编译与部署](/docs/install/source-install/compilation)。"],["","2. 运行目录下的\"run-be-ut.sh\"","","3. 编译出的可执行文件位于\"./be/ut_build_ASAN/test/tools/benchmark_tool\"","","## 使用","","#### 使用随机生成的数据集进行Segment读取测试","","会先利用数据集写入一个\"segment\"文件,然后对scan整个\"segment\"的耗时进行统计。"],["","> ./benchmark_tool --operation=SegmentScan --column_type=int,varchar --rows_number=10000 --iterations=0","","这里的\"column_type\"可以设置表结构,\"segment\"层的表结构类型目前支持\"int、char、varchar、string\",\"char\"类型的长度为\"8\",\"varchar\"和\"string\"类型长度限制都为最大值。默认值为\"int,varchar\"。","","数据集按以下规则生成。",">int: 在[1,1000000]内随机。","","字符串类型的数据字符集为大小写英文字母,长度根据类型不同。","> char: 长度在[1,8]内随机。"],["> varchar: 长度在[1,128]内随机。 ","> string: 长度在[1,100000]内随机。","","\"rows_number\"表示数据的行数,默认值为\"10000\"。","","\"iterations\"表示迭代次数,benchmark会重复进行测试,然后计算平均耗时。如果\"iterations\"为\"0\"则表示由benchmark自动选择迭代次数。默认值为\"10\"。","","#### 使用随机生成的数据集进行Segment写入测试","","对将数据集添加进segment并写入磁盘的流程进行耗时统计。"],["","> ./benchmark_tool --operation=SegmentWrite","","#### 使用从文件导入的数据集进行Segment读取测试","","> ./benchmark_tool --operation=SegmentScanByFile --input_file=./sample.dat","","这里的\"input_file\"为导入的数据集文件。","数据集文件第一行为表结构定义,之后每行分别对应一行数据,每个数据用\",\"隔开。",""],["举例: ","\"\"\"","int,char,varchar","123,hello,world","321,good,bye","\"\"\"","","类型支持同样为\"int\"、\"char\"、\"varchar\"、\"string\",注意\"char\"类型数据长度不能超过8。","","#### 使用从文件导入的数据集进行Segment写入测试"],["","> ./benchmark_tool --operation=SegmentWriteByFile --input_file=./sample.dat","","#### 使用随机生成的数据集进行page字典编码测试","","> ./benchmark_tool --operation=BinaryDictPageEncode --rows_number=10000 --iterations=0","","会随机生成长度在[1,8]之间的varchar,并对编码进行耗时统计。","","#### 使用随机生成的数据集进行page字典解码测试"],["","> ./benchmark_tool --operation=BinaryDictPageDecode","","会随机生成长度在[1,8]之间的varchar并编码,并对解码进行耗时统计。","","## Custom测试","","这里支持用户使用自己编写的函数进行性能测试,具体可以实现在\"/be/test/tools/benchmark_tool.cpp\"。","例如实现有：","\"\"\"cpp"],["void custom_run_plus() {","    int p = 100000;","    int q = 0;","    while (p--) {","        q++;","        if (UNLIKELY(q == 1024)) q = 0;","    }","}","void custom_run_mod() {","    int p = 100000;"],["    int q = 0;","    while (p--) {","        q++;","        if (q %= 1024) q = 0;","    }","}","\"\"\"","则可以通过注册\"CustomBenchmark\"来加入测试。","\"\"\"cpp","benchmarks.emplace_back("],["                    new doris::CustomBenchmark(\"custom_run_plus\", 0,","                        custom_init, custom_run_plus));","benchmarks.emplace_back(","                    new doris::CustomBenchmark(\"custom_run_mod\", 0,","                        custom_init, custom_run_mod));","\"\"\"","这里的\"init\"为每轮测试的初始化步骤(不会计入耗时),如果用户有需要初始化的对象则可以通过\"CustomBenchmark\"的派生类来实现。","运行后有如下结果:","\"\"\"","2021-08-30T10:29:35+08:00"],["Running ./benchmark_tool","Run on (96 X 3100.75 MHz CPU s)","CPU Caches:","  L1 Data 32 KiB (x48)","  L1 Instruction 32 KiB (x48)","  L2 Unified 1024 KiB (x48)","  L3 Unified 33792 KiB (x2)","Load Average: 0.55, 0.53, 0.39","----------------------------------------------------------","Benchmark                Time             CPU   Iterations"],["----------------------------------------------------------","custom_run_plus      0.812 ms        0.812 ms          861","custom_run_mod        1.30 ms         1.30 ms          539","\"\"\"","---",""]]
passages.extend([["","4. Doris 各节点认证机制","","   除了 Master FE 以外，其余角色节点（Follower FE，Observer FE，Backend），都需要通过 \"ALTER SYSTEM ADD\" 语句先注册到集群，然后才能加入集群。","","   Master FE 在第一次启动时，会在 doris-meta/image/VERSION 文件中生成一个 cluster_id。","","   FE 在第一次加入集群时，会首先从 Master FE 获取这个文件。之后每次 FE 之间的重新连接（FE 重启），都会校验自身 cluster id 是否与已存在的其它 FE 的 cluster id 相同。如果不同，则该 FE 会自动退出。","","   BE 在第一次接收到 Master FE 的心跳时，会从心跳中获取到 cluster id，并记录到数据目录的 \"cluster_id\" 文件中。之后的每次心跳都会比对 FE 发来的 cluster id。如果 cluster id 不相等，则 BE 会拒绝响应 FE 的心跳。"],["","   心跳中同时会包含 Master FE 的 ip。当 FE 切主时，新的 Master FE 会携带自身的 ip 发送心跳给 BE，BE 会更新自身保存的 Master FE 的 ip。","","   > **priority_network**","   >","   > priority_network 是 FE 和 BE 都有一个配置，其主要目的是在多网卡的情况下，协助 FE 或 BE 识别自身 ip 地址。priority_network 采用 CIDR 表示法：[RFC 4632](https://tools.ietf.org/html/rfc4632)","   >","   > 当确认 FE 和 BE 连通性正常后，如果仍然出现建表 Timeout 的情况，并且 FE 的日志中有 \"backend does not found. host: xxx.xxx.xxx.xxx\" 字样的错误信息。则表示 Doris 自动识别的 IP 地址有问题，需要手动设置 priority_network 参数。","   >","   > 出现这个问题的主要原因是：当用户通过 \"ADD BACKEND\" 语句添加 BE 后，FE 会识别该语句中指定的是 hostname 还是 IP。如果是 hostname，则 FE 会自动将其转换为 IP 地址并存储到元数据中。当 BE 在汇报任务完成信息时，会携带自己的 IP 地址。而如果 FE 发现 BE 汇报的 IP 地址和元数据中不一致时，就会出现如上错误。"],["   >","   > 这个错误的解决方法：1）分别在 FE 和 BE 设置 **priority_network** 参数。通常 FE 和 BE 都处于一个网段，所以该参数设置为相同即可。2）在 \"ADD BACKEND\" 语句中直接填写 BE 正确的 IP 地址而不是 hostname，以避免 FE 获取到错误的 IP 地址。","","5. BE 进程文件句柄数","","   BE进程文件句柄数，受min_file_descriptor_number/max_file_descriptor_number两个参数控制。","","   如果不在[min_file_descriptor_number, max_file_descriptor_number]区间内，BE进程启动会出错，可以使用ulimit进行设置。","","   min_file_descriptor_number的默认值为65536。"],["","   max_file_descriptor_number的默认值为131072.","","   举例而言：ulimit -n 65536; 表示将文件句柄设成65536。","","   启动BE进程之后，可以通过 cat /proc/$pid/limits 查看进程实际生效的句柄数","","   如果使用了supervisord，遇到句柄数错误，可以通过修改supervisord的minfds参数解决。","","   \"\"\"shell"],["   vim /etc/supervisord.conf","   ","   minfds=65535                 ; (min. avail startup file descriptors;default 1024)","   \"\"\"","---","{","      \"title\": \"基于 Doris-Operator 部署\",","      \"language\": \"zh-CN\"","}","---"],["","\x3C!--split-->","","Doris-Operator 是按照 Kubernetes 原则构建的在 Kubernetes 平台之上管理运维 Doris 集群的管理软件，允许用户按照资源定义的方式在 Kubernetes 平台之上部署管理 Doris 服务。Doris-Operator 能够管理 Doris 的所有部署形态，能够实现 Doris 大规模形态下智能化和并行化管理。","","## Kubernetes 上部署 Doris 集群","","### 环境准备","使用 Doris-Operator 部署 Doris 前提需要一个 Kubernetes (简称 K8S)集群，如果已拥有可直接跳过环境准备阶段。  ","  "],["**创建 K8S 集群**  ","  ","用户可在喜欢的云平台上申请云托管的 K8S 集群服务，例如：[阿里云的 ACK ](https://www.aliyun.com/product/kubernetes)或者[ 腾讯的 TKE ](https://cloud.tencent.com/product/tke)等等，也可以按照 [Kubernetes](https://kubernetes.io/docs/setup/) 官方推荐的方式手动搭建 K8S 集群。 ","- 创建 ACK 集群  ","您可按照阿里云官方文档在阿里云平台创建 [ACK 集群](https://help.aliyun.com/zh/ack/ack-managed-and-ack-dedicated/getting-started/getting-started/)。","- 创建 TKE 集群  ","如果你使用腾讯云可以按照腾讯云TKE相关文档创建 [TKE 集群](https://cloud.tencent.com/document/product/457/54231)。","- 创建私有集群  ","私有集群搭建，我们建议按照官方推荐的方式搭建，比如：[minikube](https://minikube.sigs.k8s.io/docs/start/)，[kOps](https://kubernetes.io/zh-cn/docs/setup/production-environment/tools/kops/)。",""],["### 部署 Doris-Operator","**1. 添加 DorisCluster [资源定义](https://kubernetes.io/zh-cn/docs/concepts/extend-kubernetes/api-extension/custom-resources/)**","\"\"\"shell","kubectl apply -f https://raw.githubusercontent.com/selectdb/doris-operator/master/config/crd/bases/doris.selectdb.com_dorisclusters.yaml    ","\"\"\"","**2. 部署 Doris-Operator**  ","**方式一：默认部署模式**  ","直接通过仓库中 Operator 的定义进行部署   ","\"\"\"shell","kubectl apply -f https://raw.githubusercontent.com/selectdb/doris-operator/master/config/operator/operator.yaml"],["\"\"\"","**方式二：自定义部署**  ","[operator.yaml](https://github.com/selectdb/doris-operator/blob/master/config/operator/operator.yaml) 中各个配置是部署 Operator 服务的最低要求。为提高管理效率或者有定制化的需求，下载 operator.yaml 进行自定义部署。  ","- 下载 Operator 的部署范例 [operator.yaml](https://raw.githubusercontent.com/selectdb/doris-operator/master/config/operator/operator.yaml)，可直接通过 wget 进行下载。","- 按期望更新 operator.yaml 中各种配置信息。","- 通过如下命令部署 Doris-Operator 服务。","\"\"\"shell","kubectl apply -f operator.yaml","\"\"\"","**3. 检查 Doris-Operator 服务部署状态**   "],["Operator 服务部署后，可通过如下命令查看服务的状态。当\"STATUS\"为\"Running\"状态，且 pod 中所有容器都为\"Ready\"状态时服务部署成功。","\"\"\""," kubectl -n doris get pods"," NAME                              READY   STATUS    RESTARTS        AGE"," doris-operator-5b9f7f57bf-tsvjz   1/1     Running   66 (164m ago)   6d22h","\"\"\"","operator.yaml 中 namespace 默认为 Doris，如果更改了 namespace，在查询服务状态的时候请替换正确的 namespace 名称。","### 部署 Doris 集群","**1. 部署集群**   ","\"Doris-Operator\"仓库的 [doc/examples](https://github.com/selectdb/doris-operator/tree/master/doc/examples) 目录提供众多场景的使用范例，可直接使用范例进行部署。以最基础的范例为例：  "],["\"\"\"","kubectl apply -f https://raw.githubusercontent.com/selectdb/doris-operator/master/doc/examples/doriscluster-sample.yaml","\"\"\"","在 Doris-Operator 仓库中，[how_to_use.md](https://github.com/selectdb/doris-operator/tree/master/doc/how_to_use_cn.md) 梳理了 Operator 管理运维 Doris 集群的主要能力，[DorisCluster](https://github.com/selectdb/doris-operator/blob/master/api/doris/v1/types.go) 展示了资源定义和从属结构，[api.md](https://github.com/selectdb/doris-operator/tree/master/doc/api.md) 可读性展示了资源定义和从属结构。可根据相关文档规划部署 Doris 集群。  ","","**2. 检测集群状态**","- 检查所有 pod 的状态  ","  集群部署资源下发后，通过如下命令检查集群状态。当所有 pod 的\"STATUS\"都是\"Running\"状态， 且所有组件的 pod 中所有容器都\"READY\"表示整个集群部署正常。","  \"\"\"shell","  kubectl get pods"],["  NAME                       READY   STATUS    RESTARTS   AGE","  doriscluster-sample-fe-0   1/1     Running   0          20m","  doriscluster-sample-be-0   1/1     Running   0          19m","  \"\"\"","- 检查部署资源状态  ","  Doris-Operator 会收集集群服务的状态显示到下发的资源中。Doris-Operator 定义了\"DorisCluster\"类型资源名称的简写\"dcr\"，在使用资源类型查看集群状态时可用简写替代。当配置的相关服务的\"STATUS\"都为\"available\"时，集群部署成功。","  \"\"\"shell","  kubectl get dcr","  NAME                  FESTATUS    BESTATUS    CNSTATUS   BROKERSTATUS","  doriscluster-sample   available   available"],["  \"\"\"","### 访问集群","Doris-Operator 为每个组件提供 K8S 的 Service 作为访问入口，可通过\"kubectl -n {namespace} get svc -l \"app.doris.ownerreference/name={dorisCluster.Name}\"\"来查看 Doris 集群有关的 Service。\"dorisCluster.Name\"为部署\"DorisCluster\"资源定义的名称。","\"\"\"shell","kubectl -n default get svc -l \"app.doris.ownerreference/name=doriscluster-sample\"","NAME                              TYPE        CLUSTER-IP       EXTERNAL-IP                                           PORT(S)                               AGE","doriscluster-sample-fe-internal   ClusterIP   None             <none>                                                9030/TCP                              30m","doriscluster-sample-fe-service    ClusterIP   10.152.183.37    a7509284bf3784983a596c6eec7fc212-618xxxxxx.com        8030/TCP,9020/TCP,9030/TCP,9010/TCP   30m","doriscluster-sample-be-internal   ClusterIP   None             <none>                                                9050/TCP                              29m","doriscluster-sample-be-service    ClusterIP   10.152.183.141   <none>                                                9060/TCP,8040/TCP,9050/TCP,8060/TCP   29m"],["\"\"\"","Doris-Operator 部署的 Service 分为两类，后缀\"-internal\"为集群内部组件通信使用的 Service，后缀\"-service\"为用户可使用的 Service。 ","  ","**集群内部访问**  ","  ","在 K8S 内部可通过 Service 的\"CLUSTER-IP\"访问对应的组件。如上图可使用访问 FE 的 Service\"doriscluster-sample-fe-service\"对应的 CLUSTER-IP 为\"10.152.183.37\"，使用如下命令连接 FE 服务。","\"\"\"shell","mysql -h 10.152.183.37 -uroot -P9030","\"\"\"","  "],["**集群外部访问**  ","  ","Doris 集群部署默认不提供 K8S 外部访问，如果集群需要被集群外部访问，需要集群能够申请 lb 资源。具备前提后，参考 [api.md](https://github.com/selectdb/doris-operator/blob/master/doc/api.md) 文档配置相关组件\"service\"字段，部署后通过对应 Service 的\"EXTERNAL-IP\"进行访问。以上图中 FE 为例，使用如下命令连接：","\"\"\"shell","mysql -h a7509284bf3784983a596c6eec7fc212-618xxxxxx.com -uroot -P9030","\"\"\"","### 后记","本文简述 Doris 在 Kubernetes 的部署使用，Doris-Operator 提供的其他能力请参看[主要能力介绍](https://github.com/selectdb/doris-operator/tree/master/doc/how_to_use_cn.md)，DorisCluster 资源的 [api](https://github.com/selectdb/doris-operator/blob/master/doc/api.md) 可读性文档定制化部署 Doris 集群。","","---"],["{","    \"title\": \"迁移tablet\",","    \"language\": \"zh-CN\"","}","---","","\x3C!--split-->","","# 迁移tablet",""],["## Request","","\"GET /api/tablet_migration?goal={enum}&tablet_id={int}&schema_hash={int}&disk={string}\"","","## Description","","在BE节点上迁移单个tablet到指定磁盘","","## Query parameters",""],["* \"goal\"","    - \"run\"：提交迁移任务","    - \"status\"：查询任务的执行状态","","* \"tablet_id\"","    需要迁移的tablet的id","","* \"schema_hash\"","    schema hash",""],["* \"disk\"","    目标磁盘。    ","","## Request body","","无","","## Response","","### 提交结果"],["","\"\"\"","    {","        status: \"Success\",","        msg: \"migration task is successfully submitted.\"","    }","\"\"\"","或","\"\"\"","    {"],["        status: \"Fail\",","        msg: \"Migration task submission failed\"","    }","\"\"\"","","### 执行状态","","\"\"\"","    {","        status: \"Success\","],["        msg: \"migration task is running\",","        dest_disk: \"xxxxxx\"","    }","\"\"\"","","或","","\"\"\"","    {","        status: \"Success\","],["        msg: \"migration task has finished successfully\",","        dest_disk: \"xxxxxx\"","    }","\"\"\"","","或","","\"\"\"","    {","        status: \"Success\","],["        msg: \"migration task failed.\",","        dest_disk: \"xxxxxx\"","    }","\"\"\"","","## Examples","","","    \"\"\"","    curl \"http://127.0.0.1:8040/api/tablet_migration?goal=run&tablet_id=123&schema_hash=333&disk=/disk1\""],["","    \"\"\"","","---","{","    \"title\": \"检查tablet文件丢失\",","    \"language\": \"zh-CN\"","}","---",""],["\x3C!--split-->","","# 检查tablet文件丢失","","## Request","","\"GET /api/check_tablet_segment_lost?repair={bool}\"","","## Description",""],["在BE节点上，可能会因为一些异常情况导致数据文件丢失，但是元数据显示正常，这种副本异常不会被FE检测到，也不能被修复。","当用户查询时，会报错\"failed to initialize storage reader\"。该接口的功能是检测出当前BE节点上所有存在文件丢失的tablet。","","## Query parameters","","* \"repair\"","","    - 设置为\"true\"时，存在文件丢失的tablet都会被设为\"SHUTDOWN\"状态，该副本会被作为坏副本处理，进而能够被FE检测和修复。","    - 设置为\"false\"时，只会返回所有存在文件丢失的tablet，并不做任何处理。",""],["## Request body","","无","","## Response","","    返回值是当前BE节点上所有存在文件丢失的tablet","","    \"\"\"","    {"],["        status: \"Success\",","        msg: \"Succeed to check all tablet segment\",","        num: 3,","        bad_tablets: [","            11190,","            11210,","            11216","        ],","        set_bad: true,","        host: \"172.3.0.101\""],["    }","    \"\"\"","","## Examples","","","    \"\"\"","    curl http://127.0.0.1:8040/api/check_tablet_segment_lost?repair=false","    \"\"\"",""],["---","{","    \"title\": \"查询元信息\",","    \"language\": \"zh-CN\"","}","---","","\x3C!--split-->","","# 查询元信息"],["","## Request","","\"GET /api/meta/header/{tablet_id}?byte_to_base64={bool}\"","","## Description","","查询tablet元信息","","## Path parameters"],["","* \"tablet_id\"","    table的id","","## Query parameters","","* \"byte_to_base64\"","    是否按base64编码，选填，默认\"false\"。","","## Request body"],["","无","","## Response","","    \"\"\"","    {","        \"table_id\": 148107,","        \"partition_id\": 148104,","        \"tablet_id\": 148193,"],["        \"schema_hash\": 2090621954,","        \"shard_id\": 38,","        \"creation_time\": 1673253868,","        \"cumulative_layer_point\": -1,","        \"tablet_state\": \"PB_RUNNING\",","        ...","    }","    \"\"\"","## Examples",""],["","    \"\"\"","    curl \"http://127.0.0.1:8040/api/meta/header/148193&byte_to_base64=true\"","","    \"\"\"","","---","{","    \"title\": \"做快照\",","    \"language\": \"zh-CN\""],["}","---","","\x3C!--split-->","","# 做快照","","## Request","","\"GET /api/snapshot?tablet_id={int}&schema_hash={int}\"\""],["","## Description","","该功能用于tablet做快照。","","## Query parameters","","* \"tablet_id\"","    需要做快照的table的id",""],["* \"schema_hash\"","    schema hash         ","","","## Request body","","无","","## Response",""],["    \"\"\"","    /path/to/snapshot","    \"\"\"","## Examples","","","    \"\"\"","    curl \"http://127.0.0.1:8040/api/snapshot?tablet_id=123456&schema_hash=1111111\"","","    \"\"\""],["","---","{","    \"title\": \"重加载tablet\",","    \"language\": \"zh-CN\"","}","---","","\x3C!--split-->",""],["# 重加载tablet","","## Request","","\"GET /api/reload_tablet?tablet_id={int}&schema_hash={int}&path={string}\"\"","","## Description","","该功能用于重加载tablet数据。",""],["## Query parameters","","* \"tablet_id\"","    需要重加载的table的id","","* \"schema_hash\"","    schema hash      ","","* \"path\"","    文件路径     "],["","","## Request body","","无","","## Response","","    \"\"\"","    load header succeed"],["    \"\"\"","## Examples","","","    \"\"\"","    curl \"http://127.0.0.1:8040/api/reload_tablet?tablet_id=123456&schema_hash=1111111&path=/abc\"","","    \"\"\"","","---"],["{","    \"title\": \"触发Compaction\",","    \"language\": \"zh-CN\"","}","---","","\x3C!--split-->","","# 触发Compaction",""],["## Request","","\"POST /api/compaction/run?tablet_id={int}&compact_type={enum}\"","\"POST /api/compaction/run?table_id={int}&compact_type=full\" 注意，table_id=xxx只有在compact_type=full时指定才会生效。","\"GET /api/compaction/run_status?tablet_id={int}\"","","","## Description","","用于手动触发 Compaction 以及状态查询。"],["","## Query parameters","","* \"tablet_id\"","    - tablet的id","","* \"table_id\"","    - table的id。注意，table_id=xxx只有在compact_type=full时指定才会生效，并且tablet_id和table_id只能指定一个，不能够同时指定，指定table_id后会自动对此table下所有tablet执行full_compaction。","","* \"compact_type\""],["    - 取值为\"base\"或\"cumulative\"或\"full\"。full_compaction的使用场景请参考[数据恢复](../../data-admin/data-recovery.md)。","","## Request body","","无","","## Response","","### 触发Compaction",""],["若 tablet 不存在，返回 JSON 格式的错误：","","\"\"\"","{","    \"status\": \"Fail\",","    \"msg\": \"Tablet not found\"","}","\"\"\"","","若 compaction 执行任务触发失败时，返回 JSON 格式的错误："],["","\"\"\"","{","    \"status\": \"Fail\",","    \"msg\": \"fail to execute compaction, error = -2000\"","}","\"\"\"","","若 compaction 执行触发成功时，则返回 JSON 格式的结果:",""],["\"\"\"","{","    \"status\": \"Success\",","    \"msg\": \"compaction task is successfully triggered.\"","}","\"\"\"","","结果说明：","","* status：触发任务状态，当成功触发时为Success；当因某些原因（比如，没有获取到合适的版本）时，返回Fail。"],["* msg：给出具体的成功或失败的信息。","","### 查询状态","","若 tablet 不存在，返回 JSON 格式：","","\"\"\"","{","    \"status\": \"Fail\",","    \"msg\": \"Tablet not found\""],["}","\"\"\"","","若 tablet 存在并且 tablet 不在正在执行 compaction，返回 JSON 格式：","","\"\"\"","{","    \"status\" : \"Success\",","    \"run_status\" : false,","    \"msg\" : \"this tablet_id is not running\","],["    \"tablet_id\" : 11308,","    \"schema_hash\" : 700967178,","    \"compact_type\" : \"\"","}","\"\"\"","","若 tablet 存在并且 tablet 正在执行 compaction，返回 JSON 格式：","","\"\"\"","{"],["    \"status\" : \"Success\",","    \"run_status\" : true,","    \"msg\" : \"this tablet_id is running\",","    \"tablet_id\" : 11308,","    \"schema_hash\" : 700967178,","    \"compact_type\" : \"cumulative\"","}","\"\"\"","","结果说明："],[""]])
passages = [' '.join([line.strip() for line in p]) for p in passages]
def do_build():
    # Load docs
    vault = get_vault_dict()
    logger.info(f'Vault length: {len(vault):,}')

    # Load tokenizer and model
    tokenizer, model = get_model_tuple()

    # Build and save embedding index and array
    builder = NumpyIndexBuilder(vault, get_doc_dict(), tokenizer, model)
    builder.build_index_mapping(vault)
    index_mapping = builder.get_index_mapping()
    logger.info(f'Embedding index length: {len(index_mapping):,}')

    start_time = time.time()

    builder.build()
    doc_embeddings_array = builder.get_embeddings_array()
    logger.info(f'Time taken for embedding all chunks: {time.time() - start_time} seconds')
    assert len(index_mapping) == builder.get_embeddings_array().shape[0], 'Length of embedding index != embedding count'

def do_query():
    # Load docs
    vault = None
    doc = None
    vault = get_vault_dict()
    doc = get_doc_dict()
    tokenizer, model = get_model_tuple()
    builder = NumpyIndexBuilder(vault, doc, tokenizer, model)
    query_list = ['行Segment写入测试', 'Segment', 'SegmentWrite', '怎么初始化 CustomBenchmark？',
                  'benchmark_tool 怎么用？']
    query_list = ['想请教一下大佬们，如果我想新增一列时间列，这张表里任一字段的数据更新，这个时间列都会自动更新时间，Doris支持这种操作么？还是说，得在数据导入前给每列数据加一个最新时间数据再进行导入Doris,',
            '如何在Apache Doris中实现节点缩容？',
            '在Doris中新建了一张表，在MySQL中可以查到，但在Doris中查不到，怎么办？',
            '在使用mysql -doris整库同步时，如果在mysql中增加字段，增加表，doris会自动同步过去吗？还是需要手动在doris中操作？测试了一下，增加字段可以同步，但增加表不行，修改字段也不同步，需要做什么操作吗？',
            '部分列更新 set enable_unique_key_partial_update=true 这个参数可以设置全局吗,',
            '大佬，doris可以把map结构的数据，里面的k和v转化为 列吗,',
            '我之前json数据我创建为text了，现在想要修改为json类型的，发现报错，不知道怎么回事,',
            'doris可以修改列的数据类型嘛,',
            '@Petrichor  大佬您好， unuqie模型可以修改列名称吗？ doris 1.2,',
            'Doris是否支持RAID？ ',
            'Doris支持哪些云存储服务？',
            '如何在Apache Doris中创建外部表访问SqlServer？有参考样例可以看吗？',
            'Apache Doris的OUTFILE支持导出阿里的OSS吗？',
            "在执行BACKUP SNAPSHOT后，通过show backup查看备份状态时，报错ERROR 2…igger than 'max_allowed_packet' bytes，有遇到过这个问题的吗？",
            'Doris中对分区数有没有限制？一张表最大能添加多少个分区？',
            '对历史数据如果要搞分区有什么好的解决方案吗？',
            'Doris 2.x版本的FE支持切换catalog吗？',
            '我使用insert into values语句插入了8819条数据，但在Apache Doris库里只查询到了7400多条数据，日志只有内存发生gc的告警。有遇到过这种情况吗？',
            '各位大佬，uniq key 或者那几个模型，对key的要求是不是不能用text之类的啊,我们要是主键包含text类型的应该怎么做啊,,',
            'Doris目前可以热挂载磁盘吗？',
            'routine load 作业配置了max_filter_ratio属性，暂停后能将这个属性给删除吗',
            '使用delete删除数据会有什么影响吗？ ',
            'Doris的BE挂的时候如果有数据通过streamload方式入库，这些数据丢失了，Doris有没有机制还原的？']
    for q in query_list:
        result = builder.query_semantic(q)
        print(f'query:{q}:')
        for i, r in enumerate(result):
            print(f'==>{i}: {r}')

        print('')

if __name__ == '__main__':
    do_query()



