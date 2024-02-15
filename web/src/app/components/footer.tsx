import { Mails } from "lucide-react";
import { FC } from "react";

export const Footer: FC = () => {
  return (
    <div className="text-center flex flex-col items-center text-xs text-zinc-700 gap-1">
      <div className="text-zinc-400">
        本知识库是基于 LLM/Transform/Vector DB/OpenSearch 等技术构建的
      </div>
      <div className="text-zinc-400">
        Retrieval Augmented Generation (RAG) 企业级产品。
      </div>
      <div className="text-zinc-400">
        部分 AIGC 幻觉（Hallucination），请酌情验证。
      </div>
      <div className="flex gap-2 justify-center">
        <div>如需 Quick Mind 在线试用，请</div>
        <div>
          <a
            className="text-blue-500 font-medium inline-flex gap-1 items-center flex-nowrap text-nowrap"
            href="mailto:gfc@dandantech.top"
          >
            <Mails size={8} />
            联系我们
          </a>
        </div>
      </div>

      <div className="flex items-center justify-center flex-wrap gap-x-4 gap-y-2 mt-2 text-zinc-400">
        <a className="hover:text-zinc-950" href="https://danflow.cn/">
          DanFlow@2024
        </a>
        <a
          className="hover:text-zinc-950"
          href="https://github.com/gongfuchang/quick_mind_doris"
        >
          Github
        </a>
        <a className="hover:text-zinc-950" href="https://twitter.com/danflow">
          X(Twitter)
        </a>
        <a
          className="hover:text-zinc-950"
          href="https://beian.miit.gov.cn/"
          target="_blank"
        >
          浙ICP备2023007228号-1
        </a>
      </div>
    </div>
  );
};
