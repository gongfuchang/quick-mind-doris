"use client";
import { getSearchUrl } from "@/app/utils/get-search-url";
import { getBotUrl } from "@/app/utils/get-bot-url";
import { RefreshCcw, MessagesSquare } from "lucide-react";
import { nanoid } from "nanoid";
import { useRouter } from "next/navigation";

export const Title = ({ query }: { query: string }) => {
  const router = useRouter();
  return (
    <div className="flex items-center pb-4 mb-6 border-b gap-4">
      <div
        className="flex-1 text-lg sm:text-xl text-black text-ellipsis overflow-hidden whitespace-nowrap"
        title={query}
      >
        {query}
      </div>
      <div className="flex-none">
        <button
          onClick={() => {
            router.push(
              getSearchUrl(encodeURIComponent(query), nanoid(), "glm"),
            );
          }}
          type="button"
          className="rounded flex gap-2 items-center bg-transparent px-2 py-1 text-xs font-semibold text-blue-500 hover:bg-zinc-100"
        >
          <RefreshCcw size={12}></RefreshCcw>重新回答
        </button>
      </div>
      <div className="flex-none">
        <button
          onClick={() => {
            router.push(
              getSearchUrl(encodeURIComponent(query), nanoid(), "gpt"),
            );
          }}
          type="button"
          className="rounded flex gap-2 items-center bg-transparent px-2 py-1 text-xs font-semibold text-blue-500 hover:bg-zinc-100"
        >
          <RefreshCcw size={12}></RefreshCcw>GPT4回答
        </button>
      </div>
      <div className="flex-none">
        <a href={getBotUrl(query)} target="_blank">
          <button
            type="button"
            className="rounded flex gap-2 items-center bg-transparent px-2 py-1 text-xs font-semibold text-blue-500 hover:bg-zinc-100"
          >
            <MessagesSquare size={12}></MessagesSquare>多轮对话
          </button>
        </a>
      </div>
    </div>
  );
};
