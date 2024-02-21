import { NextRequest } from "next/server";

async function createStream(req: NextRequest) {
  const encoder = new TextEncoder();
  const decoder = new TextDecoder();

  const query = req.nextUrl.searchParams.get("query");
  const llmType = req.nextUrl.searchParams.get("llm_type");
  const prefix = (llmType == "gpt" ? process.env.SEARCH_GPT_SERVER_URL : process.env.SEARCH_SERVER_URL);

  const query_url = prefix ?? "http://127.0.0.1:8000";
  const res = await fetch(
    `${query_url}/query?query=${query}&generate_related_questions=1`,
  );

  const contentType = res.headers.get("Content-Type") ?? "";
  if (!contentType.includes("stream")) {
    const content = await (
      await res.text()
    ).replace(/provided:.*. You/, "provided: ***. You");
    console.log("[Stream] error ", content);
    return "```json\n" + content + "```";
  }
  return res.body;
}

export async function GET(req: NextRequest) {
  try {
    const stream = await createStream(req);
    return new Response(stream);
  } catch (error) {
    console.trace("[Chat Stream]", error);
    return new Response(
      ["```json\n", JSON.stringify(error, null, "  "), "\n```"].join(""),
    );
  }
}

export const config = {
  runtime: "edge",
};
