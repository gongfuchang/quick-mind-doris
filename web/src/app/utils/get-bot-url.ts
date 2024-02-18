export const getBotUrl = (query: string) => {
  const bot_url = process.env.BOT_URL ?? "http://127.0.0.1:3000";
  return `${bot_url}?q=${encodeURIComponent(query)}`;
};
