export const getSearchUrl = (
  query: string,
  search_uuid: string,
  llm_type: string,
) => {
  const prefix = "/search";
  return `${prefix}?q=${encodeURIComponent(query)}&rid=${search_uuid}&llm_type=${llm_type}`;
};
