// Busca na web (Tavily ou Exa), gated por chave de ambiente.

import type { DexterTool } from "../../agent/types.ts";
import { config } from "../../config.ts";

interface Hit {
  title?: string;
  url?: string;
  content?: string;
}

async function tavily(query: string, maxResults: number): Promise<string> {
  const resp = await fetch("https://api.tavily.com/search", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      api_key: config.tavilyApiKey,
      query,
      max_results: maxResults,
      include_answer: true,
      search_depth: "basic",
    }),
  });
  if (!resp.ok) throw new Error(`Tavily ${resp.status}: ${(await resp.text()).slice(0, 160)}`);
  const json = (await resp.json()) as { answer?: string; results?: Hit[] };
  return JSON.stringify({
    fonte: "tavily",
    resumo: json.answer ?? null,
    resultados: (json.results ?? []).map((r) => ({
      titulo: r.title,
      url: r.url,
      trecho: (r.content ?? "").slice(0, 400),
    })),
  });
}

async function exa(query: string, maxResults: number): Promise<string> {
  const resp = await fetch("https://api.exa.ai/search", {
    method: "POST",
    headers: { "content-type": "application/json", "x-api-key": config.exaApiKey! },
    body: JSON.stringify({ query, numResults: maxResults, contents: { text: { maxCharacters: 400 } } }),
  });
  if (!resp.ok) throw new Error(`Exa ${resp.status}: ${(await resp.text()).slice(0, 160)}`);
  const json = (await resp.json()) as { results?: Array<Hit & { text?: string }> };
  return JSON.stringify({
    fonte: "exa",
    resultados: (json.results ?? []).map((r) => ({
      titulo: r.title,
      url: r.url,
      trecho: (r.text ?? r.content ?? "").slice(0, 400),
    })),
  });
}

/** Retorna a tool de busca web se houver chave; senão, lista vazia. */
export function webSearchTools(): DexterTool[] {
  const provider = config.tavilyApiKey ? "tavily" : config.exaApiKey ? "exa" : null;
  if (!provider) return [];
  return [
    {
      name: "busca_web",
      description:
        "Busca na web por notícias, contexto recente ou informação fora dos dados estruturados (ex.: decisão do Copom, fato relevante de uma empresa). Retorna resumo e links. Cite as fontes na resposta.",
      source: "web",
      inputSchema: {
        type: "object",
        properties: {
          query: { type: "string", description: "Consulta de busca." },
          max_results: { type: "integer", description: "Máximo de resultados (default 5)." },
        },
        required: ["query"],
      },
      label: (i) => `busca_web("${String(i.query ?? "").slice(0, 40)}")`,
      async run(input) {
        const query = String(input.query ?? "");
        if (!query) throw new Error("informe 'query'");
        const max = typeof input.max_results === "number" ? input.max_results : 5;
        return provider === "tavily" ? tavily(query, max) : exa(query, max);
      },
    },
  ];
}
