// Painel de KPIs via a API REST FastAPI do brazil-agent (/api/v1/...).
// Só registra as tools se a API estiver de pé (probe em /health).

import type { DexterTool } from "../../agent/types.ts";
import { config } from "../../config.ts";

async function apiGet(path: string, timeoutMs = 8000): Promise<unknown> {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const resp = await fetch(`${config.macroApiUrl}${path}`, { signal: ctrl.signal });
    if (!resp.ok) throw new Error(`API ${resp.status} em ${path}`);
    return await resp.json();
  } finally {
    clearTimeout(timer);
  }
}

async function isHealthy(): Promise<boolean> {
  try {
    const h = (await apiGet("/api/v1/health", 2000)) as { status?: string };
    return h?.status === "healthy" || h?.status === "degraded";
  } catch {
    return false;
  }
}

function trunc(v: unknown, max = 8000): string {
  const s = JSON.stringify(v);
  return s.length > max ? s.slice(0, max) + "\n… [truncado]" : s;
}

function buildTools(): DexterTool[] {
  const painel: DexterTool = {
    name: "painel_kpis",
    description:
      "Painel de KPIs macro pré-computados do brazil-agent (Selic, IPCA, câmbio, PIB, desemprego…) com valor atual, variação, tendência e status vs meta. Fonte: API REST do brazil-agent.",
    source: "rest",
    inputSchema: { type: "object", properties: {} },
    label: () => "painel_kpis()",
    async run() {
      return trunc(await apiGet("/api/v1/indicators/latest"));
    },
  };
  const lista: DexterTool = {
    name: "indicadores_lista",
    description: "Lista os indicadores disponíveis no painel do brazil-agent (id, nome, período, último valor).",
    source: "rest",
    inputSchema: { type: "object", properties: {} },
    label: () => "indicadores_lista()",
    async run() {
      return trunc(await apiGet("/api/v1/indicators/list"));
    },
  };
  const serie: DexterTool = {
    name: "indicador_serie",
    description:
      "Série temporal de um indicador do painel do brazil-agent (ex.: selic, ipca, cambio, pib). Parâmetro 'months' controla a janela.",
    source: "rest",
    inputSchema: {
      type: "object",
      properties: {
        id: { type: "string", description: "id do indicador (ex.: selic, ipca, cambio, pib)." },
        months: { type: "integer", description: "meses de histórico (default 24)." },
      },
      required: ["id"],
    },
    label: (i) => `indicador_serie(${String(i.id ?? "")})`,
    async run(input) {
      const id = String(input.id ?? "");
      if (!id) throw new Error("informe 'id'");
      const months = typeof input.months === "number" ? input.months : 24;
      return trunc(await apiGet(`/api/v1/indicators/${encodeURIComponent(id)}/timeseries?months=${months}`));
    },
  };
  return [painel, lista, serie];
}

export interface RestPanel {
  tools: DexterTool[];
  online: boolean;
}

export async function restPanelTools(): Promise<RestPanel> {
  const online = await isHealthy();
  return { tools: online ? buildTools() : [], online };
}
