// Configuração central do Dexter BR (lida de variáveis de ambiente).
// Bun carrega .env automaticamente.

import { existsSync } from "node:fs";

export interface DexterConfig {
  anthropicApiKey: string | undefined;
  model: string;
  brazilAgentPath: string | undefined;
  python: string;
  brapiToken: string | undefined;
  tavilyApiKey: string | undefined;
  /** Universo padrão de blue chips da B3 (de cointegration_gnn/config.py). */
  b3Universe: string[];
}

function firstExistingPath(...candidates: (string | undefined)[]): string | undefined {
  for (const c of candidates) {
    if (c && existsSync(c)) return c;
  }
  return undefined;
}

export const config: DexterConfig = {
  anthropicApiKey: process.env.ANTHROPIC_API_KEY,
  // Guia da skill claude-api: default para o modelo Claude mais capaz.
  model: process.env.DEXTER_MODEL || "claude-opus-4-8",
  brazilAgentPath: firstExistingPath(
    process.env.BRAZIL_AGENT_PATH,
    "/workspace/brazil-agent",
  ),
  python: process.env.DEXTER_PYTHON || "python3",
  brapiToken: process.env.BRAPI_TOKEN,
  tavilyApiKey: process.env.TAVILY_API_KEY,
  b3Universe: [
    "PETR4",
    "VALE3",
    "ITUB4",
    "BBDC4",
    "ABEV3",
    "B3SA3",
    "WEGE3",
    "RENT3",
    "BBAS3",
  ],
};
