// Monta o conjunto de ferramentas do agente: equities (brapi) + macro (bridge MCP).

import type { DexterTool } from "../agent/types.ts";
import { equitiesTools } from "./finance/brapi.ts";
import { loadMacroTools, type SpawnedMcpServer } from "./macro/mcp-bridge.ts";

export interface ToolSet {
  tools: DexterTool[];
  servers: SpawnedMcpServer[];
  warnings: string[];
}

export async function assembleTools(): Promise<ToolSet> {
  const macro = await loadMacroTools();
  const equities = equitiesTools();
  return {
    tools: [...equities, ...macro.tools],
    servers: macro.servers,
    warnings: macro.warnings,
  };
}
