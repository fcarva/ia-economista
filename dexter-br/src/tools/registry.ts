// Monta o conjunto de ferramentas do agente:
// ações (brapi) + quant (cointegração) + web + macro (MCP) + painel REST + skills.

import type { DexterTool } from "../agent/types.ts";
import { config } from "../config.ts";
import { equitiesTools } from "./finance/brapi.ts";
import { quantTools } from "./quant/cointegration.ts";
import { webSearchTools } from "./search/web-search.ts";
import { restPanelTools } from "./macro/rest-panel.ts";
import { loadMacroTools, type SpawnedMcpServer } from "./macro/mcp-bridge.ts";
import { loadSkills, skillsSummary as summarize, skillTool } from "../skills/loader.ts";

export interface ToolSet {
  tools: DexterTool[];
  servers: SpawnedMcpServer[];
  warnings: string[];
  skillsSummary: string;
  macroCount: number;
  skillCount: number;
  restOnline: boolean;
  webProvider: string | null;
}

export async function assembleTools(): Promise<ToolSet> {
  const warnings: string[] = [];

  const macro = await loadMacroTools();
  warnings.push(...macro.warnings);

  const rest = await restPanelTools();
  const skills = loadSkills();

  const tools: DexterTool[] = [
    ...equitiesTools(),
    ...quantTools(),
    ...webSearchTools(),
    ...macro.tools,
    ...rest.tools,
  ];
  if (skills.length) tools.push(skillTool(skills));

  return {
    tools,
    servers: macro.servers,
    warnings,
    skillsSummary: skills.length ? summarize(skills) : "",
    macroCount: macro.tools.length,
    skillCount: skills.length,
    restOnline: rest.online,
    webProvider: config.tavilyApiKey ? "tavily" : config.exaApiKey ? "exa" : null,
  };
}
