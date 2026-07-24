#!/usr/bin/env bun
// Smoke test do bridge MCP -> brazil-agent. Não usa LLM.
// Requer BRAZIL_AGENT_PATH apontando para o repo brazil-agent com deps Python.

import { loadMacroTools } from "../src/tools/macro/mcp-bridge.ts";
import { config } from "../src/config.ts";

async function main() {
  console.log("BRAZIL_AGENT_PATH:", config.brazilAgentPath ?? "(não definido)");
  const { tools, servers, warnings } = await loadMacroTools();

  for (const w of warnings) console.warn("⚠", w);
  console.log(`\nFerramentas macro registradas: ${tools.length}`);
  for (const t of tools) console.log(`  · ${t.name}`);

  // tenta uma chamada leve se houver o servidor bcb_focus
  const focus = tools.find((t) => t.name.includes("bcb_focus__list_endpoints"));
  if (focus) {
    console.log("\n→", focus.name, "()");
    try {
      console.log(await focus.run({}));
    } catch (e) {
      console.error("  falhou:", (e as Error).message);
    }
  }

  for (const s of servers) s.stop();
}

main();
