// Banner de abertura do Dexter BR.

import { config } from "../config.ts";
import { t } from "../theme.ts";

export function renderIntro(warnings: string[], toolCount: number, macroCount: number): string {
  const lines: string[] = [];
  lines.push("");
  lines.push(`${t.headerBar("  DEXTER BR  ")}  ${t.muted("pesquisa financeira do Brasil no terminal")}`);
  lines.push("");
  lines.push(`${t.muted("Modelo:")} ${t.blue(config.model)}`);
  const macroStatus =
    macroCount > 0
      ? t.green(`macro ${macroCount} tools`)
      : t.yellow("macro off (defina BRAZIL_AGENT_PATH)");
  lines.push(`${t.muted("Ferramentas:")} ${t.ink(String(toolCount))}  ${t.muted("·")}  ${macroStatus}`);
  if (!config.anthropicApiKey) {
    lines.push(t.red("⚠ ANTHROPIC_API_KEY não definido — defina antes de perguntar."));
  }
  for (const w of warnings) lines.push(t.yellow(`⚠ ${w}`));
  lines.push("");
  lines.push(t.muted("Pergunte sobre Selic, IPCA, Focus, PIB, câmbio, ou ações da B3."));
  lines.push(t.muted("Comandos: /sair  ·  /limpar  ·  Ctrl+C para sair."));
  lines.push("");
  return lines.join("\n");
}
