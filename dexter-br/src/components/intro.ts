// Banner de abertura do Dexter BR.

import { config } from "../config.ts";
import { t } from "../theme.ts";

export interface IntroStatus {
  warnings: string[];
  toolCount: number;
  macroCount: number;
  skillCount: number;
  restOnline: boolean;
  webProvider: string | null;
}

export function renderIntro(s: IntroStatus): string {
  const lines: string[] = [];
  lines.push("");
  lines.push(`${t.headerBar("  DEXTER BR  ")}  ${t.muted("pesquisa financeira do Brasil no terminal")}`);
  lines.push("");
  lines.push(`${t.muted("Modelo:")} ${t.blue(config.model)}`);
  lines.push(`${t.muted("Ferramentas:")} ${t.ink(String(s.toolCount))}`);

  const chip = (label: string, on: boolean, offText = "off") =>
    on ? t.green(label) : t.yellow(`${label.split(" ")[0]} ${offText}`);
  const parts = [
    chip(`macro ${s.macroCount}`, s.macroCount > 0, "off (BRAZIL_AGENT_PATH)"),
    t.green("ações 3") + t.muted(" · quant 1"),
    chip(`web ${s.webProvider ?? ""}`.trim(), !!s.webProvider, "off (TAVILY/EXA)"),
    chip("painel-REST", s.restOnline, "off (api_server.py)"),
    chip(`skills ${s.skillCount}`, s.skillCount > 0, "off"),
  ];
  lines.push(`${t.muted("Camadas:")} ${parts.join(t.muted("  ·  "))}`);

  if (!config.anthropicApiKey) {
    lines.push(t.red("⚠ ANTHROPIC_API_KEY não definido — defina antes de perguntar."));
  }
  for (const w of s.warnings) lines.push(t.yellow(`⚠ ${w}`));
  lines.push("");
  lines.push(t.muted("Pergunte sobre Selic, IPCA, Focus, PIB, câmbio, ou ações da B3."));
  lines.push(t.muted("Comandos: /sair  ·  /limpar  ·  Ctrl+C para sair."));
  lines.push("");
  return lines.join("\n");
}
