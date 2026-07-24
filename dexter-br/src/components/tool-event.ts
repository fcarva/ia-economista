// Status lines de execução de ferramenta, estilo Dexter.

import { t } from "../theme.ts";

export function toolStartLine(label: string): string {
  return `${t.blue("●")} ${t.ink(label)}`;
}

export function toolEndLine(ms: number, ok: boolean): string {
  const secs = (ms / 1000).toFixed(1);
  const mark = ok ? t.green("✓") : t.red("✗");
  const word = ok ? "concluído" : "falhou";
  return `  ${t.muted("└─")} ${mark} ${t.muted(`${word} em ${secs}s`)}`;
}

export function generatingLine(): string {
  return t.muted("· sintetizando resposta…");
}
