// Renderizador de tabela "quadrada" estilo terminal (box-drawing), tipo Dexter.
// Larguras calculadas ignorando códigos ANSI (as células podem vir coloridas).

import { t } from "../theme.ts";

export type Align = "left" | "right" | "center";

export interface Column {
  header: string;
  align?: Align; // aplicado a header e células. default: "left"
}

const ANSI_RE = /\x1b\[[0-9;]*m/g;

export function stripAnsi(s: string): string {
  return s.replace(ANSI_RE, "");
}

export function visibleWidth(s: string): number {
  // conta code points visíveis (após remover ANSI). Acentos latinos = 1 coluna.
  return [...stripAnsi(s)].length;
}

function pad(s: string, width: number, align: Align): string {
  const gap = width - visibleWidth(s);
  if (gap <= 0) return s;
  const spaces = " ".repeat(gap);
  if (align === "right") return spaces + s;
  if (align === "center") {
    const left = Math.floor(gap / 2);
    return " ".repeat(left) + s + " ".repeat(gap - left);
  }
  return s + spaces;
}

/**
 * Renderiza uma tabela box-drawing.
 * @param columns definição das colunas (header + alinhamento)
 * @param rows matriz de células já formatadas (podem conter cor ANSI)
 */
export function renderTable(columns: Column[], rows: string[][]): string {
  const b = (s: string) => t.border(s);
  const n = columns.length;

  // largura de cada coluna = máx(header, células) + 2 de padding lateral
  const widths = columns.map((col, i) => {
    let w = visibleWidth(col.header);
    for (const row of rows) {
      const cell = row[i] ?? "";
      w = Math.max(w, visibleWidth(cell));
    }
    return w;
  });

  const line = (left: string, mid: string, right: string) =>
    b(left + widths.map((w) => "─".repeat(w + 2)).join(mid) + right);

  const rowLine = (cells: string[], headerRow = false) => {
    const parts = columns.map((col, i) => {
      const align = col.align ?? "left";
      const raw = cells[i] ?? "";
      const content = headerRow ? t.bold(raw) : raw;
      return " " + pad(content, widths[i]!, align) + " ";
    });
    return b("│") + parts.join(b("│")) + b("│");
  };

  const out: string[] = [];
  out.push(line("┌", "┬", "┐"));
  out.push(rowLine(columns.map((c) => c.header), true));
  out.push(line("├", "┼", "┤"));
  for (const row of rows) out.push(rowLine(row));
  out.push(line("└", "┴", "┘"));
  return out.join("\n");
}
