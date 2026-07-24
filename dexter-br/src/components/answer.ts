// Renderiza a resposta (markdown-lite) do modelo com estética Flexoki:
// tabelas markdown viram tabelas box-drawing; **negrito**, títulos, bullets e
// percentuais com sinal ganham cor.

import { renderTable, type Align, type Column } from "../utils/box-table.ts";
import { t } from "../theme.ts";

const TABLE_ROW = /^\s*\|.*\|\s*$/;
const TABLE_SEP = /^\s*\|?[\s:|-]+\|?\s*$/;

function splitRow(line: string): string[] {
  return line
    .trim()
    .replace(/^\|/, "")
    .replace(/\|$/, "")
    .split("|")
    .map((s) => s.trim());
}

function isSeparator(line: string): boolean {
  if (!TABLE_SEP.test(line)) return false;
  return splitRow(line).every((c) => /^:?-{2,}:?$/.test(c.replace(/\s/g, "")) || c === "");
}

function alignOf(sepCell: string): Align {
  const c = sepCell.trim();
  const left = c.startsWith(":");
  const right = c.endsWith(":");
  if (left && right) return "center";
  if (right) return "right";
  return "left";
}

/** Colore percentuais com sinal (+verde / −vermelho) e aplica **negrito**. */
export function colorInline(text: string): string {
  let out = text.replace(/\*\*(.+?)\*\*/g, (_m, g1: string) => t.bold(g1));
  out = out.replace(/([+\-−]\s?\d[\d.,]*\s?%)/g, (m) => {
    const neg = m.trimStart().startsWith("-") || m.trimStart().startsWith("−");
    return neg ? t.red(m) : t.green(m);
  });
  return out;
}

function colorCell(cell: string): string {
  const c = colorInline(cell);
  const trimmed = cell.trim();
  if (/^\+/.test(trimmed)) return t.green(c);
  if (/^[-−]/.test(trimmed) && /\d/.test(trimmed)) return t.red(c);
  return c;
}

function renderMarkdownTable(block: string[]): string {
  const headerCells = splitRow(block[0]!);
  const aligns = splitRow(block[1]!).map(alignOf);
  const columns: Column[] = headerCells.map((h, i) => ({ header: h, align: aligns[i] ?? "left" }));
  const rows = block.slice(2).map((line) => {
    const cells = splitRow(line);
    return columns.map((_, i) => colorCell(cells[i] ?? ""));
  });
  return renderTable(columns, rows);
}

function renderProseLine(line: string): string {
  // títulos
  const h = line.match(/^(#{1,3})\s+(.*)$/);
  if (h) {
    const text = h[2]!;
    if (h[1] === "#") return t.bold(t.blue(text.toUpperCase()));
    return t.bold(text);
  }
  // bullets
  const b = line.match(/^(\s*)[-*]\s+(.*)$/);
  if (b) return `${b[1]}${t.blue("•")} ${colorInline(b[2]!)}`;
  return colorInline(line);
}

export function renderAnswer(markdown: string): string {
  const lines = markdown.replace(/\r/g, "").split("\n");
  const out: string[] = [];
  let i = 0;
  while (i < lines.length) {
    const line = lines[i]!;
    if (TABLE_ROW.test(line) && i + 1 < lines.length && isSeparator(lines[i + 1]!)) {
      const block: string[] = [line, lines[i + 1]!];
      i += 2;
      while (i < lines.length && TABLE_ROW.test(lines[i]!) && !isSeparator(lines[i]!)) {
        block.push(lines[i]!);
        i += 1;
      }
      out.push(renderMarkdownTable(block));
      continue;
    }
    out.push(renderProseLine(line));
    i += 1;
  }
  return out.join("\n");
}
