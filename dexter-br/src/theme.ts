// Tema Flexoki Light para o terminal.
//
// Estratégia: usar sequências OSC 10/11 para pintar o terminal inteiro de
// "papel" (creme) com tinta escura — assim o visual "flexoki light" vale para
// qualquer terminal que honre OSC (a maioria dos modernos). As cores dos
// números/acento vêm do chalk (truecolor). No exit, restauramos fg/bg.
//
// Paleta Flexoki (mesmos hex de dashboard/utils.py no ia-economista).

import { Chalk } from "chalk";

export const flexoki = {
  paper: "#FFFCF0", // fundo (Base paper)
  paper2: "#F2F0E5", // fundo secundário (Base-50)
  ink: "#100F0F", // tinta (Black)
  inkMuted: "#6F6E69", // Base-600
  border: "#B7B5AC", // Base-300 (borda visível sobre papel)
  green: "#66800B", // Green-600 (alta / positivo)
  red: "#AF3029", // Red-600 (baixa / negativo)
  blue: "#205EA6", // Blue-600 (ação / agente)
  yellow: "#AD8301", // Yellow-600 (alerta)
  purple: "#5E409D", // Purple-600 (secundário)
  cyan: "#24837B", // Cyan-600
} as const;

// Força truecolor mesmo quando o ambiente não anuncia (Bun/pipes).
const c = new Chalk({ level: 3 });

export const t = {
  ink: c.hex(flexoki.ink),
  muted: c.hex(flexoki.inkMuted),
  green: c.hex(flexoki.green),
  red: c.hex(flexoki.red),
  blue: c.hex(flexoki.blue),
  yellow: c.hex(flexoki.yellow),
  purple: c.hex(flexoki.purple),
  cyan: c.hex(flexoki.cyan),
  border: c.hex(flexoki.border),
  bold: c.bold,
  dim: c.hex(flexoki.inkMuted),
  // realce de "chip" (fundo Base-50)
  chip: c.bgHex(flexoki.paper2).hex(flexoki.ink),
  headerBar: c.bgHex(flexoki.blue).hex(flexoki.paper).bold,
};

const ESC = "\x1b";
const OSC = `${ESC}]`; // Operating System Command
const BEL = "\x07"; // terminador

/** Pinta o terminal de papel/tinta (Flexoki Light). */
export function enterTheme(): void {
  if (!process.stdout.isTTY) return;
  // OSC 11 = background, OSC 10 = foreground
  process.stdout.write(`${OSC}11;${flexoki.paper}${BEL}`);
  process.stdout.write(`${OSC}10;${flexoki.ink}${BEL}`);
}

/** Restaura fg/bg padrão do terminal. */
export function exitTheme(): void {
  if (!process.stdout.isTTY) return;
  // OSC 111 / 110 = reset background / foreground
  process.stdout.write(`${OSC}111${BEL}`);
  process.stdout.write(`${OSC}110${BEL}`);
}

/** Colore um número conforme o sinal (verde=+, vermelho=−, tinta=0). */
export function bySign(text: string, value: number): string {
  if (value > 0) return t.green(text);
  if (value < 0) return t.red(text);
  return t.ink(text);
}
