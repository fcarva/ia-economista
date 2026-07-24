// Formatação numérica em pt-BR (vírgula decimal, ponto de milhar) + cores por sinal.

import { t } from "../theme.ts";

const nf = (min: number, max: number) =>
  new Intl.NumberFormat("pt-BR", {
    minimumFractionDigits: min,
    maximumFractionDigits: max,
  });

/** 0.182 -> "18,2%" (recebe fração). Com sinal opcional (+/−). */
export function fmtPercent(fraction: number | null | undefined, decimals = 1, signed = false): string {
  if (fraction == null || Number.isNaN(fraction)) return "—";
  const pct = fraction * 100;
  const s = nf(decimals, decimals).format(pct);
  return `${signed && pct > 0 ? "+" : ""}${s}%`;
}

/** 18.2 -> "18,2%" (recebe já em pontos percentuais). */
export function fmtPercentPoints(pct: number | null | undefined, decimals = 1, signed = false): string {
  if (pct == null || Number.isNaN(pct)) return "—";
  const s = nf(decimals, decimals).format(pct);
  return `${signed && pct > 0 ? "+" : ""}${s}%`;
}

export function fmtNumber(n: number | null | undefined, decimals = 2): string {
  if (n == null || Number.isNaN(n)) return "—";
  return nf(decimals, decimals).format(n);
}

export function fmtBRL(n: number | null | undefined, decimals = 2): string {
  if (n == null || Number.isNaN(n)) return "—";
  return `R$ ${nf(decimals, decimals).format(n)}`;
}

/** Valor grande em R$ compacto: 77.3e9 -> "R$ 77,3 bi". */
export function fmtBig(n: number | null | undefined): string {
  if (n == null || Number.isNaN(n)) return "—";
  const abs = Math.abs(n);
  const sign = n < 0 ? "-" : "";
  if (abs >= 1e12) return `${sign}R$ ${nf(1, 1).format(abs / 1e12)} tri`;
  if (abs >= 1e9) return `${sign}R$ ${nf(1, 1).format(abs / 1e9)} bi`;
  if (abs >= 1e6) return `${sign}R$ ${nf(1, 1).format(abs / 1e6)} mi`;
  if (abs >= 1e3) return `${sign}R$ ${nf(1, 1).format(abs / 1e3)} mil`;
  return `${sign}R$ ${nf(2, 2).format(abs)}`;
}

/** Percentual com sinal e cor (verde=+, vermelho=−). Recebe fração. */
export function coloredPercent(fraction: number | null | undefined, decimals = 1): string {
  if (fraction == null || Number.isNaN(fraction)) return t.muted("—");
  const text = fmtPercent(fraction, decimals, true);
  if (fraction > 0) return t.green(text);
  if (fraction < 0) return t.red(text);
  return t.ink(text);
}

/** Idem, mas recebe já em pontos percentuais. */
export function coloredPercentPoints(pct: number | null | undefined, decimals = 1): string {
  if (pct == null || Number.isNaN(pct)) return t.muted("—");
  const text = fmtPercentPoints(pct, decimals, true);
  if (pct > 0) return t.green(text);
  if (pct < 0) return t.red(text);
  return t.ink(text);
}
