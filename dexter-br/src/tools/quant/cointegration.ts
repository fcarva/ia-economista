// Power tool de análise de pares / cointegração para a B3 (inspirado no
// ia-economista). TS puro sobre os fechamentos do brapi:
// Engle-Granger (hedge ratio via OLS) + meia-vida de reversão via AR(1).
//
// Heurística de pairs trading — NÃO é um teste formal de Johansen (para isso,
// use o servidor de econometria do brazil-agent). Rotulado como tal.

import type { DexterTool } from "../../agent/types.ts";
import { config } from "../../config.ts";
import { fetchCloses, type CloseSeries } from "../finance/brapi.ts";

function mean(a: number[]): number {
  return a.reduce((s, x) => s + x, 0) / a.length;
}

/** OLS y = alpha + beta*x. */
function ols(x: number[], y: number[]): { alpha: number; beta: number } {
  const mx = mean(x);
  const my = mean(y);
  let cov = 0;
  let varx = 0;
  for (let i = 0; i < x.length; i++) {
    cov += (x[i]! - mx) * (y[i]! - my);
    varx += (x[i]! - mx) ** 2;
  }
  const beta = varx > 0 ? cov / varx : 0;
  return { alpha: my - beta * mx, beta };
}

/** phi de AR(1): r_t = c + phi*r_{t-1}. */
function ar1(r: number[]): number {
  const prev = r.slice(0, -1);
  const cur = r.slice(1);
  const { beta } = ols(prev, cur);
  return beta;
}

function std(a: number[]): number {
  const m = mean(a);
  return Math.sqrt(a.reduce((s, x) => s + (x - m) ** 2, 0) / Math.max(1, a.length - 1));
}

function returnsCorr(ca: number[], cb: number[]): number {
  const ra: number[] = [];
  const rb: number[] = [];
  for (let i = 1; i < ca.length; i++) {
    ra.push(ca[i]! / ca[i - 1]! - 1);
    rb.push(cb[i]! / cb[i - 1]! - 1);
  }
  const ma = mean(ra);
  const mb = mean(rb);
  let cov = 0;
  let va = 0;
  let vb = 0;
  for (let i = 0; i < ra.length; i++) {
    cov += (ra[i]! - ma) * (rb[i]! - mb);
    va += (ra[i]! - ma) ** 2;
    vb += (rb[i]! - mb) ** 2;
  }
  return va > 0 && vb > 0 ? cov / Math.sqrt(va * vb) : 0;
}

/** Alinha duas séries por data comum. */
function align(a: CloseSeries, b: CloseSeries): { x: number[]; y: number[] } {
  const mapB = new Map<number, number>();
  b.dates.forEach((d, i) => mapB.set(d, b.closes[i]!));
  const x: number[] = [];
  const y: number[] = [];
  a.dates.forEach((d, i) => {
    const cb = mapB.get(d);
    if (cb !== undefined) {
      x.push(a.closes[i]!);
      y.push(cb);
    }
  });
  return { x, y };
}

export function quantTools(): DexterTool[] {
  const cointegracao: DexterTool = {
    name: "cointegracao_b3",
    description:
      "Análise de pares/cointegração para ações da B3: para cada par calcula hedge ratio (OLS), correlação de retornos, meia-vida de reversão à média (AR1) e z-score atual do spread. Heurística de pairs trading (não é teste formal de Johansen). Sem 'tickers', usa as blue chips. Fonte: brapi.dev.",
    source: "quant",
    inputSchema: {
      type: "object",
      properties: {
        tickers: {
          type: "array",
          items: { type: "string" },
          description: "2 a 8 tickers B3 (default = blue chips).",
        },
        range: { type: "string", description: "Janela: 6mo, 1y, 2y (default 1y)." },
      },
    },
    label: (i) => `cointegracao_b3(${(Array.isArray(i.tickers) ? i.tickers : config.b3Universe).length} ativos)`,
    async run(input) {
      const tickers = (Array.isArray(input.tickers) ? input.tickers : config.b3Universe)
        .map((s) => String(s).toUpperCase())
        .slice(0, 8);
      if (tickers.length < 2) throw new Error("informe ao menos 2 tickers");
      const range = typeof input.range === "string" ? input.range : "1y";

      const series = await fetchCloses(tickers, range, "1d");
      const valid = series.filter((s) => s.closes.length >= 30);
      if (valid.length < 2) throw new Error("dados insuficientes para os tickers informados");

      const pairs: Array<Record<string, unknown>> = [];
      for (let i = 0; i < valid.length; i++) {
        for (let j = i + 1; j < valid.length; j++) {
          const A = valid[i]!;
          const B = valid[j]!;
          const { x, y } = align(A, B);
          if (x.length < 30) continue;
          const lx = x.map(Math.log);
          const ly = y.map(Math.log);
          const { alpha, beta } = ols(lx, ly);
          const resid = ly.map((v, k) => v - alpha - beta * lx[k]!);
          const phi = ar1(resid);
          const halfLife = phi > 0 && phi < 1 ? -Math.LN2 / Math.log(phi) : null;
          const sr = std(resid);
          const z = sr > 0 ? (resid[resid.length - 1]! - mean(resid)) / sr : 0;
          const corr = returnsCorr(x, y);
          const meanReverting = phi > 0 && phi < 1 && halfLife !== null && halfLife < resid.length;
          pairs.push({
            par: `${A.ticker}/${B.ticker}`,
            hedge_ratio: Number(beta.toFixed(3)),
            corr_retornos: Number(corr.toFixed(3)),
            meia_vida_dias: halfLife !== null ? Math.round(halfLife) : null,
            phi: Number(phi.toFixed(3)),
            z_spread_atual: Number(z.toFixed(2)),
            reverte_a_media: meanReverting,
          });
        }
      }
      // ranqueia pares com reversão pela meia-vida (menor = melhor)
      pairs.sort((a, b) => {
        const ha = (a.meia_vida_dias as number | null) ?? Infinity;
        const hb = (b.meia_vida_dias as number | null) ?? Infinity;
        return ha - hb;
      });
      return JSON.stringify({
        janela: range,
        intervalo: "diario",
        metodo: "Engle-Granger (OLS) + meia-vida AR(1)",
        pares: pairs,
      });
    },
  };

  return [cointegracao];
}
