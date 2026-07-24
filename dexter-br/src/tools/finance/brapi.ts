// Camada de ações B3 via brapi.dev (cotações + fundamentos).
// Preenche a lacuna de equities do brazil-agent (que é 100% macro).

import type { DexterTool } from "../../agent/types.ts";
import { config } from "../../config.ts";

const BASE = "https://brapi.dev/api";

interface HistPoint {
  date: number;
  close: number;
}
interface BrapiResult {
  symbol: string;
  shortName?: string;
  longName?: string;
  regularMarketPrice?: number;
  regularMarketChangePercent?: number;
  historicalDataPrice?: HistPoint[];
  financialData?: {
    profitMargins?: number;
    operatingMargins?: number;
    grossMargins?: number;
    returnOnEquity?: number;
    freeCashflow?: number;
    totalRevenue?: number;
    ebitdaMargins?: number;
  };
  summaryDetail?: { trailingPE?: number; dividendYield?: number; marketCap?: number };
  defaultKeyStatistics?: { trailingEps?: number; priceToBook?: number };
  priceEarnings?: number;
  earningsPerShare?: number;
}

async function brapiGet(path: string, params: Record<string, string>): Promise<BrapiResult[]> {
  const qs = new URLSearchParams(params);
  if (config.brapiToken) qs.set("token", config.brapiToken);
  const url = `${BASE}${path}?${qs.toString()}`;
  const resp = await fetch(url, { headers: { "User-Agent": "dexter-br/0.1" } });
  if (!resp.ok) {
    const body = await resp.text().catch(() => "");
    throw new Error(
      `brapi ${resp.status} ${resp.statusText}${
        resp.status === 401 || resp.status === 429
          ? " (defina BRAPI_TOKEN para mais requisições — https://brapi.dev)"
          : ""
      }${body ? ` — ${body.slice(0, 160)}` : ""}`,
    );
  }
  const json = (await resp.json()) as { results?: BrapiResult[]; error?: boolean; message?: string };
  if (json.error) throw new Error(json.message || "erro brapi");
  return json.results ?? [];
}

function normalizeTickers(input: unknown, fallback: string[]): string[] {
  if (Array.isArray(input)) return input.map((s) => String(s).toUpperCase());
  if (typeof input === "string" && input.trim()) {
    return input.split(/[,\s]+/).filter(Boolean).map((s) => s.toUpperCase());
  }
  return fallback;
}

/** Retorno acumulado e vol anualizada a partir de closes (intervalo mensal). */
function returnAndVol(hist: HistPoint[] | undefined): { ret: number | null; vol: number | null } {
  if (!hist || hist.length < 2) return { ret: null, vol: null };
  const closes = hist.map((h) => h.close).filter((c) => typeof c === "number" && c > 0);
  if (closes.length < 2) return { ret: null, vol: null };
  const ret = closes[closes.length - 1]! / closes[0]! - 1;
  const rets: number[] = [];
  for (let i = 1; i < closes.length; i++) rets.push(closes[i]! / closes[i - 1]! - 1);
  const mean = rets.reduce((a, b) => a + b, 0) / rets.length;
  const variance = rets.reduce((a, b) => a + (b - mean) ** 2, 0) / Math.max(1, rets.length - 1);
  const vol = Math.sqrt(variance) * Math.sqrt(12); // mensal -> anual
  return { ret, vol };
}

export function equitiesTools(): DexterTool[] {
  const universe = config.b3Universe;

  const stockPrice: DexterTool = {
    name: "acoes_cotacao",
    description:
      "Cotação atual de ações da B3 (preço e variação do dia). Aceita um ou vários tickers (ex.: PETR4, VALE3). Fonte: brapi.dev.",
    source: "brapi",
    inputSchema: {
      type: "object",
      properties: {
        tickers: {
          type: "array",
          items: { type: "string" },
          description: "Tickers B3 sem sufixo (ex.: ['PETR4','VALE3']).",
        },
      },
      required: ["tickers"],
    },
    label: (i) => `acoes_cotacao(${normalizeTickers(i.tickers, []).join(",")})`,
    async run(input) {
      const tickers = normalizeTickers(input.tickers, universe);
      const results = await brapiGet(`/quote/${tickers.join(",")}`, {});
      const rows = results.map((r) => ({
        ticker: r.symbol,
        nome: r.shortName || r.longName || r.symbol,
        preco: r.regularMarketPrice ?? null,
        variacao_dia_pct: r.regularMarketChangePercent ?? null,
      }));
      return JSON.stringify({ moeda: "BRL", cotacoes: rows });
    },
  };

  const getFinancials: DexterTool = {
    name: "acoes_fundamentos",
    description:
      "Fundamentos de uma ação da B3: margens (líquida/operacional/bruta), FCF, ROE, P/L, receita, market cap. Fonte: brapi.dev (módulos financialData/summaryDetail).",
    source: "brapi",
    inputSchema: {
      type: "object",
      properties: { ticker: { type: "string", description: "Ticker B3 (ex.: PETR4)." } },
      required: ["ticker"],
    },
    label: (i) => `acoes_fundamentos(${String(i.ticker ?? "").toUpperCase()})`,
    async run(input) {
      const ticker = String(input.ticker ?? "").toUpperCase();
      if (!ticker) throw new Error("informe 'ticker'");
      const results = await brapiGet(`/quote/${ticker}`, {
        fundamental: "true",
        modules: "summaryDetail,defaultKeyStatistics,financialData",
      });
      const r = results[0];
      if (!r) throw new Error(`sem dados para ${ticker}`);
      const fd = r.financialData ?? {};
      const sd = r.summaryDetail ?? {};
      return JSON.stringify({
        ticker: r.symbol,
        nome: r.longName || r.shortName || r.symbol,
        preco: r.regularMarketPrice ?? null,
        margem_liquida: fd.profitMargins ?? null,
        margem_operacional: fd.operatingMargins ?? null,
        margem_bruta: fd.grossMargins ?? null,
        roe: fd.returnOnEquity ?? null,
        fcf: fd.freeCashflow ?? null,
        receita: fd.totalRevenue ?? null,
        market_cap: sd.marketCap ?? null,
        p_l: sd.trailingPE ?? r.priceEarnings ?? null,
        dividend_yield: sd.dividendYield ?? null,
      });
    },
  };

  const screen: DexterTool = {
    name: "acoes_screening",
    description:
      "Screening do universo de blue chips da B3: para cada ação retorna retorno em 12m, volatilidade anualizada, margem líquida, FCF e ROE. Use para rankear/comparar ações. Sem 'tickers', usa as blue chips padrão. Fonte: brapi.dev.",
    source: "brapi",
    inputSchema: {
      type: "object",
      properties: {
        tickers: {
          type: "array",
          items: { type: "string" },
          description: "Opcional. Tickers B3; default = blue chips (PETR4, VALE3, ITUB4, …).",
        },
      },
    },
    label: (i) => `acoes_screening(${normalizeTickers(i.tickers, universe).length} ativos)`,
    async run(input) {
      const tickers = normalizeTickers(input.tickers, universe);
      const results = await brapiGet(`/quote/${tickers.join(",")}`, {
        range: "1y",
        interval: "1mo",
        fundamental: "true",
        modules: "financialData",
      });
      const rows = results.map((r) => {
        const { ret, vol } = returnAndVol(r.historicalDataPrice);
        const fd = r.financialData ?? {};
        const sharpe = ret != null && vol && vol > 0 ? ret / vol : null;
        return {
          ticker: r.symbol,
          nome: r.shortName || r.symbol,
          retorno_12m: ret,
          volatilidade: vol,
          sharpe_aprox: sharpe,
          margem_liquida: fd.profitMargins ?? null,
          fcf: fd.freeCashflow ?? null,
          roe: fd.returnOnEquity ?? null,
        };
      });
      return JSON.stringify({ universo: tickers, screening: rows });
    },
  };

  return [stockPrice, getFinancials, screen];
}
