#!/usr/bin/env bun
// Smoke test da camada de ações (brapi.dev). Não usa LLM.

import { equitiesTools } from "../src/tools/finance/brapi.ts";
import { renderAnswer } from "../src/components/answer.ts";

async function main() {
  const [cotacao, fundamentos, screening] = equitiesTools();

  console.log("→ acoes_cotacao(PETR4, VALE3)");
  try {
    console.log(await cotacao!.run({ tickers: ["PETR4", "VALE3"] }));
  } catch (e) {
    console.error("  falhou:", (e as Error).message);
  }

  console.log("\n→ acoes_fundamentos(ITUB4)");
  try {
    console.log(await fundamentos!.run({ ticker: "ITUB4" }));
  } catch (e) {
    console.error("  falhou:", (e as Error).message);
  }

  console.log("\n→ acoes_screening(default)");
  try {
    console.log(await screening!.run({}));
  } catch (e) {
    console.error("  falhou:", (e as Error).message);
  }

  // demonstra o render de tabela box-drawing
  console.log("\n→ render de exemplo:");
  console.log(
    renderAnswer(
      [
        "**Blue chips B3** (exemplo)",
        "",
        "| Ticker | Retorno 12m | Vol | Margem |",
        "|:-------|------------:|----:|-------:|",
        "| PETR4  | +18,2% | 32,1% | 24,0% |",
        "| VALE3  | -4,6% | 29,8% | 31,0% |",
      ].join("\n"),
    ),
  );
}

main();
