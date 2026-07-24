#!/usr/bin/env bun
// Dexter BR — REPL de terminal (Flexoki Light).

import * as readline from "node:readline/promises";
import { stdin, stdout } from "node:process";

import { config } from "./config.ts";
import { enterTheme, exitTheme, t } from "./theme.ts";
import { assembleTools } from "./tools/registry.ts";
import { DexterAgent } from "./agent/agent.ts";
import { renderIntro } from "./components/intro.ts";
import { renderAnswer } from "./components/answer.ts";
import { toolStartLine, toolEndLine, generatingLine } from "./components/tool-event.ts";

async function main(): Promise<void> {
  enterTheme();
  stdout.write(t.muted("carregando ferramentas (macro + ações)…\n"));

  const set = await assembleTools();
  const { tools, servers, warnings, skillsSummary } = set;

  let closed = false;
  const cleanup = () => {
    if (closed) return;
    closed = true;
    for (const s of servers) s.stop();
    exitTheme();
  };
  process.on("exit", cleanup);

  stdout.write(
    renderIntro({
      warnings,
      toolCount: tools.length,
      macroCount: set.macroCount,
      skillCount: set.skillCount,
      restOnline: set.restOnline,
      webProvider: set.webProvider,
    }),
  );

  const agent = new DexterAgent(tools, skillsSummary);
  const rl = readline.createInterface({ input: stdin, output: stdout });
  rl.on("SIGINT", () => {
    rl.close();
  });

  try {
    while (true) {
      let q: string;
      try {
        q = (await rl.question(t.blue("❯ "))).trim();
      } catch {
        break; // stream fechado (Ctrl+C/EOF)
      }
      if (!q) continue;
      if (q === "/sair" || q === "/quit" || q === "/exit") break;
      if (q === "/limpar") {
        agent.reset();
        stdout.write(t.muted("(conversa reiniciada)\n\n"));
        continue;
      }
      if (!config.mock && !config.anthropicApiKey) {
        stdout.write(t.red("ANTHROPIC_API_KEY não definido. Defina no ambiente e tente de novo.\n\n"));
        continue;
      }

      try {
        const answer = await agent.ask(q, {
          onToolStart: (label) => stdout.write(`\n${toolStartLine(label)}\n`),
          onToolEnd: (ms, ok) => stdout.write(`${toolEndLine(ms, ok)}\n`),
          onGenerating: () => stdout.write(`\n${generatingLine()}\n`),
        });
        stdout.write(`\n${renderAnswer(answer)}\n\n`);
      } catch (e) {
        stdout.write(`\n${t.red(`Erro: ${(e as Error).message}`)}\n\n`);
      }
    }
  } finally {
    rl.close();
    cleanup();
  }
}

main().catch((e) => {
  exitTheme();
  console.error(e);
  process.exit(1);
});
