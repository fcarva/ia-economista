// Loop do agente Dexter BR (SDK oficial @anthropic-ai/sdk).
// Protótipo direto (sem LangChain): loop manual de tool-use com streaming.

import Anthropic from "@anthropic-ai/sdk";

import { config } from "../config.ts";
import type { DexterTool } from "./types.ts";
import { systemPrompt } from "./prompts.ts";

export interface AgentHooks {
  onToolStart?(label: string, source?: string): void;
  onToolEnd?(ms: number, ok: boolean): void;
  onGenerating?(): void;
}

const MAX_ITERATIONS = 8;

function extractText(content: Anthropic.ContentBlock[]): string {
  return content
    .filter((b): b is Anthropic.TextBlock => b.type === "text")
    .map((b) => b.text)
    .join("");
}

export class DexterAgent {
  private client: Anthropic | null = null;
  private readonly byName = new Map<string, DexterTool>();
  private readonly toolDefs: Anthropic.Tool[];
  private readonly system: string;
  private messages: Anthropic.MessageParam[] = [];

  constructor(private readonly tools: DexterTool[], skillsSummary = "") {
    for (const t of tools) this.byName.set(t.name, t);
    this.toolDefs = tools.map((t) => ({
      name: t.name,
      description: t.description,
      input_schema: t.inputSchema as Anthropic.Tool.InputSchema,
    }));
    this.system = systemPrompt(
      tools.map((t) => t.name),
      skillsSummary,
    );
  }

  private ensureClient(): Anthropic {
    if (!this.client) this.client = new Anthropic({ apiKey: config.anthropicApiKey });
    return this.client;
  }

  reset(): void {
    this.messages = [];
  }

  /** Faz uma pergunta e retorna a resposta final (markdown). */
  async ask(userInput: string, hooks: AgentHooks = {}): Promise<string> {
    const client = this.ensureClient();
    this.messages.push({ role: "user", content: userInput });

    for (let iter = 0; iter < MAX_ITERATIONS; iter++) {
      const stream = client.messages.stream({
        model: config.model,
        max_tokens: 8192,
        system: this.system,
        tools: this.toolDefs,
        messages: this.messages,
      });
      const msg = await stream.finalMessage();
      this.messages.push({ role: "assistant", content: msg.content });

      if (msg.stop_reason !== "tool_use") {
        return extractText(msg.content).trim();
      }

      const toolResults: Anthropic.ToolResultBlockParam[] = [];
      for (const block of msg.content) {
        if (block.type !== "tool_use") continue;
        const tool = this.byName.get(block.name);
        const args = (block.input ?? {}) as Record<string, unknown>;
        const label = tool?.label ? tool.label(args) : `${block.name}(${JSON.stringify(args)})`;
        hooks.onToolStart?.(label, tool?.source);

        const t0 = Date.now();
        let content: string;
        let ok = true;
        if (!tool) {
          ok = false;
          content = `ERRO: ferramenta desconhecida '${block.name}'.`;
        } else {
          try {
            content = await tool.run(args);
          } catch (e) {
            ok = false;
            content = `ERRO ao executar ${block.name}: ${(e as Error).message}`;
          }
        }
        hooks.onToolEnd?.(Date.now() - t0, ok);
        toolResults.push({ type: "tool_result", tool_use_id: block.id, content, is_error: !ok });
      }

      this.messages.push({ role: "user", content: toolResults });
      hooks.onGenerating?.();
    }

    return "Limite de passos atingido sem resposta final. Reformule a pergunta, por favor.";
  }
}
