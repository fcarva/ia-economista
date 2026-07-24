// Loop do agente Dexter BR (LangChain + Claude/Anthropic).
// Estilo Dexter: chama ferramentas em sequência, emite eventos para a UI e
// devolve a resposta final (renderizada pela CLI).

import { ChatAnthropic } from "@langchain/anthropic";
import {
  AIMessageChunk,
  HumanMessage,
  SystemMessage,
  ToolMessage,
  type BaseMessage,
} from "@langchain/core/messages";

import { config } from "../config.ts";
import type { DexterTool } from "./types.ts";
import { systemPrompt } from "./prompts.ts";

export interface AgentHooks {
  onToolStart?(label: string, source?: string): void;
  onToolEnd?(ms: number, ok: boolean): void;
  onGenerating?(): void;
}

const MAX_ITERATIONS = 8;

function extractText(content: unknown): string {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
      .map((b) => (b && typeof b === "object" && "text" in b ? String((b as { text: unknown }).text) : ""))
      .join("");
  }
  return "";
}

export class DexterAgent {
  private model: ReturnType<ChatAnthropic["bindTools"]> | null = null;
  private readonly byName = new Map<string, DexterTool>();
  private readonly messages: BaseMessage[];

  constructor(private readonly tools: DexterTool[]) {
    for (const t of tools) this.byName.set(t.name, t);
    this.messages = [new SystemMessage(systemPrompt(tools.map((t) => t.name)))];
  }

  /** Constrói o modelo sob demanda (evita lançar sem ANTHROPIC_API_KEY). */
  private ensureModel(): ReturnType<ChatAnthropic["bindTools"]> {
    if (this.model) return this.model;
    const base = new ChatAnthropic({
      model: config.model,
      maxTokens: 8192,
      streaming: true,
      apiKey: config.anthropicApiKey,
    });
    // Formato Anthropic nativo ({name, description, input_schema}) — passado
    // direto; o ChatAnthropic reconhece input_schema e não reconverte.
    const toolDefs = this.tools.map((t) => ({
      name: t.name,
      description: t.description,
      input_schema: t.inputSchema,
    }));
    this.model = base.bindTools(toolDefs as never);
    return this.model;
  }

  /** Faz uma pergunta e retorna a resposta final (markdown). */
  async ask(userInput: string, hooks: AgentHooks = {}): Promise<string> {
    this.messages.push(new HumanMessage(userInput));
    const model = this.ensureModel();

    for (let iter = 0; iter < MAX_ITERATIONS; iter++) {
      const stream = await model.stream(this.messages);
      let gathered: AIMessageChunk | undefined;
      for await (const chunk of stream) {
        gathered = gathered === undefined ? chunk : gathered.concat(chunk);
      }
      if (!gathered) throw new Error("resposta vazia do modelo");

      this.messages.push(gathered);

      const calls = gathered.tool_calls ?? [];
      if (calls.length === 0) {
        return extractText(gathered.content).trim();
      }

      for (const call of calls) {
        const tool = call.name ? this.byName.get(call.name) : undefined;
        const args = (call.args ?? {}) as Record<string, unknown>;
        const label = tool?.label ? tool.label(args) : `${call.name}(${JSON.stringify(args)})`;
        hooks.onToolStart?.(label, tool?.source);

        const t0 = Date.now();
        let content: string;
        let ok = true;
        if (!tool) {
          ok = false;
          content = `ERRO: ferramenta desconhecida '${call.name}'.`;
        } else {
          try {
            content = await tool.run(args);
          } catch (e) {
            ok = false;
            content = `ERRO ao executar ${call.name}: ${(e as Error).message}`;
          }
        }
        hooks.onToolEnd?.(Date.now() - t0, ok);

        this.messages.push(
          new ToolMessage({ content, tool_call_id: call.id ?? "", name: call.name }),
        );
      }

      hooks.onGenerating?.();
    }

    return "Limite de passos atingido sem resposta final. Reformule a pergunta, por favor.";
  }
}
