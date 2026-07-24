// Bridge para os servidores MCP (Python) do brazil-agent.
//
// Cada servidor fala um JSON-RPC mínimo por stdin/stdout (ver
// brazil-agent/mcp_servers/base.py): métodos `initialize`, `tools/list`,
// `tools/call`. Aqui damos `spawn` em `python -m mcp_servers.<server>` com
// cwd = BRAZIL_AGENT_PATH e conversamos linha-a-linha.
//
// Robustez: ignoramos qualquer linha de stdout que não seja um objeto JSON
// (os coletores às vezes imprimem logs em stdout).

import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { parse as parseYaml } from "yaml";

import type { DexterTool } from "../../agent/types.ts";
import { config } from "../../config.ts";

interface RpcPending {
  resolve: (v: unknown) => void;
  reject: (e: Error) => void;
  timer: ReturnType<typeof setTimeout>;
}

interface RemoteTool {
  name: string;
  description?: string;
  inputSchema?: Record<string, unknown>;
}

const INIT_TIMEOUT_MS = 30_000; // servidores importam pandas/statsmodels: pode demorar
const CALL_TIMEOUT_MS = 45_000;
const MAX_RESULT_CHARS = 8_000;

export class SpawnedMcpServer {
  readonly key: string;
  private readonly args: string[];
  private readonly env: Record<string, string>;
  private proc: ChildProcessWithoutNullStreams | null = null;
  private buf = "";
  private nextId = 1;
  private pending = new Map<number, RpcPending>();
  private stderr = "";

  constructor(key: string, command: string, env: Record<string, string> = {}) {
    this.key = key;
    // command ex.: "python -m mcp_servers.bcb_sgs_server" -> usa config.python
    this.args = command.trim().split(/\s+/).slice(1);
    this.env = env;
  }

  async start(): Promise<void> {
    if (!config.brazilAgentPath) throw new Error("BRAZIL_AGENT_PATH não definido");
    this.proc = spawn(config.python, this.args, {
      cwd: config.brazilAgentPath,
      env: { ...process.env, PYTHONUNBUFFERED: "1", ...this.env },
    }) as ChildProcessWithoutNullStreams;

    this.proc.stdout.setEncoding("utf8");
    this.proc.stdout.on("data", (chunk: string) => this.onData(chunk));
    this.proc.stderr.setEncoding("utf8");
    this.proc.stderr.on("data", (chunk: string) => {
      this.stderr = (this.stderr + chunk).slice(-4000);
    });
    this.proc.on("exit", (code) => {
      const tail = this.stderr.trim().split("\n").filter(Boolean).pop();
      this.failAll(new Error(`${this.key} saiu (código ${code})${tail ? `: ${tail}` : ""}`));
    });
    this.proc.on("error", (err) => this.failAll(err));

    await this.request("initialize", {}, INIT_TIMEOUT_MS);
  }

  private onData(chunk: string): void {
    this.buf += chunk;
    let idx: number;
    while ((idx = this.buf.indexOf("\n")) >= 0) {
      const line = this.buf.slice(0, idx).trim();
      this.buf = this.buf.slice(idx + 1);
      if (!line.startsWith("{")) continue; // ignora ruído/logs
      let msg: { id?: number; result?: unknown; error?: { message?: string } };
      try {
        msg = JSON.parse(line);
      } catch {
        continue;
      }
      if (typeof msg.id !== "number") continue;
      const p = this.pending.get(msg.id);
      if (!p) continue;
      this.pending.delete(msg.id);
      clearTimeout(p.timer);
      if (msg.error) p.reject(new Error(msg.error.message || "erro no servidor MCP"));
      else p.resolve(msg.result);
    }
  }

  private failAll(err: Error): void {
    for (const [, p] of this.pending) {
      clearTimeout(p.timer);
      p.reject(err);
    }
    this.pending.clear();
  }

  private request(method: string, params: Record<string, unknown>, timeoutMs = CALL_TIMEOUT_MS): Promise<unknown> {
    if (!this.proc) return Promise.reject(new Error(`${this.key} não iniciado`));
    const id = this.nextId++;
    const payload = JSON.stringify({ jsonrpc: "2.0", id, method, params }) + "\n";
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id);
        reject(new Error(`timeout em ${this.key}.${method}${this.stderr ? ` — ${this.stderr.trim().split("\n").pop()}` : ""}`));
      }, timeoutMs);
      this.pending.set(id, { resolve, reject, timer });
      this.proc!.stdin.write(payload);
    });
  }

  async listTools(): Promise<RemoteTool[]> {
    const res = (await this.request("tools/list", {})) as { tools?: RemoteTool[] };
    return res?.tools ?? [];
  }

  async callTool(name: string, args: Record<string, unknown>): Promise<unknown> {
    return this.request("tools/call", { name, arguments: args });
  }

  stop(): void {
    try {
      this.proc?.stdin.end();
      this.proc?.kill();
    } catch {
      /* noop */
    }
    this.proc = null;
  }
}

export interface LoadedMacro {
  tools: DexterTool[];
  servers: SpawnedMcpServer[];
  warnings: string[];
}

interface McpConfig {
  servers?: Record<string, { command?: string; env?: Record<string, string> }>;
}

/** Sobe os servidores MCP do brazil-agent e registra as tools deles. */
export async function loadMacroTools(): Promise<LoadedMacro> {
  const warnings: string[] = [];
  const tools: DexterTool[] = [];
  const servers: SpawnedMcpServer[] = [];

  if (!config.brazilAgentPath) {
    warnings.push(
      "BRAZIL_AGENT_PATH não definido — camada macro (BCB/IBGE/Focus/econometria) desativada.",
    );
    return { tools, servers, warnings };
  }

  let cfg: McpConfig;
  try {
    const raw = readFileSync(join(config.brazilAgentPath, "mcp_servers", "mcp_config.yaml"), "utf8");
    cfg = parseYaml(raw) as McpConfig;
  } catch (e) {
    warnings.push(`Não consegui ler mcp_config.yaml em BRAZIL_AGENT_PATH: ${(e as Error).message}`);
    return { tools, servers, warnings };
  }

  const entries = Object.entries(cfg.servers ?? {});
  for (const [key, spec] of entries) {
    const command = spec.command ?? `python -m mcp_servers.${key}_server`;
    const env = resolveEnv(spec.env ?? {});
    const server = new SpawnedMcpServer(key, command, env);
    try {
      await server.start();
      const remote = await server.listTools();
      servers.push(server);
      for (const rt of remote) {
        tools.push(wrapRemoteTool(server, key, rt));
      }
    } catch (e) {
      server.stop();
      warnings.push(`Servidor macro '${key}' indisponível: ${(e as Error).message}`);
    }
  }

  return { tools, servers, warnings };
}

/** Resolve placeholders ${VAR} no bloco env do mcp_config.yaml. */
function resolveEnv(env: Record<string, string>): Record<string, string> {
  const out: Record<string, string> = {};
  for (const [k, v] of Object.entries(env)) {
    out[k] = v.replace(/\$\{(\w+)\}/g, (_, name) => process.env[name] ?? "");
  }
  return out;
}

function wrapRemoteTool(server: SpawnedMcpServer, serverKey: string, rt: RemoteTool): DexterTool {
  const toolName = `${serverKey}__${rt.name}`;
  const schema =
    rt.inputSchema && typeof rt.inputSchema === "object"
      ? rt.inputSchema
      : { type: "object", properties: {} };
  return {
    name: toolName,
    description: `[macro:${serverKey}] ${rt.description ?? rt.name}`,
    inputSchema: schema,
    source: serverKey,
    label: (input) => `${serverKey}.${rt.name}(${previewArgs(input)})`,
    async run(input) {
      const result = await server.callTool(rt.name, input);
      let text = JSON.stringify(result);
      if (text.length > MAX_RESULT_CHARS) {
        text =
          text.slice(0, MAX_RESULT_CHARS) +
          `\n… [truncado; use 'limit' ou 'last_n_months' para reduzir a série]`;
      }
      return text;
    },
  };
}

function previewArgs(input: Record<string, unknown>): string {
  const parts = Object.entries(input)
    .slice(0, 3)
    .map(([k, v]) => `${k}=${typeof v === "string" ? v : JSON.stringify(v)}`);
  return parts.join(", ");
}
