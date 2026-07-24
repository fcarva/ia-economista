// Reaproveita as skills do brazil-agent (skills/<nome>/SKILL.md) como skills do
// Dexter, com progressive disclosure: o nome+descrição vão pro system prompt, e
// a ferramenta 'carregar_skill' devolve o corpo completo sob demanda.

import { readFileSync, readdirSync, existsSync, statSync } from "node:fs";
import { join } from "node:path";

import type { DexterTool } from "../agent/types.ts";
import { config } from "../config.ts";

export interface Skill {
  name: string;
  description: string;
  dir: string;
  body: string;
}

function parseFrontmatter(raw: string): { name?: string; description?: string; body: string } {
  const m = raw.match(/^---\s*\n([\s\S]*?)\n---\s*\n?([\s\S]*)$/);
  if (!m) return { body: raw };
  const meta: Record<string, string> = {};
  for (const line of m[1]!.split("\n")) {
    const kv = line.match(/^(\w[\w-]*):\s*(.*)$/);
    if (kv) meta[kv[1]!] = kv[2]!.trim();
  }
  return { name: meta.name, description: meta.description, body: m[2] ?? "" };
}

export function loadSkills(): Skill[] {
  if (!config.brazilAgentPath) return [];
  const dir = join(config.brazilAgentPath, "skills");
  if (!existsSync(dir)) return [];
  const skills: Skill[] = [];
  for (const entry of readdirSync(dir)) {
    const skillDir = join(dir, entry);
    const file = join(skillDir, "SKILL.md");
    if (!existsSync(file) || !statSync(skillDir).isDirectory()) continue;
    try {
      const raw = readFileSync(file, "utf8");
      const { name, description, body } = parseFrontmatter(raw);
      skills.push({ name: name || entry, description: description || "", dir: skillDir, body });
    } catch {
      /* ignora skill ilegível */
    }
  }
  return skills;
}

/** Resumo (nome: descrição) para injetar no system prompt. */
export function skillsSummary(skills: Skill[]): string {
  return skills.map((s) => `  · ${s.name}: ${s.description}`).join("\n");
}

/** Ferramenta que devolve o corpo da SKILL.md + a primeira referência (se houver). */
export function skillTool(skills: Skill[]): DexterTool {
  const byName = new Map(skills.map((s) => [s.name, s]));
  return {
    name: "carregar_skill",
    description:
      "Carrega o guia completo de uma skill de metodologia (coleta de dados macro, econometria, qualidade de dados, comunicação econômica, forecasting). Use antes de tarefas complexas do tema. Skills: " +
      skills.map((s) => s.name).join(", "),
    source: "skills",
    inputSchema: {
      type: "object",
      properties: { name: { type: "string", description: "Nome da skill (ex.: econometric-analysis)." } },
      required: ["name"],
    },
    label: (i) => `carregar_skill(${String(i.name ?? "")})`,
    async run(input) {
      const skill = byName.get(String(input.name ?? ""));
      if (!skill) {
        return `Skill não encontrada. Disponíveis: ${skills.map((s) => s.name).join(", ")}`;
      }
      let out = `# ${skill.name}\n${skill.body}`;
      // anexa a primeira referência em references/, se existir
      const refDir = join(skill.dir, "references");
      if (existsSync(refDir)) {
        for (const f of readdirSync(refDir)) {
          if (!f.endsWith(".md")) continue;
          try {
            const ref = readFileSync(join(refDir, f), "utf8");
            out += `\n\n## Referência: ${f}\n${ref.slice(0, 6000)}`;
          } catch {
            /* ignora */
          }
          break;
        }
      }
      return out.slice(0, 9000);
    },
  };
}
