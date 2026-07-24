// Contrato de uma ferramenta do agente Dexter BR.

export interface DexterTool {
  /** Nome único, casável com /^[a-zA-Z0-9_]+$/ (exigência da API de tools). */
  name: string;
  /** Descrição em PT-BR — o modelo usa isto para decidir quando chamar. */
  description: string;
  /** JSON Schema (objeto) dos argumentos. */
  inputSchema: Record<string, unknown>;
  /** Executa a ferramenta e retorna texto/JSON para o modelo. */
  run(input: Record<string, unknown>): Promise<string>;
  /** Rótulo curto dos argumentos para a status line (estilo Dexter). */
  label?(input: Record<string, unknown>): string;
  /** Origem (ex.: "bcb_focus", "brapi") para telemetria/exibição. */
  source?: string;
}
