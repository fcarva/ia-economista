# Dexter BR

Assistente de **pesquisa financeira do Brasil no terminal** — um "Dexter"
([virattt/dexter](https://github.com/virattt/dexter)) para dados brasileiros,
com estética **Flexoki Light** (fundo papel, tinta escura, números coloridos) e
**tabelas quadradas** (box-drawing) estilo terminal.

Cobre **macro** (Selic, IPCA, câmbio, PIB, atividade, emprego, **expectativas do
Focus**, econometria) via o motor [`brazil-agent`](https://github.com/fcarva/brazil-agent)
e **ações da B3** (cotações, fundamentos, screening) via [brapi.dev](https://brapi.dev).

```
❯ Qual a expectativa do Focus para Selic e IPCA?

● bcb_focus.fetch_latest(endpoint=ExpectativasMercadoInflacao12Meses)
  └─ ✓ concluído em 1.2s
· sintetizando resposta…

┌─────────────┬────────┬────────┐
│ Indicador   │   2025 │   2026 │
├─────────────┼────────┼────────┤
│ IPCA        │  +4,8% │  +4,0% │
│ Selic (fim) │ 10,50% │  9,50% │
└─────────────┴────────┴────────┘
```

## Requisitos

- **[Bun](https://bun.sh)** ≥ 1.1 (runtime do app).
- Uma **chave da API Anthropic** (`ANTHROPIC_API_KEY`).
- **Opcional (camada macro):** Python 3.11+ e o repositório `brazil-agent` com
  suas dependências instaladas. Sem isso, funcionam só as ferramentas de ações e
  a ferramenta de expectativas do Focus (que só precisa de `requests`).

## Instalação

```bash
cd dexter-br
bun install
cp .env.example .env      # preencha ANTHROPIC_API_KEY (e BRAZIL_AGENT_PATH)
bun start
```

## Configuração (`.env`)

| Variável | Obrigatório | Descrição |
|---|---|---|
| `ANTHROPIC_API_KEY` | sim | Chave Claude (console.anthropic.com). |
| `DEXTER_MODEL` | não | Modelo Claude. Default `claude-opus-4-8`. Ex.: `claude-sonnet-5`. |
| `BRAZIL_AGENT_PATH` | não | Caminho do repo `brazil-agent` (ativa a camada macro). |
| `DEXTER_PYTHON` | não | Executável Python (default `python3`). |
| `BRAPI_TOKEN` | não | Token brapi.dev (mais requisições de ações). |
| `TAVILY_API_KEY` | não | (Futuro) busca na web. |

## Camada macro (bridge MCP → `brazil-agent`)

O Dexter BR **sobe os servidores MCP em Python do `brazil-agent`** como
subprocessos e registra as ferramentas deles (BCB/SGS, **Focus**, IBGE/SIDRA,
IPEA, econometria, calendário, global macro). Para habilitar:

```bash
# 1) clone o brazil-agent e instale as deps num venv
git clone https://github.com/fcarva/brazil-agent
cd brazil-agent
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2) aponte o Dexter para ele (no .env do dexter-br)
BRAZIL_AGENT_PATH=/caminho/para/brazil-agent
# se usar venv, aponte o Python dele:
DEXTER_PYTHON=/caminho/para/brazil-agent/.venv/bin/python
```

Sem as dependências Python instaladas, cada servidor macro falha com uma warning
clara (ex.: `ModuleNotFoundError: No module named 'pandas'`) e o app segue com o
que estiver disponível.

## Camada de ações (B3)

Via [brapi.dev](https://brapi.dev): `acoes_cotacao`, `acoes_fundamentos`,
`acoes_screening` (universo padrão = blue chips PETR4, VALE3, ITUB4, BBDC4,
ABEV3, B3SA3, WEGE3, RENT3, BBAS3). `BRAPI_TOKEN` é opcional (aumenta o limite).

## Comandos do REPL

- `/limpar` — reinicia a conversa.
- `/sair` — encerra (ou `Ctrl+C`).

## Estrutura

```
dexter-br/
├── src/
│   ├── cli.ts                 # REPL (entrypoint)
│   ├── config.ts              # env + universo B3
│   ├── theme.ts               # paleta Flexoki + OSC (papel/tinta)
│   ├── agent/
│   │   ├── agent.ts           # loop de tool-use (LangChain + Claude)
│   │   ├── prompts.ts         # system prompt PT-BR
│   │   └── types.ts           # contrato de ferramenta
│   ├── tools/
│   │   ├── registry.ts        # macro + ações
│   │   ├── macro/mcp-bridge.ts# bridge JSON-RPC p/ servidores Python
│   │   └── finance/brapi.ts   # ações B3 (brapi.dev)
│   ├── components/            # intro, tool-event, answer (box tables)
│   └── utils/                 # box-table, format (pt-BR)
└── scripts/                   # smoke tests (sem LLM)
```

## Smoke tests (sem LLM)

```bash
bun run smoke:equities   # ações (brapi.dev)
bun run smoke:macro      # bridge MCP (requer BRAZIL_AGENT_PATH + deps)
```

## Notas

- **Modelo padrão:** `claude-opus-4-8` (via SDK `@langchain/anthropic`).
- **Tema:** usa OSC 10/11 para deixar o terminal papel/tinta (Flexoki Light) em
  terminais que suportam; as cores dos números vêm do `chalk` (truecolor).
- Inspirado no [Dexter](https://github.com/virattt/dexter) (virattt) e na paleta
  [Flexoki](https://stephango.com/flexoki) (Steph Ango).
