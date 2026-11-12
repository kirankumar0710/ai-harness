# Odysseus

**Odysseus** (`odysseus_v2`) is a small agent harness that works with any
model vendor. It runs the agent loop (call the model, run the tools it asks
for, repeat until it is done) and talks to Gemini, OpenAI, OpenRouter or
Anthropic through one provider interface. It uses only the Python standard
library.

## Origin

This project is based on the original Odysseus harness source code from
<https://odysseus-harness.vercel.app/>. That code was taken as the starting
point and modified to:

- **Give it a clearer structure:** an installable package (`pyproject.toml`,
  `src/` layout), with the agent loop, the providers, tests and demos each in
  their own place.
- **Improve the class layout:** an abstract `Provider` base class holds the
  shared logic (HTTP, retries, error handling), and each vendor is one
  subclass that only converts requests and responses to its own format.
- **Make it vendor-agnostic:** the loop talks to any provider through the same
  neutral message format; Gemini, OpenAI, OpenRouter and Anthropic are
  supported, and adding a vendor means one new module.
- **Move configuration out of the code:** vendor defaults (base URL, model,
  API-key variables) live in an ini file that can be overridden, and the
  provider must be chosen explicitly.
- **Add offline tests and documented demos.**

## Getting started

### 1. Clone the repo and go to this project

```bash
git clone https://github.com/kirankumar0710/ai-harness.git
cd ai-harness
cd odysseus          # this project's folder: pyproject.toml is here
```

**All the commands below are run from the `odysseus` folder**, not from the
`ai-harness` repo root. Running them from the root fails, for example with
`pip install -e .`: *does not appear to be a Python project*.

### 2. Create a virtual environment (recommended)

Requires Python 3.10 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
```

### 3. Install the package

```bash
cd odysseus          # skip if you are already in the odysseus folder
pip install -e .
```

From the `ai-harness` root, `pip install -e odysseus` does the same thing.

`-e` (editable) links the install to your checkout, so code changes take
effect without reinstalling. There are no third-party dependencies.

### 4. Check it works (offline, no API key needed)

```bash
python3 -m unittest discover tests
```

The run should end with `OK`.

### 5. Choose a provider

There is no default; you must pick one.

```bash
export ODYSSEUS_PROVIDER=anthropic     # or gemini / openai / openrouter
```

### 6. Export that provider's API key

| Provider     | Key variable                                  |
|--------------|-----------------------------------------------|
| `anthropic`  | `ANTHROPIC_API_KEY`                           |
| `gemini`     | `GEMINI_API_KEY` (or `GOOGLE_API_KEY`)        |
| `openai`     | `OPENAI_API_KEY`                              |
| `openrouter` | `OPENROUTER_API_KEY`                          |

```bash
export ANTHROPIC_API_KEY=...
```

`ODYSSEUS_API_KEY` works for any provider as a fallback. OpenAI has no
default model, so for `openai` also set one, e.g. `export ODYSSEUS_MODEL=...`.

### 7. Run a demo

```bash
python3 demos/dice/dice.py
```

Open [`demos/index.html`](demos/index.html) in a browser for the list of
demos, what each one shows, and a sample run with analysis.

Steps 1–4 are needed once. In a new terminal, activate the virtual
environment again and repeat steps 5–6 (or put the `export` lines in your
shell profile, e.g. `~/.zshrc`).

## Configuration

Settings are resolved in this order, highest first:

1. Arguments passed in code, e.g. `create_provider("gemini", model="...")`
2. The vendor's own key variable, e.g. `GEMINI_API_KEY`
3. `ODYSSEUS_API_KEY`, `ODYSSEUS_MODEL`, `ODYSSEUS_BASE_URL`
4. The vendor's section in an ini file

Each vendor's default base URL, model and key variable names live in
[`src/odysseus_v2/providers/providers.ini`](src/odysseus_v2/providers/providers.ini).
To change them without editing that file, create your own `.ini` with only
the keys you want to change and point `ODYSSEUS_CONFIG` at it:

```ini
# ~/my-odysseus.ini
[openai]
model = gpt-...
```

```bash
export ODYSSEUS_CONFIG=~/my-odysseus.ini
```

API keys are never stored in the ini files; they come from environment
variables only.

## Using it in code

```python
from odysseus_v2 import create_provider, run_loop

provider = create_provider()          # reads ODYSSEUS_PROVIDER and the key
messages = [{"role": "user", "text": "Roll 3 dice."}]
answer = run_loop(provider, "You are a helpful agent.", messages,
                  tools={"roll_dice": RollDice()})
```

A tool is any object with a `name`, a `spec` holding its JSON schema, and a
`run(**kwargs)` method; see [`demos/dice/dice.py`](demos/dice/dice.py) for a
complete example. `messages` is updated in place and holds the whole
conversation.

## Architecture

Odysseus has four parts. Applications use the first two; the provider hides
everything vendor-specific behind one method, `complete()`, so the loop and
any application built on it work unchanged with every vendor.

| Part | Where | Job |
|---|---|---|
| **Factory** | `create_provider()` in `providers/__init__.py` | picks the vendor and builds its provider with a resolved config |
| **Agent loop** | `run_loop()` in `loop.py` | calls the model, runs the tools it asks for, repeats until it is done |
| **Provider** | one subclass of `Provider` per vendor | converts the neutral conversation to the vendor's format and back |
| **Transport** | `post_json()` in `providers/base.py` | sends the HTTPS request; retries rate limits, server and network errors |

```mermaid
flowchart TB
    subgraph config["Configuration"]
        env["Environment variables<br/>ODYSSEUS_PROVIDER, API keys,<br/>ODYSSEUS_MODEL, ODYSSEUS_BASE_URL"]
        ini["providers.ini<br/>+ optional ODYSSEUS_CONFIG file"]
    end

    subgraph odysseus["odysseus_v2"]
        factory["create_provider()"]
        fromenv["e.g. GeminiConfig.from_env()"]
        loop["run_loop()"]
        provider["Provider<br/>Gemini / OpenAI / OpenRouter / Anthropic"]
        transport["post_json()<br/>retries + backoff"]
    end

    tools["Tools<br/>objects with name, spec, run()"]
    api[("Vendor API")]

    factory -- "builds config" --> fromenv
    fromenv -- "reads" --> env
    fromenv -- "reads defaults" --> ini
    factory -- "creates" --> provider
    loop -- "complete()" --> provider
    provider -- "url, headers, body" --> transport
    transport -- "HTTPS POST" --> api
    loop -- "run(args)" --> tools
```

**How the loop works.** Each turn, `run_loop()` sends the conversation to the
provider and appends the model's reply. If the reply asks for no tools, the
loop returns its text: the model decides when the work is done. Otherwise it
runs each requested tool and appends one result per call, then starts the
next turn. The rules it keeps:

- Every tool call gets exactly one result, even when it is blocked by the
  `before_tool` gate or fails.
- A failing tool never crashes the loop: the error becomes a result the
  model reads.
- `max_turns` is a hard stop. When it is reached, one last call is made with
  tools disabled, so the model reports what was done instead of stopping
  silently.

Each demo's page in [`demos/`](demos/index.html) traces a concrete run
through these parts.

## Class layout

### Providers and configuration

Each vendor has a provider class and a matching config class. The provider's
`config_class` names its config, which `create_provider()` uses to build it.
`create_provider()` and `run_loop()` are plain functions, not classes.

```mermaid
classDiagram
    direction TB

    class Provider {
        <<abstract>>
        +config_class type$
        +config ProviderConfig
        +model str
        +complete(system, messages, tools, allow_tool_calls) Completion
        #_build_request(system, messages, tools, allow_tool_calls) tuple*
        #_parse_response(raw) Completion*
    }
    class GeminiProvider {
        +_to_wire(messages) list$
    }
    class AnthropicProvider {
        +_to_wire(messages) list$
    }
    class OpenAICompatibleProvider {
        +_to_wire(system, messages) list$
    }
    class OpenAIProvider
    class OpenRouterProvider

    Provider <|-- GeminiProvider
    Provider <|-- AnthropicProvider
    Provider <|-- OpenAICompatibleProvider
    OpenAICompatibleProvider <|-- OpenAIProvider
    OpenAICompatibleProvider <|-- OpenRouterProvider

    class ProviderConfig {
        <<frozen dataclass>>
        +PROVIDER_NAME str$
        +api_key str
        +model str
        +base_url str
        +timeout float
        +max_attempts int
        +temperature float
        +max_output_tokens int
        +extra_headers dict
        +from_env(overrides) ProviderConfig$
    }
    class GeminiConfig
    class AnthropicConfig {
        +max_output_tokens int = 8192
        +anthropic_version str
    }
    class OpenAIConfig {
        +max_tokens_field str
    }
    class OpenRouterConfig {
        +max_tokens_field str
        +extra_headers dict
    }
    class VendorDefaults {
        <<frozen dataclass>>
        +base_url str
        +model str
        +api_key_env tuple
    }

    ProviderConfig <|-- GeminiConfig
    ProviderConfig <|-- AnthropicConfig
    ProviderConfig <|-- OpenAIConfig
    OpenAIConfig <|-- OpenRouterConfig

    Provider o-- ProviderConfig : config
    ProviderConfig ..> VendorDefaults : from_env() loads from providers.ini

    class create_provider {
        <<function>>
        +create_provider(name, overrides) Provider
    }
    class post_json {
        <<function>>
        +post_json(url, body, headers, timeout, max_attempts) dict
    }
    create_provider ..> Provider : creates
    create_provider ..> ProviderConfig : config_class.from_env()
    Provider ..> post_json : sends via transport
```

| Provider | Config | Notes |
|---|---|---|
| `GeminiProvider` | `GeminiConfig` | echoes reasoning signatures; generates call ids when Gemini sends none |
| `AnthropicProvider` | `AnthropicConfig` | `max_output_tokens` defaults to 8192 (Anthropic requires it) |
| `OpenAIProvider` | `OpenAIConfig` | no default model; sends `max_completion_tokens` |
| `OpenRouterProvider` | `OpenRouterConfig` | sends `max_tokens` and an `X-Title` header |

`#` marks internal methods that subclasses implement (`*` = abstract);
`$` marks class-level (static) members.

### Messages and tools

The loop and every provider exchange the same neutral types. They are plain
dicts (`TypedDict`), so a conversation can be saved as JSON and replayed on
any vendor.

```mermaid
classDiagram
    direction LR

    class Message {
        <<TypedDict>>
        +role user, assistant or tool
        +text str
        +tool_calls list of ToolCall
        +tool_call_id str
        +name str
        +is_error bool
    }
    class ToolCall {
        <<TypedDict>>
        +id str
        +name str
        +args dict
        +signature str
        +error str
    }
    class Completion {
        <<TypedDict>>
        +text str
        +tool_calls list of ToolCall
        +usage Usage
    }
    class Usage {
        <<TypedDict>>
        +input int
        +output int
    }
    class Tool {
        <<Protocol>>
        +name str
        +spec dict
        +run(kwargs) Any
    }
    class run_loop {
        <<function>>
        +run_loop(provider, system, messages, tools, on_event, before_tool, max_turns, before_turn) str
    }

    Message *-- ToolCall : assistant tool_calls
    Completion *-- ToolCall
    Completion *-- Usage
    run_loop ..> Message : appends to messages
    run_loop ..> Completion : receives from Provider.complete()
    run_loop ..> Tool : run()
```

## Project layout

```
odysseus
├── README.md                          this file
├── pyproject.toml                     packaging (pip install -e .)
├── src
│   └── odysseus_v2                    the package (import odysseus_v2)
│       ├── __init__.py                public API: create_provider, run_loop
│       ├── loop.py                    the agent loop
│       └── providers
│           ├── __init__.py            create_provider() and the vendor registry
│           ├── base.py                Provider interface, config, HTTP with retries
│           ├── providers.ini          per-vendor defaults (base URL, model, key vars)
│           ├── anthropic.py           Anthropic Messages API
│           ├── gemini.py              Google Gemini API
│           └── openai_compatible.py   OpenAI, OpenRouter and other compatible servers
├── demos
│   ├── index.html                     list of demos
│   └── dice                           one tool, two tasks: the loop end to end
│       ├── dice.py                    the demo script
│       └── dice_walkthrough.html      goal, how to run, sample output, analysis
└── tests
    └── test_harness.py                offline tests (fake transport, no network)
```
