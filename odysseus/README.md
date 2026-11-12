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
cd ai-harness/odysseus
```

All the commands below are run from the `odysseus` folder.

### 2. Create a virtual environment (recommended)

Requires Python 3.10 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
```

### 3. Install the package

```bash
pip install -e .
```

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

## Project layout

```
odysseus/
  pyproject.toml               packaging (pip install -e .)
  src/odysseus_v2/             the package (import odysseus_v2)
    loop.py                    the agent loop
    providers/
      base.py                  Provider interface, config, HTTP with retries
      providers.ini            per-vendor defaults (base URL, model, key vars)
      anthropic.py             Anthropic Messages API
      gemini.py                Google Gemini API
      openai_compatible.py     OpenAI, OpenRouter and other compatible servers
  demos/
    index.html                 list of demos
    dice/                      one tool, two tasks: the loop end to end
  tests/
    test_harness.py            offline tests (fake transport, no network)
```
