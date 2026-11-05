# plugins

The `blaze` CLI as DeepSeek Harness (`dsh`) tools. `blaze-dsh` is one dsh bundle with one Cordis plugin per command. Each plugin registers one tool that runs `blaze` and returns a JSON object. The logic stays in Python.

| tool | runs | returns |
|---|---|---|
| `blaze_scan_model` | `blaze scan-model` | spec path, config, params, op counts (total, matmul, template, per kind) |
| `blaze_gen_matrix` | `blaze gen-matrix` | `.v`/`.txt` paths, rows, cols, nonzero, adders, CSD terms, depth, scale, output width, bits, weight error |
| `blaze_sim_block` | `blaze sim-block --json` | `match`, vectors, outputs per vector, `mismatched_vectors`, `first_mismatch` (vector, element, expected, got) |
| `blaze_gen_op` | `blaze gen-op --json` | module, file, ports, output scale, formats, error vs float, cost estimate |
| `blaze_estimate` | `blaze estimate --json` | adders, depth, cycles, tok/s or evals/s, assumptions, per-op breakdown |

A failed command (bad path, wrong ports) comes back as a tool error with blaze's own message. A simulation mismatch is a normal result with `match: false`, not an error. `sim-block`, `gen-op` and `estimate` need a `blaze` that has those commands (PR #4). `scan-model` and `gen-matrix` have no `--json`, so their tools read the written `spec.json` and the summary lines.

Files: `blaze.ts` runs `blaze` (shared), one `<command>.ts` per tool, `cordis.patch.yml` lists the five plugin rows, and `package.json` declares the bundle. `codex.patch.yml` is the `blaze` profile's patch for running the block agent on Codex ([docs/block-agent.md](../docs/block-agent.md)).

## How dsh loads plugins

Checked against the installed dsh 0.2.0-rc.2 (its package READMEs, plus booting `sdk-minimal` with this bundle).

- A **profile** is a directory, `$DSH_HOME/profiles/<name>` (`DSH_HOME` defaults to `~/.dsh`). Its `package.json` lists out-of-tree plugin packages as dependencies, plus an ordered bundle list in `dsh.profile.bundles`.
- A **bundle** is a package whose `package.json` has `dsh.bundle.patch`, a YAML patch list. Here that's `cordis.patch.yml`, which `insert`s five rows. Each row is one Cordis plugin: a module exporting `name`, `inject` (the services it needs; `tools` here) and `apply(ctx)`.
- At startup dsh composes the tree from each selected bundle's patch, in list order. Then come the profile's own `cordis.patch.yml`, `$DSH_HOME/cordis.patch.yml` and any `--patch` files. `dsh --profile <name> --dump-config` prints the result.
- Row names relative to the patch file become file URLs. dsh's Node 24 strips TypeScript types, so the `.ts` files load directly with no build step. Bare imports like `@deepseek-ai/dsh-tools` resolve to the dsh installation's copy, because the package is a linked profile dependency.
- `sdk-minimal` is a standalone tree: no `dsh-base`, Web UI, settings or file tools. It has a JSON-RPC server on stdio, the DeepSeek API-key model adapter, one persistent shell (`pwsh` on Windows) and JSONL sessions under `$DSH_HOME/sessions`. Its shell runs with `danger-full-access`. It only reads patches at startup (no hot reload), so restart dsh after installing, enabling or disabling a plugin.

### Does a new chat use active plugins automatically?

Yes. `ctx.tools.register` adds a tool to the process-wide registry, and the registry feeds every visible tool's schema into each agent's model request. There's no per-chat opt-in. Any session in a dsh process that loaded the bundle sees the tools, whether it's new or resumed. `sdk-minimal` restricts no tools. Booted with this bundle, its tool list is `pwsh`, `blaze_scan_model`, `blaze_gen_matrix`, `blaze_sim_block`, `blaze_gen_op`, `blaze_estimate`.

Installing a bundle selects it. To turn it off, take it out of `dsh.profile.bundles`, or set `disabled: true` on a row in the profile's `cordis.patch.yml`, e.g. `- id: blaze-sim-block` / `disabled: true`. Then restart.

## Install

`dsh` is the launcher from the desktop install, `C:\deepseek harness\harness\resources\runtime\cli\bin\dsh.cmd`, if it isn't on PATH. `sdk-minimal` is set up from its shipped template on first use:

```powershell
dsh plugin --profile sdk-minimal add link:C:/blaze/plugins   # adds blaze-dsh and selects the bundle
dsh --profile sdk-minimal --dump-config | Select-String blaze # five blaze-* rows
```

`link:` points the profile at this folder, so edits here apply on the next dsh start. Remove it with `dsh plugin --profile sdk-minimal remove blaze-dsh`.

## Start a chat that uses the tools

The tools run `blaze` in the session's working directory, and blaze's defaults (`models/`, `build/`) are relative to the repo root. So start dsh from an activated venv and pass the repo root as `cwd`. You can also set `BLAZE_BIN` to `C:\blaze\.venv\Scripts\blaze.exe` instead of activating.

`sdk-minimal` has no chat UI: an SDK client talks JSON-RPC to it, one message per line on stdin and stdout. Its only model route is the DeepSeek API, which needs `DEEPSEEK_API_KEY` and is billed per token.

```powershell
cd C:\blaze
.venv\Scripts\Activate.ps1
$env:DEEPSEEK_API_KEY = "<your key>"
dsh --profile sdk-minimal
```

Then send, one line each:

```json
{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"cwd":"C:/blaze","provider":"deepseek-official","model":"deepseek-v4-pro"}}
{"jsonrpc":"2.0","id":2,"method":"session/prompt","params":{"sessionId":"blaze-1","contentBlocks":[{"type":"text","text":"Generate layer 0 wq with blaze, simulate it against layer0.wq, and estimate it."}]}}
```

`initialize` must finish before any prompt. Each prompt returns a `messageId` right away. The agent's work then streams back as `session.event` and `session.status` notifications, including the tool calls and their JSON results. Reuse a `sessionId` to continue that chat. Send `{"jsonrpc":"2.0","id":3,"method":"shutdown"}` to stop.
