# Block agent

The block agent is a dsh skill, [`.dsh/skills/block-agent.md`](../.dsh/skills/block-agent.md). It takes one layer of a checkpoint (default stories260K layer 0) and works only through the `blaze_*` dsh tools:

1. scans the model,
2. generates the 7 matrices (default 8 bits) and 5 ops,
3. simulates every block with `sim-block`,
4. on a mismatch, regenerates the block and retries, stopping after 3 failed attempts on one block,
5. reports per-block, layer and whole-model estimates.

There is no adder budget or weight-error limit. The report includes each block's adders and each matrix's relative RMS weight error, plus the layer total (stories260K layer 0 is about 107,000 adders at 8 bits: matmuls 93,247, ops 13,956), as information only. To trade adders for error, ask for a different width, e.g. `... with 6-bit matrices`; `gen-matrix --bits` takes 2..8. Ops have fixed int8 inputs.

dsh finds the skill on its own: any dsh chat whose working directory is inside this repo lists `block-agent`.

## Model: Codex through your ChatGPT sign-in

The agent runs in a dsh profile called `blaze`. Its model is Codex through the [`dsh-codex`](https://github.com/Yan-Zero/dsh-codex) bundle, which uses ChatGPT OAuth, not an OpenAI API key. [`plugins/codex.patch.yml`](../plugins/codex.patch.yml) pins the model (`openai-codex` / `gpt-6.1-sol`, high reasoning) and disables dsh's DeepSeek API-key, DeepSeek-account and DeepSeek-search rows, so nothing in the profile can bill DeepSeek.

`dsh-codex` stores its login in `~/.dsh/.openai-codex-auth.json`, which every profile in that `DSH_HOME` shares. If you already signed in through the desktop app's **Settings -> OpenAI Codex**, there is nothing more to do. If you haven't, open that settings page and choose **Sign in with ChatGPT**. The plugin's `status` and `login` CLI commands currently fail with dsh 0.2.0-rc.2, because `@earendil-works/pi-ai` is missing from the profile, so sign in from the desktop app.

## One-time setup

`dsh` is `C:\deepseek harness\harness\resources\runtime\cli\bin\dsh.cmd` if it isn't on PATH. Blaze setup and Verilator are in the [README](../README.md).

```powershell
cd C:\blaze
dsh --profile blaze --from-default-profile headless --help      # creates ~/.dsh/profiles/blaze from the headless template
dsh plugin --profile blaze add dsh-codex@0.3.2 link:C:/blaze/plugins
Copy-Item plugins\codex.patch.yml $HOME\.dsh\profiles\blaze\cordis.patch.yml
dsh --profile blaze --dump-config | Select-String 'openai-codex|blaze-|disabled: true'
```

The dump should show `provider: openai-codex`, five `blaze-*` rows, and `disabled: true` under `llm-deepseek`, `llm-deepseek-account` and `web-search-deepseek`.

## Run it

```powershell
cd C:\blaze
Remove-Item Env:OPENAI_API_KEY, Env:DEEPSEEK_API_KEY -ErrorAction SilentlyContinue
$env:BLAZE_BIN = "C:\blaze\.venv\Scripts\blaze.exe"
dsh --profile blaze "Use the block-agent skill on stories260K layer 0."
```

The final report goes to stdout. Add `--json` to get every tool call and result as newline-delimited events. To change the inputs, say so in the task, e.g. `... layer 1, 6-bit matrices`. Blocks go to `build/agent/`.

To continue the same chat, pass `--session-id <id>`; the `session` event of a `--json` run carries the id. The skill also works in an interactive dsh chat (Web or TUI) on a profile that has `blaze-dsh` and `dsh-codex`, started from the repo root.
