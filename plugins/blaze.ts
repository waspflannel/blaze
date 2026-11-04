// Shared by the blaze tool plugins: run the `blaze` CLI and hand its result back as JSON.
import { execFile } from 'node:child_process'
import type { ToolExecution } from '@deepseek-ai/dsh-tools'

// dsh runs blaze from its own process, so blaze must be on the PATH dsh started with
// (activate the venv first) or named by BLAZE_BIN.
const BLAZE = process.env.BLAZE_BIN ?? 'blaze'

export interface BlazeRun {
  /** Session working directory blaze ran in; relative paths in its output are relative to this. */
  cwd: string
  exitCode: number
  stdout: string
  stderr: string
}

/**
 * Run `blaze <args>` in the session's working directory, which should be the blaze repo root
 * because blaze's defaults (models/, build/) are relative to it. Resolves with any exit code and
 * rejects only when blaze can't start or the call is cancelled, which kills the blaze process.
 */
export function runBlaze(args: string[], exec: ToolExecution): Promise<BlazeRun> {
  const cwd: string = exec.agent?.session.header.cwd ?? process.cwd()
  return new Promise((resolve, reject) => {
    execFile(BLAZE, args, { cwd, signal: exec.signal, windowsHide: true, maxBuffer: 16 * 1024 * 1024 }, (error, stdout, stderr) => {
      if (error === null) resolve({ cwd, exitCode: 0, stdout, stderr })
      else if (typeof error.code === 'number') resolve({ cwd, exitCode: error.code, stdout, stderr })
      else if (error.code === 'ENOENT') reject(new Error(`${BLAZE} not found: activate the blaze venv before starting dsh, or set BLAZE_BIN`))
      else reject(error)
    })
  })
}

/** The error for a failed run: blaze's own message from stderr. */
export function blazeFailure(command: string, run: BlazeRun): Error {
  return new Error(run.stderr.trim() || `blaze ${command} exited with code ${run.exitCode}`)
}

/**
 * Run a command that prints one JSON object with `--json`. `resultExitCodes` lists the exit
 * codes that still carry that object (sim-block exits 1 on a mismatch).
 */
export async function runBlazeJson(args: string[], exec: ToolExecution, resultExitCodes = [0]): Promise<any> {
  const run = await runBlaze([...args, '--json'], exec)
  if (!resultExitCodes.includes(run.exitCode) || !run.stdout.trim().startsWith('{')) throw blazeFailure(args[0], run)
  return JSON.parse(run.stdout)
}

/** `[flag, value]`, or nothing when the model left the argument out so blaze's default applies. */
export function option(flag: string, value: string | number | undefined): string[] {
  return value === undefined ? [] : [flag, String(value)]
}

/** Every blaze tool returns one JSON object, shown to the model as indented JSON. */
export const jsonOutput = {
  schema: { type: 'json' },
  render: (_args: unknown, value: unknown) => [{ type: 'text', text: JSON.stringify(value, null, 2) }],
} as const
