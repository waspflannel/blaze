import { readFile } from 'node:fs/promises'
import { resolve } from 'node:path'
import type { Context } from '@deepseek-ai/cordis'
import { defineTool } from '@deepseek-ai/dsh-tools'
import { blazeFailure, jsonOutput, option, runBlaze } from './blaze.ts'

export const name = 'blaze-scan-model'
export const inject = ['tools']

export function apply(ctx: Context) {
  ctx.tools.register(defineTool({
    name: 'blaze_scan_model',
    description: 'Run `blaze scan-model`: read a llama2.c checkpoint and write its model spec (config plus every op of one forward pass) to <out>/<model>/spec.json. Returns the spec path, config, param count and op counts.',
    parameters: {
      checkpoint: { type: 'string', required: true, description: 'llama2.c .bin checkpoint, e.g. models/stories260K/stories260K.bin' },
      out: { type: 'string', description: 'Output root (default: build)' },
    },
    output: jsonOutput,
    async execute(args, exec) {
      const run = await runBlaze(['scan-model', args.checkpoint, ...option('--out', args.out)], exec)
      if (run.exitCode !== 0) throw blazeFailure('scan-model', run)
      // scan-model has no --json; the spec file it names on its last line is the structured result.
      const specPath = run.stdout.match(/^wrote\s+(.+?)\s*$/m)?.[1]
      if (specPath === undefined) throw new Error(`unexpected blaze scan-model output:\n${run.stdout}`)
      const spec = JSON.parse(await readFile(resolve(run.cwd, specPath), { encoding: 'utf8', signal: exec.signal }))
      const byKind: Record<string, number> = {}
      for (const op of spec.ops) byKind[op.op] = (byKind[op.op] ?? 0) + 1
      const matmul = byKind.matmul ?? 0
      return {
        spec: specPath,
        model: spec.model,
        architecture: spec.architecture,
        source: spec.source,
        config: spec.config,
        params: spec.params,
        ops: { total: spec.ops.length, matmul, template: spec.ops.length - matmul, by_kind: byKind },
      }
    },
  }))
}
