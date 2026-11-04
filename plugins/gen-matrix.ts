import type { Context } from '@deepseek-ai/cordis'
import { defineTool } from '@deepseek-ai/dsh-tools'
import { blazeFailure, jsonOutput, option, runBlaze } from './blaze.ts'

export const name = 'blaze-gen-matrix'
export const inject = ['tools']

export function apply(ctx: Context) {
  ctx.tools.register(defineTool({
    name: 'blaze_gen_matrix',
    description: 'Run `blaze gen-matrix`: quantize one weight matrix to int8, turn it into shift/add equations and write a combinational Verilog module (<out>/l<layer>_<tensor>_matvec.v plus a .txt of the equations). Returns the file paths, shape, nonzero weights, adder count and output width.',
    parameters: {
      model: { type: 'string', description: 'llama2.c .bin checkpoint (default: models/stories260K/stories260K.bin)' },
      layer: { type: 'integer', description: 'Layer index (default: 0)' },
      tensor: { type: 'string', enum: ['wq', 'wk', 'wv', 'wo', 'w1', 'w2', 'w3'], description: 'Matmul tensor (default: wq)' },
      out: { type: 'string', description: 'Output directory (default: build)' },
    },
    output: jsonOutput,
    async execute(args, exec) {
      const run = await runBlaze(['gen-matrix', ...option('--model', args.model), ...option('--layer', args.layer),
        ...option('--tensor', args.tensor), ...option('--out', args.out)], exec)
      if (run.exitCode !== 0) throw blazeFailure('gen-matrix', run)
      // ponytail: gen-matrix has no --json, so this reads its fixed summary lines. Switch to --json once it has one.
      const out = run.stdout
      const shape = out.match(/^shape\s+(\d+) x (\d+)/m)
      const nonzero = out.match(/^nonzero\s+(\d+) \/ (\d+)/m)
      const adders = out.match(/^adders\s+(\d+) \((\d+) CSD terms\)(?:, tree depth (\d+))?/m)
      const scale = out.match(/^scale\s+(\S+); y is signed (\d+)-bit/m)
      const wrote = out.match(/^wrote\s+(.+), (.+?)\s*$/m)
      if (!shape || !nonzero || !adders || !scale || !wrote) throw new Error(`unexpected blaze gen-matrix output:\n${out}`)
      return {
        verilog: wrote[1],
        equations: wrote[2],
        rows: Number(shape[1]),
        cols: Number(shape[2]),
        nonzero: Number(nonzero[1]),
        weights: Number(nonzero[2]),
        adders: Number(adders[1]),
        csd_terms: Number(adders[2]),
        // Older gen-matrix builds don't print the tree depth.
        ...(adders[3] === undefined ? {} : { depth: Number(adders[3]) }),
        scale: Number(scale[1]),
        acc_width: Number(scale[2]),
      }
    },
  }))
}
