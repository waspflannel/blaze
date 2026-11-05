import type { Context } from '@deepseek-ai/cordis'
import { defineTool } from '@deepseek-ai/dsh-tools'
import { blazeFailure, jsonOutput, option, runBlaze } from './blaze.ts'

export const name = 'blaze-gen-matrix'
export const inject = ['tools']

export function apply(ctx: Context) {
  ctx.tools.register(defineTool({
    name: 'blaze_gen_matrix',
    description: 'Run `blaze gen-matrix`: quantize one weight matrix to signed `bits`-bit weights (default 8), turn it into shift/add equations and write a combinational Verilog module (<out>/layer<layer>/l<layer>_<tensor>_matvec.v plus a .txt of the equations). Fewer bits means fewer adders and more weight error. Returns the file paths, shape, nonzero weights, adder count, depth, output width, bits and relative RMS weight error.',
    parameters: {
      model: { type: 'string', description: 'llama2.c .bin checkpoint (default: models/stories260K/stories260K.bin)' },
      layer: { type: 'integer', description: 'Layer index (default: 0)' },
      tensor: { type: 'string', enum: ['wq', 'wk', 'wv', 'wo', 'w1', 'w2', 'w3'], description: 'Matmul tensor (default: wq)' },
      out: { type: 'string', description: 'Model output root (default: build). Files go in <out>/layer<layer>/.' },
      bits: { type: 'integer', description: 'Signed weight bits, 2..8 (default: 8). sim-block reads it back from the file.' },
    },
    output: jsonOutput,
    async execute(args, exec) {
      const run = await runBlaze(['gen-matrix', ...option('--model', args.model), ...option('--layer', args.layer),
        ...option('--tensor', args.tensor), ...option('--out', args.out), ...option('--bits', args.bits)], exec)
      if (run.exitCode !== 0) throw blazeFailure('gen-matrix', run)
      // ponytail: gen-matrix has no --json, so this reads its fixed summary lines. Switch to --json once it has one.
      const out = run.stdout
      const shape = out.match(/^shape\s+(\d+) x (\d+)/m)
      const nonzero = out.match(/^nonzero\s+(\d+) \/ (\d+)/m)
      const adders = out.match(/^adders\s+(\d+) \((\d+) CSD terms\)(?:, tree depth (\d+))?/m)
      const scale = out.match(/^scale\s+(\S+); y is signed (\d+)-bit/m)
      const bits = out.match(/^bits\s+(\d+); weight error (\S+)/m)
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
        // Older gen-matrix builds have no --bits and always use 8.
        ...(bits === null ? {} : { bits: Number(bits[1]), weight_error: Number(bits[2]) }),
      }
    },
  }))
}
