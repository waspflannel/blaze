import type { Context } from '@deepseek-ai/cordis'
import { defineTool } from '@deepseek-ai/dsh-tools'
import { jsonOutput, option, runBlazeJson } from './blaze.ts'

export const name = 'blaze-sim-block'
export const inject = ['tools']

export function apply(ctx: Context) {
  ctx.tools.register(defineTool({
    name: 'blaze_sim_block',
    description: 'Run `blaze sim-block`: compile a generated Verilog block with Verilator, drive seeded random int8 vectors and compare every output bit for bit with the Python integer reference. Returns match, vectors run, outputs per vector, the mismatched vector count and the first mismatch (vector, element, expected, got). Verilog can run arbitrary code when simulated, so only simulate files blaze generated or the user trusts.',
    parameters: {
      file: { type: 'string', required: true, description: 'Generated Verilog, e.g. build/stories260K/layer0/l0_wq_matvec.v; its top module must be named after the file' },
      ref: { type: 'string', required: true, description: 'Reference as layer<N>.<name>: a matmul tensor (wq wk wv wo w1 w2 w3) or a gen-op op, e.g. layer0.wq or layer0.rmsnorm' },
      model: { type: 'string', description: 'Checkpoint the file was generated from (default: models/stories260K/stories260K.bin)' },
      vectors: { type: 'integer', description: 'Random input vectors (default: 1000)' },
      seed: { type: 'integer', description: 'Random seed (default: 0)' },
    },
    output: jsonOutput,
    async execute(args, exec) {
      // Exit 1 with a JSON summary is a mismatch result, not a failure.
      return runBlazeJson(['sim-block', args.file, '--ref', args.ref, ...option('--model', args.model),
        ...option('--vectors', args.vectors), ...option('--seed', args.seed)], exec, [0, 1])
    },
  }))
}
