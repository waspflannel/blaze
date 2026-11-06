import type { Context } from '@deepseek-ai/cordis'
import { defineTool } from '@deepseek-ai/dsh-tools'
import { jsonOutput, option, runBlazeJson } from './blaze.ts'

export const name = 'blaze-optimize-matrix'
export const inject = ['tools']

export function apply(ctx: Context) {
  ctx.tools.register(defineTool({
    name: 'blaze_optimize_matrix',
    description: 'Run `blaze optimize-matrix`: take a gen-matrix .v, rebuild its CSD terms from the same quantized weights and compute shift/add pairs that several rows share once as named wires, so the same y = W x needs fewer adders. Writes <file stem>.opt.v next to the input with the same module name and ports (a drop-in replacement). Returns the output file, module, bits, shared wire count and adders and depth before and after. The optimized file still has to pass blaze_sim_block; integrate uses it in place of the .v while it exists.',
    parameters: {
      file: { type: 'string', required: true, description: 'gen-matrix Verilog, e.g. build/stories260K/layer0/l0_wq_matvec.v; layer, tensor and bits come from its header' },
      model: { type: 'string', description: 'Checkpoint the file was generated from (default: models/stories260K/stories260K.bin)' },
      discard: { type: 'boolean', description: 'Delete the .opt.v for this file instead of writing it, e.g. after it failed blaze_sim_block. Returns the path and whether it existed.' },
    },
    output: jsonOutput,
    async execute(args, exec) {
      return runBlazeJson(['optimize-matrix', args.file, ...option('--model', args.model), ...(args.discard ? ['--discard'] : [])], exec)
    },
  }))
}
