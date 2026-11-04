import type { Context } from '@deepseek-ai/cordis'
import { defineTool } from '@deepseek-ai/dsh-tools'
import { jsonOutput, option, runBlazeJson } from './blaze.ts'

export const name = 'blaze-gen-op'
export const inject = ['tools']

export function apply(ctx: Context) {
  ctx.tools.register(defineTool({
    name: 'blaze_gen_op',
    description: 'Run `blaze gen-op`: write one fixed-point op template as Verilog for one layer (<out>/l<layer>_<op>.v), sized and parameterized from the checkpoint. Returns the module, file, ports, output scale, number formats, measured error vs float and the adder/depth cost estimate.',
    parameters: {
      op: { type: 'string', required: true, enum: ['rmsnorm', 'silu_mul', 'softmax', 'rope', 'residual_add'], description: 'Op template' },
      model: { type: 'string', description: 'llama2.c .bin checkpoint (default: models/stories260K/stories260K.bin)' },
      layer: { type: 'integer', description: 'Layer index (default: 0)' },
      out: { type: 'string', description: 'Output directory (default: build)' },
    },
    output: jsonOutput,
    async execute(args, exec) {
      return runBlazeJson(['gen-op', args.op, ...option('--model', args.model), ...option('--layer', args.layer),
        ...option('--out', args.out)], exec)
    },
  }))
}
