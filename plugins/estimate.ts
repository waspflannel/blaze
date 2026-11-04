import type { Context } from '@deepseek-ai/cordis'
import { defineTool } from '@deepseek-ai/dsh-tools'
import { jsonOutput, option, runBlazeJson } from './blaze.ts'

export const name = 'blaze-estimate'
export const inject = ['tools']

export function apply(ctx: Context) {
  ctx.tools.register(defineTool({
    name: 'blaze_estimate',
    description: 'Run `blaze estimate`: estimate adders, depth and cycles for a generated .v block, or adders, cycles per token and tok/s (SKY130 and projected 5nm) for a whole model from its spec.json. These are estimates from a simple model, not synthesis results. Returns the totals, the assumptions and, for a spec, the per-op breakdown.',
    parameters: {
      file: { type: 'string', required: true, description: 'Generated .v block or build/<model>/spec.json' },
      model: { type: 'string', description: 'Checkpoint for a spec (default: models/<model>/<source file>); its sha256 must match the spec' },
      clock_mhz: { type: 'number', description: 'Assumed SKY130 clock in MHz (default: 50)' },
      adder_ns: { type: 'number', description: 'Assumed delay of one adder level in ns (default: 5)' },
    },
    output: jsonOutput,
    async execute(args, exec) {
      return runBlazeJson(['estimate', args.file, ...option('--model', args.model), ...option('--clock-mhz', args.clock_mhz),
        ...option('--adder-ns', args.adder_ns)], exec)
    },
  }))
}
