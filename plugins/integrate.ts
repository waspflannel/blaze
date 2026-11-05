import type { Context } from '@deepseek-ai/cordis'
import { defineTool } from '@deepseek-ai/dsh-tools'
import { jsonOutput, option, runBlazeJson } from './blaze.ts'

export const name = 'blaze-integrate'
export const inject = ['tools']

export function apply(ctx: Context) {
  ctx.tools.register(defineTool({
    name: 'blaze_integrate',
    description: 'Run `blaze integrate`: stitch the verified blocks of the first N layers into one chip (<out>/blaze_chip.v) with embedding, attention + KV cache, final norm, classifier and control, then simulate it with Verilator, run BOS, the optional prompt tokens, then greedy decoding, and compare every output token (plus each token\'s logit) with the Python integer reference. Needs all 12 verified block files of every included layer in the blocks directory, exactly as the gen tools wrote them. Returns match, first_mismatch (step, expected, got), reference and chip tokens and text, cycles per token from the simulation, tok/s estimates, the bits per matrix and the float model\'s agreement. Builds and runs Verilog, so only use block files blaze generated.',
    parameters: {
      model: { type: 'string', description: 'llama2.c .bin checkpoint the blocks came from (default: models/stories260K/stories260K.bin)' },
      blocks: { type: 'string', description: 'Directory of verified block files (default: build/agent/<model file stem>)' },
      layers: { type: 'integer', description: 'Build the first N layers (default: all)' },
      steps: { type: 'integer', description: 'Positions to run, 1..seq_len (default: 32)' },
      prompt: { type: 'string', description: 'Comma-separated token ids fed after BOS before greedy decoding, e.g. 403,407,261 (default: none)' },
      out: { type: 'string', description: 'Output directory (default: build/chip/<model file stem>)' },
    },
    output: jsonOutput,
    async execute(args, exec) {
      // Exit 1 with a JSON summary is a token mismatch result, not a failure.
      return runBlazeJson(['integrate', ...option('--model', args.model), ...option('--blocks', args.blocks),
        ...option('--layers', args.layers), ...option('--steps', args.steps), ...option('--prompt', args.prompt), ...option('--out', args.out)], exec, [0, 1])
    },
  }))
}
