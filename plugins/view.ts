import type { Context } from '@deepseek-ai/cordis'
import { defineTool } from '@deepseek-ai/dsh-tools'
import { jsonOutput, option, runBlazeJson } from './blaze.ts'

export const name = 'blaze-view'
export const inject = ['tools']

export function apply(ctx: Context) {
  ctx.tools.register(defineTool({
    name: 'blaze_view',
    description: 'Run `blaze view`: write one self-contained interactive HTML page for a person to open. With `model` (build/<model>/ or its chip/), the page is a pan/zoom chip workspace of the whole model: every generated block under that directory, laid out by layer with data-flow wires, matrix tiles sized by adder count (optimized vs unoptimized), blocks not generated yet as dim placeholders; clicking a matrix zooms into its adder graph. Writes <model dir>/view/<model>.html and returns layers, blocks generated, matrices optimized, adders and the HTML path. With `file`, parses one matrix block (plain .v or optimized .opt.v) into its adder graph (default build/<model>/layer<N>/view/<block stem>.html) and returns inputs, outputs, adders, adders in nodes shared by more than one output, max depth, graph nodes and the HTML path. With `file`, `row` and `schematic`, instead synthesizes that output row\'s fan-in cone to generic gates with yosys and draws it as SVG with netlistsvg; returns the row\'s adders, depth, gate counts and the .v/.json/.svg paths. A missing yosys or netlistsvg comes back as an error with install instructions.',
    parameters: {
      file: { type: 'string', description: 'Generated block, e.g. build/stories260K/layer0/l0_wq_matvec.v. Give this or `model`' },
      model: { type: 'string', description: 'Model build directory for the whole-chip workspace, e.g. build/stories260K. Give this or `file`' },
      checkpoint: { type: 'string', description: 'With `model`: checkpoint for the layer count and placeholder estimates (default models/<model>/<model>.bin)' },
      out: { type: 'string', description: 'Output file (default: view/<block stem>.html, or view/<block stem>.row<N>.svg, next to the block)' },
      row: { type: 'integer', description: 'Output row to draw as a gate schematic; requires schematic: true' },
      schematic: { type: 'boolean', description: 'Draw `row` as a gate schematic (needs yosys and netlistsvg) instead of writing the HTML page' },
    },
    output: jsonOutput,
    async execute(args, exec) {
      return runBlazeJson(['view', ...(args.file ? [args.file] : []), ...option('--model', args.model),
        ...option('--checkpoint', args.checkpoint), ...option('--out', args.out), ...option('--row', args.row),
        ...(args.schematic ? ['--schematic'] : [])], exec)
    },
  }))
}
