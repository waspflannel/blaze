import type { Context } from '@deepseek-ai/cordis'
import { defineTool } from '@deepseek-ai/dsh-tools'
import { jsonOutput, option, runBlazeJson } from './blaze.ts'

export const name = 'blaze-fpga-export'
export const inject = ['tools']

export function apply(ctx: Context) {
  ctx.tools.register(defineTool({
    name: 'blaze_fpga_export',
    description: 'Run `blaze fpga`: the last step, one generated Verilog top -> an FPGA bitstream. Synthesizes with yosys (synth_ecp5), places and routes with nextpnr-ecp5 and packs with ecppack for the one supported target, ecp5-85f (LFE5U-85F-6BG381C, the ULX3S 85F). Needs the user\'s board LPF with a LOCATE line for every top-level port bit; blaze has no pinout and never guesses pins. A top whose ports aren\'t all pinned (a matrix block\'s wide buses) is rejected as UNSUPPORTED unless `wrap` is "serial", which puts a combinational block behind an unvalidated 5-pin shift-register top. Returns the bitstream path, target, input/LPF/weights sha256, tool versions, Fmax and utilization from nextpnr (or "unavailable"). Missing tools, a rejected interface and a failed tool run come back as errors; a bitstream is never claimed unless ecppack wrote it. Nothing is run on hardware. Only use Verilog blaze generated.',
    parameters: {
      file: { type: 'string', required: true, description: 'Generated top: build/<model>/chip/blaze_chip.v or one matrix block .v/.opt.v' },
      target: { type: 'string', required: true, enum: ['ecp5-85f'], description: 'FPGA profile; the user must name it' },
      lpf: { type: 'string', required: true, description: 'The user\'s board LPF constraints file; never write or guess one' },
      top: { type: 'string', description: 'Top module (default: file name without .v / .opt.v)' },
      wrap: { type: 'string', enum: ['serial'], description: 'Put a combinational block behind a 5-pin shift-register top (unvalidated)' },
      model: { type: 'string', description: 'Checkpoint the Verilog came from, recorded with its sha256' },
      out: { type: 'string', description: 'Output directory (default: fpga/ next to the file)' },
    },
    output: jsonOutput,
    async execute(args, exec) {
      return runBlazeJson(['fpga', args.file, '--target', args.target, '--lpf', args.lpf, ...option('--top', args.top),
        ...option('--wrap', args.wrap), ...option('--model', args.model), ...option('--out', args.out)], exec)
    },
  }))
}
