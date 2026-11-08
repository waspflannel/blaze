import type { Context } from '@deepseek-ai/cordis'
import { defineTool } from '@deepseek-ai/dsh-tools'
import { jsonOutput, option, runBlazeJson } from './blaze.ts'

export const name = 'blaze-pd'
export const inject = ['tools']

export function apply(ctx: Context) {
  ctx.tools.register(defineTool({
    name: 'blaze_pd',
    description: 'Run `blaze pd`: physical design of one generated layer (default all 12 blocks of build/<model>/layer<N>/, matvecs as .opt.v) with OpenLane 2 on SKY130 sky130_fd_sc_hd. Writes a shift-chain wrapper top and an OpenLane config.json under build/<model>/pd/<target>/, checks the toolchain, PDK and free memory, runs the flow and writes pd.json: status (complete, blocked, deferred, failed, dry-run), GDS path, area, timing (WNS/TNS, clock), power, DRC and LVS, each metric tagged measured or unavailable with where it came from, plus provenance. blocked lists each missing tool / PDK and what was checked; deferred gives the memory numbers. Those are normal results, not errors. A run can take hours and many GB of RAM; use dry_run first. `block` is a single-block substitute, allowed only after the layer run was deferred or failed, and is never a layer result.',
    parameters: {
      layer_dir: { type: 'string', required: true, description: 'Generated layer directory, e.g. build/stories260K/layer0' },
      block: { type: 'string', description: 'Substitute: one block file name in layer_dir, e.g. l0_wk_matvec.opt.v' },
      out: { type: 'string', description: 'Run directory (default build/<model>/pd/<layer or block>)' },
      clock_ns: { type: 'number', description: 'Clock period in ns (default 20)' },
      threads: { type: 'integer', description: 'Flow threads (default min(4, cores/2))' },
      max_mem_gb: { type: 'number', description: 'Memory cap; the run is deferred if less is free (default 8)' },
      dry_run: { type: 'boolean', description: 'Write the wrapper and config.json only, do not run OpenLane' },
    },
    output: jsonOutput,
    async execute(args, exec) {
      // Exit 1 with a JSON summary is a blocked/deferred/failed result, not a tool failure.
      return runBlazeJson(['pd', args.layer_dir, ...option('--block', args.block), ...option('--out', args.out),
        ...option('--clock-ns', args.clock_ns), ...option('--threads', args.threads),
        ...option('--max-mem-gb', args.max_mem_gb), ...(args.dry_run ? ['--dry-run'] : [])], exec, [0, 1])
    },
  }))
}
