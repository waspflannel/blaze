"""`blaze view`: a generated matrix block as an interactive adder graph (one self-contained HTML file),
and one output row as a gate schematic (Yosys + netlistsvg). Deterministic; no LLM involved.

The block is parsed generically from its Verilog `wire ... = ...` and `assign` statements, so a plain
gen-matrix block and an optimized one with shared intermediate wires both work. Expressions may use
+ and - (n-ary sums), unary -, constant shifts (<< <<< >> >>>), $signed(), parentheses, nets, input
port bit-selects and integer constants. Anything else stops with an error naming the statement.

Graph nodes: `in` (an input slice), `const`, `shift` (free wiring), `sum` (an n-input add/subtract,
n - 1 adders) and `out` (an output row). Identical subexpressions become one node. A node's depth is
the fewest adder levels that can compute it as a balanced tree over its inputs' depths, which equals
gen-matrix's ceil(log2 terms) for a plain block.
"""
import heapq
import json
import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path

from blaze.sim import WSL_TOOLS

TOKEN = re.compile(r"\s*(?:(\d*'[sS]?[bdhoBDHO][0-9a-fA-F_]+|\d+)|(\$?[A-Za-z_][\w$]*)|(<<<|>>>|<<|>>|[-+()\[\]:]))")
SHIFTS = {"<<": 1, "<<<": 1, ">>": -1, ">>>": -1}
KINDS = ["in", "const", "shift", "sum", "out"]


def _number(tok: str) -> int:
    if "'" not in tok:
        return int(tok)
    digits = tok.split("'")[1].lstrip("sS")
    return int(digits[1:].replace("_", ""), {"b": 2, "o": 8, "d": 10, "h": 16}[digits[0].lower()])


def _tokens(text: str) -> list[str]:
    out, pos = [], 0
    while text[pos:].strip():
        m = TOKEN.match(text, pos)
        if not m:
            raise ValueError(f"unsupported expression syntax near {text[pos:pos + 30].strip()!r}")
        out.append(m[1] or m[2] or m[3])
        pos = m.end()
    return out


def parse_block(text: str) -> dict:
    """Module name, input/output ports and the net/output statements of a matrix block."""
    text = re.sub(r"//[^\n]*|/\*.*?\*/", "", text, flags=re.S)
    m = re.search(r"\bmodule\s+(\w+)\s*\((.*?)\)\s*;(.*?)\bendmodule", text, re.S)
    if not m:
        raise ValueError("no ANSI-style module found")
    ports = {}
    for part in m[2].split(","):
        p = re.fullmatch(r"\s*(input|output)\s+(?:wire\s+)?(?:signed\s+)?(?:\[(\d+):(\d+)\]\s*)?(\w+)\s*", part)
        if not p:
            raise ValueError(f"unsupported port declaration {part.strip()!r}")
        ports[p[4]] = {"dir": p[1], "width": int(p[2]) - int(p[3]) + 1 if p[2] else 1, "text": part.strip()}
    defs, stmts, outputs = {}, [], []
    for s in (s.strip() for s in m[3].split(";")):
        if not s:
            continue
        w = re.fullmatch(r"wire\s+(?:signed\s+)?(?:\[\d+:\d+\]\s*)?(\w+)\s*(?:=\s*(.*))?", s, re.S)
        a = re.fullmatch(r"assign\s+(\w+)\s*(?:\[(\d+)(?::(\d+))?\])?\s*=\s*(.*)", s, re.S)
        if w:
            name, expr = w[1], w[2]
        elif a and ports.get(a[1], {}).get("dir") == "output":
            hi = int(a[2]) if a[2] else ports[a[1]]["width"] - 1
            lo = int(a[3] or a[2] or 0)
            width = hi - lo + 1
            label = f"{a[1]}[{lo // width}]" if a[2] else a[1]
            outputs.append({"label": label, "index": lo // width, "width": width, "expr": a[4]})
            continue
        elif a and not a[2]:
            name, expr = a[1], a[4]
        else:
            raise ValueError(f"unsupported statement {s[:60]!r}")
        if name in ports:
            raise ValueError(f"unsupported statement {s[:60]!r}")
        stmts.append((name, s + ";"))
        if expr is not None:
            if name in defs:
                raise ValueError(f"net {name} is assigned twice")
            defs[name] = expr
    if not outputs:
        raise ValueError("no output assigns found")
    outputs.sort(key=lambda o: o["index"])
    return {"module": m[1], "ports": ports, "defs": defs, "stmts": stmts, "outputs": outputs}


class Graph:
    """The DAG of a parsed block. Node ids are in topological order (inputs before their users)."""

    def __init__(self, block: dict):
        self.block = block
        self.kind, self.label, self.ins, self.shift, self.net_node = [], [], [], [], {}
        self._index, self._named, self._active = {}, set(), set()
        for name in block["defs"]:
            self._net(name)
        self.outputs = [self._node(("out", o["label"]), "out", o["label"], [(1, self._expr(o["expr"]))])
                        for o in block["outputs"]]
        self._analyze()

    def _node(self, key, kind: str, label: str, ins: list[tuple[int, int]], shift: int = 0) -> int:
        if key not in self._index:
            self._index[key] = len(self.kind)
            self.kind.append(kind)
            self.label.append(label)
            self.ins.append(ins)
            self.shift.append(shift)
        return self._index[key]

    def _net(self, name: str) -> int:
        if name not in self.net_node:
            if name in self._active:
                raise ValueError(f"net {name} depends on itself")
            if name not in self.block["defs"]:
                raise ValueError(f"net {name} is used but never assigned")
            self._active.add(name)
            n = self.net_node[name] = self._expr(self.block["defs"][name])
            self._active.discard(name)
            if n not in self._named:  # name the node after its first net, e.g. x_3 rather than x[31:24]
                self._named.add(n)
                self.label[n] = name
        return self.net_node[name]

    def _expr(self, text: str) -> int:
        self._toks, self._pos = _tokens(text), 0
        n = self._sum_node(self._sum())
        if self._pos != len(self._toks):
            raise ValueError(f"unexpected {self._toks[self._pos]!r} in {text.strip()[:60]!r}")
        return n

    def _peek(self) -> str | None:
        return self._toks[self._pos] if self._pos < len(self._toks) else None

    def _take(self, want: str | None = None) -> str:
        tok = self._peek()
        if tok is None or (want and tok != want):
            raise ValueError(f"expected {want or 'more input'}, got {tok!r}")
        self._pos += 1
        return tok

    def _sum(self) -> list[tuple[int, int]]:
        terms = self._unary()
        while self._peek() in ("+", "-"):
            sign = 1 if self._take() == "+" else -1
            terms += [(sign * s, n) for s, n in self._unary()]
        return terms

    def _unary(self) -> list[tuple[int, int]]:
        if self._peek() == "-":
            self._take()
            return [(-s, n) for s, n in self._unary()]
        n = self._primary()
        while self._peek() in SHIFTS:
            k = SHIFTS[self._take()] * _number(self._take())
            n = self._node(("shift", n, k), "shift", f"{self.label[n]} {'<<' if k > 0 else '>>'} {abs(k)}", [(1, n)], k)
        return [(1, n)]

    def _sum_node(self, terms: list[tuple[int, int]]) -> int:
        if len(terms) == 1 and terms[0][0] == 1:
            return terms[0][1]
        n = len(terms)
        return self._node(("sum", tuple(terms)), "sum", f"{'sum' if all(s > 0 for s, _ in terms) else 'add/sub'} "
                          f"{n} terms", terms)

    def _primary(self) -> int:
        tok = self._take()
        if tok in ("(", "$signed", "$unsigned"):
            if tok != "(":
                self._take("(")
            n = self._sum_node(self._sum())
            self._take(")")
            return n
        if tok[0].isdigit() or tok[0] == "'":
            return self._node(("const", _number(tok)), "const", str(_number(tok)), [])
        port = self.block["ports"].get(tok)
        if port and port["dir"] == "input":
            sel = ""
            if self._peek() == "[":
                self._take()
                hi = lo = self._take()
                if self._peek() == ":":
                    self._take()
                    lo = self._take()
                self._take("]")
                sel = f"[{hi}:{lo}]" if lo != hi else f"[{hi}]"
            return self._node(("in", tok + sel), "in", tok + sel, [])
        if port or tok.startswith("$") or not re.fullmatch(r"[A-Za-z_]\w*", tok):
            raise ValueError(f"unsupported operand {tok!r}")
        toks, pos = self._toks, self._pos  # _net parses the net's own expression with a new token stream
        n = self._net(tok)
        self._toks, self._pos = toks, pos
        if self._peek() == "[":
            raise ValueError(f"bit-select of internal net {tok} is not supported")
        return n

    def cone(self, out: int) -> list[int]:
        """Node ids feeding `out`, `out` included, ascending."""
        seen, stack = {out}, [out]
        while stack:
            for _, c in self.ins[stack.pop()]:
                if c not in seen:
                    seen.add(c)
                    stack.append(c)
        return sorted(seen)

    def _analyze(self) -> None:
        n = len(self.kind)
        self.adders = [max(len(i) - 1, 0) if k == "sum" else 0 for k, i in zip(self.kind, self.ins)]
        self.depth, self.level = [0] * n, [0] * n
        for v in range(n):
            ds = [self.depth[c] for _, c in self.ins[v]]
            if self.kind[v] == "sum" and len(ds) > 1:
                heapq.heapify(ds)  # combine the two shallowest inputs first: the optimal balanced tree
                while len(ds) > 1:
                    heapq.heappush(ds, max(heapq.heappop(ds), heapq.heappop(ds)) + 1)
            self.depth[v] = max(ds, default=0)
            self.level[v] = 1 + max((self.level[c] for _, c in self.ins[v]), default=-1)
        self.users = [0] * n
        cones = [self.cone(o) for o in self.outputs]
        for c in cones:
            for v in c:
                self.users[v] += 1
        self.rows = [{"adders": sum(self.adders[v] for v in c),
                      "shared_adders": sum(self.adders[v] for v in c if self.users[v] > 1),
                      "nodes": len(c), "depth": self.depth[o]} for o, c in zip(self.outputs, cones)]

    def stats(self) -> dict:
        return {"module": self.block["module"], "inputs": self.kind.count("in"), "outputs": len(self.outputs),
                "adders": sum(self.adders), "sum_nodes": self.kind.count("sum"),
                "shared_adders": sum(a for a, u in zip(self.adders, self.users) if u > 1),
                "max_depth": max(self.depth[o] for o in self.outputs), "nodes": len(self.kind),
                "intermediates": sum(1 for name in self.block["defs"] if self.kind[self.net_node[name]] == "sum")}


def view_dir(block_path: Path) -> Path:
    """Default output directory: view/ next to the block, e.g. build/stories260K/layer0/view/."""
    return block_path.parent / "view"


def block_stem(block_path: Path) -> str:
    """`l0_wq_matvec.v` -> `l0_wq_matvec`, `l0_wq_matvec.opt.v` -> `l0_wq_matvec.opt`."""
    return block_path.name.removesuffix(".v")


def write_html(block_path: Path, out: Path) -> dict:
    """Parse the block and write the viewer page to `out`. Returns the block stats."""
    g = Graph(parse_block(block_path.read_text()))
    stats = g.stats()
    try:
        svg_dir = Path(os.path.relpath(view_dir(block_path), out.parent)).as_posix()
    except ValueError:  # another drive on Windows
        svg_dir = view_dir(block_path).resolve().as_uri()
    data = {
        "file": block_path.as_posix(), "stats": stats,
        "kind": [KINDS.index(k) for k in g.kind], "label": g.label,
        # Each input as sign * (source id + 1), so 0 never appears.
        "ins": [[s * (c + 1) for s, c in i] for i in g.ins],
        "shift": g.shift,
        "adders": g.adders, "depth": g.depth, "level": g.level, "users": g.users,
        "outputs": [{"id": o, "label": g.label[o], "index": b["index"], **r}
                    for o, b, r in zip(g.outputs, g.block["outputs"], g.rows)],
        # Where `--row N --schematic` writes by default, relative to this page.
        "schematic": {"svg": f"{svg_dir}/{block_stem(block_path)}.row{{N}}.svg",
                      "command": f"blaze view {block_path.as_posix()} --row {{N}} --schematic"},
    }
    # ponytail: the whole graph is inlined as JSON; fine for stories260K-sized blocks (~1 MB). Large
    # models (4096x4096) would need a sidecar file loaded per row.
    page = (Path(__file__).with_name("viewer.html").read_text(encoding="utf-8")
            .replace("/*BLAZE_DATA*/null", json.dumps(data, separators=(",", ":")).replace("</", "<\\/")))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8", newline="\n")
    return stats


def cone_verilog(block: dict, g: Graph, row: int) -> tuple[str, str]:
    """(module name, Verilog) of a standalone module computing output row `row` only."""
    idx = [o["index"] for o in block["outputs"]]
    if row not in idx:
        raise ValueError(f"row {row} out of range; the block has outputs {idx[0]}..{idx[-1]}")
    k = idx.index(row)
    cone = set(g.cone(g.outputs[k]))
    nets = {name for name, n in g.net_node.items() if n in cone}
    o = block["outputs"][k]
    top = f"{block['module']}_row{row}"
    lines = [f"// Generated by blaze view: the fan-in cone of {block['module']} {o['label']}.",
             f"module {top} ("]
    lines += [f"    {p['text']}," for p in block["ports"].values() if p["dir"] == "input"]
    lines += [f"    output wire signed [{o['width'] - 1}:0] y", ");"]
    lines += [f"    {s}" for name, s in block["stmts"] if name in nets]
    lines += [f"    assign y = {o['expr'].strip()};", "endmodule", ""]
    return top, "\n".join(lines)


NETLISTSVG_MISSING = ("netlistsvg not found on PATH. Install it with npm (needs Node.js): npm install -g netlistsvg "
                      "(or install it in any folder and add its node_modules/.bin to PATH)")
YOSYS_MISSING = ("yosys not found. Install OSS CAD Suite (it includes yosys) and put its bin/ on PATH; on Windows, "
                 "unpack it in WSL at ~/tools/oss-cad-suite as in the README's Simulation setup")


def _command(args: list[str], missing: str) -> list[str]:
    """The native tool from PATH; yosys falls back to OSS CAD Suite in WSL on Windows, like sim-block."""
    exe = shutil.which(args[0])
    if exe:
        return [exe, *args[1:]]
    if args[0] == "yosys" and os.name == "nt" and shutil.which("wsl"):
        return ["wsl", "-e", "bash", "-c", f'PATH="{WSL_TOOLS}:$PATH"; {shlex.join(args)}']
    raise RuntimeError(missing)


def _run(cmd: list[str], cwd: Path, name: str, missing: str) -> None:
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if r.returncode == 127:  # WSL bash: command not found
        raise RuntimeError(missing)
    if r.returncode:
        tail = "\n".join((r.stderr or r.stdout).strip().splitlines()[-15:])
        raise RuntimeError(f"{name} failed with exit {r.returncode}:\n{tail}")


def write_schematic(block_path: Path, row: int, svg: Path) -> dict:
    """Extract output `row`'s cone, synthesize it to generic gates with Yosys, draw it with netlistsvg.

    Writes <stem>.row<N>.v and .json next to `svg`. Returns the row stats and gate counts.
    """
    block = parse_block(block_path.read_text())
    g = Graph(block)
    top, verilog = cone_verilog(block, g, row)
    svg.parent.mkdir(parents=True, exist_ok=True)
    base = svg.name.removesuffix(".svg")
    v, js = svg.with_name(base + ".v"), svg.with_name(base + ".json")
    # Both tools are looked up before either runs, so a missing netlistsvg fails fast.
    # ponytail: -noabc keeps yosys's plain gate mapping (AND/OR/XOR/NOT). ABC's optimization took over 3 minutes
    # on one stories260K row; add an --abc flag if a smaller gate count matters more than time.
    yosys = _command(["yosys", "-q", "-p", f"read_verilog {v.name}; synth -flatten -noabc -top {top}; "
                      f"opt_clean -purge; write_json {js.name}"], YOSYS_MISSING)
    netlistsvg = _command(["netlistsvg", js.name, "-o", svg.name], NETLISTSVG_MISSING)
    v.write_text(verilog, newline="\n")
    _run(yosys, svg.parent, "yosys", YOSYS_MISSING)
    _run(netlistsvg, svg.parent, "netlistsvg", NETLISTSVG_MISSING)
    cells: dict[str, int] = {}
    for c in json.loads(js.read_text())["modules"][top]["cells"].values():
        cells[c["type"]] = cells.get(c["type"], 0) + 1
    k = [o["index"] for o in block["outputs"]].index(row)
    return {"row": row, "module": top, **g.rows[k], "gates": sum(cells.values()), "gate_types": cells,
            "verilog": str(v), "json": str(js), "svg": str(svg)}
