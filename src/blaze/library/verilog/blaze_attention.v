// blaze library template: causal multi-head attention for one decode step, with this layer's KV cache.
// rst: back to idle. start (one cycle): writes k, v into cache row pos. Then three passes over t = 0..pos, one cache row
// per R1 / R2 / R3 cycles (>= 1; a row's registers load on its last cycle), all heads in parallel (head h reads kv head
// h / (NH / NKV)):
//   1. score[t] = sat_SB(round(q_h . k[t] * SM / 2^SS)) (4 fractional bits), max over t      (R1 cycles per row)
//   2. e[t] = LUT[min(max - score[t], 255)], sum over t      (LUT[k] ~= exp(-k / 16) * 2^FP)  (R2)
//   3. p[t] = floor(e[t] * 2^FP / sum), acc += p[t] * v[t]                                  (R3)
// done is high 1 + (R1 + R2 + R3) * (pos + 1) cycles after the start cycle (the cycle start is high counts as 0), for
// one cycle; y = sat8(round(acc * OM / 2^OS)) holds until the next start.
// q, y: NH*HS signed 8-bit; k, v: NKV*HS signed 8-bit. q, k, v and pos must hold steady until done.
`default_nettype none
module blaze_attention #(
    parameter NH = 1,
    parameter NKV = 1,
    parameter HS = 2,
    parameter SEQ = 2,
    parameter PB = 1,               // pos width
    parameter DW = 20,              // q.k dot product width
    parameter SB = 16,              // score width
    parameter [31:0] SM = 1,
    parameter SS = 1,
    parameter EW = 16,              // exp entry width, unsigned
    parameter FP = 15,              // probability fractional bits
    parameter SUMW = 25,            // width of sum(e)
    parameter [256*EW-1:0] LUT = 0,
    parameter AW = 32,              // p.v accumulator width
    parameter [31:0] OM = 1,
    parameter OS = 1,
    parameter R1 = 1,               // cycles per cache row in pass 1, 2, 3
    parameter R2 = 1,
    parameter R3 = 1
) (
    input  wire clk,
    input  wire rst,
    input  wire start,
    input  wire [PB-1:0] pos,
    input  wire [8*NH*HS-1:0] q,
    input  wire [8*NKV*HS-1:0] k,
    input  wire [8*NKV*HS-1:0] v,
    output wire [8*NH*HS-1:0] y,
    output reg done
);
    localparam DIM = NH * HS;
    localparam KVD = NKV * HS;
    localparam KVM = NH / NKV;
    localparam PRW = EW + FP;       // p <= 2^FP fits easily
    localparam RMAX = (R1 > R2 ? R1 : R2) > R3 ? (R1 > R2 ? R1 : R2) : R3;
    localparam HW = $clog2(RMAX + 1);
    localparam [HW-1:0] L1 = R1 - 1;
    localparam [HW-1:0] L2 = R2 - 1;
    localparam [HW-1:0] L3 = R3 - 1;

    reg [8*KVD-1:0] kc [0:SEQ-1];
    reg [8*KVD-1:0] vc [0:SEQ-1];
    reg [SB*NH-1:0] sc [0:SEQ-1];
    reg [1:0] ph;                   // 0 idle, 1 scores, 2 exp sum, 3 accumulate
    reg [PB-1:0] t;
    reg [HW-1:0] hc;                // cycle within the current row
    wire last = (ph == 2'd1) ? (hc == L1) : (ph == 2'd2) ? (hc == L2) : (hc == L3);
    reg [SB*NH-1:0] mx;
    reg [SUMW*NH-1:0] tot;
    reg [AW*DIM-1:0] acc;

    wire [8*KVD-1:0] krow = kc[t];
    wire [8*KVD-1:0] vrow = vc[t];
    wire [SB*NH-1:0] srow = sc[t];
    wire [DW*NH-1:0] dot;
    wire [SB*NH-1:0] s;
    wire [EW*NH-1:0] e;
    wire [AW*DIM-1:0] acc_next;

    blaze_requant #(.N(NH), .IW(DW), .OB(SB), .M(SM), .S(SS)) score_rq (.x(dot), .y(s));
    blaze_requant #(.N(DIM), .IW(AW), .OB(8), .M(OM), .S(OS)) out_rq (.x(acc), .y(y));

    genvar h, i;
    generate
        for (h = 0; h < NH; h = h + 1) begin : g_head
            localparam KH = h / KVM;
            reg signed [DW-1:0] d;
            integer j;
            always @* begin
                d = 0;
                for (j = 0; j < HS; j = j + 1)
                    d = d + $signed(q[8*(h*HS+j) +: 8]) * $signed(krow[8*(KH*HS+j) +: 8]);
            end
            assign dot[DW*h +: DW] = d;

            wire signed [SB:0] diff = $signed(mx[SB*h +: SB]) - $signed(srow[SB*h +: SB]);
            wire [7:0] idx = (diff > 255) ? 8'd255 : diff[7:0];
            assign e[EW*h +: EW] = LUT[EW*idx +: EW];
            wire [PRW-1:0] p = {e[EW*h +: EW], {FP{1'b0}}} / tot[SUMW*h +: SUMW];

            for (i = 0; i < HS; i = i + 1) begin : g_el
                wire signed [AW-1:0] pv = $signed({1'b0, p}) * $signed(vrow[8*(KH*HS+i) +: 8]);
                wire signed [AW-1:0] a0 = (t == 0) ? {AW{1'b0}} : acc[AW*(h*HS+i) +: AW];
                assign acc_next[AW*(h*HS+i) +: AW] = a0 + pv;
            end
        end
    endgenerate

    integer hh;
    always @(posedge clk) begin
        done <= 1'b0;
        if (rst) begin
            ph <= 2'd0;
            t <= 0;
            hc <= 0;
        end else if (start) begin
            kc[pos] <= k;
            vc[pos] <= v;
            t <= 0;
            hc <= 0;
            ph <= 2'd1;
        end else if (ph != 2'd0) begin
            if (!last) begin
                hc <= hc + 1'b1;
            end else begin
                hc <= 0;
                if (ph == 2'd1) begin
                    sc[t] <= s;
                    for (hh = 0; hh < NH; hh = hh + 1)
                        if (t == 0 || $signed(s[SB*hh +: SB]) > $signed(mx[SB*hh +: SB]))
                            mx[SB*hh +: SB] <= s[SB*hh +: SB];
                end else if (ph == 2'd2) begin
                    for (hh = 0; hh < NH; hh = hh + 1)
                        tot[SUMW*hh +: SUMW] <= (t == 0 ? {SUMW{1'b0}} : tot[SUMW*hh +: SUMW]) + {{(SUMW-EW){1'b0}}, e[EW*hh +: EW]};
                end else begin
                    acc <= acc_next;
                end
                if (t == pos) begin
                    t <= 0;
                    if (ph == 2'd3) begin
                        ph <= 2'd0;
                        done <= 1'b1;
                    end else begin
                        ph <= ph + 2'd1;
                    end
                end else begin
                    t <= t + 1'b1;
                end
            end
        end
    end
endmodule
`default_nettype wire
