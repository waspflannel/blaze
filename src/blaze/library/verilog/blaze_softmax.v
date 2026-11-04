// blaze library template: softmax, y[j] = floor(e[j] * 2^FP / sum(e)), e[j] = LUT[max(x) - x[j]].
// LUT[k] ~= exp(-k / 16) * 2^FP for k = 0..255. x: N signed 8-bit; y: N signed (FP+2)-bit, all >= 0.
`default_nettype none
module blaze_softmax #(
    parameter N = 1,
    parameter EW = 16,              // exp entry width, unsigned
    parameter FP = 15,              // output fractional bits
    parameter SUMW = 16,            // width of sum(e)
    parameter [256*EW-1:0] LUT = 0  // LUT[EW*k +: EW]
) (
    input  wire [8*N-1:0] x,
    output wire [(FP+2)*N-1:0] y
);
    reg signed [7:0] m;
    integer i;
    always @* begin
        m = x[7:0];
        for (i = 1; i < N; i = i + 1)
            if ($signed(x[8*i +: 8]) > m) m = x[8*i +: 8];
    end

    wire [EW*N-1:0] e;
    genvar j;
    generate
        for (j = 0; j < N; j = j + 1) begin : g_exp
            wire signed [8:0] d = m - $signed(x[8*j +: 8]);  // 0..255
            assign e[EW*j +: EW] = LUT[EW*d[7:0] +: EW];
        end
    endgenerate

    reg [SUMW-1:0] total;
    integer k;
    always @* begin
        total = 0;
        for (k = 0; k < N; k = k + 1) total = total + e[EW*k +: EW];
    end

    generate
        for (j = 0; j < N; j = j + 1) begin : g_div
            wire [EW+FP-1:0] num = {e[EW*j +: EW], {FP{1'b0}}};
            wire [EW+FP-1:0] p = num / total;  // total >= LUT[0] > 0
            assign y[(FP+2)*j +: FP+2] = p[FP+1:0];
        end
    endgenerate
endmodule
`default_nettype wire
