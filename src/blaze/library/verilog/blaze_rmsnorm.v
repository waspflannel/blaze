// blaze library template: RMSNorm, y[j] = trunc(x[j] * G[j] * 2^(K+FO) / D), D = isqrt(sum(x^2) << 2K).
// D ~= sqrt(sum x^2) * 2^K, so y ~= x / rms(x) * G / sqrt(N) * 2^FO; 1/sqrt(N) is folded into the
// output scale. x: N signed 8-bit; G: N signed 8-bit gamma; y: N signed OW-bit. An all-zero x gives 0.
`default_nettype none
module blaze_rmsnorm #(
    parameter N = 1,
    parameter K = 8,            // fractional bits of D
    parameter FO = 8,           // output fractional bits
    parameter SW = 15,          // width of sum(x^2)
    parameter AW = 32,          // isqrt argument width: even, >= SW + 2K
    parameter OW = 16,          // output width
    parameter [8*N-1:0] G = 0   // gamma, G[8*j +: 8]
) (
    input  wire [8*N-1:0] x,
    output wire [OW*N-1:0] y
);
    localparam DW = AW / 2;
    localparam NW = 16 + K + FO;

    // Exact floor(sqrt(v)), one result bit per iteration.
    function [DW-1:0] isqrt(input [AW-1:0] v);
        reg [AW-1:0] rem, root, b;
        integer k;
        begin
            rem = v;
            root = 0;
            b = 0;
            b[AW-2] = 1'b1;
            for (k = 0; k < DW; k = k + 1) begin
                if (rem >= root + b) begin
                    rem = rem - (root + b);
                    root = (root >> 1) + b;
                end else begin
                    root = root >> 1;
                end
                b = b >> 2;
            end
            isqrt = root[DW-1:0];
        end
    endfunction

    wire [16*N-1:0] sq;
    genvar j;
    generate
        for (j = 0; j < N; j = j + 1) begin : g_sq
            wire signed [15:0] xj = $signed(x[8*j +: 8]);
            assign sq[16*j +: 16] = xj * xj;
        end
    endgenerate

    reg [SW-1:0] s;
    integer i;
    always @* begin
        s = 0;
        for (i = 0; i < N; i = i + 1) s = s + sq[16*i +: 16];
    end

    wire [AW-1:0] arg = {{(AW-SW){1'b0}}, s} << (2*K);
    wire [DW-1:0] d = isqrt(arg);
    wire signed [DW:0] den = $signed({1'b0, d});

    generate
        for (j = 0; j < N; j = j + 1) begin : g_out
            wire signed [7:0] xj = x[8*j +: 8];
            wire signed [7:0] gj = G[8*j +: 8];
            wire signed [NW-1:0] num = (xj * gj) <<< (K + FO);
            wire signed [NW-1:0] quo = num / den;  // Verilog signed division truncates toward zero
            assign y[OW*j +: OW] = (d == 0) ? {OW{1'b0}} : quo[OW-1:0];
        end
    endgenerate
endmodule
`default_nettype wire
