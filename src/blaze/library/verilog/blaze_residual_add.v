// blaze library template: residual add, y[j] = a[j] + b[j]. Exact, no rounding.
// a, b: N signed 8-bit; y: N signed 9-bit. Element j of a port of width W sits at [W*j +: W].
`default_nettype none
module blaze_residual_add #(
    parameter N = 1
) (
    input  wire [8*N-1:0] a,
    input  wire [8*N-1:0] b,
    output wire [9*N-1:0] y
);
    genvar j;
    generate
        for (j = 0; j < N; j = j + 1) begin : g
            wire signed [8:0] aj = $signed(a[8*j +: 8]);
            wire signed [8:0] bj = $signed(b[8*j +: 8]);
            assign y[9*j +: 9] = aj + bj;
        end
    endgenerate
endmodule
`default_nettype wire
