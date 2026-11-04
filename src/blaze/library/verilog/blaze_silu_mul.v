// blaze library template: gated SiLU, y[j] = silu(a[j]) * b[j].
// a (gate, w1 x), b (up, w3 x): N signed 8-bit. silu(a) is a 256-entry lookup indexed by a's
// raw bit pattern, signed SW-bit; the product with b is exact, so y is signed (SW+8)-bit.
`default_nettype none
module blaze_silu_mul #(
    parameter N = 1,
    parameter SW = 12,
    parameter [256*SW-1:0] LUT = 0  // LUT[SW*u +: SW] = silu value for input bit pattern u
) (
    input  wire [8*N-1:0] a,
    input  wire [8*N-1:0] b,
    output wire [(SW+8)*N-1:0] y
);
    genvar j;
    generate
        for (j = 0; j < N; j = j + 1) begin : g
            wire [7:0] u = a[8*j +: 8];
            wire signed [SW+7:0] s = $signed(LUT[SW*u +: SW]);
            wire signed [SW+7:0] bj = $signed(b[8*j +: 8]);
            assign y[(SW+8)*j +: SW+8] = s * bj;
        end
    endgenerate
endmodule
`default_nettype wire
