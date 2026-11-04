// blaze library template: RoPE rotation of adjacent pairs (x[2p], x[2p+1]) by the angle for
// frequency f = p mod (HS/2). cs holds that position's cos/sin row: cos_f at cs[CW*2f +: CW],
// sin_f at cs[CW*(2f+1) +: CW], signed with 14 fractional bits. Exact products, so
// y: N signed (8+CW)-bit. x: N signed 8-bit.
`default_nettype none
module blaze_rope #(
    parameter N = 2,   // even
    parameter HS = 2,  // head size, even
    parameter CW = 16
) (
    input  wire [8*N-1:0] x,
    input  wire [CW*HS-1:0] cs,
    output wire [(8+CW)*N-1:0] y
);
    localparam OW = 8 + CW;
    genvar p;
    generate
        for (p = 0; p < N / 2; p = p + 1) begin : g
            localparam F = p % (HS / 2);
            wire signed [OW-1:0] x0 = $signed(x[16*p +: 8]);
            wire signed [OW-1:0] x1 = $signed(x[16*p+8 +: 8]);
            wire signed [OW-1:0] c = $signed(cs[CW*(2*F) +: CW]);
            wire signed [OW-1:0] s = $signed(cs[CW*(2*F+1) +: CW]);
            assign y[OW*(2*p) +: OW] = x0 * c - x1 * s;
            assign y[OW*(2*p+1) +: OW] = x0 * s + x1 * c;
        end
    endgenerate
endmodule
`default_nettype wire
