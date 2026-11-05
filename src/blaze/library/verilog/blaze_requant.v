// blaze library template: requantize, y[j] = sat_OB(floor((x[j] * M + 2^(S-1)) / 2^S)), i.e. x * M / 2^S
// rounded half up and saturated to signed OB bits. x: N signed IW-bit; y: N signed OB-bit.
// M < 2^17 and S >= 1; the product is computed wide enough that it cannot overflow.
`default_nettype none
module blaze_requant #(
    parameter N = 1,
    parameter IW = 8,
    parameter OB = 8,
    parameter [31:0] M = 1,
    parameter S = 1
) (
    input  wire [IW*N-1:0] x,
    output wire [OB*N-1:0] y
);
    localparam PW = (IW + 18 > S + 2) ? IW + 18 : S + 2;
    localparam [PW-1:0] ONE = 1;
    localparam signed [PW-1:0] RND = ONE << (S - 1);
    localparam signed [PW-1:0] MUL = M;
    localparam signed [PW-1:0] HI = (ONE << (OB - 1)) - ONE;
    localparam signed [PW-1:0] LO = -(ONE << (OB - 1));
    genvar j;
    generate
        for (j = 0; j < N; j = j + 1) begin : g
            wire signed [PW-1:0] xj = $signed(x[IW*j +: IW]);
            wire signed [PW-1:0] r = (xj * MUL + RND) >>> S;
            assign y[OB*j +: OB] = (r > HI) ? {1'b0, {(OB-1){1'b1}}} : (r < LO) ? {1'b1, {(OB-1){1'b0}}} : r[OB-1:0];
        end
    endgenerate
endmodule
`default_nettype wire
