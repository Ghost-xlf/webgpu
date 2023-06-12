
const displacementPrecisionMapVert:string = `

    struct VertexOutput {
        @builtin(position) position : vec4<f32>,
        @location(0) tex : vec2<f32>,
    }

    @vertex
    fn vert_main(
        @location(0) a_position : vec4<f32>,
        @location(1) a_uv : vec2<f32>
    ) -> VertexOutput {

        var output : VertexOutput;
        var tex = a_uv;

        output.tex = tex;
        output.position = vec4(a_position.xyz, 1.0);

        return output;
    }
`

const displacementPrecisionMapFrag:string = `

    // 计算出来的位移贴图
    @group(0) @binding(0) var displacementTex_c: texture_2d<f32>;

    @fragment
    fn frag_main(
        @builtin(position) coord : vec4<f32>,
        @location(0) tex : vec2<f32>,
    ) -> @location(0) vec4<f32> {

        var displacementColor = textureLoad(displacementTex_c, vec2<u32>(floor(256.0 * tex)), 0).rgb;

        // 从 -1 ~ 1 映射到 0 ~ 1
        displacementColor = (displacementColor + vec3(1.0)) * 0.5;
        var displacementPrecisionColor = fract(displacementColor.rgb * 255.0);

        return vec4(displacementPrecisionColor, 1.0);
    }
`
export {
    displacementPrecisionMapVert,
    displacementPrecisionMapFrag
}