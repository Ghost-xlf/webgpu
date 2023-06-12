
const gradientMapVert:string = `

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

const gradientMapFrag:string = `

    // 计算出来的梯度贴图
    @group(0) @binding(0) var gradientTex_c: texture_2d<f32>;

    @fragment
    fn frag_main(
        @builtin(position) coord : vec4<f32>,
        @location(0) tex : vec2<f32>,
    ) -> @location(0) vec4<f32> {

        var gradientColor = textureLoad(gradientTex_c, vec2<u32>(256.0 * tex), 0);

        // 从 -1 ~ 1 映射到 0 ~ 1
        gradientColor = (gradientColor + vec4(1.0)) * 0.5;

        return gradientColor;
    }
`
export {
    gradientMapVert,
    gradientMapFrag
}