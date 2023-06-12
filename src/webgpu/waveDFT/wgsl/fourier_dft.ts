const fourierDFTComp:string = `

    /***************************************************************************
     *  分两个 pass:  
     *  pass1: read_tex: heightfieldTex_c => write_tex: fourier_tex_pass1
     *  pass1: read_tex: fourier_tex_pass1 => write_tex: fourier_tex_pass2
     * 
     ****************************************************************************/

    @group(0) @binding(0) var read_tex : texture_2d<f32>;
    @group(0) @binding(1) var write_tex : texture_storage_2d<rgba32float, write>;

    // 复数相乘
    fn complexMult(a: vec2<f32>, b: vec2<f32>) -> vec2<f32>
    {
        return vec2<f32>(a.r * b.r - a.g * b.g, a.r * b.g + a.g * b.r);
    }

    // e 的复数次方 a = (x + iy)
    fn complexExp(a: vec2<f32>) -> vec2<f32>
    {
        return vec2<f32>(cos(a.y), sin(a.y)) * exp(a.x);
    }

    var<workgroup> value256: array<vec2<f32>,256>;

    @compute @workgroup_size(1, 256, 1)
    fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>, @builtin(workgroup_id) group_id : vec3<u32>, @builtin(local_invocation_id) local_id : vec3<u32>)
    {   
        var DISP_MAP_SIZE:u32 = 256;
        var TWO_PI:f32 = 6.283185307;
        var N = GlobalInvocationID.x;
        var M = GlobalInvocationID.y;

        // 第一步 先读取数据 一列 256 个数据
        value256[M]= textureLoad(read_tex, vec2<u32>(N, M), 0).rg;
        workgroupBarrier();

        // 第二步 DFT
        var result = vec2<f32>(0.0);
        for (var k:u32 = 0; k < DISP_MAP_SIZE; k++) {

            var coeff = value256[k];
    
            // X_x, X_z (0 ~ 255)
            var theta = (f32(M) * TWO_PI * f32(k)) / f32(DISP_MAP_SIZE);
            var cos_t = cos(theta);
            var sin_t = sin(theta);
    
            result.x += coeff.x * cos_t - coeff.y * sin_t;
            result.y += coeff.y * cos_t + coeff.x * sin_t; 

        }

        // 写数据(注意行和列切换了)
        textureStore(write_tex, vec2<u32>(M, N), vec4<f32>(result, 0.0, 1.0));
    }
`;

export {
    fourierDFTComp,
}