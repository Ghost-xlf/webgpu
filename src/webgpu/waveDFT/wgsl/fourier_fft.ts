const fourierFFTComp:string = `

    /***************************************************************************
     *  bitfieldReverseTex: 输入的 index 二进制倒排 
     *  分两个 pass:  
     *  pass1: read_tex: heightfieldTex_c => write_tex: fourier_tex_pass1
     *  pass1: read_tex: fourier_tex_pass1 => write_tex: fourier_tex_pass2
     * 
     ****************************************************************************/

    @group(0) @binding(0) var bitfieldReverseTex : texture_2d<f32>;
    @group(0) @binding(1) var read_tex : texture_2d<f32>;
    @group(0) @binding(2) var write_tex : texture_storage_2d<rgba32float, write>;

    // 复数相乘
    fn complexMult(a: vec2<f32>, b: vec2<f32>) -> vec2<f32>
    {
        return vec2<f32>(a.r * b.r - a.g * b.g, a.r * b.g + a.g * b.r);
    }

    var<workgroup> ping256: array<vec2<f32>,256>;
    var<workgroup> pong256: array<vec2<f32>,256>;

    @compute @workgroup_size(1, 256, 1)
    fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>, @builtin(workgroup_id) group_id : vec3<u32>, @builtin(local_invocation_id) local_id : vec3<u32>)
    {   
        // 预定义值
        var DISP_MAP_SIZE:u32 = 256;
        var LOG2_DISP_MAP_SIZE:u32 = 8;
        var TWO_PI:f32 = 6.283185307;

        // 当前 compute 运行的 workunit
        var N = GlobalInvocationID.x; // z
        var M = GlobalInvocationID.y; // x

        // 第一步 先读取数据 一列 256 个数据
        // FFT 下标索引
        var index:u32 = u32(textureLoad(bitfieldReverseTex, vec2<u32>(M, 0), 0).r);
        ping256[index] = textureLoad(read_tex, vec2<u32>(N, M), 0).rg;

        workgroupBarrier();

        // 第二步 FFT
        var src:i32 = 0;

        for (var s:u32 = 1; s <= LOG2_DISP_MAP_SIZE; s++) {

            var m:u32 = u32(1 << s);		    // butterfly 的高度
		    var mh:u32 = u32(m >> 1);			// butterfly 的半高
    
            var k:u32 = u32( f32(M) * (f32(DISP_MAP_SIZE) / f32(m) )) & (DISP_MAP_SIZE - 1);
            var i:u32 = (M & ~(m - 1));		// butterfly group starting offset
		    var j:u32 = (M & (mh - 1));		// butterfly index in group

            // twiddle factor W_N^k
		    var theta:f32 = (TWO_PI * f32(k)) / f32(DISP_MAP_SIZE);
		    var W_N_k:vec2<f32> = vec2(cos(theta), sin(theta));

            var input1:vec2<f32>;
            var input2:vec2<f32>;

            if(src > 0)
            {
                input1 = pong256[i + j + mh];
		        input2 = pong256[i + j];

                ping256[M] = input2 + complexMult(W_N_k, input1);
            }
            else{

                input1 = ping256[i + j + mh];
		        input2 = ping256[i + j];

                pong256[M] = input2 + complexMult(W_N_k, input1);
            }
            
            src = 1 - src;

            workgroupBarrier();
        }

        // 第三步 写数据(注意行和列切换了)
        var result:vec2<f32>;
        if(src > 0){
            result = pong256[M];
        }
        else{
            result = ping256[M];
        }

        textureStore(write_tex, vec2<u32>(M, N), vec4<f32>(result, 0.0, 1.0));
    }
`;

export {
    fourierFFTComp,
}