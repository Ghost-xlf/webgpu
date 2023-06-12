const updatespectrumComp:string = `

    struct spectrumParams {
        u_time : f32
    };

    @group(0) @binding(0) var<uniform> params : spectrumParams;
    @group(0) @binding(1) var tilde_h0_tex : texture_2d<f32>;
    @group(0) @binding(2) var frequence_tex : texture_2d<f32>;
    @group(0) @binding(3) var heightfield_tex : texture_storage_2d<rgba32float, write>;
    @group(0) @binding(4) var choppyfield_tex : texture_storage_2d<rgba32float, write>;

    // 复数相乘
    fn complexMult(a: vec2<f32>, b: vec2<f32>) -> vec2<f32>
    {
        return vec2<f32>(a.r * b.r - a.g * b.g, a.r * b.g + a.g * b.r);
    }

    // e 的复数次方 a= (x + iy)
    fn complexExp(a: vec2<f32>) -> vec2<f32>
    {
        return vec2<f32>(cos(a.y), sin(a.y)) * exp(a.x);
    }

    // var<workgroup> odds: array<vec2<f32>,512>;
    // var<workgroup> evens_0: array<i32,16>;

    @compute @workgroup_size(16, 16, 1)
    fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>, @builtin(workgroup_id) group_id : vec3<u32>, @builtin(local_invocation_id) local_id : vec3<u32>)
    {
        var DISP_MAP_SIZE:u32 = 256;
        var HALF_DISP_MAP_SIZE:u32 = 128;
        var time = params.u_time;

        // workgroupBarrier();
        // storageBarrier();

        // 求 compute 触发索引
        var loc1 = vec2<u32>(GlobalInvocationID.x, GlobalInvocationID.y);
        var loc2 = vec2<u32>(DISP_MAP_SIZE - loc1.x, DISP_MAP_SIZE - loc1.y);

        var h_tk: vec2<f32>;
	    var h0_k	= textureLoad(tilde_h0_tex, loc1.xy, 0).rg;
	    var h0_mk	= textureLoad(tilde_h0_tex, loc2.xy, 0).rg;   // 共轭位置(-k)
	    var w_k	= textureLoad(frequence_tex, loc1.xy, 0).r;
        
        // 读取数据
        let tilde_h0 = textureLoad(tilde_h0_tex, loc1, 0);
        let frequence = textureLoad(frequence_tex, loc1, 0);

        // 欧拉公式: e^{ix} = cos x + isin x
        var cos_wt = cos(w_k * params.u_time);
        var sin_wt = sin(w_k * params.u_time);

        // height 高度场 波普 h(k, t) = h(k) * e(iω(k)t) + h(-k) * e(-iω(k)t)
        h_tk.x = cos_wt * (h0_k.x + h0_mk.x) - sin_wt * (h0_k.y + h0_mk.y);
        h_tk.y = cos_wt * (h0_k.y - h0_mk.y) + sin_wt * (h0_k.x - h0_mk.x);

        //  choppy 波普

        var k:vec2<f32>;

	    k.x = f32(HALF_DISP_MAP_SIZE - loc1.x);
	    k.y = f32(HALF_DISP_MAP_SIZE - loc1.y);

        var kn2 = dot(k, k);
        var nk = vec2(0.0, 0.0);

        if (kn2 > 1e-12){
            nk = normalize(k);
        }

        // take advantage of DFT's linearity
	    var Dt_x = vec2(h_tk.y * nk.x, -h_tk.x * nk.x);
	    var iDt_z = vec2(h_tk.x * nk.y, h_tk.y * nk.y);

        // 提取时间
        let colorH = vec3<f32>(tilde_h0.rg, 0.0);
        // let colorH = vec3<f32>(sin(params.u_time), cos(params.u_time), 0.0);
        let colorFreq = vec3<f32>(frequence.r, 0.0, 0.0);

        // 写数据
        textureStore(heightfield_tex, loc1.xy, vec4<f32>(h_tk, 0.0, 1.0));
        textureStore(choppyfield_tex, loc1.xy, vec4<f32>(Dt_x + iDt_z, 0.0,  1.0));
    }

`;

export {
    updatespectrumComp,
}