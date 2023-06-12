const createDisplacementComp:string = `

    /***************************************************************************
     *  
     *  heightfield_tex: fourier_h_tex_pass2
     *  choppyfield_tex: fourier_choppy_tex_pass2
     * 
     ****************************************************************************/

    @group(0) @binding(0) var heightfield_tex : texture_2d<f32>;
    @group(0) @binding(1) var choppyfield_tex : texture_2d<f32>;
    @group(0) @binding(2) var displacement_tex : texture_storage_2d<rgba32float, write>;

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

    @compute @workgroup_size(16, 16, 1)
    fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>, @builtin(workgroup_id) group_id : vec3<u32>, @builtin(local_invocation_id) local_id : vec3<u32>)
    {
        var lambda = 1.3;

        // 求 compute 触发索引
        var loc = vec2<u32>(GlobalInvocationID.x, GlobalInvocationID.y);

        var sign_correction = 1.0;

        if( ( (loc.x + loc.y) & 1) == 1)
        {
            sign_correction = -1.0;
        }

        // 读取数据
        var heightfield = textureLoad(heightfield_tex, loc, 0).x;
        var choppyfield = textureLoad(choppyfield_tex, loc, 0).xy;

        var h = sign_correction * heightfield;
	    var D = sign_correction * choppyfield;

        // 写数据
        textureStore(displacement_tex, loc, vec4<f32>(D.x * lambda, h, D.y * lambda, 1.0));
    }
`;

export {
    createDisplacementComp,
}