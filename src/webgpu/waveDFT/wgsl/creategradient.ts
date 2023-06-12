const createGradientComp:string = `

    @group(0) @binding(0) var displacement_tex : texture_2d<f32>;
    @group(0) @binding(1) var gradient_Tex : texture_storage_2d<rgba32float, write>;

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

    @compute @workgroup_size(16, 16, 1)
    fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>, @builtin(workgroup_id) group_id : vec3<u32>, @builtin(local_invocation_id) local_id : vec3<u32>)
    {
        var DISP_MAP_SIZE:u32 = 256;
        var PATCH_SIZE:u32 = 10;
        var TWO_PI:f32 = 6.283185307;
        var INV_TILE_SIZE = f32(DISP_MAP_SIZE) / f32(PATCH_SIZE);
        var TILE_SIZE_X2 = f32(PATCH_SIZE) * 2.0 / f32(DISP_MAP_SIZE);

        // 求 compute 触发索引
        var loc = vec2<u32>(GlobalInvocationID.x, GlobalInvocationID.y);

        // gradient
        var left = loc - vec2<u32>(1, 0);
            left.x = left.x & (DISP_MAP_SIZE - 1);
            left.y = left.y & (DISP_MAP_SIZE - 1);

        var right = loc + vec2<u32>(1, 0);
            right.x = right.x & (DISP_MAP_SIZE - 1);
            right.y = right.y & (DISP_MAP_SIZE - 1);

        var bottom = loc - vec2<u32>(0, 1);
            bottom.x = bottom.x & (DISP_MAP_SIZE - 1);
            bottom.y = bottom.y & (DISP_MAP_SIZE - 1);

        var top	= loc + vec2<u32>(0, 1);
            top.x = top.x & (DISP_MAP_SIZE - 1);
            top.y = top.y & (DISP_MAP_SIZE - 1);

        var disp_left = textureLoad(displacement_tex, left, 0).xyz;
        var disp_right	= textureLoad(displacement_tex, right, 0).xyz;
        var disp_bottom	= textureLoad(displacement_tex, bottom, 0).xyz;
        var disp_top = textureLoad(displacement_tex, top, 0).xyz;

        var gradient = vec2<f32>(disp_left.y - disp_right.y, disp_bottom.y - disp_top.y);

        // Jacobian
	    var dDx = (disp_right.xz - disp_left.xz) * INV_TILE_SIZE;
	    var dDy = (disp_top.xz - disp_bottom.xz) * INV_TILE_SIZE;

        var J = (1.0 + dDx.x) * (1.0 + dDy.y) - dDx.y * dDy.x;

        // 写数据
        textureStore(gradient_Tex, loc, vec4<f32>(gradient, TILE_SIZE_X2, J));
    }
`;

export {
    createGradientComp,
}