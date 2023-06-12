
const oceanVert:string = `

    struct Uniforms {
        projectiveMatrix : mat4x4<f32>,
        viewMatrix : mat4x4<f32>
    }

    // vec3 占用 vec4 的位置
    struct oceanParams {
        u_choppyDxScale: f32,
        u_choppyDzScale : f32,
        u_yScale: f32,
        u_foamScale : f32,

        u_foamStatus : f32,
        u_patch_size: f32,
        u_disp_map_size: f32,
        u_xxx:f32,

        u_sunColor : vec3<f32>,
        u_sunDir: vec3<f32>,
        u_oceanColor:  vec3<f32>,
        u_cameraPosition: vec3<f32>
    }

    @group(0) @binding(0) var<uniform> uniforms : Uniforms;
    @group(0) @binding(1) var<uniform> params : oceanParams;
    @group(0) @binding(2) var perlinTexSampler: sampler;
    @group(0) @binding(3) var perlinTex: texture_2d<f32>;

    // 计算出来的
    @group(0) @binding(6) var displacementTex_c: texture_2d<f32>;
    @group(0) @binding(7) var gradientTex_c: texture_2d<f32>;

    struct VertexOutput {
        @builtin(position) position : vec4<f32>,
        @location(0) tex : vec2<f32>,
        @location(1) vdir : vec3<f32>,
    }

    @vertex
    fn vert_main(
        @location(0) a_position : vec4<f32>
    ) -> VertexOutput {

        var BLEND_START = 8.0;		// m
        var BLEND_END = 200.0;		// m
        var perlinFrequency	= vec3(1.12, 0.59, 0.23);
        var perlinAmplitude	= vec3(0.35, 0.42, 0.57);

        var output : VertexOutput;
        var tex = vec2(a_position.x / params.u_patch_size, a_position.z / params.u_patch_size);
        output.tex = tex;

        // 取位移像素
        var uvCoord = vec2<i32>(fract(tex) * 256.0);
        var disp = textureLoad(displacementTex_c, uvCoord, 0).xyz;
        
        // 求世界坐标
        var pos_local = vec4(a_position.xyz, 1.0);

        // 求视线方向
        var vdir = params.u_cameraPosition - pos_local.xyz;
        output.vdir = vdir;

        // 到眼睛的距离
        var dist = length(vdir.xz);

        var factor = clamp((BLEND_END - dist) / (BLEND_END - BLEND_START), 0.0, 1.0);
	    var perl = 0.0;

        if (factor < 1.0) {

            var ptex = fract(tex);
            
            var uvCoordP0 = vec2<i32>(floor(ptex * perlinFrequency.x * 256.0));
            var uvCoordP1 = vec2<i32>(floor(ptex * perlinFrequency.y * 256.0));
            var uvCoordP2 = vec2<i32>(floor(ptex * perlinFrequency.z * 256.0));

            var p0 = textureLoad(perlinTex, uvCoordP0, 0).a;
            var p1 = textureLoad(perlinTex, uvCoordP1, 0).a;
            var p2 = textureLoad(perlinTex, uvCoordP2, 0).a;

            perl = dot(vec3(p0, p1, p2), perlinAmplitude);
        }

        output.position = uniforms.projectiveMatrix * uniforms.viewMatrix * vec4(pos_local.xyz + vec3(disp.x * params.u_choppyDxScale, disp.y * params.u_yScale, disp.z * params.u_choppyDzScale), 1.0);

        return output;
    }
`

const oceanFrag:string = `

    // vec3 占用 vec4 的位置
    struct oceanParams {
        u_choppyDxScale: f32,
        u_choppyDzScale : f32,
        u_yScale: f32,
        u_foamScale : f32,

        u_foamStatus : f32,
        u_patch_size: f32,
        u_disp_map_size: f32,
        u_xxx:f32,

        u_sunColor : vec3<f32>,
        u_sunDir: vec3<f32>,
        u_oceanColor:  vec3<f32>,
        u_cameraPosition: vec3<f32>
        
    }
    @group(0) @binding(1) var<uniform> params : oceanParams;
    @group(0) @binding(2) var perlinTexSampler: sampler;
    @group(0) @binding(3) var perlinTex: texture_2d<f32>;
    @group(0) @binding(4) var cubeTexSampler: sampler;
    @group(0) @binding(5) var cubeTex: texture_cube<f32>;

    // 计算出来的
    @group(0) @binding(6) var displacementTex_c: texture_2d<f32>;
    @group(0) @binding(7) var gradientTex_c: texture_2d<f32>;

    @fragment
    fn frag_main(
        @builtin(position) coord : vec4<f32>,
        @location(0) tex : vec2<f32>,
        @location(1) vdir : vec3<f32>,
    ) -> @location(0) vec4<f32> {

        var BLEND_START = 8.0;		// m
        var BLEND_END = 200.0;		// m
        var perlinFrequency	= vec3(1.12, 0.59, 0.23);
        var perlinGradient	= vec3(0.014, 0.016, 0.022);

        var ONE_OVER_4PI = 0.0795774715459476;

        var dist = length(vdir.xz);
	    var factor = (BLEND_END - dist) / (BLEND_END - BLEND_START);
	    factor = clamp(factor * factor * factor, 0.0, 1.0);

        var ptex = tex;
        var p0uv = ptex * perlinFrequency.x;
        var p1uv = ptex * perlinFrequency.y;
        var p2uv = ptex * perlinFrequency.z;

        var p0 = textureSample(perlinTex, perlinTexSampler, p0uv).rg;
        var p1 = textureSample(perlinTex, perlinTexSampler, p1uv).rg;
        var p2 = textureSample(perlinTex, perlinTexSampler, p2uv).rg;
        
        var perl = vec2(0.0);

        if (factor < 1.0) {
            perl = (p0 * perlinGradient.x);
        }

        // 根据梯度计算法向量等
	    // var grad = textureSample(gradientTex, gradientTexSampler, tex);

        var uvCoord = vec2<i32>(fract(tex) * 256.0);
        var grad = textureLoad(gradientTex_c, uvCoord, 0);
        var jacob = grad.w;

        var temp = mix(vec2(perl), vec2(grad.xy), factor);
		    grad = vec4(temp.xy, grad.zw);

        var n = normalize(grad.xzy);
        var v = normalize(vdir);
        var l = reflect(-v, n);

        // 计算 fresnel 效果
        var F0 = 0.020018673;
        var F = F0 + (1.0 - F0) * pow(1.0 - dot(n, l), 5.0);

        // 环境反射光
        var refl = textureSample(cubeTex, cubeTexSampler, l).rgb;

        // tweaked from ARM/Mali's sample
        var turbulence = max(1.6 - grad.w , 0.0);
        var color_mod = 1.0 + 3.0 * smoothstep(1.2, 1.8, turbulence);

        color_mod = mix(1.0, color_mod, factor);

        // 添加一些额外的高光 some additional specular (Ward model)
        var rho = 0.3;
        var ax = 0.2;
        var ay = 0.1;

        var h = params.u_sunDir + v;
        var x = cross(params.u_sunDir, n);
        var y = cross(x, n);

        var mult = (ONE_OVER_4PI * rho / (ax * ay * sqrt(max(1e-5, dot(params.u_sunDir, n) * dot(v, n)))));

        var hdotx = dot(h, x) / ax;
        var hdoty = dot(h, y) / ay;
        var hdotn = dot(h, n);

        var spec = mult * exp(-((hdotx * hdotx) + (hdoty * hdoty)) / (hdotn * hdotn));
        var finalColor = mix(params.u_oceanColor, refl * color_mod, F) + params.u_sunColor * spec;

        if(jacob < 0.6){

            return vec4(1.0, 1.0, 1.0, 1.0);
            // return vec4(finalColor, 1.0);
        }
        else{
            return vec4(finalColor, 1.0);
        }
    }
`
export {
    oceanVert,
    oceanFrag
}