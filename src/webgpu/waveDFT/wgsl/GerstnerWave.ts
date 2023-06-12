const gerstnerWaveComp:string = `

    // 配置 参数
    struct configParams {
        u_rowCnt: f32,
        u_colCnt: f32,
        u_time: f32,
    }

    // 波1
    struct Wave1Params {
        u_wavenumber: f32,         // 波数 
        c_amplitude: f32,          // 波幅
        u_amplitude_scale: f32,    // 波幅调整比例
        u_frequency: f32,          // 频率
        c_angular_velocity: f32,   // 角速度
        u_direct_x: f32,           // 波的传播方向    
        u_direct_z: f32,           // 波的传播方向  
        u_animation: f32,          // 是否动画
    }

    // 波2
    struct Wave2Params {
        u_wavenumber: f32,         // 波数 
        c_amplitude: f32,          // 波幅
        u_amplitude_scale: f32,    // 波幅调整比例
        u_frequency: f32,          // 频率
        c_angular_velocity: f32,   // 角速度
        u_direct_x: f32,           // 波的传播方向    
        u_direct_z: f32,           // 波的传播方向  
        u_animation: f32,          // 是否动画
    }

    // 波3
    struct Wave3Params {
        u_wavenumber: f32,         // 波数 
        c_amplitude: f32,          // 波幅
        u_amplitude_scale: f32,    // 波幅调整比例
        u_frequency: f32,          // 频率
        c_angular_velocity: f32,   // 角速度
        u_direct_x: f32,           // 波的传播方向    
        u_direct_z: f32,           // 波的传播方向  
        u_animation: f32,          // 是否动画
    }

    // 波4
    struct Wave4Params {
        u_wavenumber: f32,         // 波数 
        c_amplitude: f32,          // 波幅
        u_amplitude_scale: f32,    // 波幅调整比例
        u_frequency: f32,          // 频率
        c_angular_velocity: f32,   // 角速度
        u_direct_x: f32,           // 波的传播方向    
        u_direct_z: f32,           // 波的传播方向  
        u_animation: f32,          // 是否动画
    }


    // 因为 compute shader 里面 传 vec3 也是占用 vec4 的空间
    struct vert {
        pos : vec3<f32>,
    }

    struct MeshGrid {
        positions : array<vert>,
    }
    
    @group(0) @binding(0)  var<uniform> param : configParams;
    @group(0) @binding(1)  var<uniform> wave1 : Wave1Params;
    @group(0) @binding(2)  var<uniform> wave2 : Wave2Params;
    @group(0) @binding(3)  var<uniform> wave3 : Wave3Params;
    @group(0) @binding(4)  var<uniform> wave4 : Wave4Params;
    @group(0) @binding(5)  var<storage, read> vertexBufferSrc : MeshGrid;
    @group(0) @binding(6)  var<storage, read_write> vertexBufferDst : MeshGrid;

    // 计算出来的
    @group(0) @binding(7) var displacementTex_c: texture_2d<f32>;

    @compute @workgroup_size(256, 1, 1)
    fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>) {

        var index = GlobalInvocationID.x;

        var pi = 3.14159265;

        // 波的方向
        var wave1Dir = normalize(vec2(wave1.u_direct_x, wave1.u_direct_z));
        var wave2Dir = normalize(vec2(wave2.u_direct_x, wave2.u_direct_z));
        var wave3Dir = normalize(vec2(wave3.u_direct_x, wave3.u_direct_z));
        var wave4Dir = normalize(vec2(wave4.u_direct_x, wave4.u_direct_z));

        // 当前顶点
        var vPos = vertexBufferSrc.positions[index].pos;

        // 当前顶点的高程
        var sinVal1 = sin(wave1.u_wavenumber * (wave1Dir.x * vPos.x + wave1Dir.y * vPos.z) + wave1.c_angular_velocity * param.u_time * wave1.u_animation);
        var sinVal2 = sin(wave2.u_wavenumber * (wave2Dir.x * vPos.x + wave2Dir.y * vPos.z) + wave2.c_angular_velocity * param.u_time * wave2.u_animation);
        var sinVal3 = sin(wave3.u_wavenumber * (wave3Dir.x * vPos.x + wave3Dir.y * vPos.z) + wave3.c_angular_velocity * param.u_time * wave3.u_animation);
        var sinVal4 = sin(wave4.u_wavenumber * (wave4Dir.x * vPos.x + wave4Dir.y * vPos.z) + wave4.c_angular_velocity * param.u_time * wave4.u_animation);

        // 波幅
        var amplitude1 = wave1.u_amplitude_scale * wave1.c_amplitude;
        var amplitude2 = wave2.u_amplitude_scale * wave2.c_amplitude;
        var amplitude3 = wave3.u_amplitude_scale * wave3.c_amplitude;
        var amplitude4 = wave4.u_amplitude_scale * wave4.c_amplitude;

        var y1:f32 =  amplitude1 * sinVal1;
        var y2:f32 =  amplitude2 * sinVal2;
        var y3:f32 =  amplitude3 * sinVal3;
        var y4:f32 =  amplitude4 * sinVal4;

        // 累计高程
        var y:f32 = y1 + y2 + y3 + y4;
        // var y:f32 = y1;

        // 当前顶点的 xz 方向的平移
        
        var cosVal1 = cos(wave1.u_wavenumber * (wave1Dir.x * vPos.x + wave1Dir.y * vPos.z) + wave1.c_angular_velocity * param.u_time * wave1.u_animation);
        var cosVal2 = cos(wave2.u_wavenumber * (wave2Dir.x * vPos.x + wave2Dir.y * vPos.z) + wave2.c_angular_velocity * param.u_time * wave2.u_animation);
        var cosVal3 = cos(wave3.u_wavenumber * (wave3Dir.x * vPos.x + wave3Dir.y * vPos.z) + wave3.c_angular_velocity * param.u_time * wave3.u_animation);
        var cosVal4 = cos(wave4.u_wavenumber * (wave4Dir.x * vPos.x + wave4Dir.y * vPos.z) + wave4.c_angular_velocity * param.u_time * wave4.u_animation);
        var xz1 = amplitude1 * cosVal1;
        var xz2 = amplitude2 * cosVal2;
        var xz3 = amplitude3 * cosVal3;
        var xz4 = amplitude4 * cosVal4;

        var offsetXZ = vec2(0.0, 0.0);
        var offsetXZ1 = xz1 * wave1Dir;
        var offsetXZ2 = xz2 * wave2Dir;
        var offsetXZ3 = xz3 * wave3Dir;
        var offsetXZ4 = xz4 * wave4Dir;
        offsetXZ = offsetXZ1 + offsetXZ2 + offsetXZ3 + offsetXZ4;
        // offsetXZ = offsetXZ1;

        // var diffV3 = vec3(0.0, y, 0.0);

        // 求贴图坐标
        var tex = vec2(vPos.x / 20.0, vPos.z / 20.0);

        var uvCoord = vec2<i32>(floor(fract(tex) * 256.0));
        var disp = textureLoad(displacementTex_c, uvCoord, 0).xyz;


        var diffV3 = vec3(offsetXZ.x, y, offsetXZ.y);

        var newVPos = vPos + disp;

        // 写到结果 bufferB
        vertexBufferDst.positions[index].pos = vPos;
        // vertexBufferDst.positions[index].pos = newVPos;
    }  
`;

export {
    gerstnerWaveComp,
}