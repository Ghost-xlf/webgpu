/**
    shader material  vertex shader 内置的变量
    // = object.matrixWorld
    uniform mat4 modelMatrix;

    // = camera.matrixWorldInverse * object.matrixWorld
    uniform mat4 modelViewMatrix;

    // = camera.projectionMatrix
    uniform mat4 projectionMatrix;

    // = camera.matrixWorldInverse
    uniform mat4 viewMatrix;

    // = inverse transpose of modelViewMatrix
    uniform mat3 normalMatrix;

    // = camera position in world space
    uniform vec3 cameraPosition;

    attribute vec3 position;
    attribute vec3 normal;
    attribute vec2 uv;

**/

const vertStr:string = `

    #define BLEND_START		8.0		    // m
    #define BLEND_END		200.0		// m

    uniform sampler2D displacementTex;              // 位移贴图
    uniform sampler2D displacementPrecisionTex;      // 位移贴图
    uniform sampler2D perlinTex;            // 柏林噪音贴图

    // uniform vec4 uvParams;
    uniform float u_patch_size;             // 20m, 也代表最大波长
    uniform float u_disp_map_size;

    uniform float u_choppyDxScale;          // 平移 x 方向比例
    uniform float u_choppyDzScale;          // 平移 z 方向比例
    uniform float u_yScale;                 // 垂直 y 方向比例

    out vec3 vdir;                          // 视线方向 viewDirection
    out vec2 tex;

    void main()
    {
        // fragment shader 里面也有
        const vec3 perlinFrequency	= vec3(1.12, 0.59, 0.23);
        const vec3 perlinAmplitude	= vec3(0.35, 0.42, 0.57);

        // 没 20 米距离贴一张 512*512 的位移贴图, 将贴图半个像素, 用像素中心对齐
        // tex = vec2(position.xz) / u_patch_size + vec2(0.5 / u_disp_map_size);
        tex = vec2(position.xz) / u_patch_size + vec2(0.5 / u_disp_map_size);

        // 取位移像素
        vec3 disp = texture(displacementTex, tex).xyz;
        vec3 dispPrecision = texture(displacementPrecisionTex, tex).xyz;

        // 0 ~ 1 映射到 -2 到 2 
        disp = (disp + dispPrecision * (1.0 / 255.0)) * 4.0 - 2.0;
        // disp = disp + dispPrecision;

        // 求世界坐标
        vec4 pos_local = modelMatrix * vec4(position, 1.0);

        // 求视线方向
        vdir = cameraPosition - pos_local.xyz;

        // 到眼睛的距离
        float dist = length(vdir.xz);

        float factor = clamp((BLEND_END - dist) / (BLEND_END - BLEND_START), 0.0, 1.0);
	    float perl = 0.0;

        if (factor < 1.0) {

            vec2 ptex = tex;
            
            float p0 = texture(perlinTex, ptex * perlinFrequency.x).a;
            float p1 = texture(perlinTex, ptex * perlinFrequency.y).a;
            float p2 = texture(perlinTex, ptex * perlinFrequency.z).a;

            perl = dot(vec3(p0, p1, p2), perlinAmplitude);
        }

        disp = mix(vec3(0.0, perl, 0.0), disp, factor);

        gl_Position = projectionMatrix * viewMatrix * modelMatrix * vec4(pos_local.xyz + vec3(disp.x * u_choppyDxScale, disp.y * u_yScale, disp.z * u_choppyDzScale) , 1.0);
    }  
`;


/**
    shader material  fragment shader 内置的变量
    uniform mat4 viewMatrix;
    uniform vec3 cameraPosition;
 */

const fragStr:string = `

    #define BLEND_START		8.0		    // m
    #define BLEND_END		200.0		// m
    #define ONE_OVER_4PI	0.0795774715459476

    const vec3 sundir = vec3(0.603, 0.240, -0.761);
    uniform sampler2D gradientTex;                  // 生成白泡沫的贴图
    uniform sampler2D gradientPrecisionTex;          // 生成白泡沫的贴图
    
    uniform sampler2D perlinTex;            // 柏林噪音贴图
    
    uniform samplerCube envmapTex;          // 环境贴图
    
    uniform vec3 u_sunColor;                // 太阳的颜色
    uniform vec3 u_sunDir;                  // 太阳的方向

    uniform vec4 uvParams;
    // uniform vec2 perlinOffset;
    uniform vec3 u_oceanColor;              // 海水的颜色

    uniform float u_foamScale;              // 白沫比例
    uniform float u_foamStatus;             // 白沫状态

    in vec3 vdir;       // 视线方向 viewDirection
    in vec2 tex;

    out vec4 FragColor;

    void main()
    {
        // uniform 保留
        float foamScale = u_foamScale;
        float foamStatus = u_foamStatus;
        
        const vec3 perlinFrequency	= vec3(1.12, 0.59, 0.23);
        const vec3 perlinGradient	= vec3(0.014, 0.016, 0.022);

        float dist = length(vdir.xz);
	    float factor = (BLEND_END - dist) / (BLEND_END - BLEND_START);
	    factor = clamp(factor * factor * factor, 0.0, 1.0);
        
        vec2 perl = vec2(0.0);

        if (factor < 1.0) {

            vec2 ptex = tex;
    
            vec2 p0 = texture(perlinTex, ptex * perlinFrequency.x).rg;
            vec2 p1 = texture(perlinTex, ptex * perlinFrequency.y).rg;
            vec2 p2 = texture(perlinTex, ptex * perlinFrequency.z).rg;
    
            perl = (p0 * perlinGradient.x + p1 * perlinGradient.y + p2 * perlinGradient.z);
        }

        // 根据梯度计算法向量等  
	    vec4 grad = texture(gradientTex, tex);
        vec4 gradPrecision = texture(gradientPrecisionTex, tex);
        grad = grad + gradPrecision * (1.0 / 255.0);

        // 0 ~ 1 映射到 -1 到 1
        grad = grad * 2.0 - 1.0;

		grad.xy = mix(perl, grad.xy, factor);

        // When the Jacobian factor is close to 0, the water is very "turbulent" and when larger than 1, the water mesh has been "stretched" out. A Jacobian of 1 is the "normal" state.
        // grad.w = grad.w * 0.2; 

        vec3 n = normalize(grad.xzy);
        vec3 v = normalize(vdir);
        vec3 l = reflect(-v, n);

        // 计算 fresnel 效果
        float F0 = 0.020018673;
        float F = F0 + (1.0 - F0) * pow(1.0 - dot(n, l), 5.0);

        // 环境反射光
        vec3 refl = texture(envmapTex, l).rgb;

        // tweaked from ARM/Mali's sample
        float turbulence = max(1.6 - grad.w, 0.0);
        float color_mod = 1.0 + 3.0 * smoothstep(1.2, 1.8, turbulence);

        color_mod = mix(1.0, color_mod, factor);

        // 添加一些额外的高光 some additional specular (Ward model)
        const float rho = 0.3;
        const float ax = 0.2;
        const float ay = 0.1;

        vec3 h = u_sunDir + v;
        vec3 x = cross(u_sunDir, n);
        vec3 y = cross(x, n);

        float mult = (ONE_OVER_4PI * rho / (ax * ay * sqrt(max(1e-5, dot(u_sunDir, n) * dot(v, n)))));

        // vec3 h = sundir + v;
        // vec3 x = cross(sundir, n);
        // vec3 y = cross(x, n);

        // float mult = (ONE_OVER_4PI * rho / (ax * ay * sqrt(max(1e-5, dot(sundir, n) * dot(v, n)))));
  
        float hdotx = dot(h, x) / ax;
        float hdoty = dot(h, y) / ay;
        float hdotn = dot(h, n);

        float spec = mult * exp(-((hdotx * hdotx) + (hdoty * hdoty)) / (hdotn * hdotn));

        vec3 finalColor = mix(u_oceanColor, refl * color_mod, F) + u_sunColor * spec;
        // vec3 HDRColor = finalColor / (finalColor + vec3(1.0));
        FragColor =  vec4(finalColor.rgb, 1.0);
        // FragColor = vec4(foamScale, foamStatus, 0.0, 1.0);
    }
`
export {
    vertStr,
    fragStr
}

