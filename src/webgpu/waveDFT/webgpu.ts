import {
    DISP_MAP_SIZE,
    PATCH_SIZE,
    WIND_SPEED,
    WIND_X,
    WIND_Z,
    AMPLITUDE_CONSTANT,
    GRAV_ACCELERATION,
    ONE_OVER_SQRT_2
} from './config';

import * as THREE from 'three';
import { MapControls } from './examples/jsm/controls/OrbitControls';

// import { mat4, vec3 } from 'gl-matrix';
import { H_KGen, omegaGen, rgba32floatGen, bitfieldReverse256 } from './utils';
import { updatespectrumComp } from './wgsl/updatespectrum';
import { fourierDFTComp } from './wgsl/fourier_dft';
import { fourierFFTComp } from './wgsl/fourier_fft';
import { createDisplacementComp } from './wgsl/createdisplacement';
import { createGradientComp } from './wgsl/creategradient';

import { displacementMapVert, displacementMapFrag } from './wgsl/displacementMapGPU';
import { displacementPrecisionMapVert, displacementPrecisionMapFrag } from './wgsl/displacementPrecisionMapGPU';
import { gradientMapVert, gradientMapFrag } from './wgsl/gradientMapGPU';
import { gradientPrecisionMapVert, gradientPrecisionMapFrag } from './wgsl/gradientPrecisionMapGPU';
import { oceanVert, oceanFrag } from './wgsl/OceanGPU';

import MeshGrid from './meshgrid'
import Quad from './quad'

let myMeshGrid = new MeshGrid(2000, 2000, 0.5);
let myQuad = new Quad();

export default class WebGPU_DLM{

    // webGPU
    public webGPU:GPU;

    // webgl
    public webGLInstance:any;   // webgl 实例

    // 相机
    public camera:any;          //  相机
    public controls:any;        // 相机控制器

    // 成员变量
    public container:HTMLElement;
    public pLabel:HTMLElement;
    public canvas:HTMLCanvasElement;
    public displacementCanvas:HTMLCanvasElement;
    public displacementPrecisionCanvas:HTMLCanvasElement;
    public gradientCanvas:HTMLCanvasElement;
    public gradientPrecisionCanvas:HTMLCanvasElement;

    public oceanContext:GPUCanvasContext;
    public displacementContext:GPUCanvasContext;
    public displacementPrecisionContext:GPUCanvasContext;
    public gradientContext:GPUCanvasContext;
    public gradientPrecisionContext:GPUCanvasContext;

    // adapter, logical device, queue （参考 WebGPU 大图）https://www.yuque.com/denglimin/uw1511/rriuk2nhut6vnhth
    public adapter:GPUAdapter;
    public device:GPUDevice;
    public queue:GPUQueue;

    // uniform buffer
    public matrixUniformBuffer:GPUBuffer;       // 矩阵 uniform buffer
    public updateSpectrumParamBuffer:GPUBuffer;       // 矩阵 uniform buffer
    public oceanParamBuffer:GPUBuffer;       // 矩阵 uniform buffer

    // bindGroup
    public updatespectrumBindGroup:GPUBindGroup;    // gpu bind group
    public fourierDFTBindGroup:GPUBindGroup;        // gpu bind group

    public fourierDFT_H_Pass1BindGroup:GPUBindGroup;   // gpu bind group
    public fourierDFT_H_Pass2BindGroup:GPUBindGroup;   // gpu bind group
    public fourierDFT_Choppy_Pass1BindGroup:GPUBindGroup;   // gpu bind group
    public fourierDFT_Choppy_Pass2BindGroup:GPUBindGroup;   // gpu bind group

    public fourierFFT_H_Pass1BindGroup:GPUBindGroup;   // gpu bind group
    public fourierFFT_H_Pass2BindGroup:GPUBindGroup;   // gpu bind group
    public fourierFFT_Choppy_Pass1BindGroup:GPUBindGroup;   // gpu bind group
    public fourierFFT_Choppy_Pass2BindGroup:GPUBindGroup;   // gpu bind group

    public displacementComputeBindGroup:GPUBindGroup;      // compute bind group
    public gradientComputeBindGroup:GPUBindGroup;          // compute bind group

    public displacementMapRenderBindGroup:GPUBindGroup;     // render bind group
    public displacementPrecisionMapRenderBindGroup:GPUBindGroup;     // render bind group
    public gradientMapRenderBindGroup:GPUBindGroup;          // render bind group
    public gradientPrecisionMapRenderBindGroup:GPUBindGroup;          // render bind group
    public oceanBindGroup:GPUBindGroup;             // render bind group

    // 贴图
    public cubemapTex:GPUTexture;
    public perlinTex:GPUTexture;

    // 有 compute Shader 计算出来的贴图
    public tilde_h0Tex_c:GPUTexture;            // rgba32f      257 * 257
    public frequenceTex_c:GPUTexture;           // rgba32f      257 * 257

    public fourier_tex_h_pass1: GPUTexture;                 // rgba32f      256* 256
    public fourier_tex_h_pass2: GPUTexture;                 // rgba32f      256* 256
    public fourier_tex_choppy_pass1: GPUTexture;            // rgba32f      256* 256
    public fourier_tex_choppy_pass2: GPUTexture;            // rgba32f      256* 256

    public heightfieldTex_c:GPUTexture;         // rgba32f      256* 256
    public choppyfieldTex_c:GPUTexture;         // rgba32f      256* 256

    public displacementTex_c:GPUTexture;        // rgba32f      256* 256
    public gradientTex_c:GPUTexture;            // rgba16f      256* 256
    public bitfieldReverseTex:GPUTexture;       // rgba16f      256 * 1

    // 数据 buffer
    public rowCnt:number;                       // 每行的顶点数目
    public colCnt:number;                       // 每列的顶点数目
    public pointCnt:number;                     // 顶点个数
    public indexCnt:number;                     // 顶点索引个数
    public pointByteSize:number;                // 顶点数据 byte 长度
    public vertexBufferSrc:GPUBuffer;           // 源 vertex buffer
    public vertexBufferDst:GPUBuffer;           // 目的 vertex buffer
    public quadVertexBuffer:GPUBuffer;          // quad 的顶点
    public quadIndexBuffer:GPUBuffer;           // quad 的顶点索引
    public readBuffer:GPUBuffer;                // 把数据从 vertexBufferDst 复制出来
    public indexBuffer:GPUBuffer;               // index buffer

    // float32array 顶点数据
    public vertexData:ArrayBuffer;              // 从 readBuffer 取出来的 float32array 顶点数据

    // compute pipeline
    public updatespectrumComputePipeline: GPUComputePipeline;
    public fourierDFTComputePipeline: GPUComputePipeline;
    public fourierFFTComputePipeline: GPUComputePipeline;
    public displacementComputePipeline: GPUComputePipeline;
    public gradientComputePipeline: GPUComputePipeline;

    // render pipeline
    public displacementMapRenderPipeline: GPURenderPipeline;
    public displacementPrecisionMapRenderPipeline: GPURenderPipeline;
    public gradientMapRenderPipeline: GPURenderPipeline;
    public gradientPrecisionMapRenderPipeline: GPURenderPipeline;
    public oceanRenderPipeline: GPURenderPipeline;
    public commandEncoder: GPUCommandEncoder;

    // 配置参数
    public configParams = {
        u_time: 0.0,
        u_rowCnt: 0.0,
        u_colCnt: 0.0,
    };

    // 波形数据
    public data = {

        // 波1
        // 波1 100m ~ 600m
        u_wavenumber1:0.0628, 			// 波数(可以理解为角速度)
        c_waveLength1:100, 				// 波长 
        c_amplitude1:15.92,				// KA = 1 算出来的波幅
        u_amplitude_scale1: 0.8,		// 波幅的放缩比例
        c_frequency1: 0.01,				// 频率
        c_angular_velocity1: 0.0628,  	// 角速度
        u_direct_x1: 1.0, 				// 波的方向(方向 x 分量)
        u_direct_z1: 0.0,				// 波的方向(方向 z 分量)
        u_animation1: true,

        // 波2 20m ~ 100m
        u_wavenumber2:0.2513, 			// 波数(可以理解为角速度)
        c_waveLength2:25, 				// 波长 
        c_amplitude2:3.9793,			// KA = 1 算出来的波幅
        u_amplitude_scale2: 0.6,		// 波幅的放缩比例
        c_frequency2: 0.04,				// 频率
        c_angular_velocity2: 0.2513,  	// 角速度
        u_direct_x2: 0.0, 				// 波的方向(方向 x 分量)
        u_direct_z2: 1.0,				// 波的方向(方向 z 分量)
        u_animation2: true,

        // 波3 10m ~ 20m
        u_wavenumber3:0.50265, 			// 波数(可以理解为角速度)
        c_waveLength3:12.5, 			// 波长 
        c_amplitude3:1.98,				// KA = 1 算出来的波幅
        u_amplitude_scale3: 0.6,		// 波幅的放缩比例
        c_frequency3: 0.08,				// 频率
        c_angular_velocity3: 0.50265,  	// 角速度
        u_direct_x3: 1.0, 				// 波的方向(方向 x 分量)
        u_direct_z3: 1.0,				// 波的方向(方向 z 分量)
        u_animation3: true,

        // 波4 0.1m ~ 10m
        u_wavenumber4:1.005309, 		// 波数(可以理解为角速度)
        c_waveLength4:6.25, 			// 波长 
        c_amplitude4:0.994719,			// KA = 1 算出来的波幅
        u_amplitude_scale4: 0.5,		// 波幅的放缩比例
        c_frequency4: 0.16,				// 频率
        c_angular_velocity4: 1.0053,  	// 角速度
        u_direct_x4: 1.0, 				// 波的方向(方向 x 分量)
        u_direct_z4: -1.0,				// 波的方向(方向 z 分量)
        u_animation4: true,

        u_choppyDxScale:1.0,
        u_choppyDzScale:1.0,
        u_yScale:1.0,
        u_foamScale:0.5,
        u_foamStatus:true,
        u_sunColor:[1.000,1.000,0.471],             // 太阳颜色
        u_sunDir:[-0.707,0.309,-0.636],                // 太阳方向
        u_oceanColor:[0.004,0.020,0.031],           // 海水颜色
    };

    // 帧率
    private lastFrameTimestamp:number = 0;
    private frameRate:number = 60.0;
    private currentFrameId:number = 0;     // 当前帧
    private iTime:number = 0;               // 当前时间 单位为秒

    // 构造函数
    public constructor(){

        // 空的构造函数
        this.configParams = {
            u_rowCnt: 0.0,
            u_colCnt: 0.0,
            u_time: 0.0
        };
    }

    public init(canvas:HTMLCanvasElement, displacementCanvas:HTMLCanvasElement, displacementPrecisionCanvas:HTMLCanvasElement, gradientCanvas:HTMLCanvasElement, gradientPrecisionCanvas:HTMLCanvasElement, p:HTMLElement,container:HTMLElement){

        this.canvas = canvas;  
        this.displacementCanvas = displacementCanvas;
        this.displacementPrecisionCanvas = displacementPrecisionCanvas;
        this.gradientCanvas = gradientCanvas;
        this.gradientPrecisionCanvas = gradientPrecisionCanvas;
        this.pLabel = p;


        // 设置透视相机
        // 高宽比
        const aspect = window.innerWidth / window.innerHeight;
        const camera = new THREE.PerspectiveCamera( 60, aspect, 1, 20000 );

        const controls = new MapControls( camera, container );
        // eslint-disable-next-line no-lone-blocks
        {
            controls.screenSpacePanning = false;
            controls.panSpeed = 5;
            // controls.rotateSpeed = 0.4;
            controls.zoomSpeed = 2;
            
            // controls.enabled = false;   // 启用或禁用
            // controls.enableDamping = true;
            // controls.mouseButtons = {
            // 	LEFT: THREE.MOUSE.PAN,
            // 	MIDDLE: THREE.MOUSE.DOLLY,
            // 	RIGHT: THREE.MOUSE.ROTATE
            // }
            controls.minDistance = -100;
            controls.maxDistance = 1000;
            controls.maxPolarAngle = Math.PI * 0.50 ;
            
            controls.object.position.set(915.44, 7.15, 816.24);
            controls.target = new THREE.Vector3(912.14, 5.5, 813.31);
            controls.update();
        }
        this.camera = camera;
        this.controls = controls;
    }

    // 初始化WebGPU 
    public async initWebGPU(): Promise<Boolean>{

        // 试试是不是支持 WebGPU
        try{
            const webGPU:GPU = navigator.gpu;
            if(!webGPU){

                console.group("webgpu 初始化失败");
                console.error("浏览器不支持 WebGPU");
                console.groupEnd();
                return false;
            }

            this.webGPU = webGPU;

            // 物理设备适配器 => vulcan, directX, mental
            this.adapter = await webGPU.requestAdapter();

            // 逻辑设备
            this.device = await this.adapter.requestDevice();

            // 绘制队列
            this.queue = this.device.queue;

            // 上下文
            this.oceanContext = this.canvas.getContext('webgpu');

            // 设置上下文 context
            this.oceanContext.configure({
                device: this.device,
                format: webGPU.getPreferredCanvasFormat() || 'bgra8unorm',
                alphaMode: 'opaque'
            });

            // 设置 displacement 上下文
            this.displacementContext = this.displacementCanvas.getContext('webgpu');

            // 设置上下文 context
            this.displacementContext.configure({
                device: this.device,
                format: webGPU.getPreferredCanvasFormat() || 'bgra8unorm',
                alphaMode: 'opaque'
            });

            // 设置 displacement percision上下文
            this.displacementPrecisionContext = this.displacementPrecisionCanvas.getContext('webgpu');

            // 设置上下文 context
            this.displacementPrecisionContext.configure({
                device: this.device,
                format: webGPU.getPreferredCanvasFormat() || 'bgra8unorm',
                alphaMode: 'premultiplied'
            });
            
            // 设置 gradient 上下文
            this.gradientContext = this.gradientCanvas.getContext('webgpu');

            // 设置上下文 context
            this.gradientContext.configure({
                device: this.device,
                format: webGPU.getPreferredCanvasFormat() || 'bgra8unorm',
                alphaMode: 'premultiplied'
            });

            // 设置 gradient percision上下文上下文
            this.gradientPrecisionContext = this.gradientPrecisionCanvas.getContext('webgpu');

            // 设置上下文 context
            this.gradientPrecisionContext.configure({
                device: this.device,
                format: webGPU.getPreferredCanvasFormat() || 'bgra8unorm',
                alphaMode: 'premultiplied'
            });
            
        }
        catch(e){
            console.group("webgpu 初始化失败");
            console.error(e);
            console.groupEnd();
            return false;
        }

        console.group("webgpu 初始化");
        console.log("WebGPU 初始化成功！");
        console.groupEnd();
        return true;
    }

    // 加载贴图图片(天空盒子, perlin 噪音等)
    public async loadTextures():Promise<boolean>{

        let { device } = this;

        // cubemap 6 张图片的加载地址是 [+X, -X, +Y, -Y, +Z, -Z]
        // 3 张其他的贴图
        const imgSrcs = [
            new URL(`/wave/0_positive_x.jpg`,'http://localhost:3000/').toString(),
            new URL(`/wave/1_negative_x.jpg`,'http://localhost:3000/').toString(),
            new URL(`/wave/2_positive_y.jpg`,'http://localhost:3000/').toString(),
            new URL(`/wave/3_negative_y.jpg`,'http://localhost:3000/').toString(),
            new URL(`/wave/4_positive_z.jpg`,'http://localhost:3000/').toString(),
            new URL(`/wave/5_negative_z.jpg`,'http://localhost:3000/').toString(),

            new URL(`/wave/perlin_noise.png`,'http://localhost:3000/').toString(),  // 6
        ];
        console.log(imgSrcs);
        
        // 加载并解析图片
        const promises = imgSrcs.map((src) => {
            const img = document.createElement('img');
            img.src = src;
            return img.decode().then(() => createImageBitmap(img));
        });
        const imageBitmaps = await Promise.all(promises);
        console.log(imageBitmaps);
        // 创建一个 2d array texture.
        // 每张图片尺寸大小一致
        const cubemapTex = device.createTexture({
            dimension: '2d',
            size: [imageBitmaps[0].width, imageBitmaps[0].height, 6],
            format: 'rgba8unorm',
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
        });
      
        // 复制到 6 张cubemap 图片到 GPU
        for (let i = 0; i < 6; i++) {
            const imageBitmap = imageBitmaps[i];
            device.queue.copyExternalImageToTexture(
                { source: imageBitmap },
                { texture: cubemapTex, origin: [0, 0, i] },
                [imageBitmap.width, imageBitmap.height]
            );
        }

        this.cubemapTex = cubemapTex;
        console.log(cubemapTex);
        
        // perlinTex
        {
            const imageBitmap = imageBitmaps[6];
            const texture = device.createTexture({
                size: [imageBitmap.width, imageBitmap.height, 1],
                format: 'rgba8unorm',
                usage: GPUTextureUsage.TEXTURE_BINDING |  GPUTextureUsage.COPY_DST |  GPUTextureUsage.RENDER_ATTACHMENT
            });
    
            device.queue.copyExternalImageToTexture(
                { source: imageBitmap },
                { texture: texture },
                [imageBitmap.width, imageBitmap.height]
            );
    
            this.perlinTex = texture;
        }
        
        return true;
    }

    // 创建贴图 compute shader 计算和导出数据给 渲染 pipeline 用
    public createTextureAndSpectrum(){
        let { device } = this;
        //  tilde_h0Tex_c                   // rg32float    256* 256
        //  frequenceTex_c                  // r32float     256* 256
        //  heightfieldTex_c                // rg32float    256* 256
        //  choppyfieldTex_c                // rg32float    256* 256
        //  displacementTex_c               // rgba32float  256* 256
        //  gradientTex_c                   // rgba16float  256* 256
        const imageW = 256;
        const imageH = 256;

        //1. 波幅数据贴图
        const h0TexW = DISP_MAP_SIZE + 1;
        const h0TexH = DISP_MAP_SIZE + 1;
        const tilde_h0Tex_c = device.createTexture({
            size: { width: h0TexW, height: h0TexH },
            format: "rgba32float",
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        });

        // 写数据 H0 = h(k) 数据
        // const tilde_h0_data = H_KGen(-0.406138480, -0.913811505, 6.5);
        const tilde_h0_data = H_KGen(WIND_X, WIND_Z, WIND_SPEED);
        device.queue.writeTexture(
            { texture: tilde_h0Tex_c } , 
            tilde_h0_data,
            { bytesPerRow: h0TexW * 16 },
            { width: h0TexW, height: h0TexH }
        );
        this.tilde_h0Tex_c = tilde_h0Tex_c;

        //2. 波的频率贴图 ω 
        const frequenceTexW = DISP_MAP_SIZE + 1;
        const frequenceTexH = DISP_MAP_SIZE + 1;
        const frequenceTex_c = device.createTexture({
            size: [frequenceTexW, frequenceTexH],
            format: 'rgba32float',
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        });

        // 写数据
        const frequenceTex_data = omegaGen();
        device.queue.writeTexture(
            { texture: frequenceTex_c } , 
            frequenceTex_data,
            { bytesPerRow: frequenceTexW * 16 },
            { width: frequenceTexW, height: frequenceTexH }
        );
        this.frequenceTex_c = frequenceTex_c;

        // test image 数据
        const image_data = rgba32floatGen();

        // dft 和 fft tex
        const fourier_tex_h_pass1 = device.createTexture({
            size: [imageW, imageH],
            format: 'rgba32float',
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        });
        this.fourier_tex_h_pass1 = fourier_tex_h_pass1;

        const fourier_tex_h_pass2 = device.createTexture({
            size: [imageW, imageH],
            format: 'rgba32float',
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        });
        this.fourier_tex_h_pass2 = fourier_tex_h_pass2;

        const fourier_tex_choppy_pass1 = device.createTexture({
            size: [imageW, imageH],
            format: 'rgba32float',
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        });
        this.fourier_tex_choppy_pass1 = fourier_tex_choppy_pass1;

        const fourier_tex_choppy_pass2 = device.createTexture({
            size: [imageW, imageH],
            format: 'rgba32float',
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        });
        this.fourier_tex_choppy_pass2 = fourier_tex_choppy_pass2;

        // 波的高度场贴图
        const heightfieldTex_c = device.createTexture({
            size: [imageW, imageH],
            format: 'rgba32float',
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        });

        // 写数据
        device.queue.writeTexture(
            { texture: heightfieldTex_c } , 
            image_data,
            { bytesPerRow: imageW * 16 },
            { width: imageW, height: imageH }
        );
        this.heightfieldTex_c = heightfieldTex_c;

        // 波的平移场贴图
        const choppyfieldTex_c = device.createTexture({
            size: [imageW, imageH],
            format: 'rgba32float',
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        });

        // 写数据
        device.queue.writeTexture(
            { texture: choppyfieldTex_c }, 
            image_data,
            { bytesPerRow: imageW * 16 },
            { width: imageW, height: imageH }
        );
        this.choppyfieldTex_c = choppyfieldTex_c;

        // 波的位移场贴图
        const displacementTex_c = device.createTexture({
            size: [imageW, imageH],
            format: 'rgba32float',
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        });

        // 写数据
        device.queue.writeTexture(
            { texture: displacementTex_c } , 
            image_data,
            { bytesPerRow: imageW * 16 },
            { width: imageW, height: imageH }
        );
        this.displacementTex_c = displacementTex_c;

        // 10 秒后写入新的数据
        // const image_data1 = rgba32floatGen1();
        // setTimeout(()=>{
        //     device.queue.writeTexture(
        //         { texture: displacementTex_c }, 
        //         image_data1,
        //         { bytesPerRow: imageW * 16 },
        //         { width: imageW, height: imageH }
        //     );
        //     this.displacementTex_c = displacementTex_c;
        // }, 5000);
        
        // 波的梯度场贴图
        const gradientTex_c = device.createTexture({
            size: [imageW, imageH],
            format: 'rgba32float',
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        });
        device.queue.writeTexture(
            { texture: gradientTex_c } , 
            image_data,
            { bytesPerRow: imageW * 16 },
            { width: imageW, height: imageH }
        );
        this.gradientTex_c = gradientTex_c;

        // bitfieldReverse FFT 索引下标(二进制数倒排)
        var bitIndexData = bitfieldReverse256();
        const bitfieldReverseTex = device.createTexture({
            size: [imageW, 1],
            format: 'rgba32float',
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        });
        device.queue.writeTexture(
            { texture: bitfieldReverseTex }, 
            bitIndexData,
            { bytesPerRow: imageW * 16 },
            { width: imageW, height: 1 }
        );
        this.bitfieldReverseTex = bitfieldReverseTex;
    }

    // resize color texture, depth texture
    public resize():void{

        let { webGPU, canvas } = this;

        // 设置尺寸
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;

    }

    // 初始化 ShaderModule 和 pipeline
    public initShaderModuleAndPipeline():void{

        let { device } = this;

        /**************************************************************************
         * 
         * 1. 生成 shader module
         * 
        **************************************************************************/
        
        // 1.1 计算 module
        // 创建 updatespectrumCompute  计算 module
        const updatespectrumComputeModule = device.createShaderModule({
            label:'updatespectrum Compute Module 着色器模块',
            code: updatespectrumComp
        });

        // 创建 fourier_DFTCompute  计算 module
        const fourier_DFTComputeModule = device.createShaderModule({
            label:'fourier_DFT Compute Module 着色器模块',
            code:fourierDFTComp
        });

        // 创建 fourier_FFTCompute  计算 module
        const fourier_FFTComputeModule = device.createShaderModule({
            label:'fourier_FFT Compute Module 着色器模块',
            code:fourierFFTComp
        });

        // 创建 displacementCompute  计算 module
        const displacementComputeModule = device.createShaderModule({
            label:'displacement Compute Module 着色器模块',
            code:createDisplacementComp
        });

        // 创建 createGradient Compute  计算 module
        const GradientComputeModule = device.createShaderModule({
            label:'create Gradient Compute Module 着色器模块',
            code:createGradientComp
        });

        // 1.2 渲染 module
        // 创建 displacementMap shader module 渲染
        const displacementMapVertexModule = device.createShaderModule({
            label:'位移贴图渲染 Vertex Module 顶点着色器模块',
            code:displacementMapVert
        });

        const displacementMapFragmentModule = device.createShaderModule({
            label:'位移贴图渲染 fragment Module 片元着色器模块',
            code:displacementMapFrag
        });

        // 创建 displacementPrecisionMap shader module 渲染
        const displacementPrecisionMapVertexModule = device.createShaderModule({
            label:'位移高精贴图渲染 Vertex Module 顶点着色器模块',
            code:displacementPrecisionMapVert
        });

        const displacementPrecisionMapFragmentModule = device.createShaderModule({
            label:'位移高精贴图渲染 fragment Module 片元着色器模块',
            code:displacementPrecisionMapFrag
        });

        // 创建 gradientMap shader module 渲染
        const gradientMapVertexModule = device.createShaderModule({
            label:'梯度贴图渲染 Vertex Module 顶点着色器模块',
            code:gradientMapVert
        });

        const gradientMapFragmentModule = device.createShaderModule({
            label:'梯度贴图渲染 fragment Module 片元着色器模块',
            code:gradientMapFrag
        });

        // 创建 gradientPrecisionMap shader module 渲染
        const gradientPrecisionMapVertexModule = device.createShaderModule({
            label:'梯度高精贴图渲染 Vertex Module 顶点着色器模块',
            code:gradientPrecisionMapVert
        });

        const gradientPrecisionMapFragmentModule = device.createShaderModule({
            label:'梯度高精贴图渲染 fragment Module 片元着色器模块',
            code:gradientPrecisionMapFrag
        });

        // 创建 ocean shader module 渲染
        const oceanVertexModule = device.createShaderModule({
            label:'ocean Vertex Module 顶点着色器模块',
            code:oceanVert
        });

        const oceanFragmentModule = device.createShaderModule({
            label:'ocean fragment Module 片元着色器模块',
            code:oceanFrag
        });
        
        /**************************************************************************
         * 
         * 2. 生成 pipeline
         * 
        **************************************************************************/
        // 2.1 计算管线
        // updatespectrum 计算管线
        const updatespectrumComputePipeline = device.createComputePipeline({
            layout: 'auto', 
            compute: {
                module: updatespectrumComputeModule,
                entryPoint: 'main',
            },
        });

        this.updatespectrumComputePipeline = updatespectrumComputePipeline;

        // fourier_dft 计算管线
        const fourierDFTComputePipeline = device.createComputePipeline({
            layout: 'auto', 
            compute: {
                module: fourier_DFTComputeModule,
                entryPoint: 'main',
            },
        });
        this.fourierDFTComputePipeline = fourierDFTComputePipeline;

        // fourier_fft 计算管线
        const fourierFFTComputePipeline = device.createComputePipeline({
            layout: 'auto', 
            compute: {
                module: fourier_FFTComputeModule,
                entryPoint: 'main',
            },
        });
        this.fourierFFTComputePipeline = fourierFFTComputePipeline;

        // displacement 计算管线
        const displacementComputePipeline = device.createComputePipeline({
            layout: 'auto', 
            compute: {
                module: displacementComputeModule,
                entryPoint: 'main',
            },
        });

        this.displacementComputePipeline = displacementComputePipeline;

        // displacement 计算管线
        const gradientComputePipeline = device.createComputePipeline({
            layout: 'auto', 
            compute: {
                module: GradientComputeModule,
                entryPoint: 'main',
            },
        });

        this.gradientComputePipeline = gradientComputePipeline;

        // 2.2 渲染管线
        // 渲染 位移贴图 displacementMap 
        const displacementMapRenderPipeline = device.createRenderPipeline({
            layout: 'auto',
            vertex: {
                module: displacementMapVertexModule,
                entryPoint: 'vert_main',
                buffers: [
                    {
                        // 顶点 vertex buffer // 因为 compute shader 里面 传 vec3 也是占用 vec4 的空间
                        arrayStride: Float32Array.BYTES_PER_ELEMENT * 6,         // sizeof(float) * 6 x,y,z,w,uv
                        stepMode: 'vertex',
                        attributes: [
                            {
                                // 顶点位置 vertex positions
                                shaderLocation: 0,
                                offset: 0,
                                format: 'float32x4'
                            },
                            {
                                // 顶点贴图 vertex uv
                                shaderLocation: 1,
                                offset: 16,
                                format: 'float32x2'
                            },
                        ],
                    },
                ],
            },
            fragment: {
                module: displacementMapFragmentModule,
                entryPoint: 'frag_main',
                targets: [
                    {
                        format: navigator.gpu.getPreferredCanvasFormat(),
                    },
                ],
            },
            primitive: {
                // topology: 'line-list',   // 线框
                topology: 'triangle-list',  // 三角面
            },
        });

        this.displacementMapRenderPipeline = displacementMapRenderPipeline;

        const displacementPrecisionMapRenderPipeline = device.createRenderPipeline({
            layout: 'auto',
            vertex: {
                module: displacementPrecisionMapVertexModule,
                entryPoint: 'vert_main',
                buffers: [
                    {
                        // 顶点 vertex buffer // 因为 compute shader 里面 传 vec3 也是占用 vec4 的空间
                        arrayStride: Float32Array.BYTES_PER_ELEMENT * 6,         // sizeof(float) * 6 x,y,z,w,uv
                        stepMode: 'vertex',
                        attributes: [
                            {
                                // 顶点位置 vertex positions
                                shaderLocation: 0,
                                offset: 0,
                                format: 'float32x4'
                            },
                            {
                                // 顶点贴图 vertex uv
                                shaderLocation: 1,
                                offset: 16,
                                format: 'float32x2'
                            },
                        ],
                    },
                ],
            },
            fragment: {
                module: displacementPrecisionMapFragmentModule,
                entryPoint: 'frag_main',
                targets: [
                    {
                        format: navigator.gpu.getPreferredCanvasFormat(),
                    },
                ],
            },
            primitive: {
                // topology: 'line-list',   // 线框
                topology: 'triangle-list',  // 三角面
            },
        });

        this.displacementPrecisionMapRenderPipeline = displacementPrecisionMapRenderPipeline;


        // 渲染 梯度贴图 gradientMap 
        const gradientMapRenderPipeline = device.createRenderPipeline({
            layout: 'auto',
            vertex: {
                module: gradientMapVertexModule,
                entryPoint: 'vert_main',
                buffers: [
                    {
                        // 顶点 vertex buffer // 因为 compute shader 里面 传 vec3 也是占用 vec4 的空间
                        arrayStride: Float32Array.BYTES_PER_ELEMENT * 6,         // sizeof(float) * 6 x,y,z,w,u,v
                        stepMode: 'vertex',
                        attributes: [
                            {
                                // 顶点位置 vertex positions
                                shaderLocation: 0,
                                offset: 0,
                                format: 'float32x4'
                            },
                            {
                                // 顶点贴图 vertex uv
                                shaderLocation: 1,
                                offset: 16,
                                format: 'float32x2'
                            },
                        ],
                    },
                ],
            },
            fragment: {
                module: gradientMapFragmentModule,
                entryPoint: 'frag_main',
                targets: [
                    {
                        format: navigator.gpu.getPreferredCanvasFormat(),
                    },
                ],
            },
            primitive: {
                // topology: 'line-list',   // 线框
                topology: 'triangle-list',  // 三角面
            },
        });

        this.gradientMapRenderPipeline = gradientMapRenderPipeline;

        const gradientPrecisionMapRenderPipeline = device.createRenderPipeline({
            layout: 'auto',
            vertex: {
                module: gradientPrecisionMapVertexModule,
                entryPoint: 'vert_main',
                buffers: [
                    {
                        // 顶点 vertex buffer // 因为 compute shader 里面 传 vec3 也是占用 vec4 的空间
                        arrayStride: Float32Array.BYTES_PER_ELEMENT * 6,         // sizeof(float) * 6 x,y,z,w,u,v
                        stepMode: 'vertex',
                        attributes: [
                            {
                                // 顶点位置 vertex positions
                                shaderLocation: 0,
                                offset: 0,
                                format: 'float32x4'
                            },
                            {
                                // 顶点贴图 vertex uv
                                shaderLocation: 1,
                                offset: 16,
                                format: 'float32x2'
                            },
                        ],
                    },
                ],
            },
            fragment: {
                module: gradientPrecisionMapFragmentModule,
                entryPoint: 'frag_main',
                targets: [
                    {
                        format: navigator.gpu.getPreferredCanvasFormat(),
                    },
                ],
            },
            primitive: {
                // topology: 'line-list',   // 线框
                topology: 'triangle-list',  // 三角面
            },
        });

        this.gradientPrecisionMapRenderPipeline = gradientPrecisionMapRenderPipeline;

        // 渲染海洋表面
        const oceanRenderPipeline = device.createRenderPipeline({
            layout: 'auto',
            vertex: {
                module: oceanVertexModule,
                entryPoint: 'vert_main',
                buffers: [
                    {
                        // 顶点 vertex buffer // 因为 compute shader 里面 传 vec3 也是占用 vec4 的空间
                        arrayStride: Float32Array.BYTES_PER_ELEMENT * 4,         // sizeof(float) * 4 x,y,z,w
                        stepMode: 'vertex',
                        attributes: [
                            {
                                // 顶点位置 vertex positions
                                shaderLocation: 0,
                                offset: 0,
                                format: 'float32x4'
                            },
                        ],
                    },
                ],
            },
            fragment: {
                module: oceanFragmentModule,
                entryPoint: 'frag_main',
                targets: [
                    {
                        format: navigator.gpu.getPreferredCanvasFormat(),
                    },
                ],
            },
            primitive: {
                // topology: 'line-list',   // 线框
                topology: 'triangle-list',  // 三角面
            },
        });

        this.oceanRenderPipeline = oceanRenderPipeline;
    }

    // 初始化数据 buffer 
    public initDataBufferAndBindGroup():void{
    // public initDataBufferAndBindGroup(positions:Float32Array, indices:Uint32Array, pointCnt:number, indexCnt:number, rowCnt:number, colCnt:number):void{

        let {   device, configParams } = this;
        let {   cubemapTex, perlinTex } = this;
        let {   tilde_h0Tex_c, frequenceTex_c, 
                fourier_tex_h_pass1, fourier_tex_h_pass2, fourier_tex_choppy_pass1, fourier_tex_choppy_pass2,
                bitfieldReverseTex,
                heightfieldTex_c, choppyfieldTex_c, displacementTex_c, gradientTex_c } = this;

        let {   updatespectrumComputePipeline, fourierDFTComputePipeline, fourierFFTComputePipeline, 
                displacementComputePipeline,  gradientComputePipeline } = this;

        let {   displacementMapRenderPipeline, 
                displacementPrecisionMapRenderPipeline, 
                gradientMapRenderPipeline, 
                gradientPrecisionMapRenderPipeline, 
                oceanRenderPipeline } = this;


        // 网格顶点数据和索引数据
        const positions = new Float32Array(myMeshGrid.vertexArrayBuffer);
        const indices = new Uint32Array(myMeshGrid.indexTriangleArrayBuffer);
        const pointCnt = myMeshGrid.pointCnt;
        const indexCnt = myMeshGrid.indexTriangleCnt;
        const rowCnt = myMeshGrid.rowCnt;
        const colCnt = myMeshGrid.colCnt

        // 记录一下顶点个数和索引个数
        this.pointCnt = pointCnt;
        this.indexCnt = indexCnt;
        this.rowCnt = rowCnt;
        this.colCnt = colCnt;

        this.configParams.u_rowCnt = rowCnt;
        this.configParams.u_colCnt = colCnt;

        /**************************************************************************
         * 
         * 1. meshgrid vertex buffer & index buffer
         * 
        **************************************************************************/
        const kVertexStride = 4;   // x,y,z,w
        const pointByteSize = pointCnt * kVertexStride * Float32Array.BYTES_PER_ELEMENT;
        this.pointByteSize = pointByteSize;
 
        // 1.1 源 vertexBuffer 用于读
        const vertexBufferSrc = device.createBuffer({

            // position: vec3
            size: pointByteSize,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.STORAGE,
            mappedAtCreation: true,
        });

        // 赋值
        {
            const writeArray = new Float32Array(vertexBufferSrc.getMappedRange());
                writeArray.set(positions);

            // 注入到显存
            vertexBufferSrc.unmap();  
        }
        this.vertexBufferSrc = vertexBufferSrc;

        // 1.2 目的 vertexBuffer 用于写
        const vertexBufferDst = device.createBuffer({

            // position: vec3
            size: pointByteSize,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
            // usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
            mappedAtCreation: true,
        });

        // 赋值
        {
            const writeArray = new Float32Array(vertexBufferDst.getMappedRange());
                writeArray.set(positions);

            // 注入到显存
            vertexBufferDst.unmap();  
        }
        this.vertexBufferDst = vertexBufferDst;

        // 1.3 用于读取结果数据
        const gpuReadBuffer = device.createBuffer({
            size: pointByteSize,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        });

            gpuReadBuffer.unmap();  
        this.readBuffer = gpuReadBuffer;

        // 1.4 创建模型的索引 buffer
        const indexBuffer = device.createBuffer({
            size: indexCnt * Uint32Array.BYTES_PER_ELEMENT,
            usage: GPUBufferUsage.INDEX,
            mappedAtCreation: true,
        });

        {
            const writeArray = new Uint32Array(indexBuffer.getMappedRange());
            writeArray.set(indices);

            // 注入到显存
            indexBuffer.unmap();
        }

        this.indexBuffer = indexBuffer;

        /**************************************************************************
         * 
         * 2. quad vertex buffer & index buffer
         * 
        **************************************************************************/
        const quadPositions = new Float32Array(myQuad.vertexArrayBuffer);
        const quadIndices = new Float32Array(myQuad.indexArrayBuffer);

        const quadPointCnt = 4;
        const quadVertexStride = 6;  // point + uv 占用两个 vec4 的空间
        const quadPointByteSize = quadPointCnt * quadVertexStride * Float32Array.BYTES_PER_ELEMENT;
 
        // 1.1 quad vertexBuffer 用于渲染 canvasTexture
        const quadVertexBuffer = device.createBuffer({

            // position: vec3 但是占用 vec4 的空间
            size: quadPointByteSize,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.STORAGE,
            mappedAtCreation: true,
        });

        // 赋值
        {
            const writeArray = new Float32Array(quadVertexBuffer.getMappedRange());
                writeArray.set(quadPositions);

            // 注入到显存
            quadVertexBuffer.unmap();  
        }
        this.quadVertexBuffer = quadVertexBuffer;

        // 1.4 创建模型的索引 buffer
        const quadIndexBuffer = device.createBuffer({
            size: 6 * Uint32Array.BYTES_PER_ELEMENT,
            usage: GPUBufferUsage.INDEX,
            mappedAtCreation: true,
        });

        {
            const writeArray = new Uint32Array(quadIndexBuffer.getMappedRange());
            writeArray.set(quadIndices);

            // 注入到显存
            quadIndexBuffer.unmap();
        }

        this.quadIndexBuffer = quadIndexBuffer;

        

        /**************************************************************************
         * 
         * 3. uniform buffer
         * 
        **************************************************************************/
        // 3.1 矩阵 uniform buffer
        const uniformBufferSize = 2 * 4 * 4 * Float32Array.BYTES_PER_ELEMENT;                       // 2个 4x4 matrix
        const matrixUniformBuffer = device.createBuffer({
            size: uniformBufferSize,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.matrixUniformBuffer = matrixUniformBuffer;

        // 3.2 创建一个 updateSpectrum uniform buffer 
        const updateSpectrumParamsBufferSize = 1.0 * Float32Array.BYTES_PER_ELEMENT;
        const updateSpectrumParamBuffer = device.createBuffer({
            size: updateSpectrumParamsBufferSize,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // 初始化 uniform 数据
        device.queue.writeBuffer(
            updateSpectrumParamBuffer,
            0,
            new Float32Array([
                configParams.u_time
            ])
        );

        this.updateSpectrumParamBuffer = updateSpectrumParamBuffer;

        // 3.3 创建一个 ocean uniform buffer 
        const oceanParamsBufferSize = 6 * 4  * Float32Array.BYTES_PER_ELEMENT;
        const oceanParamBuffer = device.createBuffer({
            size: oceanParamsBufferSize,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // 暂时不初始化 uniform 数据
        this.oceanParamBuffer = oceanParamBuffer;


        /**************************************************************************
         * 
         * 4. uniform bind group
         * 
        **************************************************************************/

        /**************************************************************************
         * 4.1 updatespectrum compute binding group
        **************************************************************************/
        const updatespectrumBindGroup = device.createBindGroup({
            layout: updatespectrumComputePipeline.getBindGroupLayout(0),
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: updateSpectrumParamBuffer,
                    },
                },
                {
                    binding: 1,
                    resource: tilde_h0Tex_c.createView(),
                },
                {
                    binding: 2,
                    resource: frequenceTex_c.createView(),
                },
                {
                    binding: 3,
                    resource: heightfieldTex_c.createView(),
                },
                {
                    binding: 4,
                    resource: choppyfieldTex_c.createView()
                }
            ],
        });
        this.updatespectrumBindGroup = updatespectrumBindGroup;

        /**************************************************************************
         * 4.2 DFT binding group
        **************************************************************************/

        // dft compute h binding group
        const fourierDFT_H_Pass1BindGroup = device.createBindGroup({
            layout: fourierDFTComputePipeline.getBindGroupLayout(0),
            entries: [
                {
                    binding: 0,
                    resource: heightfieldTex_c.createView(),
                },
                {
                    binding: 1,
                    resource: fourier_tex_h_pass1.createView(),
                }
            ],
        });
        this.fourierDFT_H_Pass1BindGroup = fourierDFT_H_Pass1BindGroup;

        // dft compute binding group
        const fourierDFT_H_Pass2BindGroup = device.createBindGroup({
            layout: fourierDFTComputePipeline.getBindGroupLayout(0),
            entries: [
                {
                    binding: 0,
                    resource: fourier_tex_h_pass1.createView(),
                    // resource: heightfieldTex_c.createView(),
                },
                {
                    binding: 1,
                    resource: fourier_tex_h_pass2.createView(),
                }
            ],
        });
        this.fourierDFT_H_Pass2BindGroup = fourierDFT_H_Pass2BindGroup;

        // dft compute Choppy binding group
        const fourierDFT_Choppy_Pass1BindGroup = device.createBindGroup({
            layout: fourierDFTComputePipeline.getBindGroupLayout(0),
            entries: [
                {
                    binding: 0,
                    resource: choppyfieldTex_c.createView(),
                },
                {
                    binding: 1,
                    resource: fourier_tex_choppy_pass1.createView(),
                }
            ],
        });
        this.fourierDFT_Choppy_Pass1BindGroup = fourierDFT_Choppy_Pass1BindGroup;

        // dft compute Choppy binding group
        const fourierDFT_Choppy_Pass2BindGroup = device.createBindGroup({
            layout: fourierDFTComputePipeline.getBindGroupLayout(0),
            entries: [
                {
                    binding: 0,
                    resource: fourier_tex_choppy_pass1.createView(),
                },
                {
                    binding: 1,
                    resource: fourier_tex_choppy_pass2.createView(),
                }
            ],
        });
        this.fourierDFT_Choppy_Pass2BindGroup = fourierDFT_Choppy_Pass2BindGroup;

        /**************************************************************************
         * 4.3 FFT binding group
        **************************************************************************/
        // FFT compute pass 1 binding group
        const fourierFFT_H_Pass1BindGroup = device.createBindGroup({
            layout: fourierFFTComputePipeline.getBindGroupLayout(0),
            entries: [
                {
                    binding: 0,
                    resource: bitfieldReverseTex.createView(),
                },
                {
                    binding: 1,
                    resource: heightfieldTex_c.createView(),
                },
                {
                    binding: 2,
                    resource: fourier_tex_h_pass1.createView(),
                }
            ],
        });
        this.fourierFFT_H_Pass1BindGroup = fourierFFT_H_Pass1BindGroup;

        // FFT compute pass 2 binding group
        const fourierFFT_H_Pass2BindGroup = device.createBindGroup({
            layout: fourierFFTComputePipeline.getBindGroupLayout(0),
            entries: [
                {
                    binding: 0,
                    resource: bitfieldReverseTex.createView(),
                },
                {
                    binding: 1,
                    resource: fourier_tex_h_pass1.createView(),
                },
                {
                    binding: 2,
                    resource: fourier_tex_h_pass2.createView(),
                }
            ],
        });
        this.fourierFFT_H_Pass2BindGroup = fourierFFT_H_Pass2BindGroup;

        // fft compute Choppy binding group
        const fourierFFT_Choppy_Pass1BindGroup = device.createBindGroup({
            layout: fourierFFTComputePipeline.getBindGroupLayout(0),
            entries: [
                {
                    binding: 0,
                    resource: bitfieldReverseTex.createView(),
                },
                {
                    binding: 1,
                    resource: choppyfieldTex_c.createView(),
                },
                {
                    binding: 2,
                    resource: fourier_tex_choppy_pass1.createView(),
                }
            ],
        });
        this.fourierFFT_Choppy_Pass1BindGroup = fourierFFT_Choppy_Pass1BindGroup;

        // fft compute Choppy binding group
        const fourierFFT_Choppy_Pass2BindGroup = device.createBindGroup({
            layout: fourierFFTComputePipeline.getBindGroupLayout(0),
            entries: [
                {
                    binding: 0,
                    resource: bitfieldReverseTex.createView(),
                },
                {
                    binding: 1,
                    resource: fourier_tex_choppy_pass1.createView(),
                },
                {
                    binding: 2,
                    resource: fourier_tex_choppy_pass2.createView(),
                }
            ],
        });
        this.fourierFFT_Choppy_Pass2BindGroup = fourierFFT_Choppy_Pass2BindGroup;

        /**************************************************************************
         * 4.4 displacement compute binding group
        **************************************************************************/
        // displacement compute binding group
        const displacementComputeBindGroup = device.createBindGroup({
            layout: displacementComputePipeline.getBindGroupLayout(0),
            entries: [
                {
                    binding: 0,
                    resource: fourier_tex_h_pass2.createView(),
                },
                {
                    binding: 1,
                    resource: fourier_tex_choppy_pass2.createView(),
                },
                {
                    binding: 2,
                    resource: displacementTex_c.createView(),
                }
            ],
        });
        this.displacementComputeBindGroup = displacementComputeBindGroup;

        /**************************************************************************
         * 4.5 gradient compute binding group
        **************************************************************************/
        const gradientComputeBindGroup = device.createBindGroup({
            layout: gradientComputePipeline.getBindGroupLayout(0),
            entries: [
                {
                    binding: 0,
                    resource: displacementTex_c.createView(),
                },
                {
                    binding: 1,
                    resource: gradientTex_c.createView(),
                }
            ],
        });
        this.gradientComputeBindGroup = gradientComputeBindGroup;

        /**************************************************************************
         * 4.6 render shader 的 binding group
        **************************************************************************/
        const displacementMapRenderBindGroup = device.createBindGroup({
            layout: displacementMapRenderPipeline.getBindGroupLayout(0),
            entries: [
                {
                    binding: 0,
                    resource: displacementTex_c.createView(),
                },
            ],
        });
        this.displacementMapRenderBindGroup = displacementMapRenderBindGroup;

        const displacementPrecisionMapRenderBindGroup = device.createBindGroup({
            layout: displacementPrecisionMapRenderPipeline.getBindGroupLayout(0),
            entries: [
                {
                    binding: 0,
                    resource: displacementTex_c.createView(),
                },
            ],
        });
        this.displacementPrecisionMapRenderBindGroup = displacementPrecisionMapRenderBindGroup;

        const gradientMapRenderBindGroup = device.createBindGroup({
            layout: gradientMapRenderPipeline.getBindGroupLayout(0),
            entries: [
                {
                    binding: 0,
                    resource: gradientTex_c.createView(),
                },
            ],
        });
        this.gradientMapRenderBindGroup = gradientMapRenderBindGroup;

        const gradientPrecisionMapRenderBindGroup = device.createBindGroup({
            layout: gradientPrecisionMapRenderPipeline.getBindGroupLayout(0),
            entries: [
                {
                    binding: 0,
                    resource: gradientTex_c.createView(),
                },
            ],
        });
        this.gradientPrecisionMapRenderBindGroup = gradientPrecisionMapRenderBindGroup;

        const perlinTexSampler = device.createSampler({
            addressModeU: 'repeat',
            addressModeV: 'repeat',
            magFilter: 'linear',
            minFilter: 'linear',
        });
        const cubemapTexSampler = device.createSampler({
            magFilter: 'linear',
            minFilter: 'linear',
        });

        const oceanBindGroup = device.createBindGroup({
            layout: oceanRenderPipeline.getBindGroupLayout(0),
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: matrixUniformBuffer,
                    },
                },
                {
                    binding: 1,
                    resource: {
                        buffer: oceanParamBuffer,
                    },
                },
                {
                    binding: 2,
                    resource: perlinTexSampler,
                },
                {
                    binding: 3,
                    resource: perlinTex.createView(),
                },
                {
                    binding: 4,
                    resource: cubemapTexSampler,
                },
                {
                    binding: 5,
                    resource: cubemapTex.createView({
                        dimension: 'cube',
                    }),
                },

                // 计算出来的材质
                {
                    binding: 6,
                    resource: displacementTex_c.createView(),
                    // resource: fourier_tex_h_pass2.createView(),
                    // resource: fourier_tex_h_pass2.createView(),
                    // resource: heightfieldTex_c.createView(),
                },
                {
                    binding: 7,
                    resource: gradientTex_c.createView(),
                }
            ],
        });
        this.oceanBindGroup = oceanBindGroup;
    }

    // 更新 uniform 数据 
    public updateUniformData():void{

        let { matrixUniformBuffer, queue, camera } = this;
        let { updateSpectrumParamBuffer, oceanParamBuffer } = this;

        /**************************************************************************
         * 
         * 1. 更新矩阵信息
         * 
        **************************************************************************/
        const matrixByteLength = 4 * 4 * Float32Array.BYTES_PER_ELEMENT;   
        const offsetByte = 4 * 4 * Float32Array.BYTES_PER_ELEMENT;   
        
        const projectiveMatrix = new Float32Array(camera.projectionMatrix.elements);
        const viewMatrix = new Float32Array(camera.matrixWorldInverse.elements);

        // 写入 projectiveMatrix
        queue.writeBuffer(
            matrixUniformBuffer,
            0,
            projectiveMatrix.buffer,
            0,
            matrixByteLength
        );

        // 写入 viewMatrix
        queue.writeBuffer(
            matrixUniformBuffer,
            offsetByte,
            viewMatrix.buffer,
            0,
            matrixByteLength
        ); 

        /**************************************************************************
         * 
         * 2. 更新 updatespectrum compute shader 的 时间 u_time
         * 
        **************************************************************************/
        let timeParaArr = new Float32Array([ this.configParams.u_time]);
        queue.writeBuffer(
            updateSpectrumParamBuffer,
            0,
            timeParaArr.buffer,
            0,
            1 * Float32Array.BYTES_PER_ELEMENT
        ); 

        /**************************************************************************
         * 
         * 4. 更新 ocean rendering shader 的 oceanParams
         * 
        **************************************************************************/
        let { u_choppyDxScale, u_choppyDzScale, u_yScale, u_foamScale, u_foamStatus, u_sunColor, u_sunDir, u_oceanColor } = this.data;
        let cameraPosition = this.camera.position;

        let oceanParaArr = new Float32Array([
            u_choppyDxScale, u_choppyDzScale, u_yScale, u_foamScale,

            Number(u_foamStatus),
            PATCH_SIZE,
            DISP_MAP_SIZE,
            0.0, 

            ...u_sunColor, 0.0,
            ...u_sunDir, 0.0,
            ...u_oceanColor, 0.0,
            cameraPosition.x, cameraPosition.y, cameraPosition.z, 0.0,
            
        ]);

        // 更新参数
        queue.writeBuffer(
            oceanParamBuffer,
            0,
            oceanParaArr.buffer,
            0,
            6 * 4  * Float32Array.BYTES_PER_ELEMENT 
        ); 

    }

    // 集合绘制命令
    public assembleCommands():void{

        let {   canvas, oceanContext, 
                device, queue, 

                displacementCanvas,
                displacementPrecisionCanvas,
                gradientCanvas,
                gradientPrecisionCanvas,

                displacementContext, 
                displacementPrecisionContext,
                gradientContext,
                gradientPrecisionContext,
        
                updatespectrumBindGroup,

                fourierDFT_H_Pass1BindGroup,
                fourierDFT_H_Pass2BindGroup,
                fourierDFT_Choppy_Pass1BindGroup,
                fourierDFT_Choppy_Pass2BindGroup,

                fourierFFT_H_Pass1BindGroup,
                fourierFFT_H_Pass2BindGroup,
                fourierFFT_Choppy_Pass1BindGroup,
                fourierFFT_Choppy_Pass2BindGroup,

                displacementComputeBindGroup,
                gradientComputeBindGroup,

                displacementMapRenderBindGroup, 
                displacementPrecisionMapRenderBindGroup,
                gradientMapRenderBindGroup,
                gradientPrecisionMapRenderBindGroup,
                oceanBindGroup,

                indexCnt,
                pointByteSize,
                vertexBufferDst,
                readBuffer,
                indexBuffer,

                quadVertexBuffer, 
                quadIndexBuffer,

                updatespectrumComputePipeline,
                fourierDFTComputePipeline,
                fourierFFTComputePipeline,
                displacementComputePipeline, 
                gradientComputePipeline,

                displacementMapRenderPipeline, 
                displacementPrecisionMapRenderPipeline,
                gradientMapRenderPipeline,
                gradientPrecisionMapRenderPipeline,
                oceanRenderPipeline, 
            } = this;
        
        // 创建 conmmandEncoder
        const commandEncoder = device.createCommandEncoder();
        /******************************************************************* 
         * 
         * 0. 从 vertexBufferDst 把数据读到 readBuffer
         * 
         *******************************************************************/ 

        commandEncoder.copyBufferToBuffer(
            vertexBufferDst /* source buffer */,
            0 /* source offset */,
            readBuffer /* destination buffer */,
            0 /* destination offset */,
            pointByteSize /* size */
        );   
        
        /******************************************************************* 
         * 
         * 1. 运行 updatespectrum h(k, t)compute pipeline 得到结果
         * 
         *******************************************************************/ 
        // eslint-disable-next-line no-lone-blocks
        {
            // 创建 compute pass
            const passEncoder = commandEncoder.beginComputePass();

                // 设置管线 pipeline
                passEncoder.setPipeline(updatespectrumComputePipeline);
                passEncoder.setBindGroup(0, updatespectrumBindGroup);
                passEncoder.dispatchWorkgroups(16, 16, 1);
                passEncoder.end();
        }

        /******************************************************************* 
         * 
         * 2.1 运行 fourier_dft pass 1 & pass2 compute pipeline 
         * 
         *******************************************************************/ 
        // DFT H pass 1
        // eslint-disable-next-line no-lone-blocks
        {
            // // 创建 compute pass
            // const passEncoder = commandEncoder.beginComputePass();

            //     // 设置管线 pipeline
            //     passEncoder.setPipeline(fourierDFTComputePipeline);
            //     passEncoder.setBindGroup(0, fourierDFT_H_Pass1BindGroup);
            //     passEncoder.dispatchWorkgroups(256, 1, 1);
            //     passEncoder.end();
        }

        // DFT H pass 2
        // eslint-disable-next-line no-lone-blocks
        {
            // // 创建 compute pass
            // const passEncoder = commandEncoder.beginComputePass();

            //     // 设置管线 pipeline
            //     passEncoder.setPipeline(fourierDFTComputePipeline);
            //     passEncoder.setBindGroup(0, fourierDFT_H_Pass2BindGroup);
            //     passEncoder.dispatchWorkgroups(256, 1, 1);
            //     passEncoder.end();
        }

        // DFT choppy pass 1
        // eslint-disable-next-line no-lone-blocks
        {
            // // 创建 compute pass
            // const passEncoder = commandEncoder.beginComputePass();

            //     // 设置管线 pipeline
            //     passEncoder.setPipeline(fourierDFTComputePipeline);
            //     passEncoder.setBindGroup(0, fourierDFT_Choppy_Pass1BindGroup);
            //     passEncoder.dispatchWorkgroups(256, 1, 1);
            //     passEncoder.end();
        }

        // DFT choppy pass 2
        // eslint-disable-next-line no-lone-blocks
        {
            // // 创建 compute pass
            // const passEncoder = commandEncoder.beginComputePass();

            //     // 设置管线 pipeline
            //     passEncoder.setPipeline(fourierDFTComputePipeline);
            //     passEncoder.setBindGroup(0, fourierDFT_Choppy_Pass2BindGroup);
            //     passEncoder.dispatchWorkgroups(256, 1, 1);
            //     passEncoder.end();
        }

        /******************************************************************* 
         * 
         * 2.2 运行 fourier_fft pass 1 & pass2 compute pipeline 
         * 
         *******************************************************************/ 
        // FFT H pass 1
        {
            // 创建 compute pass
            const passEncoder = commandEncoder.beginComputePass();

                // 设置管线 pipeline
                passEncoder.setPipeline(fourierFFTComputePipeline);
                passEncoder.setBindGroup(0, fourierFFT_H_Pass1BindGroup);
                passEncoder.dispatchWorkgroups(256, 1, 1);
                passEncoder.end();
        }

        // FFT H pass2
        {
            // 创建 compute pass
            const passEncoder = commandEncoder.beginComputePass();

                // 设置管线 pipeline
                passEncoder.setPipeline(fourierFFTComputePipeline);
                passEncoder.setBindGroup(0, fourierFFT_H_Pass2BindGroup);
                passEncoder.dispatchWorkgroups(256, 1, 1);
                passEncoder.end();
        }

        // FFT choppy pass 1
        // eslint-disable-next-line no-lone-blocks
        {
            // 创建 compute pass
            const passEncoder = commandEncoder.beginComputePass();

                // 设置管线 pipeline
                passEncoder.setPipeline(fourierFFTComputePipeline);
                passEncoder.setBindGroup(0, fourierFFT_Choppy_Pass1BindGroup);
                passEncoder.dispatchWorkgroups(256, 1, 1);
                passEncoder.end();
        }

        // FFT choppy pass 2
        // eslint-disable-next-line no-lone-blocks
        {
            // 创建 compute pass
            const passEncoder = commandEncoder.beginComputePass();

                // 设置管线 pipeline
                passEncoder.setPipeline(fourierFFTComputePipeline);
                passEncoder.setBindGroup(0, fourierFFT_Choppy_Pass2BindGroup);
                passEncoder.dispatchWorkgroups(256, 1, 1);
                passEncoder.end();
        }

        /******************************************************************* 
         * 
         * 3. 运行 displacement compute pipeline 得到结果
         * 
         *******************************************************************/ 
        // eslint-disable-next-line no-lone-blocks
        {
            // 创建 compute pass
            const passEncoder = commandEncoder.beginComputePass();

                // 设置管线 pipeline
                passEncoder.setPipeline(displacementComputePipeline);
                passEncoder.setBindGroup(0, displacementComputeBindGroup);
                passEncoder.dispatchWorkgroups(16, 16, 1);
                passEncoder.end();
        }

        /******************************************************************* 
         * 
         * 4. 绘制到 displacementCanvas
         * 
         *******************************************************************/ 
        // 4.1  大于 1/255 的分量
        {
            const displacementPassDescriptor: GPURenderPassDescriptor = {
                colorAttachments: [
                    {
                        view: displacementContext.getCurrentTexture().createView(),     
                        clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
                        loadOp: 'clear',
                        storeOp: 'store',
                    },
                ],
            };

            const passEncoder = commandEncoder.beginRenderPass(displacementPassDescriptor);


            // 设置管线 pipeline
            passEncoder.setPipeline(displacementMapRenderPipeline);
            passEncoder.setBindGroup(0, displacementMapRenderBindGroup);
            passEncoder.setVertexBuffer(0, quadVertexBuffer);
            passEncoder.setIndexBuffer(quadIndexBuffer, 'uint32');
            
            // 设置窗口大小
            passEncoder.setViewport(0, 0, displacementCanvas.width, displacementCanvas.height, 0, 1);

            // 设置剪裁区域大小
            passEncoder.setScissorRect(0, 0, displacementCanvas.width, displacementCanvas.height);

            // 绘制网格
            passEncoder.drawIndexed(6);
            passEncoder.end();
        }

        // 4.2  小于 1/255 的分量
        {
            const displacementPrecisionPassDescriptor: GPURenderPassDescriptor = {
                colorAttachments: [
                    {
                        view: displacementPrecisionContext.getCurrentTexture().createView(),     
                        clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
                        loadOp: 'clear',
                        storeOp: 'store',
                    },
                ],
            };

            const passEncoder = commandEncoder.beginRenderPass(displacementPrecisionPassDescriptor);


            // 设置管线 pipeline
            passEncoder.setPipeline(displacementPrecisionMapRenderPipeline);
            passEncoder.setBindGroup(0, displacementPrecisionMapRenderBindGroup);
            passEncoder.setVertexBuffer(0, quadVertexBuffer);
            passEncoder.setIndexBuffer(quadIndexBuffer, 'uint32');
            
            // 设置窗口大小
            passEncoder.setViewport(0, 0, displacementPrecisionCanvas.width, displacementPrecisionCanvas.height, 0, 1);

            // 设置剪裁区域大小
            passEncoder.setScissorRect(0, 0, displacementPrecisionCanvas.width, displacementPrecisionCanvas.height);

            // 绘制网格
            passEncoder.drawIndexed(6);
            passEncoder.end();
        }

        /******************************************************************* 
         * 
         * 5. 运行 gradient compute pipeline 得到结果
         * 
         *******************************************************************/ 
        // eslint-disable-next-line no-lone-blocks
        {
            // 创建 compute pass
            const passEncoder = commandEncoder.beginComputePass();

                // 设置管线 pipeline
                passEncoder.setPipeline(gradientComputePipeline);
                passEncoder.setBindGroup(0, gradientComputeBindGroup);
                passEncoder.dispatchWorkgroups(16, 16, 1);
                passEncoder.end();
        }

        /******************************************************************* 
         * 
         * 6. 绘制到 gradientCanvas
         * 
         *******************************************************************/ 
        {
            const gradientPassDescriptor: GPURenderPassDescriptor = {
                colorAttachments: [
                    {
                        view: gradientContext.getCurrentTexture().createView(),     
                        clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
                        loadOp: 'clear',
                        storeOp: 'store',
                    },
                ],
            };

            const passEncoder = commandEncoder.beginRenderPass(gradientPassDescriptor);


            // 设置管线 pipeline
            passEncoder.setPipeline(gradientMapRenderPipeline);
            passEncoder.setBindGroup(0, gradientMapRenderBindGroup);
            passEncoder.setVertexBuffer(0, quadVertexBuffer);
            passEncoder.setIndexBuffer(quadIndexBuffer, 'uint32');
            
            // 设置窗口大小
            passEncoder.setViewport(0, 0, gradientCanvas.width, gradientCanvas.height, 0, 1);

            // 设置剪裁区域大小
            passEncoder.setScissorRect(0, 0, gradientCanvas.width, gradientCanvas.height);

            // 绘制网格
            passEncoder.drawIndexed(6);
            passEncoder.end();
        }

        {
            const gradientPrecisionPassDescriptor: GPURenderPassDescriptor = {
                colorAttachments: [
                    {
                        view: gradientPrecisionContext.getCurrentTexture().createView(),     
                        clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
                        loadOp: 'clear',
                        storeOp: 'store',
                    },
                ],
            };

            const passEncoder = commandEncoder.beginRenderPass(gradientPrecisionPassDescriptor);


            // 设置管线 pipeline
            passEncoder.setPipeline(gradientPrecisionMapRenderPipeline);
            passEncoder.setBindGroup(0, gradientPrecisionMapRenderBindGroup);
            passEncoder.setVertexBuffer(0, quadVertexBuffer);
            passEncoder.setIndexBuffer(quadIndexBuffer, 'uint32');
            
            // 设置窗口大小
            passEncoder.setViewport(0, 0, gradientPrecisionCanvas.width, gradientPrecisionCanvas.height, 0, 1);

            // 设置剪裁区域大小
            passEncoder.setScissorRect(0, 0, gradientPrecisionCanvas.width, gradientPrecisionCanvas.height);

            // 绘制网格
            passEncoder.drawIndexed(6);
            passEncoder.end();
        }


        /******************************************************************* 
         * 
         * 7. 再运行一下 render pipeline
         * 
         *******************************************************************/ 

        // eslint-disable-next-line no-lone-blocks
        {
            // 渲染 pass 配置描述
            const oceanRenderPassDescriptor: GPURenderPassDescriptor = {
                colorAttachments: [
                    {
                        view: oceanContext.getCurrentTexture().createView(),        
                        clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
                        loadOp: 'clear',
                        storeOp: 'store',
                    },
                ],
            };
            const passEncoder = commandEncoder.beginRenderPass(oceanRenderPassDescriptor);

            // 设置管线 pipeline
            passEncoder.setPipeline(oceanRenderPipeline);
            passEncoder.setBindGroup(0, oceanBindGroup);
            passEncoder.setVertexBuffer(0, vertexBufferDst);
            passEncoder.setIndexBuffer(indexBuffer, 'uint32');
            
            // 设置窗口大小
            passEncoder.setViewport(0, 0, canvas.width, canvas.height, 0, 1);

            // 设置剪裁区域大小
            passEncoder.setScissorRect(0, 0, canvas.width, canvas.height);

            // 绘制网格
            passEncoder.drawIndexed(indexCnt);
            passEncoder.end();
        }


        // 保存一下
        this.commandEncoder = commandEncoder;

        // 提交 绘制命令
        queue.submit([this.commandEncoder.finish()]);

    }

    // 从现存里面读取数据
    public async readBufferData(){

        let { readBuffer, webGLInstance } = this;

        // const timeStart = Date.now();
        await readBuffer.mapAsync(GPUMapMode.READ);

        // 读取全量数据
        const vertexData = readBuffer.getMappedRange();

        // 处理数据
        this.vertexData = vertexData;

        // 复制数据到 webgl
        let dst = new ArrayBuffer(vertexData.byteLength);
        new Float32Array(dst).set(new Float32Array(vertexData));
        webGLInstance.vertexData = dst
        
        // 把 vertexBufferDst 的所有权交回 GPU
        readBuffer.unmap();
        // const timeEnd = Date.now();

        // console.log(webGLInstance.vertexData.byteLength);

        // const time = timeEnd - timeStart;
        // console.warn('读取 readBuffer 的时间是:' + time + " ms");

        return true;
    }

    // 更新 动画数据
    public update(data){
        this.data = {...this.data, ...data}
    }


    // 渲染
    public render = (dt:number) => {

        let { currentFrameId } = this;

        // 更新帧率
        this.frameRate = 1000 / (dt - this.lastFrameTimestamp);
        this.pLabel.innerText = `帧率为:${this.frameRate.toFixed(0)} fps`;

        // 更新时间戳
        this.lastFrameTimestamp = dt;

        // 更新尺寸(如果要用webgpu 作为 webgl 的canvasTexture, 中途就不能更新 webgpu canvas 的尺寸)
        this.resize();

        // 更新矩阵数据
        this.updateUniformData();

        // 提交绘制命令到 queue
        this.assembleCommands();

        this.controls.update();

        // // 异步读取数据
        // (async () => {

        //     await this.readBufferData();

            // 帧加 1
            this.currentFrameId = currentFrameId + 1;

            // 当前时间秒
            this.iTime = this.lastFrameTimestamp / 1000.0;

            if(!isNaN(this.iTime)){
                this.configParams.u_time = this.iTime;
            }

            // 刷新 canvas, 再绘制一帧
            requestAnimationFrame(this.render);

        // })() 
    }
    
}