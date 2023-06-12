
import * as THREE from 'three';
// import { vertStr, fragStr } from './glsl/meshgridGLSL';
import { MapControls } from './examples/jsm/controls/OrbitControls';
import { vertStr, fragStr } from './glsl/oceanGL'
import {     
    DISP_MAP_SIZE,
    PATCH_SIZE, 
} from './config'

export default class WebGL_DLM{

    // dom 元素
    public container:HTMLElement;           // 用于 mapcontrol 绑定
    public pLabel:HTMLElement;              // 用于 显示帧率
    public canvas:HTMLCanvasElement;        // webgl 画布
    public displacementCanvas:HTMLCanvasElement;            // webgpu 画布(用作 webgl 的canvasTexture)
    public displacementPrecisionCanvas:HTMLCanvasElement;   // webgpu 画布(用作 webgl 的canvasTexture, 存储 displacementCanvas 留下的不足 1/255 的小数部分)
    public gradientCanvas:HTMLCanvasElement;                // webgpu 画布(用作 webgl 的canvasTexture)
    public gradientPrecisionCanvas:HTMLCanvasElement;       // webgpu 画布(用作 webgl 的canvasTexture, 存储 gradientCanvas 留下的不足 1/255 的小数部分)
    
    public context:WebGL2RenderingContext;  // webgl 上下文

    // three 
    public camera:any;
    public scene:any;
    public program:WebGLProgram;
    public renderer:any;
    public controls:any;

    // geometry 和材质
    public geometry:any;
    public material:any;
    public lines:any;
    public mesh:any;

    // float32array 顶点数据
    public vertexData:ArrayBuffer;              // 从 webGPU readBuffer 取出来的 float32array 顶点数据

    // 帧率
    private lastFrameTimestamp:number = 0;
    private frameRate:number;
    private currentFrameId:number = 0;     // 当前帧

    // 水参数
    public data:any;

    // 当前时间和当前分辨率
    public iTime:number = 0.0;
    public iResolution:any;

    // 构建函数
    public constructor(){

        this.data = {

        }
    }

    public init(canvas:HTMLCanvasElement, p:HTMLElement, container:HTMLElement, displacementCanvas:HTMLCanvasElement, displacementPrecisionCanvas:HTMLCanvasElement, gradientCanvas:HTMLCanvasElement, gradientPrecisionCanvas:HTMLCanvasElement){

        // 画布
        this.canvas = canvas;  
        this.displacementCanvas = displacementCanvas;
        this.displacementPrecisionCanvas = displacementPrecisionCanvas;
        this.gradientCanvas = gradientCanvas;
        this.gradientPrecisionCanvas = gradientPrecisionCanvas;

        // 标签
        this.pLabel = p;

        // 容器
        this.container = container;

        // 渲染器
        this.renderer = new THREE.WebGLRenderer({   canvas: canvas,
                                                    antialias: true });
    }

    // 初始化 threejs
    public initThree(){

        // 设置透视相机
        // 高宽比
        const aspect = 0.5 * window.innerWidth / window.innerHeight;
        this.camera = new THREE.PerspectiveCamera( 60, aspect, 0.1, 2000 );

        // 相机位置
		this.camera.position.z = 20;
        this.camera.position.y = 5;

		this.scene = new THREE.Scene();
		this.scene.background = new THREE.Color( 0x353535 );

        this.scene.background = new THREE.CubeTextureLoader()
            .setPath( 'wave/' )
            .load( [
                '0_positive_x.jpg',
                '1_negative_x.jpg',
                '2_positive_y.jpg',
                '3_negative_y.jpg',
                '4_positive_z.jpg',
                '5_negative_z.jpg'
            ] );

        // this.controls = new MapControls( this.camera, this.renderer.domElement );
        this.controls = new MapControls( this.camera, this.container );
        // eslint-disable-next-line no-lone-blocks
        {
            this.controls.screenSpacePanning = false;
            this.controls.panSpeed = 5;
            // controls.rotateSpeed = 0.4;
            this.controls.zoomSpeed = 2;
            
            // controls.enabled = false;   // 启用或禁用
            // controls.enableDamping = true;
            // controls.mouseButtons = {
            // 	LEFT: THREE.MOUSE.PAN,
            // 	MIDDLE: THREE.MOUSE.DOLLY,
            // 	RIGHT: THREE.MOUSE.ROTATE
            // }
            this.controls.minDistance = -100;
            this.controls.maxDistance = 1000;
            this.controls.maxPolarAngle = Math.PI * 0.50 ;
            
            this.controls.object.position.set(0, 0, 0);
            this.controls.target = new THREE.Vector3(7.9, 3.811, 7.6);
            this.controls.update();
        }
    }

    // 初始化 grid 数据
    public initGrid(vertices:ArrayBuffer, indices:ArrayBuffer){

        let { scene } = this;
        this.geometry = new THREE.BufferGeometry();

        // 因为 compute shader 里面 传 vec3 也是占用 vec4 的空间
        this.geometry.setAttribute( 'position', new THREE.BufferAttribute( vertices, 4 ) );
        this.geometry.setIndex(new THREE.BufferAttribute(indices , 1));

        this.material = new THREE.LineBasicMaterial( { color: 0x00ffff, vertexColors: false } );
        this.lines = new THREE.LineSegments( this.geometry, this.material );
        this.lines.material.needsUpdate = true;

        // 把 mesh 添加到场景
        scene.add(this.lines);

        // 添加坐标轴
        let axesHelper = new THREE.AxesHelper( 500 );
        axesHelper.position.y = 0.1;
        scene.add(axesHelper);

        // 添加 grid
        const size = 2000;
        const divisions = 2;

        const gridHelper = new THREE.GridHelper( size, divisions );
        scene.add( gridHelper );
    }

    // 初始化 三角面数据
    public initMesh(vertices:ArrayBuffer, indices:ArrayBuffer){

        this.resize();

        let { scene, displacementCanvas, displacementPrecisionCanvas, gradientCanvas, gradientPrecisionCanvas} = this;
        this.geometry = new THREE.BufferGeometry();

        // 因为 compute shader 里面 传 vec3 也是占用 vec4 的空间
        this.geometry.setAttribute( 'position', new THREE.BufferAttribute( vertices, 4 ) );
        this.geometry.setIndex(new THREE.BufferAttribute(indices , 1));

        const material = new THREE.ShaderMaterial( {

            uniforms: {
                u_time: { value: 1.0 },
                
                u_sunColor: { value: new THREE.Vector3(1.000, 1.000, 0.471) },
                u_sunDir: { value: new THREE.Vector3(0.0, 0.174, -0.985) },
                u_oceanColor: { value: new THREE.Vector3(0.004, 0.020, 0.031) },
                u_choppyDxScale:{ value: 1.0 },
                u_choppyDzScale:{ value: 1.0 },
                u_yScale:{ value: 1.0 },
                u_foamScale:{value: 0.5},
                u_foamStatus:{value: 1.0},
                u_patch_size:{value: PATCH_SIZE },
                u_disp_map_size:{value: DISP_MAP_SIZE },

                envmapTex:{ type: "t", value: new THREE.CubeTextureLoader()
                .setPath( 'wave/' )
                .load( [
                    '0_positive_x.jpg',
                    '1_negative_x.jpg',
                    '2_positive_y.jpg',
                    '3_negative_y.jpg',
                    '4_positive_z.jpg',
                    '5_negative_z.jpg'
                ] ) },

                // canvasTex:  new THREE.CanvasTexture(webgpuCanvas),
                // (如果要用webgpu 作为 webgl 的canvasTexture, 中途就不能更新webgpu canvas 的尺寸)
                // canvasTex:  { type: "t", value: new THREE.CanvasTexture(displacementCanvas)},
                // displacementTex: { type: "t", value: new THREE.TextureLoader().load("wave/displacement.jpg" ) },
                displacementTex: { type: "t", value: new THREE.CanvasTexture(displacementCanvas) },
                displacementPrecisionTex: { type: "t", value: new THREE.CanvasTexture(displacementPrecisionCanvas) },
                // gradientTex: { type: "t", value: new THREE.TextureLoader().load("wave/gradient.png") },
                gradientTex: { type: "t", value: new THREE.CanvasTexture(gradientCanvas) },
                gradientPrecisionTex: { type: "t", value: new THREE.CanvasTexture(gradientPrecisionCanvas) },
                perlinTex: { type: "t", value: new THREE.TextureLoader().load("wave/perlin_noise.png") }
            },

            glslVersion:THREE.GLSL3,
            wireframe:false,
            vertexShader: vertStr,
            fragmentShader: fragStr
        } );

        this.material = material;


        this.material.uniforms.displacementTex.value.wrapS = THREE.RepeatWrapping;
        this.material.uniforms.displacementTex.value.wrapT  = THREE.RepeatWrapping;
        this.material.uniforms.displacementTex.value.magFilter = THREE.LinearFilter;
        this.material.uniforms.displacementTex.value.minFilter = THREE.LinearFilter;
        this.material.uniforms.displacementTex.value.generateMipmaps = false;
        this.material.uniforms.displacementTex.value.repeat.set( 2000, 2000 );
        this.material.uniforms.displacementTex.value.flipY = true;
        this.material.uniforms.displacementTex.value.needsUpdate = true;

        this.material.uniforms.displacementPrecisionTex.value.wrapS = THREE.RepeatWrapping;
        this.material.uniforms.displacementPrecisionTex.value.wrapT  = THREE.RepeatWrapping;
        this.material.uniforms.displacementPrecisionTex.value.magFilter = THREE.LinearFilter;
        this.material.uniforms.displacementPrecisionTex.value.minFilter = THREE.LinearFilter;
        this.material.uniforms.displacementPrecisionTex.value.generateMipmaps = false;
        this.material.uniforms.displacementPrecisionTex.value.repeat.set( 2000, 2000 );
        this.material.uniforms.displacementPrecisionTex.value.flipY = true;
        this.material.uniforms.displacementPrecisionTex.value.needsUpdate = true;

        this.material.uniforms.gradientTex.value.wrapS = THREE.RepeatWrapping;
        this.material.uniforms.gradientTex.value.wrapT  = THREE.RepeatWrapping;
        this.material.uniforms.gradientTex.value.magFilter = THREE.LinearFilter;
        this.material.uniforms.gradientTex.value.minFilter = THREE.LinearFilter;
        this.material.uniforms.gradientTex.value.generateMipmaps = false;
        this.material.uniforms.gradientTex.value.repeat.set( 2000, 2000 );
        this.material.uniforms.gradientTex.value.flipY = true;
        this.material.uniforms.gradientTex.value.needsUpdate = true;

        this.material.uniforms.gradientPrecisionTex.value.wrapS = THREE.RepeatWrapping;
        this.material.uniforms.gradientPrecisionTex.value.wrapT  = THREE.RepeatWrapping;
        this.material.uniforms.gradientPrecisionTex.value.magFilter = THREE.LinearFilter;
        this.material.uniforms.gradientPrecisionTex.value.minFilter = THREE.LinearFilter;
        this.material.uniforms.gradientPrecisionTex.value.generateMipmaps = false;
        this.material.uniforms.gradientPrecisionTex.value.repeat.set( 2000, 2000 );
        this.material.uniforms.gradientPrecisionTex.value.flipY = true;
        this.material.uniforms.gradientPrecisionTex.value.needsUpdate = true;

        this.material.uniforms.perlinTex.value.wrapS = THREE.RepeatWrapping;
        this.material.uniforms.perlinTex.value.wrapT  = THREE.RepeatWrapping;
        this.material.uniforms.perlinTex.value.magFilter = THREE.LinearFilter;
        this.material.uniforms.perlinTex.value.minFilter = THREE.LinearFilter;
        this.material.uniforms.perlinTex.value.generateMipmaps = false;
        this.material.uniforms.perlinTex.value.repeat.set( 2000, 2000 );
        this.material.uniforms.perlinTex.value.flipY = true;

        this.mesh = new THREE.Mesh( this.geometry, this.material );
        this.mesh.material.needsUpdate = true;

        // 把 mesh 添加到场景
        scene.add(this.mesh);

        // 添加坐标轴
        let axesHelper = new THREE.AxesHelper( 500 );
        axesHelper.position.y = 0.1;
        scene.add(axesHelper);

        // 添加 grid
        const size = 2000;
        const divisions = 2;

        const gridHelper = new THREE.GridHelper( size, divisions );
        scene.add( gridHelper );
    }

    // resize 大小
    public resize():void{

        let { canvas } = this;

        // 设置尺寸
        canvas.width = canvas.clientWidth ;
        canvas.height = canvas.clientHeight ;

        this.camera.aspect = 0.5 * window.innerWidth / window.innerHeight;
        this.camera.updateProjectionMatrix();

        this.renderer.setSize( 0.5 * window.innerWidth , window.innerHeight );

        // 设置分辨率
        this.iResolution = {
                                x:0.5 * window.innerWidth,
                                y:window.innerHeight,
                            };
    }

    // 更新 动画数据
    public update(data){
        this.data = {...this.data, ...data}
    }

    // 渲染
    public render = (dt:number) => {

        let { vertexData } = this;

        // 更新帧率
        this.frameRate = 1000 / (dt - this.lastFrameTimestamp);
        this.pLabel.innerText = `帧率为:${this.frameRate.toFixed(0)} fps`;

        // 更新时间戳
        this.lastFrameTimestamp = dt;

        // 更新尺寸
        this.resize();

        // 更新时间
        // let time = dt / 1000.0;
        let time = dt / 1000.0;
        if(!isNaN(time)){
            this.iTime = time;

            // this.lines.geometry.attributes.position.array = vertexData;
            // this.lines.geometry.attributes.position.needsUpdate = true;

            // 更新材质
            if(this.data.wireframe)
            {
                this.mesh.material.wireframe = true;
            }
            else{
                this.mesh.material.wireframe = false;
            }

            // 更新 uniform 
            this.mesh.material.uniforms.u_sunColor.value = new THREE.Vector3(...this.data.u_sunColor);
            this.mesh.material.uniforms.u_oceanColor.value = new THREE.Vector3(...this.data.u_oceanColor);
            this.mesh.material.uniforms.u_sunDir.value = new THREE.Vector3(...this.data.u_sunDir);
            this.mesh.material.uniforms.u_choppyDxScale.value = this.data.u_choppyDxScale;
            this.mesh.material.uniforms.u_choppyDzScale.value = this.data.u_choppyDzScale;
            this.mesh.material.uniforms.u_yScale.value = this.data.u_yScale;
            
            this.mesh.material.uniforms.u_foamScale.value = this.data.u_foamScale;
            this.mesh.material.uniforms.u_foamStatus.value = Number(this.data.u_foamStatus);
            
            this.mesh.material.uniforms.displacementTex.value.needsUpdate = true;
            this.mesh.material.uniforms.displacementPrecisionTex.value.needsUpdate = true;
            this.mesh.material.uniforms.gradientTex.value.needsUpdate = true;
            this.mesh.material.uniforms.gradientPrecisionTex.value.needsUpdate = true;

            this.mesh.material.needsUpdate = true;

            

            // 更新 geometry
            this.mesh.geometry.attributes.position.array = vertexData;
            this.mesh.geometry.attributes.position.needsUpdate = true;
        }

        // 渲染
        this.renderer.render( this.scene, this.camera );

        // 刷新 canvas, 再绘制一帧
        requestAnimationFrame(this.render);
    }
    
}