import React from "react";
import DatGui, {
  DatColor,
  DatNumber,
  DatString,
  DatFolder,
} from "react-dat-gui";
import * as THREE from "three";
import "./dat-gui-css/index.css";
import "./OceanWave.css";
import WebGPU_DLM from "./webgpu/waveDFT/webgpu";

let myWebGPU = new WebGPU_DLM();

// https://medium.com/@fa19-bcs-087/component-life-cycle-mounting-unmounting-updating-ca8c82bafbbe
class OceanWave extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      webgpuMessage: "webGPU DFT 波浪 和 FFT 波浪",
      webglMessage: "WebGL 渲染",
      count: 1,
      data: {
        // 太阳方位和颜色
        u_sunColorHex: "#ffff78", // 太阳颜色
        c_sunColor: "1.000,1.000,0.471", // 太阳颜色
        u_sunColor: [1.0, 1.0, 0.471], // 太阳颜色
        u_sunAngleXZ: 222,
        u_sunAngleY: 18,
        c_sunDir: "-0.707,0.309,-0.636", // 太阳方向
        u_sunDir: [-0.707, 0.309, -0.636], // 太阳方向

        // 风速和风向
        u_windSpeed: 6.5,
        u_windAngleXZ: 246, // { -0.4067f, -0.9135f }
        c_windDir: "-0.4067, 0.0, -0.9135", // 风方向
        n_windDir: [-0.4067, 0.0, -0.9135], // 风方向

        // 重力
        u_G: 9.81, // 重力加速度
        c_L: 4.3, // 风速 u_windSpeed 下 最大波长

        // Phillips AMPLITUDE_CONSTANT 参数  A, 风与 K 的夹角 cos指数次方 N
        u_A: 0.00045, // 0.0081

        // 海水颜色
        u_oceanColorHex: "#010508", // 海水颜色
        c_oceanColor: "0.004,0.020,0.031", // 海水颜色
        u_oceanColor: [0.004, 0.02, 0.031], // 海水颜色

        // 泡沫
        u_foamScale: 0.5,
        u_foamStatus: true,

        // 波是否带水平位移
        u_choppy: true,
        u_choppyDxScale: 1.0,
        u_choppyDzScale: 1.0,
        u_yScale: 1.0,

        // 是否显示 grid
        wireframe: false,
      },
    };
  }
  shouldComponentUpdate(nextProps, nextState) {
    if (this.state.count !== nextState.count) {
      return true;
    }
    return false;
  }
  // componentWillMount(){
  // 	console.log("WillMount");
  // }

  // 参考 WebGPU 大图
  async componentDidMount() {
    // 更新标题
    document.title = this.state.webgpuMessage;
    // let webglCanvas = document.getElementById("webglCanvas");
    // let webglP = document.getElementById('webglFrameRate');
    let webgpuCanvas = document.getElementById("webgpuCanvas");
    let webgpuP = document.getElementById("webgpuFrameRate");

    // webglCanvas.width = webglCanvas.clientWidth ;
    // webglCanvas.height = webglCanvas.clientHeight ;
    webgpuCanvas.width = webgpuCanvas.clientWidth;
    webgpuCanvas.height = webgpuCanvas.clientHeight;

    let displacementCanvas = document.getElementById("displacementCanvas");
    let displacementPrecisionCanvas = document.getElementById(
      "displacementPrecisionCanvas"
    );
    let gradientCanvas = document.getElementById("gradientCanvas");
    let gradientPrecisionCanvas = document.getElementById(
      "gradientPrecisionCanvas"
    );

    displacementCanvas.width = displacementCanvas.clientWidth;
    gradientCanvas.width = gradientCanvas.clientWidth;

    // 初始化 画布和帧率 元素 p

    /**************************************************************************
     *
     *  1. 初始化
     *
     * ***********************************************************************/
    let container = document.getElementById("canvasContainer");

    /**************************************************************************
     *
     *  2. 初始化 webgpu 实例
     *
     * ***********************************************************************/
    myWebGPU.init(
      webgpuCanvas,
      displacementCanvas,
      displacementPrecisionCanvas,
      gradientCanvas,
      gradientPrecisionCanvas,
      webgpuP,
      container
    );

    // 初始化 webgpu
    if (await myWebGPU.initWebGPU()) {
      // 先加载图片
      if (await myWebGPU.loadTextures()) {
        // 创建贴图和波普
        myWebGPU.createTextureAndSpectrum();

        // 2.1 初始化 pipeline
        myWebGPU.initShaderModuleAndPipeline();

        // 2.2 初始化数据
        // myWebGPU.initDataBufferAndBindGroup(myMeshGrid.vertexArrayBuffer, myMeshGrid.indexLineArrayBuffer, myMeshGrid.pointCnt, myMeshGrid.indexLineCnt, myMeshGrid.rowCnt, myMeshGrid.colCnt);
        // myWebGPU.initDataBufferAndBindGroup(myMeshGrid.vertexArrayBuffer, myMeshGrid.indexTriangleArrayBuffer, myMeshGrid.pointCnt, myMeshGrid.indexTriangleCnt, myMeshGrid.rowCnt, myMeshGrid.colCnt);
        myWebGPU.initDataBufferAndBindGroup();

        // 2.3 同步数据
        // myWebGPU.update(this.state.data);

        myWebGPU.resize();

        // 2.4 渲染
        myWebGPU.render();

        // console.group("myWebGPU");
        // console.dir(myWebGPU);
        // console.groupEnd();

        window.myWebGPU = myWebGPU;
      }
    }
  }

  componentDidUpdate() {
    // console.log("DidUpdate");
    // console.log(this.state.count);
  }

  // Update current state with changes from controls
  handleUpdate = (newData) => {
    this.setState((prevState) => {
      var PI = Math.PI;

      // 计算太阳、海水颜色
      var sunColor = new THREE.Color(newData.u_sunColorHex);
      newData.c_sunColor =
        sunColor.r.toFixed(3) +
        "," +
        sunColor.g.toFixed(3) +
        "," +
        sunColor.b.toFixed(3);
      newData.u_sunColor = [sunColor.r, sunColor.g, sunColor.b];
      var oceanColor = new THREE.Color(newData.u_oceanColorHex);
      newData.c_oceanColor =
        oceanColor.r.toFixed(3) +
        "," +
        oceanColor.g.toFixed(3) +
        "," +
        oceanColor.b.toFixed(3);
      newData.u_oceanColor = [oceanColor.r, oceanColor.g, oceanColor.b];

      // 计算太阳方向
      var radian = (newData.u_sunAngleY / 180) * PI;
      var sunY = Math.sin(radian);
      var sunXZR = Math.cos(radian);
      radian = (newData.u_sunAngleXZ / 180) * PI;
      var sunX = sunXZR * Math.cos(radian);
      var sunZ = sunXZR * Math.sin(radian);
      newData.c_sunDir =
        sunX.toFixed(3) + "," + sunY.toFixed(3) + "," + sunZ.toFixed(3);
      newData.u_sunDir = [sunX, sunY, sunZ];

      // 计算风方向
      radian = (newData.u_windAngleXZ / 180) * PI;
      var windX = Math.cos(radian);
      var windZ = Math.sin(radian);
      newData.c_windDir = windX.toFixed(3) + ", 0.0, " + windZ.toFixed(3);
      newData.n_windDir = [windX, 0.0, windZ];

      // 计算最大波长
      newData.c_L = Number(
        ((newData.u_windSpeed * newData.u_windSpeed) / newData.u_G).toFixed(3)
      );
      let data = { ...prevState.data, ...newData }; // creating copy of state variable jasper

      return { data };
    });

    // 强制更新
    this.forceUpdate();

    // 更新 webgpu 里面的数据
    myWebGPU.update(newData);
  };

  render() {
    const { data } = this.state;
    return (
      <div className="App">
        <div>
          <h2 id="webgpuTitle" className="Header-h2">
            {this.state.webgpuMessage}{" "}
          </h2>
          {/* <h2 id = "webglTitle" className="Header-h2">{this.state.webglMessage} </h2> */}
          <p id="webgpuFrameRate">{this.state.count} fps</p>
          {/* <p id="webglFrameRate">{this.state.count} fps</p> */}
          <div id="canvasContainer">
            <canvas id="webgpuCanvas"></canvas>
            {/* <canvas id="webglCanvas"></canvas> */}
          </div>

          <div id="textureContainer">
            <canvas id="displacementCanvas"></canvas>
            <canvas id="displacementPrecisionCanvas"></canvas>
            <canvas id="gradientCanvas"></canvas>
            <canvas id="gradientPrecisionCanvas"></canvas>
          </div>
        </div>

        <DatGui data={data} onUpdate={this.handleUpdate}>
          <DatFolder title="波形参数设置">
            <DatColor path="u_sunColorHex" label="太阳颜色" />
            <DatString path="c_sunColor" label="太阳颜色" />

            <DatNumber
              path="u_sunAngleXZ"
              label="太阳水平旋转角°"
              min={0}
              max={360}
              step={2}
            />
            <DatNumber
              path="u_sunAngleY"
              label="太阳垂直升角°"
              min={0}
              max={90}
              step={1}
            />
            <DatString path="c_sunDir" label="太阳方向" />
            <DatColor path="u_oceanColorHex" label="海水颜色" />
            <DatString path="c_oceanColor" label="海水颜色" />
          </DatFolder>
        </DatGui>
      </div>
    );
  }
}

export default OceanWave;
