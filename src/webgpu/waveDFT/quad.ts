// 四边形
export default class Quad{

    public vertexArrayBuffer:ArrayBuffer;           // 顶点 position 数据 
    public indexArrayBuffer:ArrayBuffer;            // 线顶点 index 数据 

    // 构建函数
    public constructor(){

        // 四个点
        // float4 position, float2 uv,
        const p0 = [-1, -1, 0, 0, 0, 0];
        const p1 = [1,  -1, 0, 0, 1, 0];
        const p2 = [1,  1,  0, 0, 1, 1];
        const p3 = [-1, 1,  0, 0, 0, 1];

        // 创建 arraybuffer
        const positions = new Float32Array([...p0, ...p1, ...p2, ...p3]);
        const indices = new Uint32Array([0, 1, 2, 0, 2, 3]);

        // 创建 arraybuffer
        this.vertexArrayBuffer = positions;
        this.indexArrayBuffer = indices;
    }
}