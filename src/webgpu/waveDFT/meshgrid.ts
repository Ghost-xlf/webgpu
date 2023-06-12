export default class MeshGrid{

    // dom 元素
    public rowCnt:number;                           // 多少行顶点
    public colCnt:number;                           // 多少列顶点
    public pointCnt:number;                         // 顶点个数
    public indexLineCnt:number;                     // 线顶点索引个数
    public indexTriangleCnt:number;                 // 三角形顶点索引个数
    public step:number;                             // 步长
    public vertexArrayBuffer:ArrayBuffer;           // 顶点 position 数据 
    public indexLineArrayBuffer:ArrayBuffer;         // 线的顶点 index 数据 
    public indexTriangleArrayBuffer:ArrayBuffer;         // 三角面的顶点 index 数据 

    // 构建函数
    public constructor(rowCnt:number, colCnt:number, step:number){

        this.rowCnt = rowCnt;               // 2n 比如 512 或 1024
        this.colCnt = colCnt;               // 2n 比如 512 或 1024
        this.pointCnt = rowCnt * colCnt;    // 顶点个数 比如 512 * 512 = 262144 个

        // 半长和步长
        const halfX = (colCnt - 1) * 0.5 * step;
        const halfZ = (rowCnt - 1) * 0.5 * step;
        this.step = step;

        // 计算 arraybuffer 的长度
        const posLen = this.pointCnt * 4;        // 每个顶点由 x,y,z,w 4个分量组成

        // 计算线条的索引个数
        const indexLineLen = 2 * ((rowCnt - 1) * colCnt + rowCnt * (colCnt - 1));  // 每条线段有2个点组成
        this.indexLineCnt = indexLineLen;

        // 计算三角形的索引个数
        const indexTriangleLen = 6 * (rowCnt - 1) * (colCnt - 1) ;  // 每个四边形由6个点组成
        this.indexTriangleCnt = indexTriangleLen;

        // 创建 arraybuffer
        const positions = new Float32Array(posLen);
        const lineIndices = new Uint32Array(indexLineLen);
        const triIndices = new Uint32Array(indexTriangleLen);

        // 循环赋值 position
        for(let i = 0; i < rowCnt; i++){        // z 方向
            for(let j = 0; j < colCnt; j++){    // x 方向

                const index = (i * colCnt + j) * 4;
                // const x = -halfX + j * step;
                const x = j * step;
                const y = 0;
                // const z = -halfZ + i * step;
                const z = i * step;
                const w = 0;
                positions[index] = x;
                positions[index + 1] = y;
                positions[index + 2] = z;
                positions[index + 3] = w;
            }
        }

        /*****************************************************************
            
                                建线段索引

        *****************************************************************/
        let curIndex = 0;
        for(let i = 0; i < rowCnt; i++)         // 512 行
        {
            for(let j = 0; j < colCnt -1 ; j++) // 511 个线段
            {
                let index0 = i * colCnt + j;
                let index1 = index0 + 1;

                // 加入一条线段
                lineIndices[curIndex] = index0;
                lineIndices[curIndex + 1] = index1;
                curIndex +=2;
            }
        }

        // 循环赋值 index 列线段 512 * 512 个点； 每行有511条线段,总共有512列
        for(let i = 0; i < colCnt; i++)         // 512 列
        {
            for(let j = 0; j < rowCnt -1 ; j++) // 511 个线段
            {
                let index0 = i + j * colCnt;
                let index1 = index0 + colCnt;

                // 加入一条线段
                lineIndices[curIndex] = index0;
                lineIndices[curIndex + 1] = index1;
                curIndex +=2;
            }
        }

        /*****************************************************************
            
                                建三角面索引

        *****************************************************************/
        curIndex = 0;
        for(let i = 0; i < rowCnt - 1; i++)         // z 方向 512 行点, 511 行三角形
        {
            for(let j = 0; j < colCnt - 1 ; j++)     // x 方向 512 列点, 511 列三角形
            {
                let p0index = rowCnt * i + j;
                let p1index = rowCnt * i + (j + 1);
                let p2index = rowCnt * (i + 1) + (j + 1);
                let p3index = rowCnt * (i + 1) + j;

                // 三角形1 p0 p2 p1
                triIndices[curIndex]     = p0index;
                triIndices[curIndex + 1] = p2index;
                triIndices[curIndex + 2] = p1index;

                // 三角形1 p0 p3 p2
                triIndices[curIndex + 3] = p0index;
                triIndices[curIndex + 4] = p3index;
                triIndices[curIndex + 5] = p2index;

                curIndex +=6;  // 每个 quad 2个三角形 6 个顶点
            }
        }

        // 创建 arraybuffer
        this.vertexArrayBuffer = positions;
        this.indexLineArrayBuffer = lineIndices;
        this.indexTriangleArrayBuffer = triIndices;
    }
}