
import {
    DISP_MAP_SIZE,
    PATCH_SIZE,
    AMPLITUDE_CONSTANT,
    GRAV_ACCELERATION,
    ONE_OVER_SQRT_2,
    TWO_PI
} from './config';

// 高斯正态分布随机数
const gaussianRandom = (mean=0, stdev=1) => {

    // 从 [0,1) 转到 (0,1]
    const u = 1 - Math.random(); 
    const v = Math.random();
    const z = Math.sqrt( -2.0 * Math.log( u ) ) * Math.cos( 2.0 * Math.PI * v );

    // 转到标准的正态分布:
    return z * stdev + mean;
}

// 飞利浦波谱
/********************
 * @k (kx, kz) 为 wavenumber 波数(二维向量), 没有归一化
 * @wind (windX, windZ) 风向 (二维向量), 有归一化
 * @windV 风速, 标量
 * @A 飞利浦波普 的 A 常量 Phillips spectrum
 *******************/
const Phillips = (kx:number, kz:number, windX:number, windZ:number, windV:number) =>
{
	var L = (windV * windV) / GRAV_ACCELERATION;	// 风速 V 下 最大的波长 L
	var smallL = L / 1000.0;					    // 波长小于 最大的波长 L 千分之一的波舍弃掉

    var k2 = kx*kx + kz * kz;			    // K 的平方
    var kLen = Math.sqrt(k2);
    var knX = kx / kLen;
    var knZ = kz / kLen;
	var kdotw = knX * windX +  knZ * windZ;   // k 点积 w

	var P_h = AMPLITUDE_CONSTANT * (Math.pow(Math.E, -1.0 / (k2 * L * L))) / (k2 * k2) * (kdotw * kdotw);
	//float P_h = A * (expf(-1.0f / (k2 * L * L))) / (k2 * k2) * (kdotw * kdotw);

	if (kdotw < 0.0) {

		// 如果波的方向跟风的方向相反 
		P_h *= 0.07;
        // P_h *= 0.7;
	}

	return P_h * Math.pow(Math.E, -k2 * smallL * smallL);
}

// var p = Math.sqrt(Phillips(0.314159244, 0.314159244, -0.406138480,-0.913811505, 6.5 ));
// debugger;

// 写入 H0 = H(k) 数据
/*****************************************************
 * @wind (windX, windZ) 风向 (二维向量), 有归一化
 * @windV 风速, 标量
 * @A 飞利浦波普 的 A 常量 Phillips spectrum
 *****************************************************/
export const H_KGen = (windX:number, windZ:number, windV:number)=> {

    let dataArr = [];

    // n, m 应该在 [-N / 2, N / 2] 区间， 这里为[-256, 256]
    const start = DISP_MAP_SIZE / 2;
    const L = PATCH_SIZE;     // 20m

    // 从上到下边 -z => +z
    for(let m = 0; m <= DISP_MAP_SIZE; ++m){     // 从上到下：

        let kz = (TWO_PI * (start - m)) / L;

        // 从左到右边 -x => +x
        for (let n = 0; n <= DISP_MAP_SIZE; ++n) {

            let kx = (TWO_PI * (start - n)) / L;

            // int index = m * (DISP_MAP_SIZE + 1) + n;
            var sqrt_P_h = 0;
            if (kx != 0.0 || kz != 0.0)
            {
                sqrt_P_h = Math.sqrt(Phillips(kx, kz, windX, windZ, windV));
            }
			
            let r = sqrt_P_h * gaussianRandom() * ONE_OVER_SQRT_2;    
            let g = sqrt_P_h * gaussianRandom() * ONE_OVER_SQRT_2;    

            let color = [r, g, 0.0, 1.0];

            dataArr.push(color);
        }
    }
    var data = dataArr.flat();
    // return new Float32Array(dataArr.flat());
    // console.dir(data);
    return new Float32Array(data);
}

// 写入 频率贴图 ω 数据
export const omegaGen = () => {

    let dataArr = [];

    // n, m 应该在 [-N / 2, N / 2] 区间， 这里为[-256, 256]
    const start = DISP_MAP_SIZE / 2;
    const L = PATCH_SIZE;     // 20m

   // 从上到下边 -z => +z
    for(let m = 0; m <= DISP_MAP_SIZE; ++m){     

        let kz = (TWO_PI * (start - m)) / L;

        // 从左到右边 -x => +x
        for (let n = 0; n <= DISP_MAP_SIZE; ++n) {

            let kx = (TWO_PI * (start - n)) / L;
            let kLen = Math.sqrt(kx * kx + kz * kz);

            // ω(k)^2 = gk => ω = Math.sqrt(gk)
            let omega = Math.sqrt(GRAV_ACCELERATION * kLen);    
            let color = [omega, 0.0, 0.0, 1.0];

            dataArr.push(color);
        }
    }

    return new Float32Array(dataArr.flat());
}

// 测试数据 rgba32float
export const rgba32floatGen = () => {

    let dataArr = [];

    // 从左上到右下由 黑 => 黄
    for(let i = 0; i<512; i++){     // 从上到下：
        for(let j = 0; j<512; j++)  // 从左到右
        {
            let r = i / 511;    // 从上到下：黑 => 红
            let g = j / 511;    // 从左到右：黑 => 绿
            // let b = ( i + j ) / 1022;
            let b = 0.0;
            let color = [r, g, b, 1.0];
            dataArr.push(color);
        }
    }

    return new Float32Array(dataArr.flat());
}

// 测试数据 rgba32float
export const rgba32floatGen1 = () => {

    let dataArr = [];

    // 从左上到右下由 黑 => 黄
    for(let i = 0; i<512; i++){     // 从上到下：
        for(let j = 0; j<512; j++)  // 从左到右
        {
            let r = (511 - i) / 511;    // 从上到下：黑 => 红
            let g = (511 - j) / 511;    // 从左到右：黑 => 绿
            // let b = ( i + j ) / 1022;
            let b = 0;
            let color = [r, g, b, 1.0];
            dataArr.push(color);
        }
    }

    return new Float32Array(dataArr.flat());
}

// 256 bit reverse index  3 => 011 => 1100 => 
export const bitfieldReverse256 =() =>{

    let dataArr = [];

    // 8 位二进制数
    for(let i = 0; i < 256; i++)
    {
        // 从高到低
        let bit7 = (i & 1) << 7;    // 倒数第一位放到前面第一位
        let bit6 = (i & 2) << 5;    // 倒数第二位放到前面第二位
        let bit5 = (i & 4) << 3;    // 倒数第三位放到前面第三位
        let bit4 = (i & 8) << 1;    // 倒数第四位放到前面第四位
        let bit3 = (i & 16) >> 1;   // 倒数第五位放到前面第五位
        let bit2 = (i & 32) >> 3;   // 倒数第六位放到前面第六位
        let bit1 = (i & 64) >> 5;   // 倒数第七位放到前面第七位
        let bit0 = (i & 128) >> 7;  // 倒数第八位放到前面第八位

        let val = bit7 | bit6 | bit5 | bit4 | bit3 | bit2 | bit1 | bit0;

        let color = [val, 0, 0, 1.0];

        dataArr.push(color);
    }

    return new Float32Array(dataArr.flat());
}

// 512 bit reverse index
export const bitfieldReverse512 =() =>{

    let dataArr = [];

    // 9 位二进制数
    for(let i = 0; i<512; i++)
    {
        // 从高到低
        let bit8 = (i & 1) << 8;    // 倒数第一位放到前面第一位
        let bit7 = (i & 2) << 6;    // 倒数第二位放到前面第二位
        let bit6 = (i & 4) << 4;    // 倒数第三位放到前面第三位
        let bit5 = (i & 8) << 2;    // 倒数第四位放到前面第四位
        let bit4 = (i & 16);        // 倒数第五位放到前面第五位
        let bit3 = (i & 32) >> 2;   // 倒数第六位放到前面第六位
        let bit2 = (i & 64) >> 4;   // 倒数第七位放到前面第七位
        let bit1 = (i & 128) >> 6;  // 倒数第八位放到前面第八位
        let bit0 = (i & 256) >> 8;  // 倒数第九位放到前面第九位

        let val =  bit8 |bit7 | bit6 | bit5 | bit4 | bit3 | bit2 | bit1 | bit0;

        let color = [val, 0, 0, 1.0];
        dataArr.push(color);
    }

    return new Float32Array(dataArr.flat());
}
