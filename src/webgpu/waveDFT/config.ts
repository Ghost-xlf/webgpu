const DISP_MAP_SIZE = 256;                              // 位移贴图的尺寸
const PATCH_SIZE = 10.0;                                // 一个 patch 的尺寸 m
const MESH_SIZE = 128;                                  // 用于渲染的 mesh grid 等份数量
const WIND_SPEED = 10;                                 // 风速
const WIND_X = -0.406138480;                            // 风向X
const WIND_Z = -0.913811505;                            // 风向X
const tilesize = PATCH_SIZE / MESH_SIZE;                // 单个网格格子尺寸
const AMPLITUDE_CONSTANT = (0.45 * 1e-3);		        // Phillips spectrum 飞利浦波普 的 A 常量 Phillips spectrum
const GRAV_ACCELERATION = 9.81;                         // 引力加速度
const ONE_OVER_SQRT_2 = Math.SQRT1_2;                   // 1 / sqrt(2)
const TWO_PI = 2 * Math.PI;                             // 2 * PI

export {
    DISP_MAP_SIZE,
    PATCH_SIZE,
    MESH_SIZE,
    WIND_SPEED,
    WIND_X,
    WIND_Z,
    tilesize,
    AMPLITUDE_CONSTANT,
    GRAV_ACCELERATION,
    ONE_OVER_SQRT_2,
    TWO_PI
}