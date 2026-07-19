# Anya Pixel-Art Shader Variant

这是基于原始三角网格光线追踪器新增的艺术化版本，原始文件 `anya_final_shadertoy.glsl` 和 `preview.html` 保持不变。

## 新增文件

- `anya_final_pixel_art_shadertoy.glsl`：提交到 Shadertoy 的艺术化 Fragment Shader。
- `preview_pixel_art.html`：与 Shadertoy 自定义纹理扩展保持一致的本地预览。

四个 Channel 继续共用 `channels/` 中的原始数据：

| Channel | 文件 | Filter | Wrap | VFlip |
| --- | --- | --- | --- | --- |
| iChannel0 | `anya_bvh.png` | Nearest | Clamp | Off |
| iChannel1 | `anya_vertices.png` | Nearest | Clamp | Off |
| iChannel2 | `anya_triangles.png` | Nearest | Clamp | Off |
| iChannel3 | `anya_albedo.png` | Mipmap | Repeat | Off |

## 艺术处理

- 显式 sRGB 到线性颜色转换，统一本地预览和 Shadertoy 的颜色。
- 距离相关 Mipmap，减少头发上的摩尔纹。
- ACES 色调映射与重新平衡的环境光、主光和补光。
- 3×3 像素网格射线。
- 五档卡通光照。
- 4×4 Bayer 有序抖动与 12 档颜色量化。
- 基于法线和视线夹角的低成本轮廓强调。

把 Shader 顶部的：

```glsl
#define PIXEL_ART_MODE 1
```

改为 `0`，可以关闭像素网格、分级光照、抖动和轮廓，只保留颜色校正、Mipmap 和新光照，用于课堂上的前后对比。

## 本地预览

```bash
python3 -m http.server 8765 --bind 127.0.0.1
```

打开：

`http://127.0.0.1:8765/preview_pixel_art.html`

编译检查：

```bash
./compile.sh anya_final_pixel_art_shadertoy.glsl
QUALITY=0 ./compile.sh anya_final_pixel_art_shadertoy.glsl
```
