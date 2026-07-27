# Anya Final Shadertoy

这是从 `anya_final.glb` 重新生成的最终版本。Fragment Shader 直接读取真实三角网格、法线、UV 和底色贴图，并使用 BVH 在每个像素执行光线追踪；不使用旧的程序化 SDF 近似模型。

## 在线作品网站

GitHub Pages：<https://rickyyy10.github.io/anya-shader/>

网站首页提供项目说明、三种实时渲染版本、Channel 数据结构说明和源码入口。

## 最终文件

- `anya_final.glb`：唯一模型源文件。
- `anya_final_shadertoy.glsl`：提交的 Fragment Shader。
- `index.html`：GitHub Pages 作品首页。
- `preview.html`：本地 WebGL2 / Shadertoy 兼容预览器。
- `preview_front.jpg`：网站首页使用的最终正面效果截图。
- `channels/`：四个 iChannel 文件和 `manifest.json`。
- `tools/glb_to_shadertoy.py`：重新生成 Channels 的转换工具。
- `compile.sh`：本地 GLSL 语法检查。

## Channel 设置

| Channel | 文件 | Filter | Wrap | VFlip |
| --- | --- | --- | --- | --- |
| iChannel0 | `channels/anya_bvh.png` | Nearest | Clamp | Off |
| iChannel1 | `channels/anya_vertices.png` | Nearest | Clamp | Off |
| iChannel2 | `channels/anya_triangles.png` | Nearest | Clamp | Off |
| iChannel3 | `channels/anya_albedo.png` | Mipmap | Repeat | Off |

模型数据：1,006,473 个顶点、1,955,630 个三角形、524,287 个 BVH 节点。三个数据 Channel 必须保持 Nearest、Clamp、VFlip Off，否则索引和坐标会被破坏。

## 本地查看

在本目录运行：

```bash
python3 -m http.server 8765 --bind 127.0.0.1
```

然后打开：

`http://127.0.0.1:8765/preview.html`

预览默认使用完整分辨率。左右拖动可连续 360° 环绕；上下拖动采用常见的“拖动物体”方向，可从低角度旋转到约 80° 的头顶俯视。鼠标滚轮或触控板上下滚动用于缩放，松开后保留视角，双击画面重置旋转和缩放并恢复自动旋转。

性能不足时可以使用：

`http://127.0.0.1:8765/preview.html?scale=0.5`

## 编译检查

```bash
./compile.sh
QUALITY=0 ./compile.sh
```

## 重新生成 Channels

使用带有 NumPy 和 Pillow 的 Python：

```bash
python3 tools/glb_to_shadertoy.py anya_final.glb channels
```

提交作业方法二时，提交 `anya_final_shadertoy.glsl`、`preview.html`、整个 `channels/` 和本 README。运行时不需要 Blender，也不需要原始 GLB。
