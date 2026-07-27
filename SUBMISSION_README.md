# Anya Bonus Submission

This package contains a WebGL2/Shadertoy fragment-shader ray tracer for the
Anya GLB mesh. The four `iChannel` images are input data, not pre-rendered
frames: the shader performs BVH traversal, triangle intersection, attribute
interpolation, texture sampling, camera movement, and lighting in real time.

Online project: <https://rickyyy10.github.io/anya-shader/>

## Run in VS Code

1. Open the extracted submission folder in VS Code.
2. Open **Terminal > New Terminal**.
3. Run:

   ```bash
   python3 -m http.server 8765 --bind 127.0.0.1
   ```

4. Open <http://127.0.0.1:8765/preview.html> in Chrome or Edge.

Do not open `preview.html` directly with a `file://` URL. Browsers block the
shader and Channel `fetch()` requests in that mode. VS Code's **Live Server**
extension is also suitable: right-click `preview.html` and choose
**Open with Live Server**.

The preview reads `channels/manifest.json`, uploads all four images as WebGL2
textures, and binds them automatically:

| Input | File | Purpose | Required settings |
| --- | --- | --- | --- |
| `iChannel0` | `channels/anya_bvh.png` | BVH acceleration nodes | Nearest, Clamp, VFlip Off |
| `iChannel1` | `channels/anya_vertices.png` | Positions, normals, and UVs | Nearest, Clamp, VFlip Off |
| `iChannel2` | `channels/anya_triangles.png` | Triangle vertex indices | Nearest, Clamp, VFlip Off |
| `iChannel3` | `channels/anya_albedo.png` | Base-color texture | Mipmap, Repeat, VFlip Off |

Mouse drag orbits the camera, the wheel zooms, and double-click restores the
automatic camera. For a slower computer, use
<http://127.0.0.1:8765/preview.html?scale=0.5>.

## Important files

- `anya_final_shadertoy.glsl`: submitted fragment shader.
- `preview.html`: WebGL2 runner and Channel-binding code.
- `channels/`: packed model data and texture inputs.
- `tools/glb_to_shadertoy.py`: source-to-Channel conversion tool.
- `preview_front.jpg`: reference image of the expected result.

The original GLB is not required at runtime because its rendering data has
already been packed into the three data Channels. The included conversion tool
documents how those Channels were generated.
