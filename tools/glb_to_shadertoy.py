#!/usr/bin/env python3
"""Pack a textured GLB mesh into Shadertoy-friendly RGBA8 data textures.

The generated assets are consumed by ``anya_final_shadertoy.glsl``:

    iChannel0  BVH nodes
    iChannel1  vertex position / octahedral normal / UV data
    iChannel2  triangle vertex indices
    iChannel3  original base-color image

The converter intentionally uses lossless mesh topology by default. Positions,
normals and UVs are quantized to 16 bits; vertex/triangle/BVH indices use 24
bits.  This keeps the model precise while making the data uploadable as normal
PNG files on a fragment-shader-only platform.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


GLB_MAGIC = 0x46546C67
JSON_CHUNK = 0x4E4F534A
BIN_CHUNK = 0x004E4942

COMPONENT_DTYPES = {
    5120: np.int8,
    5121: np.uint8,
    5122: np.int16,
    5123: np.uint16,
    5125: np.uint32,
    5126: np.float32,
}

TYPE_COMPONENTS = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT2": 4,
    "MAT3": 9,
    "MAT4": 16,
}


@dataclass
class MeshData:
    positions: np.ndarray
    normals: np.ndarray
    uvs: np.ndarray
    triangles: np.ndarray
    image_bytes: bytes
    image_mime: str


@dataclass
class BvhData:
    bounds_min: np.ndarray
    bounds_max: np.ndarray
    left_or_first: np.ndarray
    right: np.ndarray
    counts: np.ndarray
    ordered_triangles: np.ndarray
    max_depth: int


def read_glb(path: Path) -> tuple[dict, bytes]:
    raw = path.read_bytes()
    if len(raw) < 12:
        raise ValueError("GLB is shorter than its 12-byte header")
    magic, version, declared_length = struct.unpack_from("<III", raw, 0)
    if magic != GLB_MAGIC:
        raise ValueError(f"Not a GLB file: bad magic 0x{magic:08x}")
    if version != 2:
        raise ValueError(f"Only glTF 2.0 is supported, found version {version}")
    if declared_length > len(raw):
        raise ValueError("GLB declares more bytes than the file contains")

    gltf = None
    binary = None
    offset = 12
    while offset + 8 <= declared_length:
        chunk_length, chunk_type = struct.unpack_from("<II", raw, offset)
        offset += 8
        chunk = raw[offset : offset + chunk_length]
        offset += chunk_length
        if chunk_type == JSON_CHUNK:
            gltf = json.loads(chunk.rstrip(b"\x00 \t\r\n").decode("utf-8"))
        elif chunk_type == BIN_CHUNK:
            binary = bytes(chunk)

    if gltf is None or binary is None:
        raise ValueError("GLB must contain both JSON and BIN chunks")
    return gltf, binary


def accessor_array(gltf: dict, binary: bytes, accessor_index: int) -> np.ndarray:
    accessor = gltf["accessors"][accessor_index]
    if "sparse" in accessor:
        raise ValueError("Sparse glTF accessors are not supported by this converter")
    view = gltf["bufferViews"][accessor["bufferView"]]
    dtype = np.dtype(COMPONENT_DTYPES[accessor["componentType"]]).newbyteorder("<")
    components = TYPE_COMPONENTS[accessor["type"]]
    count = accessor["count"]
    offset = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
    packed_stride = dtype.itemsize * components
    stride = view.get("byteStride", packed_stride)
    if stride < packed_stride:
        raise ValueError("Accessor byteStride is smaller than one packed element")

    array = np.ndarray(
        shape=(count, components),
        dtype=dtype,
        buffer=binary,
        offset=offset,
        strides=(stride, dtype.itemsize),
    )
    return np.ascontiguousarray(array)


def load_mesh(path: Path) -> MeshData:
    gltf, binary = read_glb(path)
    if len(gltf.get("meshes", [])) != 1:
        raise ValueError("This packer currently expects one GLB mesh")
    primitives = gltf["meshes"][0].get("primitives", [])
    if len(primitives) != 1:
        raise ValueError("This packer currently expects one mesh primitive")

    primitive = primitives[0]
    attributes = primitive["attributes"]
    required = ("POSITION", "NORMAL", "TEXCOORD_0")
    missing = [name for name in required if name not in attributes]
    if missing:
        raise ValueError(f"GLB is missing attributes: {', '.join(missing)}")
    if "indices" not in primitive:
        raise ValueError("An indexed triangle primitive is required")
    if primitive.get("mode", 4) != 4:
        raise ValueError("Only TRIANGLES primitives are supported")

    position_glb = accessor_array(gltf, binary, attributes["POSITION"]).astype(np.float32)
    normal_glb = accessor_array(gltf, binary, attributes["NORMAL"]).astype(np.float32)
    uvs = accessor_array(gltf, binary, attributes["TEXCOORD_0"]).astype(np.float32)
    indices = accessor_array(gltf, binary, primitive["indices"]).reshape(-1)
    if indices.size % 3:
        raise ValueError("Triangle index count is not divisible by three")

    # anya_final.glb is already Y-up and faces +Z after the Blender export.
    # Keep the GLB axes unchanged so the default shader camera sees the front.
    positions = position_glb.astype(np.float32)
    normals = normal_glb.astype(np.float32)
    normal_length = np.linalg.norm(normals, axis=1, keepdims=True)
    normals /= np.maximum(normal_length, 1.0e-20)
    triangles = indices.reshape(-1, 3).astype(np.uint32)

    material_index = primitive.get("material", 0)
    material = gltf.get("materials", [{}])[material_index]
    pbr = material.get("pbrMetallicRoughness", {})
    texture_index = pbr.get("baseColorTexture", {}).get("index")
    if texture_index is None:
        raise ValueError("The mesh does not have a base-color texture")
    image_index = gltf["textures"][texture_index]["source"]
    image_info = gltf["images"][image_index]
    if "bufferView" not in image_info:
        raise ValueError("Only GLB-embedded images are supported")
    image_view = gltf["bufferViews"][image_info["bufferView"]]
    image_offset = image_view.get("byteOffset", 0)
    image_bytes = binary[image_offset : image_offset + image_view["byteLength"]]

    return MeshData(
        positions=positions,
        normals=normals,
        uvs=uvs,
        triangles=triangles,
        image_bytes=image_bytes,
        image_mime=image_info.get("mimeType", "image/jpeg"),
    )


def build_bvh(mesh: MeshData, leaf_size: int) -> BvhData:
    triangles = mesh.triangles
    points = mesh.positions[triangles]
    tri_min = points.min(axis=1)
    tri_max = points.max(axis=1)
    centroids = (tri_min + tri_max) * 0.5
    order = np.arange(triangles.shape[0], dtype=np.uint32)

    node_min: list[np.ndarray] = []
    node_max: list[np.ndarray] = []
    node_left: list[int] = []
    node_right: list[int] = []
    node_count: list[int] = []
    max_depth = 0
    last_report = time.monotonic()

    sys.setrecursionlimit(max(sys.getrecursionlimit(), 10000))

    def build(start: int, end: int, depth: int) -> int:
        nonlocal max_depth, last_report
        node_index = len(node_min)
        ids = order[start:end]
        bounds_min = tri_min[ids].min(axis=0)
        bounds_max = tri_max[ids].max(axis=0)
        node_min.append(bounds_min)
        node_max.append(bounds_max)
        node_left.append(0)
        node_right.append(0)
        node_count.append(0)
        max_depth = max(max_depth, depth)

        count = end - start
        if count <= leaf_size:
            node_left[node_index] = start
            node_count[node_index] = count
            return node_index

        center_slice = centroids[ids]
        extent = np.ptp(center_slice, axis=0)
        axis = int(np.argmax(extent))
        middle_local = count // 2
        partition = np.argpartition(center_slice[:, axis], middle_local)
        order[start:end] = ids[partition]
        middle = start + middle_local

        left = build(start, middle, depth + 1)
        right = build(middle, end, depth + 1)
        node_left[node_index] = left
        node_right[node_index] = right

        now = time.monotonic()
        if depth == 0 or now - last_report > 5.0:
            print(f"  BVH: {len(node_min):,} nodes built", flush=True)
            last_report = now
        return node_index

    build(0, triangles.shape[0], 0)
    ordered_triangles = triangles[order]
    return BvhData(
        bounds_min=np.asarray(node_min, dtype=np.float32),
        bounds_max=np.asarray(node_max, dtype=np.float32),
        left_or_first=np.asarray(node_left, dtype=np.uint32),
        right=np.asarray(node_right, dtype=np.uint32),
        counts=np.asarray(node_count, dtype=np.uint8),
        ordered_triangles=ordered_triangles,
        max_depth=max_depth,
    )


def encode_u16(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.uint16)
    return (values >> 8).astype(np.uint8), (values & 255).astype(np.uint8)


def encode_u24(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.uint32)
    if values.size and int(values.max()) >= (1 << 24):
        raise ValueError("24-bit packed integer overflow")
    return np.column_stack(
        (
            (values >> 16) & 255,
            (values >> 8) & 255,
            values & 255,
        )
    ).astype(np.uint8)


def quantize_unorm16(values: np.ndarray) -> np.ndarray:
    return np.rint(np.clip(values, 0.0, 1.0) * 65535.0).astype(np.uint16)


def oct_encode(normals: np.ndarray) -> np.ndarray:
    projected = normals / np.maximum(np.abs(normals).sum(axis=1, keepdims=True), 1.0e-20)
    octa = projected[:, :2].copy()
    lower = projected[:, 2] < 0.0
    folded = (1.0 - np.abs(octa[lower][:, ::-1])) * np.where(
        octa[lower] >= 0.0, 1.0, -1.0
    )
    octa[lower] = folded
    return octa * 0.5 + 0.5


def texture_extent(texel_count: int, preferred_width: int) -> tuple[int, int]:
    width = preferred_width
    height_needed = max(1, math.ceil(texel_count / width))
    height = 1 << math.ceil(math.log2(height_needed))
    if height > 4096:
        raise ValueError(f"Data needs a {width}x{height} texture, above the 4096 limit")
    return width, height


def save_rgba_texture(
    path: Path, texels: np.ndarray, preferred_width: int
) -> tuple[int, int, str]:
    width, height = texture_extent(texels.shape[0], preferred_width)
    image_data = np.zeros((height * width, 4), dtype=np.uint8)
    image_data[: texels.shape[0]] = texels
    image = Image.fromarray(image_data.reshape(height, width, 4), mode="RGBA")
    image.save(path, format="PNG", compress_level=6, optimize=False)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return width, height, digest


def pack_vertices(
    mesh: MeshData, bounds_min: np.ndarray, bounds_max: np.ndarray
) -> np.ndarray:
    extent = np.maximum(bounds_max - bounds_min, 1.0e-20)
    position_unorm = (mesh.positions - bounds_min) / extent
    q_position = quantize_unorm16(position_unorm)
    q_normal = quantize_unorm16(oct_encode(mesh.normals))
    q_uv = quantize_unorm16(mesh.uvs)

    pxh, pxl = encode_u16(q_position[:, 0])
    pyh, pyl = encode_u16(q_position[:, 1])
    pzh, pzl = encode_u16(q_position[:, 2])
    nxh, nxl = encode_u16(q_normal[:, 0])
    nyh, nyl = encode_u16(q_normal[:, 1])
    uh, ul = encode_u16(q_uv[:, 0])
    vh, vl = encode_u16(q_uv[:, 1])

    texels = np.zeros((mesh.positions.shape[0], 4, 4), dtype=np.uint8)
    texels[:, 0, :] = np.column_stack((pxh, pxl, pyh, pyl))
    texels[:, 1, :] = np.column_stack((pzh, pzl, nxh, nxl))
    texels[:, 2, :] = np.column_stack((nyh, nyl, uh, ul))
    texels[:, 3, :] = np.column_stack((vh, vl, np.zeros_like(vh), np.zeros_like(vh)))
    return texels.reshape(-1, 4)


def pack_triangles(triangles: np.ndarray) -> np.ndarray:
    packed = encode_u24(triangles.reshape(-1))
    texels = np.zeros((packed.shape[0], 4), dtype=np.uint8)
    texels[:, :3] = packed
    return texels


def pack_bvh(
    bvh: BvhData, bounds_min: np.ndarray, bounds_max: np.ndarray
) -> np.ndarray:
    extent = np.maximum(bounds_max - bounds_min, 1.0e-20)
    q_min = quantize_unorm16((bvh.bounds_min - bounds_min) / extent)
    q_max = quantize_unorm16((bvh.bounds_max - bounds_min) / extent)
    # Expand quantized boxes by one unit to avoid missing boundary triangles.
    q_min = np.maximum(q_min.astype(np.int32) - 1, 0).astype(np.uint16)
    q_max = np.minimum(q_max.astype(np.int32) + 1, 65535).astype(np.uint16)

    min_xh, min_xl = encode_u16(q_min[:, 0])
    min_yh, min_yl = encode_u16(q_min[:, 1])
    min_zh, min_zl = encode_u16(q_min[:, 2])
    max_xh, max_xl = encode_u16(q_max[:, 0])
    max_yh, max_yl = encode_u16(q_max[:, 1])
    max_zh, max_zl = encode_u16(q_max[:, 2])
    left = encode_u24(bvh.left_or_first)
    right = encode_u24(bvh.right)

    texels = np.zeros((bvh.bounds_min.shape[0], 5, 4), dtype=np.uint8)
    texels[:, 0, :] = np.column_stack((min_xh, min_xl, min_yh, min_yl))
    texels[:, 1, :] = np.column_stack((min_zh, min_zl, max_xh, max_xl))
    texels[:, 2, :] = np.column_stack((max_yh, max_yl, max_zh, max_zl))
    texels[:, 3, :3] = left
    texels[:, 3, 3] = bvh.counts
    texels[:, 4, :3] = right
    return texels.reshape(-1, 4)


def write_albedo(mesh: MeshData, output_dir: Path) -> tuple[str, int, int, str]:
    extension = ".png" if "png" in mesh.image_mime else ".jpg"
    filename = "anya_albedo" + extension
    path = output_dir / filename
    path.write_bytes(mesh.image_bytes)
    with Image.open(path) as image:
        width, height = image.size
    digest = hashlib.sha256(mesh.image_bytes).hexdigest()
    return filename, width, height, digest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Source .glb file")
    parser.add_argument("output", type=Path, help="Output asset directory")
    parser.add_argument(
        "--leaf-size",
        type=int,
        default=8,
        help="Maximum triangles per BVH leaf (default: 8)",
    )
    args = parser.parse_args()
    if not 1 <= args.leaf_size <= 32:
        parser.error("--leaf-size must be between 1 and 32")

    args.output.mkdir(parents=True, exist_ok=True)
    print(f"Loading {args.input}", flush=True)
    mesh = load_mesh(args.input)
    bounds_min = mesh.positions.min(axis=0)
    bounds_max = mesh.positions.max(axis=0)
    print(
        f"  {mesh.positions.shape[0]:,} vertices, "
        f"{mesh.triangles.shape[0]:,} triangles",
        flush=True,
    )
    print(
        "  shader bounds "
        f"min={bounds_min.tolist()} max={bounds_max.tolist()}",
        flush=True,
    )

    started = time.monotonic()
    print(f"Building median BVH (leaf size {args.leaf_size})", flush=True)
    bvh = build_bvh(mesh, args.leaf_size)
    print(
        f"  {bvh.bounds_min.shape[0]:,} nodes, max depth {bvh.max_depth}, "
        f"{time.monotonic() - started:.1f}s",
        flush=True,
    )

    print("Packing RGBA8 data textures", flush=True)
    bvh_texels = pack_bvh(bvh, bounds_min, bounds_max)
    vertex_texels = pack_vertices(mesh, bounds_min, bounds_max)
    triangle_texels = pack_triangles(bvh.ordered_triangles)

    bvh_path = args.output / "anya_bvh.png"
    vertex_path = args.output / "anya_vertices.png"
    triangle_path = args.output / "anya_triangles.png"
    bvh_size = save_rgba_texture(bvh_path, bvh_texels, 1024)
    vertex_size = save_rgba_texture(vertex_path, vertex_texels, 1024)
    triangle_size = save_rgba_texture(triangle_path, triangle_texels, 2048)
    albedo_name, albedo_width, albedo_height, albedo_hash = write_albedo(
        mesh, args.output
    )

    manifest = {
        "format": "anya-shadertoy-bvh-v1",
        "source": str(args.input),
        "source_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "orientation": {
            "shader_x": "glb_x",
            "shader_y": "glb_y",
            "shader_z": "glb_z",
        },
        "bounds_min": [float(v) for v in bounds_min],
        "bounds_max": [float(v) for v in bounds_max],
        "vertex_count": int(mesh.positions.shape[0]),
        "triangle_count": int(mesh.triangles.shape[0]),
        "bvh_node_count": int(bvh.bounds_min.shape[0]),
        "bvh_max_depth": int(bvh.max_depth),
        "bvh_leaf_size": int(args.leaf_size),
        "channels": {
            "iChannel0": {
                "file": bvh_path.name,
                "width": bvh_size[0],
                "height": bvh_size[1],
                "sha256": bvh_size[2],
                "filter": "nearest",
                "wrap": "clamp",
                "vflip": False,
            },
            "iChannel1": {
                "file": vertex_path.name,
                "width": vertex_size[0],
                "height": vertex_size[1],
                "sha256": vertex_size[2],
                "filter": "nearest",
                "wrap": "clamp",
                "vflip": False,
            },
            "iChannel2": {
                "file": triangle_path.name,
                "width": triangle_size[0],
                "height": triangle_size[1],
                "sha256": triangle_size[2],
                "filter": "nearest",
                "wrap": "clamp",
                "vflip": False,
            },
            "iChannel3": {
                "file": albedo_name,
                "width": albedo_width,
                "height": albedo_height,
                "sha256": albedo_hash,
                "filter": "mipmap",
                "wrap": "repeat",
                "vflip": False,
            },
        },
    }
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print("Generated:", flush=True)
    for path in (bvh_path, vertex_path, triangle_path, args.output / albedo_name, manifest_path):
        print(f"  {path.name:24s} {path.stat().st_size / 1024 / 1024:7.2f} MiB", flush=True)


if __name__ == "__main__":
    main()
