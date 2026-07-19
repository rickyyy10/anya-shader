#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SHADER_FILE="${1:-$SCRIPT_DIR/anya_final_shadertoy.glsl}"
QUALITY="${QUALITY:-1}"

if [[ "$QUALITY" != "0" && "$QUALITY" != "1" ]]; then
  echo "QUALITY must be 0 or 1" >&2
  exit 2
fi

if [[ ! -f "$SHADER_FILE" ]]; then
  echo "Shader not found: $SHADER_FILE" >&2
  exit 2
fi

if ! command -v glslangValidator >/dev/null 2>&1; then
  echo "glslangValidator is not installed or not in PATH" >&2
  echo "Install it on macOS with: brew install glslang" >&2
  exit 127
fi

{
  printf '%s\n' \
    '#version 300 es' \
    'precision highp float;' \
    'precision highp int;' \
    'uniform vec3 iResolution;' \
    'uniform float iTime;' \
    'uniform vec4 iMouse;' \
    'uniform sampler2D iChannel0;' \
    'uniform sampler2D iChannel1;' \
    'uniform sampler2D iChannel2;' \
    'uniform sampler2D iChannel3;' \
    'uniform vec3 iChannelResolution[4];' \
    'out vec4 outColor;'
  sed "s/^#define HIGH_QUALITY 1/#define HIGH_QUALITY $QUALITY/" "$SHADER_FILE"
  printf '%s\n' 'void main(){mainImage(outColor,gl_FragCoord.xy);}'
} | glslangValidator --stdin -S frag

echo "Compile OK: $SHADER_FILE (HIGH_QUALITY=$QUALITY)"
