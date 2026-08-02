/*
    Anya GLB mesh ray tracer for Shadertoy
    ---------------------------------------
    This shader intersects the real triangles extracted from anya_final.glb. It does
    not approximate the figurine with SDF primitives.

    Channel setup (Nearest/Clamp, VFlip off unless noted):
      iChannel0 = channels/anya_bvh.png
      iChannel1 = channels/anya_vertices.png
      iChannel2 = channels/anya_triangles.png
      iChannel3 = channels/anya_albedo.png
                  Filter=Mipmap, Wrap=Repeat, VFlip=OFF

    Mouse drag orbits the camera. Without input the model turns automatically.
*/

#define HIGH_QUALITY 1

const float PI = 3.141592653589793;
const float TAU = 6.283185307179586;
const float MIN_ELEVATION = -0.75;
const float MAX_ELEVATION = 1.40;
const float DEFAULT_CAMERA_RADIUS = 1.63;
const float MIN_CAMERA_RADIUS = 0.90;
const float MAX_CAMERA_RADIUS = 2.80;
const float FAR_CLIP = 5.0;

const vec3 MODEL_BOUNDS_MIN = vec3(-0.3583831787, 0.0000000000, -0.2966766357);
const vec3 MODEL_BOUNDS_MAX = vec3( 0.3583831787, 0.9797363281,  0.2966766357);
const float MODEL_CENTER_Y = 0.4898681641;
const float FLOOR_Y = -0.005;

const int BVH_STACK_SIZE = 32;
const int MAX_LEAF_TRIANGLES = 8;

#if HIGH_QUALITY
const int MAX_BVH_STEPS = 512;
#else
const int MAX_BVH_STEPS = 288;
#endif

struct Hit
{
    float distance;
    int triangle;
    vec2 barycentric;
};


// -----------------------------------------------------------------------------
// Packed data texture decoding
// -----------------------------------------------------------------------------

uvec4 byteTexel(sampler2D channel, int linearIndex)
{
    ivec2 size = textureSize(channel, 0);
    ivec2 coordinate = ivec2(linearIndex % size.x, linearIndex / size.x);
    vec4 value = texelFetch(channel, coordinate, 0);
    return uvec4(floor(value * 255.0 + 0.5));
}

uint unpack16(uvec2 value)
{
    return value.x * 256u + value.y;
}

uint unpack24(uvec3 value)
{
    return value.x * 65536u + value.y * 256u + value.z;
}

float unorm16(uvec2 value)
{
    return float(unpack16(value)) * (1.0 / 65535.0);
}

vec3 modelPointFromUnorm(vec3 value)
{
    return mix(MODEL_BOUNDS_MIN, MODEL_BOUNDS_MAX, value);
}

vec3 octahedralNormal(vec2 encoded)
{
    vec2 f = encoded * 2.0 - 1.0;
    vec3 normal = vec3(f, 1.0 - abs(f.x) - abs(f.y));
    float fold = clamp(-normal.z, 0.0, 1.0);
    normal.xy += mix(vec2(fold), vec2(-fold), step(vec2(0.0), normal.xy));
    return normalize(normal);
}

vec3 loadVertexPosition(int vertexIndex)
{
    int base = vertexIndex * 4;
    uvec4 a = byteTexel(iChannel1, base + 0);
    uvec4 b = byteTexel(iChannel1, base + 1);
    vec3 encoded = vec3(
        unorm16(a.rg),
        unorm16(a.ba),
        unorm16(b.rg)
    );
    return modelPointFromUnorm(encoded);
}

void loadVertexAttributes(int vertexIndex, out vec3 normal, out vec2 uv)
{
    int base = vertexIndex * 4;
    uvec4 b = byteTexel(iChannel1, base + 1);
    uvec4 c = byteTexel(iChannel1, base + 2);
    uvec4 d = byteTexel(iChannel1, base + 3);
    vec2 octa = vec2(unorm16(b.ba), unorm16(c.rg));
    normal = octahedralNormal(octa);
    uv = vec2(unorm16(c.ba), unorm16(d.rg));
}

ivec3 loadTriangle(int triangleIndex)
{
    int base = triangleIndex * 3;
    uint a = unpack24(byteTexel(iChannel2, base + 0).rgb);
    uint b = unpack24(byteTexel(iChannel2, base + 1).rgb);
    uint c = unpack24(byteTexel(iChannel2, base + 2).rgb);
    return ivec3(int(a), int(b), int(c));
}

void loadNodeBounds(int nodeIndex, out vec3 boundsMin, out vec3 boundsMax)
{
    int base = nodeIndex * 5;
    uvec4 a = byteTexel(iChannel0, base + 0);
    uvec4 b = byteTexel(iChannel0, base + 1);
    uvec4 c = byteTexel(iChannel0, base + 2);
    vec3 encodedMin = vec3(unorm16(a.rg), unorm16(a.ba), unorm16(b.rg));
    vec3 encodedMax = vec3(unorm16(b.ba), unorm16(c.rg), unorm16(c.ba));
    boundsMin = modelPointFromUnorm(encodedMin);
    boundsMax = modelPointFromUnorm(encodedMax);
}

void loadNodeMetadata(int nodeIndex, out int leftOrFirst,
                      out int right, out int triangleCount)
{
    int base = nodeIndex * 5;
    uvec4 a = byteTexel(iChannel0, base + 3);
    uvec4 b = byteTexel(iChannel0, base + 4);
    leftOrFirst = int(unpack24(a.rgb));
    right = int(unpack24(b.rgb));
    triangleCount = int(a.a);
}


// -----------------------------------------------------------------------------
// Triangle and BVH intersection
// -----------------------------------------------------------------------------

bool intersectBox(vec3 rayOrigin, vec3 inverseRayDirection,
                  vec3 boundsMin, vec3 boundsMax,
                  float maximumDistance, out float nearDistance)
{
    vec3 t0 = (boundsMin - rayOrigin) * inverseRayDirection;
    vec3 t1 = (boundsMax - rayOrigin) * inverseRayDirection;
    vec3 near3 = min(t0, t1);
    vec3 far3 = max(t0, t1);
    nearDistance = max(max(near3.x, near3.y), near3.z);
    float farDistance = min(min(far3.x, far3.y), far3.z);
    return farDistance >= max(nearDistance, 0.0)
        && nearDistance < maximumDistance;
}

bool intersectTriangle(vec3 rayOrigin, vec3 rayDirection,
                       vec3 a, vec3 b, vec3 c,
                       float maximumDistance,
                       out float distanceValue, out vec2 barycentric)
{
    vec3 edge1 = b - a;
    vec3 edge2 = c - a;
    vec3 p = cross(rayDirection, edge2);
    float determinant = dot(edge1, p);

    // Complex rejection flow is intentional: it avoids the remaining work for
    // the overwhelming majority of BVH leaf triangle tests.
    if (abs(determinant) < 1.0e-9)
        return false;

    float inverseDeterminant = 1.0 / determinant;
    vec3 t = rayOrigin - a;
    float u = dot(t, p) * inverseDeterminant;
    if (u < 0.0 || u > 1.0)
        return false;

    vec3 q = cross(t, edge1);
    float v = dot(rayDirection, q) * inverseDeterminant;
    if (v < 0.0 || u + v > 1.0)
        return false;

    float hitDistance = dot(edge2, q) * inverseDeterminant;
    if (hitDistance <= 0.00005 || hitDistance >= maximumDistance)
        return false;

    distanceValue = hitDistance;
    barycentric = vec2(u, v);
    return true;
}

Hit traceMesh(vec3 rayOrigin, vec3 rayDirection)
{
    Hit hit = Hit(FAR_CLIP, -1, vec2(0.0));
    vec3 inverseRayDirection = 1.0 / rayDirection;
    int stack[BVH_STACK_SIZE];
    int stackSize = 0;
    int nodeIndex = 0;

    for (int traversalStep = 0;
         traversalStep < MAX_BVH_STEPS;
         ++traversalStep)
    {
        vec3 boundsMin;
        vec3 boundsMax;
        loadNodeBounds(nodeIndex, boundsMin, boundsMax);
        float nodeNear;
        bool nodeHit = intersectBox(rayOrigin, inverseRayDirection,
                                    boundsMin, boundsMax,
                                    hit.distance, nodeNear);

        // BVH traversal is inherently stateful. These branches prevent entire
        // subtrees and their texture fetches from being evaluated.
        if (!nodeHit)
        {
            if (stackSize == 0)
                break;
            nodeIndex = stack[--stackSize];
            continue;
        }

        int leftOrFirst;
        int right;
        int triangleCount;
        loadNodeMetadata(nodeIndex, leftOrFirst, right, triangleCount);

        if (triangleCount > 0)
        {
            for (int leafIndex = 0;
                 leafIndex < MAX_LEAF_TRIANGLES;
                 ++leafIndex)
            {
                // Leaf counts vary from one to eight; stop before fetching
                // padding triangles from the next leaf.
                if (leafIndex >= triangleCount)
                    break;

                int triangleIndex = leftOrFirst + leafIndex;
                ivec3 indices = loadTriangle(triangleIndex);
                vec3 a = loadVertexPosition(indices.x);
                vec3 b = loadVertexPosition(indices.y);
                vec3 c = loadVertexPosition(indices.z);
                float triangleDistance;
                vec2 barycentric;
                if (intersectTriangle(rayOrigin, rayDirection, a, b, c,
                                      hit.distance,
                                      triangleDistance, barycentric))
                {
                    hit.distance = triangleDistance;
                    hit.triangle = triangleIndex;
                    hit.barycentric = barycentric;
                }
            }

            if (stackSize == 0)
                break;
            nodeIndex = stack[--stackSize];
            continue;
        }

        vec3 leftMin;
        vec3 leftMax;
        vec3 rightMin;
        vec3 rightMax;
        loadNodeBounds(leftOrFirst, leftMin, leftMax);
        loadNodeBounds(right, rightMin, rightMax);
        float leftNear;
        float rightNear;
        bool leftHit = intersectBox(rayOrigin, inverseRayDirection,
                                    leftMin, leftMax, hit.distance, leftNear);
        bool rightHit = intersectBox(rayOrigin, inverseRayDirection,
                                     rightMin, rightMax, hit.distance, rightNear);

        if (leftHit && rightHit)
        {
            bool leftFirst = leftNear <= rightNear;
            int nearChild = leftFirst ? leftOrFirst : right;
            int farChild = leftFirst ? right : leftOrFirst;
            if (stackSize < BVH_STACK_SIZE)
                stack[stackSize++] = farChild;
            nodeIndex = nearChild;
        }
        else if (leftHit)
        {
            nodeIndex = leftOrFirst;
        }
        else if (rightHit)
        {
            nodeIndex = right;
        }
        else
        {
            if (stackSize == 0)
                break;
            nodeIndex = stack[--stackSize];
        }
    }
    return hit;
}


// -----------------------------------------------------------------------------
// Surface reconstruction and shading
// -----------------------------------------------------------------------------

void reconstructSurface(Hit hit, vec3 rayDirection,
                        out vec3 normal, out vec2 uv)
{
    ivec3 indices = loadTriangle(hit.triangle);
    vec3 normalA;
    vec3 normalB;
    vec3 normalC;
    vec2 uvA;
    vec2 uvB;
    vec2 uvC;
    loadVertexAttributes(indices.x, normalA, uvA);
    loadVertexAttributes(indices.y, normalB, uvB);
    loadVertexAttributes(indices.z, normalC, uvC);
    float weightA = 1.0 - hit.barycentric.x - hit.barycentric.y;
    normal = normalize(normalA * weightA
                     + normalB * hit.barycentric.x
                     + normalC * hit.barycentric.y);
    normal *= mix(-1.0, 1.0, step(dot(normal, -rayDirection), 0.0));
    uv = uvA * weightA
       + uvB * hit.barycentric.x
       + uvC * hit.barycentric.y;
}

vec3 shadeMesh(vec3 point, vec3 rayDirection, Hit hit)
{
    vec3 normal;
    vec2 uv;
    reconstructSurface(hit, rayDirection, normal, uv);

    // Keep iChannel3 VFlip disabled; the exported glTF UVs match the image as-is.
    // Use a ray-footprint LOD for the atlas. The pink hair is replaced with a
    // stable material color below, so neither its baked weave nor bright colors
    // leaking through coarse atlas mips can become visible on the model.
    float atlasWidth = float(textureSize(iChannel3, 0).x);
    float texelFootprint = hit.distance * atlasWidth
        / max(iResolution.y, 1.0);
    float lod = clamp(log2(max(texelFootprint * 0.45, 1.0)), 0.0, 4.0);
    vec3 albedo = textureLod(iChannel3, uv, lod).rgb;

    float redLead = albedo.r - max(albedo.g, albedo.b);
    float pinkBalance = 1.0 - smoothstep(0.08, 0.18,
        abs(albedo.g - albedo.b));
    float upperHead = smoothstep(0.40, 0.47, point.y);
    float relativeRedness = redLead / max(albedo.r, 1.0e-4);
    float hairMask = smoothstep(0.04, 0.10, redLead)
        * smoothstep(0.42, 0.58, relativeRedness)
        * pinkBalance * upperHead;
    const vec3 HAIR_ALBEDO = vec3(0.42, 0.14, 0.17);
    albedo = mix(albedo, HAIR_ALBEDO, hairMask);
    vec3 viewDirection = -rayDirection;
    vec3 keyDirection = normalize(vec3(-0.55, 0.82, 0.68));
    vec3 fillDirection = normalize(vec3(0.72, 0.20, 0.52));
    float diffuse = max(dot(normal, keyDirection), 0.0);
    float wrap = max(dot(normal, keyDirection) * 0.68 + 0.32, 0.0);
    float fill = max(dot(normal, fillDirection), 0.0);
    vec3 halfVector = normalize(keyDirection + viewDirection);
    float specular = pow(max(dot(normal, halfVector), 0.0), 52.0);
    float rim = pow(1.0 - max(dot(normal, viewDirection), 0.0), 3.0);

    vec3 lighting = vec3(0.64, 0.68, 0.76)
                  + vec3(1.00, 0.91, 0.80) * diffuse * 0.68
                  + vec3(0.28, 0.40, 0.62) * fill * 0.18
                  + vec3(0.18, 0.20, 0.24) * wrap * 0.14;
    vec3 color = albedo * lighting;
    color += vec3(1.0, 0.92, 0.82) * specular
        * mix(0.20, 0.07, hairMask);
    color += vec3(0.25, 0.38, 0.62) * rim * 0.08;

    // Subtle contact darkening toward the model's physical base.
    float baseDarkening = smoothstep(0.00, 0.16, point.y);
    color *= mix(0.78, 1.0, baseDarkening);
    return color;
}

vec3 backgroundColor(vec3 rayDirection)
{
    float height = clamp(rayDirection.y * 0.5 + 0.5, 0.0, 1.0);
    return mix(vec3(0.040, 0.047, 0.064),
               vec3(0.28, 0.31, 0.37),
               smoothstep(0.08, 0.92, height));
}

vec3 shadeFloor(vec3 point, vec3 rayDirection)
{
    vec3 normal = vec3(0.0, 1.0, 0.0);
    float key = max(dot(normal, normalize(vec3(-0.55, 0.82, 0.68))), 0.0);
    float radial = length(point.xz);
    float contact = exp(-radial * radial * 5.2);
    float grain = sin(point.x * 38.0 + sin(point.z * 11.0)) * 0.012;
    vec3 floorColor = vec3(0.085, 0.075, 0.078) * (0.50 + key * 0.50 + grain);
    floorColor *= 1.0 - contact * 0.46;
    float fresnel = pow(1.0 - max(dot(normal, -rayDirection), 0.0), 4.0);
    return floorColor + vec3(0.11, 0.14, 0.19) * fresnel * 0.25;
}


// -----------------------------------------------------------------------------
// Camera and final image
// -----------------------------------------------------------------------------

mat3 cameraBasis(vec3 cameraPosition, vec3 target)
{
    vec3 forward = normalize(target - cameraPosition);
    vec3 right = normalize(cross(forward, vec3(0.0, 1.0, 0.0)));
    vec3 up = cross(right, forward);
    return mat3(right, up, forward);
}

void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
    vec2 uv = (2.0 * fragCoord - iResolution.xy) / iResolution.y;
    float mouseActive = step(0.001, iMouse.z);
    float autoAzimuth = iTime * 0.16;
    float mouseAzimuth = (0.5 - iMouse.x / max(iResolution.x, 1.0)) * TAU;
    float azimuth = mix(autoAzimuth, mouseAzimuth, mouseActive);
    float autoElevation = 0.10 + sin(iTime * 0.21) * 0.025;
    float mouseElevation = mix(MIN_ELEVATION, MAX_ELEVATION,
        1.0 - clamp(iMouse.y / max(iResolution.y, 1.0), 0.0, 1.0));
    float elevation = mix(autoElevation, mouseElevation, mouseActive);

    vec3 target = vec3(0.0, MODEL_CENTER_Y, 0.0);
    // preview.html places its wheel-controlled radius in iMouse.w. Native
    // Shadertoy mouse values fall back to the default unless they are in this
    // deliberately narrow camera-radius range.
    float previewZoomActive = step(MIN_CAMERA_RADIUS, iMouse.w)
        * (1.0 - step(MAX_CAMERA_RADIUS + 0.01, iMouse.w));
    float cameraRadius = mix(DEFAULT_CAMERA_RADIUS,
        clamp(iMouse.w, MIN_CAMERA_RADIUS, MAX_CAMERA_RADIUS),
        previewZoomActive);
    float elevationCosine = cos(elevation);
    vec3 orbitDirection = vec3(
        sin(azimuth) * elevationCosine,
        sin(elevation),
        cos(azimuth) * elevationCosine
    );
    vec3 cameraPosition = target + orbitDirection * cameraRadius;
    vec3 rayDirection = normalize(
        cameraBasis(cameraPosition, target) * vec3(uv, 2.20)
    );

    Hit hit = traceMesh(cameraPosition, rayDirection);
    vec3 color = backgroundColor(rayDirection);

    float floorDistance = FAR_CLIP;
    float floorDenominator = rayDirection.y;
    float floorValid = step(1.0e-6, -floorDenominator);
    float candidateFloor = (FLOOR_Y - cameraPosition.y) / floorDenominator;
    floorDistance = mix(FAR_CLIP, candidateFloor,
                        floorValid * step(0.0, candidateFloor));

    // A hit branch prevents expensive normal/UV/albedo reconstruction on misses.
    if (hit.triangle >= 0 && hit.distance < floorDistance)
    {
        vec3 point = cameraPosition + rayDirection * hit.distance;
        color = shadeMesh(point, rayDirection, hit);
    }
    else if (floorDistance < FAR_CLIP)
    {
        vec3 point = cameraPosition + rayDirection * floorDistance;
        color = shadeFloor(point, rayDirection);
        float fog = 1.0 - exp(-0.20 * floorDistance * floorDistance);
        color = mix(color, backgroundColor(rayDirection), fog);
    }

    // Visible diagnostic if one of the three packed data channels is missing.
    float dataReady = step(1.0, iChannelResolution[0].x)
                    * step(1.0, iChannelResolution[1].x)
                    * step(1.0, iChannelResolution[2].x);
    color = mix(vec3(0.65, 0.02, 0.42), color, dataReady);

    color = 1.0 - exp(-color * 1.08);
    color = pow(max(color, 0.0), vec3(0.4545));
    float vignette = 1.0 - 0.18 * dot(uv * 0.58, uv * 0.58);
    color *= clamp(vignette, 0.76, 1.0);
    fragColor = vec4(color, 1.0);
}
