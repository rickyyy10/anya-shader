/*
    Anya desk-scene mesh ray tracer for Shadertoy
    ------------------------------------------------
    This shader intersects the real triangles extracted from anya_final.glb. It does
    not approximate the figurine with SDF primitives.

    Channel setup (Nearest/Clamp, VFlip off unless noted):
      iChannel0 = channels/anya_bvh.png
      iChannel1 = channels/anya_vertices.png
      iChannel2 = channels/anya_triangles.png
      iChannel3 = channels/anya_albedo.png
                  Filter=Mipmap, Wrap=Repeat, VFlip=OFF

    Scene additions: a procedural wooden desk, a fixed warm point light and
    secondary BVH shadow rays for both the figurine and the tabletop.

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
const vec3 FIXED_LIGHT_POSITION = vec3(-1.15, 1.72, 0.70);
const vec3 FIXED_LIGHT_COLOR = vec3(1.00, 0.82, 0.68);
const vec3 FILL_LIGHT_DIRECTION = vec3(0.76, 0.32, -0.56);
const vec3 FILL_LIGHT_COLOR = vec3(0.42, 0.58, 0.92);
const vec3 RIM_LIGHT_DIRECTION = vec3(0.18, 0.62, -0.76);
const vec3 RIM_LIGHT_COLOR = vec3(1.00, 0.58, 0.34);

const vec3 AREA_LIGHT_OFFSETS[3] = vec3[3](
    vec3(-0.10,  0.03, -0.05),
    vec3( 0.09,  0.01, -0.07),
    vec3( 0.02, -0.03,  0.11)
);

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

vec3 srgbToLinear(vec3 color)
{
    vec3 low = color / 12.92;
    vec3 high = pow((color + 0.055) / 1.055, vec3(2.4));
    return mix(low, high, step(vec3(0.04045), color));
}

vec3 linearToSrgb(vec3 color)
{
    color = max(color, 0.0);
    vec3 low = color * 12.92;
    vec3 high = 1.055 * pow(color, vec3(1.0 / 2.4)) - 0.055;
    return mix(low, high, step(vec3(0.0031308), color));
}

vec3 acesTonemap(vec3 color)
{
    vec3 numerator = color * (2.51 * color + 0.03);
    vec3 denominator = color * (2.43 * color + 0.59) + 0.14;
    return clamp(numerator / denominator, 0.0, 1.0);
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

Hit traceMeshLimited(vec3 rayOrigin, vec3 rayDirection,
                     float maximumDistance)
{
    Hit hit = Hit(maximumDistance, -1, vec2(0.0));
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

Hit traceMesh(vec3 rayOrigin, vec3 rayDirection)
{
    return traceMeshLimited(rayOrigin, rayDirection, FAR_CLIP);
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

float meshLightVisibility(vec3 point, vec3 normal,
                          vec3 lightDirection, float lightDistance)
{
    float shadowLimit = max(lightDistance - 0.003, 0.001);
    Hit blocker = traceMeshLimited(point + normal * 0.0012,
                                   lightDirection, shadowLimit);
    return 1.0 - step(-0.5, float(blocker.triangle));
}

float areaLightVisibility(vec3 point, vec3 normal)
{
#if HIGH_QUALITY
    float visibility = 0.0;
    for (int sampleIndex = 0; sampleIndex < 3; ++sampleIndex)
    {
        vec3 sampleVector = FIXED_LIGHT_POSITION
            + AREA_LIGHT_OFFSETS[sampleIndex] - point;
        float sampleDistance = length(sampleVector);
        visibility += meshLightVisibility(point, normal,
            sampleVector / max(sampleDistance, 1.0e-6), sampleDistance);
    }
    return visibility * (1.0 / 3.0);
#else
    vec3 sampleVector = FIXED_LIGHT_POSITION - point;
    float sampleDistance = length(sampleVector);
    return meshLightVisibility(point, normal,
        sampleVector / max(sampleDistance, 1.0e-6), sampleDistance);
#endif
}

vec3 shadeMesh(vec3 point, vec3 rayDirection, Hit hit)
{
    vec3 normal;
    vec2 uv;
    reconstructSurface(hit, rayDirection, normal, uv);

    // The custom-texture extension provides raw sRGB bytes. Decode explicitly
    // and select a distance-dependent mip to preserve color while suppressing
    // the high-frequency hair moire.
    float atlasWidth = float(textureSize(iChannel3, 0).x);
    float texelFootprint = hit.distance * atlasWidth
        / max(iResolution.y, 1.0);
    float lod = clamp(log2(max(texelFootprint * 0.45, 1.0)), 0.0, 4.0);
    vec3 albedo = srgbToLinear(textureLod(iChannel3, uv, lod).rgb);
    vec3 viewDirection = -rayDirection;
    vec3 lightVector = FIXED_LIGHT_POSITION - point;
    float lightDistance = length(lightVector);
    vec3 lightDirection = lightVector / max(lightDistance, 1.0e-6);
    float diffuse = max(dot(normal, lightDirection), 0.0);
    float visibility = 1.0;

    // Skip a complete secondary BVH traversal on back-facing surfaces that
    // cannot receive direct light from the fixed source.
    if (diffuse > 0.0001)
        visibility = areaLightVisibility(point, normal);

    float attenuation = 1.0 / (1.0 + lightDistance * lightDistance * 0.30);
    float directLight = diffuse * visibility * attenuation;
    float fill = max(dot(normal, normalize(FILL_LIGHT_DIRECTION)), 0.0);
    float hemisphere = normal.y * 0.5 + 0.5;
    vec3 halfVector = normalize(lightDirection + viewDirection);
    float specular = pow(max(dot(normal, halfVector), 0.0), 56.0)
        * visibility * attenuation;
    float rim = pow(1.0 - max(dot(normal, viewDirection), 0.0), 3.0);

    float groundBounce = max(-normal.y, 0.0);
    float rimLight = max(dot(normal, normalize(RIM_LIGHT_DIRECTION)), 0.0)
        * pow(1.0 - max(dot(normal, viewDirection), 0.0), 2.0);
    vec3 lighting = vec3(0.31, 0.34, 0.40) * (0.68 + hemisphere * 0.32)
                  + FIXED_LIGHT_COLOR * directLight * 1.72
                  + FILL_LIGHT_COLOR * fill * 0.24
                  + vec3(0.42, 0.19, 0.08) * groundBounce * 0.16;
    vec3 color = albedo * lighting;
    color += FIXED_LIGHT_COLOR * specular * 0.28;
    color += vec3(0.25, 0.38, 0.62) * rim * 0.05;
    color += RIM_LIGHT_COLOR * rimLight * 0.16;

    // Subtle contact darkening toward the model's physical base.
    float baseDarkening = smoothstep(0.00, 0.16, point.y);
    color *= mix(0.78, 1.0, baseDarkening);
    return color;
}

vec3 backgroundColor(vec3 rayDirection)
{
    float height = clamp(rayDirection.y * 0.5 + 0.5, 0.0, 1.0);
    vec3 color = mix(vec3(0.105, 0.125, 0.165),
                     vec3(0.30, 0.29, 0.31),
                     smoothstep(0.06, 0.94, height));
    vec3 glowDirection = normalize(FIXED_LIGHT_POSITION
        - vec3(0.0, MODEL_CENTER_Y, 0.0));
    float studioGlow = pow(max(dot(rayDirection, glowDirection), 0.0), 7.0);
    return color + vec3(0.16, 0.105, 0.070) * studioGlow;
}

vec3 shadeDesk(vec3 point, vec3 rayDirection)
{
    vec3 normal = vec3(0.0, 1.0, 0.0);
    float plankCoordinate = point.z * 4.2;
    float plankIndex = floor(plankCoordinate);
    float plankRandom = fract(sin(plankIndex * 91.73 + 17.19) * 43758.5453);
    float grain = sin(point.x * 72.0
        + sin(point.x * 13.0 + plankRandom * TAU) * 2.8
        + point.z * 4.0) * 0.5 + 0.5;
    float seamDistance = abs(fract(plankCoordinate) - 0.5);
    float seam = smoothstep(0.475, 0.495, seamDistance);
    vec3 darkWood = vec3(0.105, 0.041, 0.018);
    vec3 lightWood = vec3(0.255, 0.112, 0.040);
    vec3 woodAlbedo = mix(darkWood, lightWood,
        0.32 + grain * 0.45 + plankRandom * 0.18);
    woodAlbedo *= 1.0 - seam * 0.42;

    vec3 lightVector = FIXED_LIGHT_POSITION - point;
    float lightDistance = length(lightVector);
    vec3 lightDirection = lightVector / max(lightDistance, 1.0e-6);
    float attenuation = 1.0 / (1.0 + lightDistance * lightDistance * 0.30);
    float visibility = 1.0;

    // Only this conservative region can receive the figurine's projected
    // shadow. The branch avoids an extra BVH traversal over the rest of the desk.
    if (point.x > -0.60 && point.x < 2.80
        && point.z > -2.20 && point.z < 0.55)
    {
        visibility = areaLightVisibility(point, normal);
    }

    float diffuse = max(dot(normal, lightDirection), 0.0)
        * mix(0.16, 1.0, visibility) * attenuation;
    vec3 color = woodAlbedo
        * (vec3(0.18, 0.20, 0.25) + FIXED_LIGHT_COLOR * diffuse * 1.90);

    vec3 halfVector = normalize(lightDirection - rayDirection);
    float specular = pow(max(dot(normal, halfVector), 0.0), 42.0)
        * visibility * attenuation;
    color += FIXED_LIGHT_COLOR * specular * 0.12;
    float fresnel = pow(1.0 - max(dot(normal, -rayDirection), 0.0), 4.0);
    return color + vec3(0.10, 0.13, 0.18) * fresnel * 0.18;
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
        color = shadeDesk(point, rayDirection);
        float fog = 1.0 - exp(-0.20 * floorDistance * floorDistance);
        color = mix(color, backgroundColor(rayDirection), fog);
    }

    // Visible diagnostic if one of the three packed data channels is missing.
    float dataReady = step(1.0, iChannelResolution[0].x)
                    * step(1.0, iChannelResolution[1].x)
                    * step(1.0, iChannelResolution[2].x);
    color = mix(vec3(0.65, 0.02, 0.42), color, dataReady);

    float vignette = 1.0 - 0.18 * dot(uv * 0.58, uv * 0.58);
    color *= clamp(vignette, 0.76, 1.0);
    color = linearToSrgb(acesTonemap(color * 1.10));
    fragColor = vec4(color, 1.0);
}
