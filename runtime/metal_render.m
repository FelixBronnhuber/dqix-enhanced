/* Metal renderer (macOS) for the DS 3D engine, plus the final composition of
 * 3D with the 2D layers and presentation: the same work and interface as
 * gl_render.c (gl_render.h), pass for pass, so both draw the same picture.
 *
 * GL keeps a texture's row 0 at the bottom of what it draws; Metal at the
 * top. Every pass into a texture here flips y in its vertex shader, so each
 * texture holds its rows exactly as under GL: uploads, texelFetch-style reads
 * and read-backs all carry over unchanged. Only the window itself is drawn
 * unflipped (it is shown with row 0 at the top).
 *
 * Depth is a float buffer (Apple GPUs have no 24-bit depth); the shaders
 * round it to 24 bits, so depth tests compare what GL's D24 would.
 * The shaders are compiled from source at start (no Xcode needed to build). */
#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>
#include "SDL.h"
#include "SDL_metal.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "nds.h"
#include "gpu3d.h"
#include "gl_render.h"

#define VTX_FLOATS 18                    /* floats per vertex sent to the GPU */

/* ---------------------------------------------------------------- shaders */
static const char* MSL =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "\n"
    /* ---- polygons */
    "struct PolyV { float4 pos [[position]]; float4 color; float2 tex; float4 attr [[flat]];\n"
    "  float3 bary [[center_no_perspective]]; int ext [[flat]]; };\n"
    "vertex PolyV vs_poly(uint vid [[vertex_id]], const device float* vb [[buffer(0)]], constant float2& texsize [[buffer(1)]]) {\n"
    "  const device float* o = vb + vid * 18;\n"
    "  PolyV r;\n"
    "  r.pos = float4(o[0], -o[1], (o[2] + o[3]) * 0.5, o[3]);\n"   /* GL's -1..1 depth -> 0..1; y flipped (see the top) */
    "  r.color = float4(o[4], o[5], o[6], o[7]); r.tex = float2(o[8], o[9]) / texsize;\n"
    "  r.attr = float4(o[10], o[11], o[12], o[13]); r.bary = float3(o[14], o[15], o[16]); r.ext = int(o[17] + 0.5);\n"
    "  return r; }\n"
    "struct PolyU { float4 toon[32]; float aref, edgew; int highlight, alphatest, wbuf, part, sharpa, pad; };\n"
    "struct PolyOut { float4 color [[color(0)]]; float4 attr [[color(1)]]; float w [[color(2)]]; float depth [[depth(any)]]; };\n"
    /* DS pixel pipeline, in 5-bit color steps like the hardware. */
    "fragment PolyOut fs_poly(PolyV in [[stage_in]], constant PolyU& u [[buffer(0)]], texture2d<float> tex [[texture(0)]], sampler smp [[sampler(0)]]) {\n"
    "  float3 gx = dfdx(in.bary), gy = dfdy(in.bary);\n"
    "  float3 dist = in.bary / max(sqrt(gx * gx + gy * gy), float3(1e-6));\n"
    "  float eflag = 0.0;\n"
    "  for (int k = 0; k < 3; k++) if (((in.ext >> k) & 1) != 0 && dist[k] < u.edgew) eflag = 1.0;\n"
    "  int3 vc = int3(clamp(floor(in.color.rgb + 0.5), 0.0, 31.0));\n"
    "  int pa = int(in.color.a + 0.5);\n"
    "  int mode = int(in.attr.x + 0.5); bool textured = in.attr.y > 0.5;\n"
    "  if (mode == 2) { int3 t = int3(u.toon[vc.r].rgb * 31.0 + 0.5); if (u.highlight == 0) vc = t; else vc = int3(vc.r); }\n"
    "  int3 c = vc; int a = pa;\n"
    "  if (textured) {\n"
    "    float4 t = tex.sample(smp, in.tex);\n"
    "    if (u.sharpa != 0) t.a = t.a >= 0.5 ? 1.0 : 0.0;\n"
    "    int3 tc = int3(t.rgb * 31.0 + 0.5); int ta = int(t.a * 31.0 + 0.5);\n"
    "    if (mode == 1) { c = (tc * ta + vc * (31 - ta)) / 31; a = pa; }\n"
    "    else { c = ((tc + 1) * (vc + 1) - 1) >> 5; a = ((ta + 1) * (pa + 1) - 1) >> 5; }\n"
    "  }\n"
    "  if (mode == 2 && u.highlight != 0) { int3 t = int3(u.toon[int(in.color.r + 0.5)].rgb * 31.0 + 0.5); c = min(c + t, int3(31)); }\n"
    "  if (a == 0) discard_fragment();\n"
    "  if (u.alphatest != 0 && float(a) <= u.aref) discard_fragment();\n"
    "  if ((u.part == 1 && a < 31) || (u.part == 2 && a >= 31)) discard_fragment();\n"
    "  float fa = a >= 31 ? 1.0 : float(a) / 31.0;\n"
    "  PolyOut o;\n"
    "  o.color = float4(float3(c) / 31.0, fa);\n"
    "  o.attr = float4(in.attr.z, in.attr.w, eflag, 1.0);\n"
    "  o.w = 1.0 / in.pos.w;\n"                                 /* distance from the eye (HD-2D: depth of field) */
    "  float d = u.wbuf != 0 ? clamp((1.0 / in.pos.w) / 4096.0, 0.0, 1.0) : in.pos.z;\n"
    "  o.depth = rint(d * 16777215.0) / 16777215.0;\n"       /* as a 24-bit depth buffer keeps it */
    "  return o; }\n"
    "\n"
    /* ---- full-screen quads */
    "struct QuadV { float4 pos [[position]]; float2 uv; };\n"
    "struct QuadU { float4 rect, uv; float flip, p0, p1, p2; };\n"
    "vertex QuadV vs_quad(uint vid [[vertex_id]], constant QuadU& u [[buffer(0)]]) {\n"
    "  const float2 q[6] = { float2(0, 0), float2(1, 0), float2(1, 1), float2(0, 0), float2(1, 1), float2(0, 1) };\n"
    "  float2 a = q[vid]; QuadV r; r.uv = u.uv.xy + a * u.uv.zw;\n"
    "  float2 p = u.rect.xy + a * u.rect.zw; r.pos = float4(p.x, p.y * u.flip, 0.0, 1.0); return r; }\n"
    "fragment float4 fs_nothing(QuadV in [[stage_in]]) { return float4(0.0); }\n"
    "\n"
    /* ---- edge marking and fog on the finished 3D frame */
    "struct PostU { float4 edgecol[8]; float4 fogcol; float fogtab[32]; int edge, scale, aa, fog, fog_alpha_only, fogoff, fogshift, wbuf; };\n"
    "fragment float4 fs_post(QuadV in [[stage_in]], constant PostU& u [[buffer(0)]], texture2d<float> tcol [[texture(0)]],\n"
    "                        texture2d<float> tattr [[texture(1)]], depth2d<float> tdep [[texture(2)]]) {\n"
    "  int2 p = int2(in.pos.xy);\n"
    "  float4 c = tcol.read(uint2(p)); float4 at = tattr.read(uint2(p));\n"
    "  if (u.edge != 0 && at.b > 0.5 && at.a > 0.5) {\n"
    "    float id = at.r; float d = tdep.read(uint2(p));\n"
    "    int2 sz = int2(tcol.get_width(), tcol.get_height());\n"
    "    int2 o[4] = { int2(-u.scale, 0), int2(u.scale, 0), int2(0, -u.scale), int2(0, u.scale) };\n"
    "    for (int k = 0; k < 4; k++) {\n"
    "      int2 q = p + o[k];\n"
    "      if (q.x < 0 || q.y < 0 || q.x >= sz.x || q.y >= sz.y) continue;\n"
    "      float4 an = tattr.read(uint2(q));\n"
    "      if (abs(an.r - id) > 0.004 && d < tdep.read(uint2(q))) { int i = int(id * 63.0 + 0.5) >> 3;\n"
    "        if (u.aa != 0) { float4 b = tcol.read(uint2(q)); c = float4(b.a > 0.0 ? mix(b.rgb, u.edgecol[i].rgb, 17.0 / 32.0) : u.edgecol[i].rgb, b.a > 0.0 ? max(c.a, b.a) : c.a); }\n"
    "        else c = float4(u.edgecol[i].rgb, c.a);\n"
    "        break; }\n"
    "    }\n"
    "  }\n"
    "  if (u.fog != 0 && at.g > 0.5) {\n"
    "    float d = tdep.read(uint2(p));\n"
    "    int z15 = u.wbuf != 0 ? int(d * 4096.0 * 4096.0) >> 9 : int(d * 16777215.0) >> 9;\n"
    "    int step = 0x400 >> u.fogshift; float dens;\n"
    "    if (step <= 0) dens = u.fogtab[31];\n"
    "    else { int i = (z15 - u.fogoff) / step;\n"
    "      if (z15 - u.fogoff < 0) dens = u.fogtab[0]; else if (i >= 32) dens = u.fogtab[31];\n"
    "      else { float f = float((z15 - u.fogoff) - i * step) / float(step); dens = mix(u.fogtab[max(i - 1, 0)], u.fogtab[i], f); } }\n"
    "    if (dens >= 127.0) dens = 128.0; dens /= 128.0;\n"
    "    float a = c.a; float3 rgb = c.rgb;\n"
    "    if (u.fog_alpha_only == 0) rgb = mix(rgb, u.fogcol.rgb, dens);\n"
    "    a = mix(a, u.fogcol.a, dens);\n"
    "    c = float4(rgb, a);\n"
    "  }\n"
    "  return c; }\n"
    "\n"
    /* ---- one screen's final picture: 2D layers (CPU, 256x192) + the 3D layer */
    "struct CompU { int has3d, wide; float wtot, wside; };\n"
    "fragment float4 fs_comp(QuadV in [[stage_in]], constant CompU& u [[buffer(0)]], texture2d<float> tcol [[texture(0)]],\n"
    "                        texture2d<float> tc2 [[texture(1)]], texture2d<uint> tinfo [[texture(2)]], texture2d<float> t3d [[texture(3)]]) {\n"
    "  float2 uv = float2(in.uv.x, 1.0 - in.uv.y);\n"
    "  float dsx = u.wide != 0 ? uv.x * u.wtot - u.wside : uv.x * 256.0;\n"
    "  int2 p2 = int2(min(float2(dsx, uv.y * 192.0), float2(255.0, 191.0)));\n"
    "  bool side = u.wide != 0 && (dsx < 0.0 || dsx >= 256.0);\n"
    "  if (side) {\n"
    "    int found = -1;\n"
    "    for (int k = 0; k < 8; k++) { int x = dsx < 0.0 ? k : 255 - k;\n"
    "      if ((tinfo.read(uint2(x, p2.y)).r & 3u) != 0u) { found = x; break; } }\n"
    "    if (found < 0) {\n"
    "      int mb0 = int(tc2.read(uint2(dsx < 0.0 ? 0 : 255, p2.y)).a * 255.0 + 0.5);\n"
    "      float3 o0 = float3(0.0); if ((mb0 >> 5) == 1) o0 = float3(float(mb0 & 31) / 16.0);\n"
    "      return float4(o0, 1.0);\n"
    "    }\n"
    "    p2.x = found;\n"
    "  }\n"
    "  float3 col = tcol.read(uint2(p2)).rgb;\n"
    "  if (u.has3d == 0) return float4(col, 1.0);\n"
    "  uint4 info = tinfo.read(uint2(p2));\n"
    "  int mode = int(info.r & 3u);\n"
    "  if (mode == 0) return float4(col, 1.0);\n"
    "  float4 c2t = tc2.read(uint2(p2));\n"
    "  float3 c2 = c2t.rgb; int mb = int(c2t.a * 255.0 + 0.5);\n"
    "  int2 sz = int2(t3d.get_width(), t3d.get_height());\n"
    "  float4 s = t3d.read(uint2(min(int2(in.uv * float2(sz)), sz - 1)));\n"
    "  float A = s.a; float3 c3 = s.rgb;\n"
    "  float eva = float(info.g) / 16.0, evb = float(info.b) / 16.0, evy = float(info.a) / 16.0;\n"
    "  float3 o;\n"
    "  if (mode == 1) {\n"
    "    bool sec = (info.r & 4u) != 0u; int eff = int((info.r >> 3) & 3u);\n"
    "    if (A <= 0.0) o = col;\n"
    "    else if (sec) o = A < 0.999 ? c3 * A + c2 * (1.0 - A) : c3;\n"
    "    else if (eff == 2) o = c3 + (1.0 - c3) * evy;\n"
    "    else if (eff == 3) o = c3 - c3 * evy;\n"
    "    else o = c3;\n"
    "  } else {\n"
    "    float3 under = A > 0.0 ? c3 : c2;\n"
    "    o = min(col * eva + under * evb, float3(1.0));\n"
    "  }\n"
    "  int mbm = mb >> 5; float mbf = float(mb & 31) / 16.0;\n"
    "  if (mbm == 1) o = o + (1.0 - o) * mbf; else if (mbm == 2) o = o - o * mbf;\n"
    "  return float4(o, 1.0); }\n"
    "\n"
    /* ---- LCD motion blur: this frame mixed with what the screen still shows (linear light) */
    "struct GhostU { float keep, p0, p1, p2; };\n"
    "fragment float4 fs_ghost(QuadV in [[stage_in]], constant GhostU& u [[buffer(0)]], texture2d<float> cur [[texture(0)]],\n"
    "                         texture2d<float> prev [[texture(1)]], sampler sm [[sampler(0)]]) {\n"
    "  float3 c = pow(cur.sample(sm, in.uv).rgb, float3(2.2)); float3 p = prev.sample(sm, in.uv).rgb;\n"
    "  return float4(mix(c, p, u.keep), 1.0); }\n"
    "\n"
    /* ---- the screens in the window: exact pixels, or the DS Lite screen (its LCD
     * colours, pixel grid, backlight), optionally inside a DS Lite. The colour
     * profile is Pokefan531's DS Lite measurement (public domain, libretro
     * \"dslite-color\"). Canvas units are DS pixels. See gl_render.c's FS_SCREEN. */
    "struct ScreenU {\n"
    "  float4 view;                 // canvas -> window: window = xy + canvas * z\n"
    "  float4 win_canvas;           // drawable size, canvas size\n"
    "  float4 top, bot, ctop, cbot; // screen rects; the console's upper and lower screen windows\n"
    "  float4 topx_botx;            // each screen in the source texture: x offset, width (DS pixels)\n"
    "  float4 lid, base, btn, hinge;\n"
    "  float4 skin_texw_alpha;      // skin scale, the source's width (DS pixels), how solid the front screen is\n"
    "  int4 f0;                     // src_linear, front, lcd, lcdcolor\n"
    "  int4 f1;                     // grid, shell, skinmode, -\n"
    "  float4 f2;                   // backlight\n"
    "};\n"
    "struct S {\n"
    "  float2 win; float3 view; float2 canvas; float4 top, bot, ctop, cbot; float2 topx, botx; float texw; int front; float alpha;\n"
    "  int src_linear, lcd, lcdcolor, grid, shell, skinmode; float backlight; float3 lid, base, btn, hinge; float2 skinscale;\n"
    "  float px;                    // one window pixel in canvas units (for anti-aliasing)\n"
    "  float2 lo, hi;               // the current screen's part of the texture\n"
    "  float2 duv; float dp;        // how far one window pixel moves in the texture (and in DS pixels)\n"
    "};\n"
    "constant float3x3 DSL = float3x3(float3(0.93, 0.025, 0.008), float3(0.14, 0.90, -0.03), float3(-0.07, 0.075, 1.022));\n"
    "constant float DSL_LUM = 0.935;\n"
    "float cover(thread S& s, float d) { return clamp(0.5 - d / s.px, 0.0, 1.0); }\n"
    "float sd_box(float2 p, float2 c, float2 h, float4 r) {\n"
    "  p -= c;\n"
    "  float rr = p.x < 0.0 ? (p.y < 0.0 ? r.x : r.w) : (p.y < 0.0 ? r.y : r.z);\n"
    "  float2 q = abs(p) - h + rr;\n"
    "  return min(max(q.x, q.y), 0.0) + length(max(q, 0.0)) - rr; }\n"
    "float sd_circle(float2 p, float2 c, float r) { return length(p - c) - r; }\n"
    "float3 to_lin(float3 c) { return pow(max(c, 0.0), float3(2.2)); }\n"
    "float2 keep_in(thread S& s, texture2d<float> src, float2 uv, float2 duv) {\n"
    "  float2 tsz = float2(src.get_width(), src.get_height());\n"
    "  float f = max(abs(duv.x) * tsz.x, abs(duv.y) * tsz.y);\n"
    "  float2 m = 0.5 * (f <= 1.001 ? 1.0 : exp2(ceil(log2(f) - 0.001))) / tsz;\n"
    "  return clamp(uv, s.lo + m, s.hi - m); }\n"
    "float3 fetch(thread S& s, texture2d<float> src, sampler sm, float2 uv) {\n"
    "  float3 c = src.sample(sm, keep_in(s, src, uv, s.duv), gradient2d(float2(s.duv.x, 0.0), float2(0.0, s.duv.y))).rgb;\n"
    "  return s.src_linear != 0 ? c : to_lin(c); }\n"
    "float3 screen(thread S& s, texture2d<float> src, sampler sm, float2 p, int bottom) {\n"
    "  float2 tsz = float2(src.get_width(), src.get_height());\n"
    "  float2 uv = float2((p.x + (bottom != 0 ? s.botx.x : s.topx.x)) / s.texw, (bottom != 0 ? 0.5 : 1.0) - p.y / 384.0);\n"
    "  float3 lin;\n"
    "  {\n"
    "    float2 t = uv * tsz - 0.5;\n"
    "    float2 i = floor(t), f = t - i;\n"
    "    float2 sc = 1.0 / max(s.duv * tsz, float2(1e-4));\n"
    "    f = clamp((f - 0.5) * sc + 0.5, 0.0, 1.0);\n"
    "    lin = fetch(s, src, sm, (i + 0.5 + f) / tsz);\n"
    "  }\n"
    "  if (s.lcdcolor != 0) lin = clamp(DSL * lin * DSL_LUM, 0.0, 1.0);\n"
    "  if (s.grid != 0) {\n"
    "    float strength = clamp((s.view.z - 1.5) / 2.0, 0.0, 1.0);\n"
    "    if (strength > 0.0) {\n"
    "      float2 fp = fract(p);\n"
    "      float w = clamp(s.dp * 0.75, 0.0, 0.3);\n"
    "      float3 sub;\n"
    "      for (int k = 0; k < 3; k++) {\n"
    "        float d = abs(fp.x - (float(k) + 0.5) / 3.0);\n"
    "        float m = 1.0 - smoothstep(1.0 / 6.0 - w, 1.0 / 6.0 + w, d);\n"
    "        sub[k] = mix(0.62, 1.0, m);\n"
    "      }\n"
    "      float gx = min(fp.x, 1.0 - fp.x), gy = min(fp.y, 1.0 - fp.y);\n"
    "      float line = mix(0.55, 1.0, smoothstep(0.035 - w, 0.035 + w, gx)) * mix(0.45, 1.0, smoothstep(0.06 - w, 0.06 + w, gy));\n"
    "      float3 mask = sub * line;\n"
    "      mask /= 0.74;\n"
    "      lin *= mix(float3(1.0), mask, strength);\n"
    "    }\n"
    "  }\n"
    "  lin = lin * s.backlight;\n"
    "  lin = lin * (1.0 - 0.002) + 0.002 * s.backlight;\n"
    "  return lin; }\n"
    "float3 shade_plastic(thread S& s, float3 col, float d, float2 p, float glossy) {\n"
    "  float in_ = -d;\n"
    "  float bevel = smoothstep(0.0, 7.0, in_);\n"
    "  float3 c = col * mix(0.78, 1.0, bevel);\n"
    "  c += (1.0 - smoothstep(4.0, 9.0, abs(in_ - 5.0))) * 0.04;\n"
    "  float g = glossy * smoothstep(0.0, 1.0, 1.0 - abs((p.x - p.y * 0.6) / s.canvas.x - 0.2) * 2.5) * 0.06;\n"
    "  return c + g; }\n"
    "float3 dslite(thread S& s, float2 p) {\n"
    "  float3 bg = float3(0.055, 0.058, 0.065) * (1.0 - 0.35 * length((p - s.canvas * 0.5) / s.canvas));\n"
    "  float3 col = bg;\n"
    "  float W = s.canvas.x;\n"
    "  float hingeY = s.ctop.y + s.ctop.w + 40.0;\n"
    "  float sh = sd_box(p, float2(W * 0.5, s.canvas.y * 0.5 + 6.0), float2(W * 0.5 - 4.0, s.canvas.y * 0.5 - 8.0), float4(40.0));\n"
    "  col *= mix(0.55, 1.0, smoothstep(-4.0, 22.0, sh));\n"
    "  float dbase = sd_box(p, float2(W * 0.5, (hingeY + s.canvas.y) * 0.5), float2(W * 0.5 - 2.0, (s.canvas.y - hingeY) * 0.5 - 2.0), float4(10.0, 10.0, 38.0, 38.0));\n"
    "  col = mix(col, shade_plastic(s, s.base, dbase, p, 0.6), cover(s, dbase));\n"
    "  float dlid = sd_box(p, float2(W * 0.5, hingeY * 0.5), float2(W * 0.5 - 2.0, hingeY * 0.5 - 2.0), float4(38.0, 38.0, 10.0, 10.0));\n"
    "  col = mix(col, shade_plastic(s, s.lid, dlid, p, 1.0), cover(s, dlid));\n"
    "  float dh = sd_box(p, float2(W * 0.5, hingeY), float2(W * 0.5 - 18.0, 9.0), float4(9.0));\n"
    "  float hy = clamp((p.y - hingeY) / 9.0, -1.0, 1.0);\n"
    "  float3 hc = s.hinge * (0.75 + 0.35 * sqrt(max(1.0 - hy * hy, 0.0)));\n"
    "  col = mix(col, hc, cover(s, dh));\n"
    "  col = mix(col, s.hinge * 0.35, cover(s, sd_box(p, float2(W * 0.5, hingeY), float2(1.2, 7.0), float4(1.0))) * 0.6);\n"
    "  col = mix(col, float3(0.08), cover(s, sd_circle(p, float2(W * 0.5 + 24.0, hingeY), 2.4)));\n"
    "  float led = sd_circle(p, float2(W - 46.0, hingeY), 3.0);\n"
    "  col = mix(col, float3(0.25, 0.95, 0.35), cover(s, led));\n"
    "  col += float3(0.1, 0.5, 0.15) * exp(-max(led, 0.0) / 4.0) * 0.35;\n"
    "  for (int k = 0; k < 2; k++) {\n"
    "    float cx = k == 0 ? s.ctop.x * 0.5 : W - s.ctop.x * 0.5;\n"
    "    float cy = s.ctop.y + s.ctop.w * 0.5;\n"
    "    for (int r = 0; r < 5; r++) for (int c = 0; c < 3; c++) {\n"
    "      float2 hc2 = float2(cx + (float(c) - 1.0) * 9.0, cy + (float(r) - 2.0) * 9.0);\n"
    "      col = mix(col, s.lid * 0.25, cover(s, sd_circle(p, hc2, 2.4)));\n"
    "    }\n"
    "  }\n"
    "  for (int k = 0; k < 2; k++) {\n"
    "    float4 r = k == 0 ? s.ctop : s.cbot;\n"
    "    float m = k == 0 ? 13.0 : 9.0;\n"
    "    float dw = sd_box(p, r.xy + r.zw * 0.5, r.zw * 0.5 + m, float4(k == 0 ? 7.0 : 4.0));\n"
    "    if (k == 1) {\n"
    "      float dl = sd_box(p, r.xy + r.zw * 0.5, r.zw * 0.5 + m + 5.0, float4(8.0));\n"
    "      col = mix(col, s.base * 0.86, cover(s, dl));\n"
    "    }\n"
    "    float3 glass = float3(0.045, 0.047, 0.05) + 0.03 * smoothstep(0.0, 1.0, 1.0 - (p.y - r.y) / r.w);\n"
    "    col = mix(col, glass, cover(s, dw));\n"
    "  }\n"
    "  float2 dp = float2(s.cbot.x * 0.5, s.cbot.y + 58.0);\n"
    "  col = mix(col, s.base * 0.8, cover(s, min(sd_box(p, dp, float2(45.0, 16.0), float4(6.0)), sd_box(p, dp, float2(16.0, 45.0), float4(6.0)))) * 0.6);\n"
    "  float darm = min(sd_box(p, dp, float2(42.0, 13.0), float4(4.0)), sd_box(p, dp, float2(13.0, 42.0), float4(4.0)));\n"
    "  float3 btn = shade_plastic(s, s.btn, darm, p, 0.5);\n"
    "  btn *= 1.0 - 0.18 * smoothstep(10.0, 0.0, length(p - dp));\n"
    "  col = mix(col, btn, cover(s, darm));\n"
    "  float2 ab = float2(W - s.cbot.x * 0.5, s.cbot.y + 58.0);\n"
    "  float2 offs[4] = { float2(30.0, 0.0), float2(0.0, 30.0), float2(0.0, -30.0), float2(-30.0, 0.0) };\n"
    "  for (int k = 0; k < 4; k++) {\n"
    "    float2 c = ab + offs[k];\n"
    "    col = mix(col, s.base * 0.75, cover(s, sd_circle(p, c, 15.5)) * 0.6);\n"
    "    float db = sd_circle(p, c, 13.0);\n"
    "    float3 bc = s.btn * (1.0 + 0.12 * clamp(-(p.y - c.y + p.x - c.x) / 18.0, -1.0, 1.0));\n"
    "    col = mix(col, shade_plastic(s, bc, db, p, 0.3), cover(s, db));\n"
    "  }\n"
    "  for (int k = 0; k < 2; k++) {\n"
    "    float2 c = ab + float2(4.0, 78.0 + 26.0 * float(k));\n"
    "    col = mix(col, s.base * 0.75, cover(s, sd_circle(p, c, 8.5)) * 0.6);\n"
    "    float db = sd_circle(p, c, 6.5);\n"
    "    col = mix(col, shade_plastic(s, s.btn, db, p, 0.3), cover(s, db));\n"
    "  }\n"
    "  return col; }\n"
    "float3 shade(thread S& s, texture2d<float> src, sampler sm, bool in_bot, float2 p) {\n"
    "  float4 r = in_bot ? s.bot : s.top;\n"
    "  float2 tx = in_bot ? s.botx : s.topx;\n"
    "  s.lo = float2(tx.x / s.texw, in_bot ? 0.0 : 0.5); s.hi = float2((tx.x + tx.y) / s.texw, in_bot ? 0.5 : 1.0);\n"
    "  s.duv = float2(tx.y / (r.z * s.view.z * s.texw), 192.0 / (r.w * s.view.z * 384.0)); s.dp = tx.y / (r.z * s.view.z);\n"
    "  float2 sp = (p - r.xy) * float2(tx.y, 192.0) / r.zw;\n"
    "  if (s.lcd == 0) {\n"
    "    float2 tsz = float2(src.get_width(), src.get_height());\n"
    "    float2 t = float2((sp.x + tx.x) / s.texw, (in_bot ? 0.5 : 1.0) - sp.y / 384.0) * tsz - 0.5;\n"
    "    float2 i = floor(t), f = t - i;\n"
    "    float2 sc = 1.0 / max(s.duv * tsz, float2(1e-4));\n"
    "    f = clamp((f - 0.5) * sc + 0.5, 0.0, 1.0);\n"
    "    return src.sample(sm, keep_in(s, src, (i + 0.5 + f) / tsz, s.duv), gradient2d(float2(s.duv.x, 0.0), float2(0.0, s.duv.y))).rgb;\n"
    "  }\n"
    "  return pow(clamp(screen(s, src, sm, sp, in_bot ? 1 : 0), 0.0, 1.0), float3(1.0 / 2.2)); }\n"
    "fragment float4 fs_screen(QuadV in [[stage_in]], constant ScreenU& u [[buffer(0)]], texture2d<float> src [[texture(0)]],\n"
    "                          texture2d<float> skin [[texture(1)]], sampler sm [[sampler(0)]], sampler skinsm [[sampler(1)]]) {\n"
    "  S s;\n"
    "  s.view = u.view.xyz; s.win = u.win_canvas.xy; s.canvas = u.win_canvas.zw;\n"
    "  s.top = u.top; s.bot = u.bot; s.ctop = u.ctop; s.cbot = u.cbot; s.topx = u.topx_botx.xy; s.botx = u.topx_botx.zw;\n"
    "  s.lid = u.lid.rgb; s.base = u.base.rgb; s.btn = u.btn.rgb; s.hinge = u.hinge.rgb;\n"
    "  s.skinscale = u.skin_texw_alpha.xy; s.texw = u.skin_texw_alpha.z; s.alpha = u.skin_texw_alpha.w;\n"
    "  s.src_linear = u.f0.x; s.front = u.f0.y; s.lcd = u.f0.z; s.lcdcolor = u.f0.w;\n"
    "  s.grid = u.f1.x; s.shell = u.f1.y; s.skinmode = u.f1.z; s.backlight = u.f2.x;\n"
    "  s.lo = float2(0.0); s.hi = float2(1.0); s.duv = float2(0.0); s.dp = 0.0;\n"
    "  float2 wp = in.pos.xy;\n"                                /* window pixels, top-down (the window isn't flipped) */
    "  float2 p = (wp - s.view.xy) / s.view.z;\n"
    "  s.px = 1.0 / s.view.z;\n"
    "  float3 outc;\n"
    "  bool in_top = all(p >= s.top.xy) && all(p < s.top.xy + s.top.zw);\n"
    "  bool in_bot = all(p >= s.bot.xy) && all(p < s.bot.xy + s.bot.zw);\n"
    "  if (in_top || in_bot) {\n"
    "    bool both = in_top && in_bot;\n"
    "    bool fb = both ? s.front != 0 : in_bot;\n"
    "    outc = shade(s, src, sm, fb, p);\n"
    "    if (both && s.alpha < 0.999) outc = mix(shade(s, src, sm, !fb, p), outc, s.alpha);\n"
    "  } else if (s.shell != 0 && s.skinmode != 0) {\n"
    "    float3 bg = float3(0.055, 0.058, 0.065) * (1.0 - 0.35 * length((p - s.canvas * 0.5) / s.canvas));\n"
    "    float4 c = skin.sample(skinsm, p * s.skinscale);\n"
    "    bool inside = all(p >= float2(0.0)) && all(p < s.canvas);\n"
    "    outc = inside ? mix(bg, c.rgb, c.a) : bg;\n"
    "  } else if (s.shell != 0) {\n"
    "    outc = dslite(s, p);\n"
    "  } else {\n"
    "    outc = float3(0.0);\n"
    "  }\n"
    "  return float4(outc, 1.0); }\n"
    "\n"
    /* ---- HD-2D: depth of field, bloom, colour grade, vignette on the 3D layer.
     * Textures hold rows as under GL (row 0 = the bottom of the picture). */
    "struct Hd2dU { float scale, focus_keep, bloom, pad; };\n"
    /* the focus: the distance at the middle of the picture (where the hero or the
     * monsters are, in front of the background): the nearer third of 15 samples,
     * eased over frames */
    "fragment float fs_focus(QuadV in [[stage_in]], constant Hd2dU& u [[buffer(0)]], texture2d<float> wd [[texture(0)]], texture2d<float> prev [[texture(1)]]) {\n"
    "  float v[15]; int n = 0;\n"
    "  float2 sz = float2(wd.get_width(), wd.get_height());\n"
    "  for (int j = 0; j < 3; j++) for (int i = 0; i < 5; i++) {\n"
    "    float2 t = float2(0.34 + 0.08 * float(i), 1.0 - (0.45 + 0.12 * float(j)));\n"   /* x across, y from the top */
    "    float w = wd.read(uint2(t * sz)).r;\n"
    "    if (w < 1e8) v[n++] = w;\n"
    "  }\n"
    "  float old = prev.read(uint2(0)).r;\n"
    "  if (n == 0) return old;\n"
    "  for (int a = 1; a < n; a++) for (int b = a; b > 0 && v[b - 1] > v[b]; b--) { float t = v[b]; v[b] = v[b - 1]; v[b - 1] = t; }\n"
    "  float m = v[n / 3];\n"
    "  return old <= 0.0 ? m : mix(m, old, u.focus_keep); }\n"
    /* blur size in DS pixels: grows with the distance from the focus, more behind
     * it than in front; plus a little toward the top and bottom (tilt-shift) */
    "float coc(float w, float f, float ytd) {\n"
    "  if (w >= 1e8) return 3.0;\n"                              /* nothing drawn there (sky): far */
    "  float d = (w - f) / max(f, 1e-3);\n"
    "  float c = d > 0.0 ? d * 3.0 : -d * 2.5;\n"
    "  float tilt = smoothstep(0.42, 0.0, ytd) * 1.2 + smoothstep(0.8, 1.0, ytd) * 0.8;\n"
    "  return clamp(max(c - 0.15, 0.0) + tilt, 0.0, 2.5); }\n"
    "fragment float4 fs_dof(QuadV in [[stage_in]], constant Hd2dU& u [[buffer(0)]], texture2d<float> col [[texture(0)]],\n"
    "                       texture2d<float> wd [[texture(1)]], texture2d<float> foc [[texture(2)]]) {\n"
    "  int2 p = int2(in.pos.xy); int2 sz = int2(col.get_width(), col.get_height());\n"
    "  float4 c0 = col.read(uint2(p));\n"
    "  if (c0.a <= 0.0) return c0;\n"
    "  float f = foc.read(uint2(0)).r;\n"
    "  float ytd = 1.0 - (float(p.y) + 0.5) / float(sz.y);\n"
    "  float r0 = coc(wd.read(uint2(p)).r, f, ytd) * u.scale;\n"   /* in pixels of this texture */
    "  if (r0 < 0.5) return c0;\n"
    "  float3 sum = c0.rgb; float ws = 1.0;\n"
    "  for (int i = 1; i < 24; i++) {\n"                          /* a golden-angle spiral */
    "    float rr = sqrt(float(i) / 24.0) * r0, a = float(i) * 2.39996;\n"
    "    int2 q = clamp(p + int2(round(float2(cos(a), sin(a)) * rr)), int2(0), sz - 1);\n"
    "    float4 c = col.read(uint2(q));\n"
    "    if (c.a <= 0.0) continue;\n"
    "    float rq = coc(wd.read(uint2(q)).r, f, 1.0 - (float(q.y) + 0.5) / float(sz.y)) * u.scale;\n"
    "    float k = clamp(min(rq, r0) - rr + 1.0, 0.0, 1.0);\n"   /* a sharp pixel nearby doesn't spread over this one */
    "    sum += c.rgb * k; ws += k;\n"
    "  }\n"
    "  return float4(sum / ws, c0.a); }\n"
    /* bloom: the bright parts, down to a few sizes and back up */
    "fragment float4 fs_bright(QuadV in [[stage_in]], constant Hd2dU& u [[buffer(0)]], texture2d<float> src [[texture(0)]], sampler sm [[sampler(0)]]) {\n"
    "  float2 dsz = u.scale / float2(src.get_width(), src.get_height());\n"   /* one DS pixel (this texel) in the source */
    "  float3 acc = float3(0.0);\n"
    "  for (int j = 0; j < 4; j++) for (int i = 0; i < 4; i++) {\n"
    "    float4 c = src.sample(sm, in.uv + (float2(i, j) - 1.5) / 4.0 * dsz);\n"
    "    float l = dot(c.rgb, float3(0.299, 0.587, 0.114));\n"
    "    acc += c.rgb * smoothstep(0.72, 1.0, l) * step(0.001, c.a);\n"
    "  }\n"
    "  return float4(acc / 16.0, 1.0); }\n"
    "fragment float4 fs_down(QuadV in [[stage_in]], texture2d<float> src [[texture(0)]], sampler sm [[sampler(0)]]) {\n"
    "  float2 d = 1.0 / float2(src.get_width(), src.get_height());\n"
    "  float2 uv = in.uv;\n"
    "  return float4((src.sample(sm, uv + float2(-d.x, -d.y)).rgb + src.sample(sm, uv + float2(d.x, -d.y)).rgb +\n"
    "                 src.sample(sm, uv + float2(-d.x, d.y)).rgb + src.sample(sm, uv + float2(d.x, d.y)).rgb) * 0.25, 1.0); }\n"
    "fragment float4 fs_up(QuadV in [[stage_in]], texture2d<float> src [[texture(0)]], sampler sm [[sampler(0)]]) {\n"
    "  float2 d = 1.0 / float2(src.get_width(), src.get_height());\n"
    "  float2 uv = in.uv;\n"
    "  float3 c = src.sample(sm, uv).rgb * 4.0;\n"                 /* a tent: smooth when it grows */
    "  c += (src.sample(sm, uv + float2(d.x, 0)).rgb + src.sample(sm, uv - float2(d.x, 0)).rgb + src.sample(sm, uv + float2(0, d.y)).rgb + src.sample(sm, uv - float2(0, d.y)).rgb) * 2.0;\n"
    "  c += src.sample(sm, uv + d).rgb + src.sample(sm, uv - d).rgb + src.sample(sm, uv + float2(d.x, -d.y)).rgb + src.sample(sm, uv - float2(d.x, -d.y)).rgb;\n"
    "  return float4(c / 16.0, 1.0); }\n"
    /* the finish: bloom added, a warm grade, a vignette */
    "fragment float4 fs_hd2d(QuadV in [[stage_in]], constant Hd2dU& u [[buffer(0)]], texture2d<float> col [[texture(0)]],\n"
    "                        texture2d<float> bloom [[texture(1)]], sampler sm [[sampler(0)]]) {\n"
    "  float2 sz = float2(col.get_width(), col.get_height());\n"
    "  float4 c0 = col.read(uint2(in.pos.xy));\n"
    "  if (c0.a <= 0.0) return c0;\n"
    "  float2 uv = in.pos.xy / sz;\n"
    "  float3 c = c0.rgb + bloom.sample(sm, uv).rgb * (u.bloom / 5.0);\n"
    "  float l = dot(c, float3(0.299, 0.587, 0.114));\n"
    "  c = mix(float3(l), c, 1.12);\n"                             /* a little more colour */
    "  c *= mix(float3(0.97, 0.99, 1.04), float3(1.03, 1.0, 0.95), smoothstep(0.2, 0.9, l));\n"   /* cool shadows, warm light */
    "  float2 v = (uv - 0.5) * float2(1.0, 0.8);\n"
    "  c *= 1.0 - 0.35 * smoothstep(0.3, 0.75, length(v));\n"
    "  return float4(clamp(c, 0.0, 1.0), c0.a); }\n";

/* Uniforms, laid out as the shaders' structs (16-byte multiples). */
typedef struct { float toon[32][4]; float aref, edgew; int highlight, alphatest, wbuf, part, sharpa, pad; } PolyU;
typedef struct { float rect[4], uv[4]; float flip, p0, p1, p2; } QuadU;
typedef struct { float edgecol[8][4]; float fogcol[4]; float fogtab[32]; int edge, scale, aa, fog, fog_alpha_only, fogoff, fogshift, wbuf; } PostU;
typedef struct { int has3d, wide; float wtot, wside; } CompU;
typedef struct { float keep, p0, p1, p2; } GhostU;
typedef struct { float scale, focus_keep, bloom, pad; } Hd2dU;
typedef struct {
    float view[4], win_canvas[4], top[4], bot[4], ctop[4], cbot[4], topx_botx[4], lid[4], base[4], btn[4], hinge[4], skin_texw_alpha[4];
    int f0[4], f1[4]; float f2[4];
} ScreenU;

/* ---------------------------------------------------------------- state */
static id<MTLDevice> dev;
static id<MTLCommandQueue> queue;
static id<MTLCommandBuffer> cb;                               /* being recorded; committed at swap / read-back */
static id<MTLRenderPipelineState> pl_post, pl_comp, pl_ghost, pl_screen, pl_stclear;
static id<MTLRenderPipelineState> pl_focus, pl_dof, pl_bright, pl_down, pl_up, pl_hd2d;   /* the HD-2D look */
static id<MTLRenderPipelineState> pl_poly[3][2][3];             /* [blend: off, DS, replace][colour 0: none, all][colour 1: none, all, fog flag only] */
static NSMutableDictionary<NSNumber*, id<MTLDepthStencilState>>* ds_cache;
static id<MTLDepthStencilState> ds_none;
static id<MTLSamplerState> samplers[9];                         /* [s_mode * 3 + t_mode]: clamp, repeat, mirror */
static id<MTLSamplerState> smp_nearest, smp_linear, smp_mip;
static id<MTLTexture> white_tex, skin_tex;
static id<MTLTexture> ghost_tex[2]; static int ghost_cur = -1, ghost_w, ghost_h;
static int g_scale = 2;

typedef struct {
    __unsafe_unretained id<MTLTexture> color, attr, depth, post; /* owned through fb_keep */
    __unsafe_unretained id<MTLTexture> wdepth, hd2d;             /* distance from the eye; the HD-2D picture */
    int valid, hd2d_valid;
} Fb3D;
static Fb3D g_fb[2];
static NSMutableArray* fb_keep;
static int g_fb_latest = -1, g_fb_shown = -1;                  /* rendered last / visible this frame */
static int g_fb_w, g_fb_h;                                     /* one DS screen at scale */
static int g_wside, g_fb3_w;                                   /* widescreen: DS pixels added each side; 3D / final width at scale */
static int g_wide_shown = -1;                                  /* the screen composed wide last: 0 top, 1 bottom, -1 none */
static float g_half_w = 128.0f;                                /* the frame being drawn: half its width (DS pixels) */
static id<MTLTexture> final_tex;                               /* both screens, stacked, at scale (mipmapped) */
static int g_ready;

/* The HD-2D look (Graphics menu): depth of field, bloom, a warm grade and a
 * vignette on the 3D layer. Off: the picture is exactly the DS's. */
static int g_hd2d;
#define BLOOM_LEVELS 5
static id<MTLTexture> bloom_tex[BLOOM_LEVELS];                 /* DS resolution, then halves */
static id<MTLTexture> focus_tex[2]; static int focus_cur;      /* the focus distance, eased over frames */

const int* g_gl_scale = &g_scale;

/* The windows: each has its own layer; gfx_begin() picks the one drawn next. */
typedef struct { SDL_Window* win; SDL_MetalView view; __unsafe_unretained CAMetalLayer* layer; __unsafe_unretained id<CAMetalDrawable> drawable; } MtlWin;
static MtlWin g_wins[2];
static int g_cur_win;
static NSMutableArray* win_keep;                               /* keeps the layers and drawables */

static id<MTLCommandBuffer> cur_cb(void) {
    if (!cb) cb = [queue commandBuffer];
    return cb;
}
static void flush(int wait) {
    if (!cb) return;
    [cb commit];
    if (wait) [cb waitUntilCompleted];
    cb = nil;
}

/* ---------------------------------------------------------------- textures */
typedef struct { uint32_t ti, pl; uint64_t hash; void* tex; int w, h; uint64_t checked, used; int live; } TexEntry;   /* tex: retained */
#define TEXCACHE 4096
static TexEntry g_texcache[TEXCACHE];
static int g_smooth;                                           /* smooth textures (bilinear filtering) */
static uint64_t g_frameno;
static uint8_t* g_decode;

static void tex_release(TexEntry* e) { if (e->tex) CFRelease(e->tex); e->tex = NULL; }

static TexEntry* tex_lookup(uint32_t ti, uint32_t pl) {
    uint32_t key = (ti & 0x3FF0FFFFu);                        /* without repeat/flip/transform bits */
    uint32_t h = (key * 2654435761u) ^ (pl * 40503u);
    TexEntry* free_slot = NULL;
    for (int k = 0; k < 16; k++) {
        TexEntry* e = &g_texcache[(h + (uint32_t)k) & (TEXCACHE - 1)];
        if (e->live && e->ti == key && e->pl == pl) return e;
        if (!e->live && !free_slot) free_slot = e;
    }
    if (!free_slot) {                                          /* evict the least recently used in the probe window */
        TexEntry* lru = NULL;
        for (int k = 0; k < 16; k++) { TexEntry* e = &g_texcache[(h + (uint32_t)k) & (TEXCACHE - 1)]; if (!lru || e->used < lru->used) lru = e; }
        tex_release(lru);
        lru->live = 0;
        free_slot = lru;
    }
    memset(free_slot, 0, sizeof *free_slot);
    free_slot->ti = key; free_slot->pl = pl; free_slot->live = 1; free_slot->checked = 0;
    return free_slot;
}

/* Performance counters, read and reset by the frontend each frame (bug reports). */
GlStats g_gl_stats;
static char g_gl_name[300];
const char* gl_renderer_name(void) { return g_gl_name; }
int gl_render_scale(void) { return g_scale; }
static double perf_ms(void) { return (double)SDL_GetPerformanceCounter() * 1000.0 / (double)SDL_GetPerformanceFrequency(); }

/* Smooth textures: a see-through texel's colour is never shown, but filtering
 * mixes it into its neighbours'. Give each one the average colour of the
 * solid texels around it (as gl_render.c). */
static void bleed(uint8_t* px, int w, int h) {
    for (int round = 0; round < 2; round++) {
        int changed = 0;
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++) {
                uint8_t* p = px + ((size_t)y * w + x) * 4;
                if (p[3]) continue;
                int r = 0, g = 0, b = 0, n = 0;
                for (int dy = -1; dy <= 1; dy++)
                    for (int dx = -1; dx <= 1; dx++) {
                        int xx = (x + dx + w) % w, yy = (y + dy + h) % h;
                        const uint8_t* q = px + ((size_t)yy * w + xx) * 4;
                        if (q[3]) { r += q[0]; g += q[1]; b += q[2]; n++; }
                    }
                if (n) { p[0] = (uint8_t)(r / n); p[1] = (uint8_t)(g / n); p[2] = (uint8_t)(b / n); p[3] = 1; changed = 1; }
            }
        if (!changed) break;
    }
    for (size_t i = 0; i < (size_t)w * h; i++) if (px[i * 4 + 3] == 1) px[i * 4 + 3] = 0;
}

static id<MTLTexture> new_tex(MTLPixelFormat fmt, int w, int h, int target, int mips) {
    MTLTextureDescriptor* d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:fmt width:(NSUInteger)w height:(NSUInteger)h mipmapped:mips ? YES : NO];
    d.usage = MTLTextureUsageShaderRead | (target ? MTLTextureUsageRenderTarget : 0);
    d.storageMode = target ? MTLStorageModePrivate : MTLStorageModeShared;
    return [dev newTextureWithDescriptor:d];
}
/* A texture with these pixels (rows as given). A new one each time: the GPU
 * may still be drawing with the previous contents. */
static id<MTLTexture> upload(MTLPixelFormat fmt, int w, int h, const void* px, int bpp) {
    id<MTLTexture> t = new_tex(fmt, w, h, 0, 0);
    [t replaceRegion:MTLRegionMake2D(0, 0, (NSUInteger)w, (NSUInteger)h) mipmapLevel:0 withBytes:px bytesPerRow:(NSUInteger)(w * bpp)];
    return t;
}

static id<MTLTexture> texture_for(uint32_t ti, uint32_t pl, int* w, int* h) {
    TexEntry* e = tex_lookup(ti, pl);
    gpu3d_tex_size(ti, w, h);
    e->used = g_frameno;
    if (e->checked != g_frameno) {
        e->checked = g_frameno;
        uint64_t hs = gpu3d_texture_hash(ti, pl);
        if (hs != e->hash || e->w != *w || e->h != *h || !e->tex) {
            double t0 = perf_ms();
            e->hash = hs; e->w = *w; e->h = *h;
            gpu3d_decode_texture(ti, pl, g_decode);
            if (g_smooth) bleed(g_decode, *w, *h);
            tex_release(e);
            e->tex = (void*)CFBridgingRetain(upload(MTLPixelFormatRGBA8Unorm, *w, *h, g_decode, 4));
            g_gl_stats.uploads++; g_gl_stats.upload_texels += *w * *h; g_gl_stats.upload_ms += perf_ms() - t0;
        }
    }
    return (__bridge id<MTLTexture>)e->tex;
}

/* ---------------------------------------------------------------- setup */
static void make_fbs(void) {
    g_fb_w = 256 * g_scale; g_fb_h = 192 * g_scale;
    g_fb3_w = (256 + 2 * g_wside) * g_scale;
    fb_keep = [NSMutableArray array];
    for (int i = 0; i < 2; i++) {
        Fb3D* f = &g_fb[i];
        id<MTLTexture> c = new_tex(MTLPixelFormatRGBA8Unorm, g_fb3_w, g_fb_h, 1, 0);
        id<MTLTexture> a = new_tex(MTLPixelFormatRGBA8Unorm, g_fb3_w, g_fb_h, 1, 0);
        id<MTLTexture> d = new_tex(MTLPixelFormatDepth32Float_Stencil8, g_fb3_w, g_fb_h, 1, 0);   /* stencil: the DS's per-pixel flags */
        id<MTLTexture> p = new_tex(MTLPixelFormatRGBA8Unorm, g_fb3_w, g_fb_h, 1, 0);
        id<MTLTexture> wd = new_tex(MTLPixelFormatR32Float, g_fb3_w, g_fb_h, 1, 0);
        id<MTLTexture> hd = new_tex(MTLPixelFormatRGBA8Unorm, g_fb3_w, g_fb_h, 1, 0);
        [fb_keep addObjectsFromArray:@[c, a, d, p, wd, hd]];
        f->color = c; f->attr = a; f->depth = d; f->post = p; f->wdepth = wd; f->hd2d = hd;
        f->valid = f->hd2d_valid = 0;
    }
    for (int i = 0; i < BLOOM_LEVELS; i++)
        bloom_tex[i] = new_tex(MTLPixelFormatRGBA16Float, (256 + 2 * g_wside) >> i, 192 >> i, 1, 0);
    final_tex = new_tex(MTLPixelFormatRGBA8Unorm, g_fb3_w, g_fb_h * 2, 1, 1);
    g_fb_latest = g_fb_shown = -1;
}

static void free_fbs(void) {
    for (int i = 0; i < 2; i++) memset(&g_fb[i], 0, sizeof g_fb[i]);
    fb_keep = nil; final_tex = nil;                            /* command buffers in flight keep what they use */
    for (int i = 0; i < BLOOM_LEVELS; i++) bloom_tex[i] = nil;
}

/* The largest 3D resolution this GPU can hold: widescreen up to 512 DS
 * pixels across, both screens 384 high in one texture. */
static int max_scale(void) {
    int m = [dev supportsFamily:MTLGPUFamilyApple3] || [dev supportsFamily:MTLGPUFamilyMac2] ? 16384 : 8192;
    int s = m / 512;
    return s < 1 ? 1 : s > GL_MAX_SCALE ? GL_MAX_SCALE : s;
}
int gl_render_max_scale(void) { return g_ready ? max_scale() : GL_MAX_SCALE; }

static id<MTLSamplerState> sampler(MTLSamplerMinMagFilter f, MTLSamplerMipFilter mip, MTLSamplerAddressMode s, MTLSamplerAddressMode t) {
    MTLSamplerDescriptor* d = [MTLSamplerDescriptor new];
    d.minFilter = d.magFilter = f; d.mipFilter = mip;
    d.sAddressMode = s; d.tAddressMode = t;
    return [dev newSamplerStateWithDescriptor:d];
}
static void make_samplers(void) {
    static const MTLSamplerAddressMode wrap[3] = { MTLSamplerAddressModeClampToEdge, MTLSamplerAddressModeRepeat, MTLSamplerAddressModeMirrorRepeat };
    for (int i = 0; i < 9; i++)
        samplers[i] = sampler(g_smooth ? MTLSamplerMinMagFilterLinear : MTLSamplerMinMagFilterNearest, MTLSamplerMipFilterNotMipmapped, wrap[i / 3], wrap[i % 3]);
}

static id<MTLRenderPipelineState> pipeline_add(id<MTLLibrary> lib, NSString* vs, NSString* fs, MTLPixelFormat fmt, int add) {
    MTLRenderPipelineDescriptor* d = [MTLRenderPipelineDescriptor new];
    d.vertexFunction = [lib newFunctionWithName:vs];
    d.fragmentFunction = [lib newFunctionWithName:fs];
    d.colorAttachments[0].pixelFormat = fmt;
    if (add) {                                                 /* added to what is there */
        d.colorAttachments[0].blendingEnabled = YES;
        d.colorAttachments[0].sourceRGBBlendFactor = d.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOne;
        d.colorAttachments[0].sourceAlphaBlendFactor = d.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorOne;
    }
    NSError* err = nil;
    id<MTLRenderPipelineState> p = [dev newRenderPipelineStateWithDescriptor:d error:&err];
    if (!p) fprintf(stderr, "Metal: pipeline %s: %s\n", fs.UTF8String, err.localizedDescription.UTF8String);
    return p;
}
static id<MTLRenderPipelineState> pipeline(id<MTLLibrary> lib, NSString* vs, NSString* fs, MTLPixelFormat fmt) { return pipeline_add(lib, vs, fs, fmt, 0); }

/* The polygon pass's pipelines: blending of colour 0, and which channels each target keeps. */
static id<MTLRenderPipelineState> poly_pipeline(id<MTLLibrary> lib, NSString* fs, int blend, int mask0, int mask1) {
    MTLRenderPipelineDescriptor* d = [MTLRenderPipelineDescriptor new];
    d.vertexFunction = [lib newFunctionWithName:fs == nil ? @"vs_poly" : @"vs_quad"];
    d.fragmentFunction = [lib newFunctionWithName:fs == nil ? @"fs_poly" : fs];
    d.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
    d.colorAttachments[1].pixelFormat = MTLPixelFormatRGBA8Unorm;
    d.colorAttachments[2].pixelFormat = MTLPixelFormatR32Float;
    d.colorAttachments[2].writeMask = mask1 == 1 ? MTLColorWriteMaskRed : MTLColorWriteMaskNone;   /* the distance: with the pixel's attributes */
    d.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float_Stencil8;
    d.stencilAttachmentPixelFormat = MTLPixelFormatDepth32Float_Stencil8;
    d.colorAttachments[0].writeMask = mask0 ? MTLColorWriteMaskAll : MTLColorWriteMaskNone;
    d.colorAttachments[1].writeMask = mask1 == 0 ? MTLColorWriteMaskNone : mask1 == 1 ? MTLColorWriteMaskAll : MTLColorWriteMaskGreen;
    if (blend) {                                               /* alpha: the larger of the two (the factors don't count for max) */
        MTLRenderPipelineColorAttachmentDescriptor* a = d.colorAttachments[0];
        a.blendingEnabled = YES;
        a.rgbBlendOperation = MTLBlendOperationAdd;
        a.alphaBlendOperation = MTLBlendOperationMax;
        a.sourceRGBBlendFactor = blend == 1 ? MTLBlendFactorSourceAlpha : MTLBlendFactorOne;
        a.destinationRGBBlendFactor = blend == 1 ? MTLBlendFactorOneMinusSourceAlpha : MTLBlendFactorZero;
        a.sourceAlphaBlendFactor = a.destinationAlphaBlendFactor = MTLBlendFactorOne;
    }
    NSError* err = nil;
    id<MTLRenderPipelineState> p = [dev newRenderPipelineStateWithDescriptor:d error:&err];
    if (!p) fprintf(stderr, "Metal: polygon pipeline: %s\n", err.localizedDescription.UTF8String);
    return p;
}

int gl_render_init(int scale) {
    @autoreleasepool {
    if (!dev) dev = MTLCreateSystemDefaultDevice();
    if (!dev) { fprintf(stderr, "Metal: no GPU\n"); return 0; }
    queue = [dev newCommandQueue];
    g_scale = scale < 1 ? 1 : scale > max_scale() ? max_scale() : scale;
    MTLCompileOptions* opt = [MTLCompileOptions new];
    if (@available(macOS 15.0, *)) opt.mathMode = MTLMathModeSafe;   /* the same arithmetic as GL's shaders */
    else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
        opt.fastMathEnabled = NO;
#pragma clang diagnostic pop
    }
    NSError* err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:@(MSL) options:opt error:&err];
    if (!lib) { fprintf(stderr, "Metal shader error:\n%s\n", err.localizedDescription.UTF8String); return 0; }
    for (int b = 0; b < 3; b++) for (int m0 = 0; m0 < 2; m0++) for (int m1 = 0; m1 < 3; m1++)
        if (!(pl_poly[b][m0][m1] = poly_pipeline(lib, nil, b, m0, m1))) return 0;
    pl_stclear = poly_pipeline(lib, @"fs_nothing", 0, 0, 0);
    pl_post = pipeline(lib, @"vs_quad", @"fs_post", MTLPixelFormatRGBA8Unorm);
    pl_comp = pipeline(lib, @"vs_quad", @"fs_comp", MTLPixelFormatRGBA8Unorm);
    pl_ghost = pipeline(lib, @"vs_quad", @"fs_ghost", MTLPixelFormatRGBA16Float);
    pl_screen = pipeline(lib, @"vs_quad", @"fs_screen", MTLPixelFormatBGRA8Unorm);
    if (!pl_stclear || !pl_post || !pl_comp || !pl_ghost || !pl_screen) return 0;
    pl_focus = pipeline(lib, @"vs_quad", @"fs_focus", MTLPixelFormatR32Float);
    pl_dof = pipeline(lib, @"vs_quad", @"fs_dof", MTLPixelFormatRGBA8Unorm);
    pl_bright = pipeline(lib, @"vs_quad", @"fs_bright", MTLPixelFormatRGBA16Float);
    pl_down = pipeline(lib, @"vs_quad", @"fs_down", MTLPixelFormatRGBA16Float);
    pl_up = pipeline_add(lib, @"vs_quad", @"fs_up", MTLPixelFormatRGBA16Float, 1);
    pl_hd2d = pipeline(lib, @"vs_quad", @"fs_hd2d", MTLPixelFormatRGBA8Unorm);
    for (int i = 0; i < 2; i++) focus_tex[i] = new_tex(MTLPixelFormatR32Float, 1, 1, 1, 0);
    ds_cache = [NSMutableDictionary dictionary];
    ds_none = [dev newDepthStencilStateWithDescriptor:[MTLDepthStencilDescriptor new]];
    make_samplers();
    smp_nearest = sampler(MTLSamplerMinMagFilterNearest, MTLSamplerMipFilterNotMipmapped, MTLSamplerAddressModeClampToEdge, MTLSamplerAddressModeClampToEdge);
    smp_linear = sampler(MTLSamplerMinMagFilterLinear, MTLSamplerMipFilterNotMipmapped, MTLSamplerAddressModeClampToEdge, MTLSamplerAddressModeClampToEdge);
    smp_mip = sampler(MTLSamplerMinMagFilterLinear, MTLSamplerMipFilterLinear, MTLSamplerAddressModeClampToEdge, MTLSamplerAddressModeClampToEdge);
    static const uint8_t white[4] = { 255, 255, 255, 255 };
    white_tex = upload(MTLPixelFormatRGBA8Unorm, 1, 1, white, 4);
    g_decode = malloc(1024 * 1024 * 4);
    make_fbs();
    g_ready = 1;
    snprintf(g_gl_name, sizeof g_gl_name, "%s / Metal", dev.name.UTF8String);
    fprintf(stderr, "Metal renderer: %s, 3D at %dx\n", g_gl_name, g_scale);
    return 1;
    }
}

/* Smooth textures: filter the DS's textures (the DS shows each texel as a
 * block). The textures are decoded again (see bleed). */
void gl_set_smooth(int on) {
    on = on != 0;
    if (!g_ready) { g_smooth = on; return; }
    if (on == g_smooth) return;
    g_smooth = on;
    make_samplers();
    for (int i = 0; i < TEXCACHE; i++) g_texcache[i].hash = 0;   /* decode again */
}

void gl_render_set_scale(int scale) {
    if (scale < 1) scale = 1;
    if (g_ready && scale > max_scale()) scale = max_scale();
    if (!g_ready || scale == g_scale) return;
    free_fbs();
    g_scale = scale;
    make_fbs();
    fprintf(stderr, "3D at %dx\n", g_scale);
}

/* ---------------------------------------------------------------- 3D frame */
#define IO16_(o) (*(uint16_t*)&g_io[o])
#define IO32_(o) (*(uint32_t*)&g_io[o])

static float* g_verts;
static int g_vcap;

/* corner: which corner of its triangle this vertex is (0-2); ext: bit k set if the
 * triangle edge opposite corner k is an edge of the DS polygon. */
static void push_vertex(int* nv, const SVtx* v, float alpha, const Poly* p, float textured, int corner, int ext) {
    if (*nv + 1 > g_vcap) { g_vcap = g_vcap ? g_vcap * 2 : 16384; g_verts = realloc(g_verts, sizeof(float) * VTX_FLOATS * (size_t)g_vcap); }
    float* o = g_verts + (size_t)*nv * VTX_FLOATS;
    float w = v->iw > 1e-12f ? 1.0f / v->iw : 1e12f;
    float nx = v->x / g_half_w - 1.0f, ny = 1.0f - v->y / 96.0f, nz = v->zn * 2.0f - 1.0f;
    o[0] = nx * w; o[1] = ny * w; o[2] = nz * w; o[3] = w;
    o[4] = v->r; o[5] = v->g; o[6] = v->b; o[7] = alpha;
    o[8] = v->s; o[9] = v->t;
    o[10] = (float)((p->attr >> 4) & 3); o[11] = textured;
    o[12] = (float)((p->attr >> 24) & 63) / 63.0f; o[13] = (float)((p->attr >> 15) & 1);
    o[14] = corner == 0; o[15] = corner == 1; o[16] = corner == 2;
    o[17] = (float)ext;
    (*nv)++;
}

#define ST_SOLID 0x80                    /* stencil: the pixel has a colour (alpha > 0) */
#define ST_TRANS 0x40                    /* a translucent polygon is on it (its id: bits 0-4, of the DS's 0-5) */
#define ST_MASKBIT 0x20                  /* marked by the latest shadow mask */
#define ST_ID 0x1F
typedef struct { int first, count; __unsafe_unretained id<MTLTexture> tex; int sampler; float tw, th; int translucent, depth_write, depth_eq, lines, pid, shadow, fog, alpha1; } Batch;
static Batch* g_batches; static int g_bcap;

/* The GL state gl_render.c sets, kept as it is and turned into Metal objects at each draw. */
typedef struct {
    int blend;                           /* 0 off, 1 the DS's blending, 2 replace (alpha still the larger) */
    int mask0, mask1;                    /* colour 0: none / all; colour 1: none / all / fog flag only */
    MTLCompareFunction depth_func; int depth_write;
    MTLCompareFunction st_func; uint32_t st_ref, st_read, st_write;
    MTLStencilOperation st_fail, st_zfail, st_pass;
} DrawState;

static id<MTLDepthStencilState> depth_stencil(const DrawState* s) {
    uint64_t key = (uint64_t)s->depth_func | (uint64_t)s->depth_write << 3 | (uint64_t)s->st_func << 4 | (uint64_t)s->st_read << 8 |
                   (uint64_t)s->st_write << 16 | (uint64_t)s->st_fail << 24 | (uint64_t)s->st_zfail << 28 | (uint64_t)s->st_pass << 32;
    NSNumber* k = @(key);
    id<MTLDepthStencilState> st = ds_cache[k];
    if (st) return st;
    MTLDepthStencilDescriptor* d = [MTLDepthStencilDescriptor new];
    d.depthCompareFunction = s->depth_func;
    d.depthWriteEnabled = s->depth_write ? YES : NO;
    MTLStencilDescriptor* sd = [MTLStencilDescriptor new];
    sd.stencilCompareFunction = s->st_func;
    sd.readMask = s->st_read; sd.writeMask = s->st_write;
    sd.stencilFailureOperation = s->st_fail; sd.depthFailureOperation = s->st_zfail; sd.depthStencilPassOperation = s->st_pass;
    d.frontFaceStencil = sd; d.backFaceStencil = sd;
    st = [dev newDepthStencilStateWithDescriptor:d];
    ds_cache[k] = st;
    return st;
}

static void draw(id<MTLRenderCommandEncoder> enc, const DrawState* s, const PolyU* u, MTLPrimitiveType prim, int first, int count) {
    [enc setRenderPipelineState:pl_poly[s->blend][s->mask0][s->mask1]];
    [enc setDepthStencilState:depth_stencil(s)];
    [enc setStencilReferenceValue:s->st_ref];
    [enc setFragmentBytes:u length:sizeof *u atIndex:0];
    [enc drawPrimitives:prim vertexStart:(NSUInteger)first vertexCount:(NSUInteger)count];
}

/* one full-target quad pass (rows as under GL) */
static id<MTLRenderCommandEncoder> quad_pass(id<MTLTexture> target, id<MTLRenderPipelineState> pl, int load) {
    MTLRenderPassDescriptor* rp = [MTLRenderPassDescriptor renderPassDescriptor];
    rp.colorAttachments[0].texture = target;
    rp.colorAttachments[0].loadAction = load ? MTLLoadActionLoad : MTLLoadActionDontCare;
    rp.colorAttachments[0].storeAction = MTLStoreActionStore;
    id<MTLRenderCommandEncoder> enc = [cur_cb() renderCommandEncoderWithDescriptor:rp];
    [enc setRenderPipelineState:pl];
    QuadU q = { { -1, -1, 2, 2 }, { 0, 0, 1, 1 }, -1, 0, 0, 0 };
    [enc setVertexBytes:&q length:sizeof q atIndex:0];
    return enc;
}
static void quad_end(id<MTLRenderCommandEncoder> enc) {
    [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:6];
    [enc endEncoding];
}

/* The HD-2D look on the finished 3D layer: post -> depth of field -> + bloom, grade, vignette -> hd2d.
 * The colour goes through fb->color as scratch (the polygon pass is done with it). */
static void hd2d_passes(Fb3D* fb) {
    Hd2dU u = { (float)g_scale, 0.85f, 0.8f, 0 };
    int prev = focus_cur, cur = focus_cur ^ 1;
    id<MTLRenderCommandEncoder> enc = quad_pass(focus_tex[cur], pl_focus, 0);
    [enc setFragmentBytes:&u length:sizeof u atIndex:0];
    [enc setFragmentTexture:fb->wdepth atIndex:0];
    [enc setFragmentTexture:focus_tex[prev] atIndex:1];
    quad_end(enc);
    focus_cur = cur;
    enc = quad_pass(fb->color, pl_dof, 0);
    [enc setFragmentBytes:&u length:sizeof u atIndex:0];
    [enc setFragmentTexture:fb->post atIndex:0];
    [enc setFragmentTexture:fb->wdepth atIndex:1];
    [enc setFragmentTexture:focus_tex[cur] atIndex:2];
    quad_end(enc);
    enc = quad_pass(bloom_tex[0], pl_bright, 0);
    [enc setFragmentBytes:&u length:sizeof u atIndex:0];
    [enc setFragmentTexture:fb->color atIndex:0];
    [enc setFragmentSamplerState:smp_linear atIndex:0];
    quad_end(enc);
    for (int i = 1; i < BLOOM_LEVELS; i++) {
        enc = quad_pass(bloom_tex[i], pl_down, 0);
        [enc setFragmentTexture:bloom_tex[i - 1] atIndex:0];
        [enc setFragmentSamplerState:smp_linear atIndex:0];
        quad_end(enc);
    }
    for (int i = BLOOM_LEVELS - 2; i >= 0; i--) {
        enc = quad_pass(bloom_tex[i], pl_up, 1);
        [enc setFragmentTexture:bloom_tex[i + 1] atIndex:0];
        [enc setFragmentSamplerState:smp_linear atIndex:0];
        quad_end(enc);
    }
    enc = quad_pass(fb->hd2d, pl_hd2d, 0);
    [enc setFragmentBytes:&u length:sizeof u atIndex:0];
    [enc setFragmentTexture:fb->color atIndex:0];
    [enc setFragmentTexture:bloom_tex[0] atIndex:1];
    [enc setFragmentSamplerState:smp_linear atIndex:0];
    quad_end(enc);
}

void gl_set_hd2d(int on) { g_hd2d = on != 0; }
int gl_effects(void) { return GL_EFFECT_HD2D; }

static void gl_vblank(void) {
    g_fb_shown = g_fb_latest;                                  /* what was rendered last is visible now */
}

static void gl_render_(const Frame3D* f);
static void gl_render(const Frame3D* f) {
    double t0 = perf_ms();
    @autoreleasepool { gl_render_(f); }
    g_gl_stats.render_ms += perf_ms() - t0;
    g_gl_stats.polys += f->n;
}
static void gl_render_(const Frame3D* f) {
    g_frameno++;
    {   /* tests: DQIX_GL_PROBE="f,x,y": on the first 3D frame from frame f, list the polygons over DS pixel x,y */
        static int pn = -2, px, py; const char* e;
        if (pn == -2) { pn = -1; if ((e = getenv("DQIX_GL_PROBE"))) sscanf(e, "%d,%d,%d", &pn, &px, &py); }
        static int done; if (pn >= 0 && (int)g_frame >= pn && !done && (done = 1)) {
            fprintf(stderr, "probe: frame %llu disp3d %04x wbuf %d bldcnt %04x bldalpha %04x clear %08x fogcol %04x/%02x fogoff %04x fogtab", (unsigned long long)g_frame, IO16_(0x060), f->wbuf, IO16_(0x050), IO16_(0x052), IO32_(0x350), IO16_(0x358), g_io[0x35A], IO16_(0x35C));
            for (int k = 0; k < 32; k++) fprintf(stderr, " %d", g_io[0x360 + k] & 127);
            fprintf(stderr, "\n");
        }
        if (pn >= 0 && done == 1 && (done = 2)) for (int i = 0; i < f->n; i++) {
            const Poly* p = f->order[i]; float x0 = 1e9f, x1 = -1e9f;
            for (int k = 0; k < p->n; k++) { if (p->v[k].x < x0) x0 = p->v[k].x; if (p->v[k].x > x1) x1 = p->v[k].x; }
            if (px < x0 || px > x1 || py < p->ymin || py > p->ymax) continue;
            fprintf(stderr, "probe %d: %s attr %08x alpha %u mode %u id %u fmt %u tex %08x pal %04x n %d col %.0f,%.0f,%.0f z %.4f\n", i,
                    i < f->n_opaque ? "opaque" : "transl", p->attr, (p->attr >> 16) & 31, (p->attr >> 4) & 3, (p->attr >> 24) & 63,
                    (p->teximg >> 26) & 7, p->teximg, p->pltt, p->n, p->v[0].r, p->v[0].g, p->v[0].b, p->v[0].zn);
        }
    }
    int side = f->width > 256 ? (f->width - 256) / 2 : 0;
    if (side != g_wside) { free_fbs(); g_wside = side; make_fbs(); }   /* widescreen on/off: buffers of the new width */
    g_half_w = (float)(256 + 2 * side) / 2.0f;
    uint16_t disp3d = IO16_(0x060);
    int textures_on = disp3d & 1;
    /* vertices + batches */
    int nv = 0, nb = 0;
    NSMutableArray* used = [NSMutableArray array];             /* this frame's textures, even if the cache drops one meanwhile */
    for (int i = 0; i < f->n; i++) {
        const Poly* p = f->order[i];
        int fmt = (p->teximg >> 26) & 7;
        int textured = textures_on && fmt != 0;
        int alpha = (p->attr >> 16) & 31;
        int wire = alpha == 0;
        Batch b;
        memset(&b, 0, sizeof b);
        b.tex = white_tex; b.tw = 1; b.th = 1;
        if (textured) {
            int w, h;
            b.tex = texture_for(p->teximg, p->pltt, &w, &h);
            if (nb == 0 || g_batches[nb - 1].tex != b.tex) [used addObject:b.tex];
            b.tw = (float)w; b.th = (float)h;
            int sm = (p->teximg >> 16) & 1 ? ((p->teximg >> 18) & 1 ? 2 : 1) : 0;
            int tm = (p->teximg >> 17) & 1 ? ((p->teximg >> 19) & 1 ? 2 : 1) : 0;
            b.sampler = sm * 3 + tm;
            b.alpha1 = fmt != 1 && fmt != 6;                      /* no alpha levels (only A3I5 and A5I3 have them) */
        }
        b.translucent = i >= f->n_opaque;
        b.depth_write = !b.translucent || (p->attr & 0x800);
        b.depth_eq = (p->attr >> 14) & 1;
        b.lines = wire;
        b.pid = (p->attr >> 24) & 63;
        b.fog = (p->attr >> 15) & 1;
        b.shadow = ((p->attr >> 4) & 3) == 3 ? (b.pid == 0 ? 1 : 2) : 0;   /* 1 the shadow's mask, 2 the shadow */
        b.first = nv;
        float a = wire ? 31.0f : (float)alpha;
        if (wire) {
            for (int k = 0; k < p->n; k++) { push_vertex(&nv, &p->v[k], a, p, (float)textured, -1, 7); push_vertex(&nv, &p->v[(k + 1) % p->n], a, p, (float)textured, -1, 7); }
        } else {
            for (int k = 1; k + 1 < p->n; k++) {                  /* fan: (v0, vk, vk+1) */
                int ext = 1 | (k + 2 == p->n ? 2 : 0) | (k == 1 ? 4 : 0);
                push_vertex(&nv, &p->v[0], a, p, (float)textured, 0, ext);
                push_vertex(&nv, &p->v[k], a, p, (float)textured, 1, ext);
                push_vertex(&nv, &p->v[k + 1], a, p, (float)textured, 2, ext);
            }
        }
        b.count = nv - b.first;
        if (nb > 0) {
            Batch* l = &g_batches[nb - 1];
            if (l->tex == b.tex && l->sampler == b.sampler && l->translucent == b.translucent && l->depth_write == b.depth_write &&
                l->depth_eq == b.depth_eq && l->lines == b.lines && l->first + l->count == b.first &&
                (!b.translucent || (l->pid == b.pid && l->fog == b.fog)) && l->shadow == b.shadow && !b.shadow) { l->count += b.count; continue; }
        }
        if (nb == g_bcap) { g_bcap = g_bcap ? g_bcap * 2 : 1024; g_batches = realloc(g_batches, sizeof(Batch) * (size_t)g_bcap); }
        g_batches[nb++] = b;
    }

    int target = g_fb_latest < 0 ? 0 : (g_fb_latest ^ 1);
    if (target == g_fb_shown) target ^= 1;
    Fb3D* fb = &g_fb[target];

    /* clear: color (straight), attributes (clear polygon id, fog), depth and the per-pixel flags */
    uint32_t clear = IO32_(0x350);
    uint16_t cdepth = IO16_(0x354) & 0x7FFF;
    float ca = (float)((clear >> 16) & 31) / 31.0f;
    uint32_t d24 = (uint32_t)cdepth * 0x200 + ((cdepth + 1) / 0x8000) * 0x1FF;
    float cd = cdepth == 0x7FFF ? 1.0f : f->wbuf ? fminf((float)d24 / 4096.0f / 4096.0f, 1.0f) : (float)d24 / 16777215.0f;
    cd = (float)(rint((double)cd * 16777215.0) / 16777215.0);   /* as a 24-bit depth buffer keeps it */
    MTLRenderPassDescriptor* rp = [MTLRenderPassDescriptor renderPassDescriptor];
    rp.colorAttachments[0].texture = fb->color;
    rp.colorAttachments[0].loadAction = MTLLoadActionClear;
    rp.colorAttachments[0].clearColor = MTLClearColorMake((clear & 31) / 31.0, ((clear >> 5) & 31) / 31.0, ((clear >> 10) & 31) / 31.0, ca);
    rp.colorAttachments[0].storeAction = MTLStoreActionStore;
    rp.colorAttachments[1].texture = fb->attr;
    rp.colorAttachments[1].loadAction = MTLLoadActionClear;
    rp.colorAttachments[1].clearColor = MTLClearColorMake(((clear >> 24) & 63) / 63.0, (clear >> 15) & 1, 0, 0);
    rp.colorAttachments[1].storeAction = MTLStoreActionStore;
    rp.colorAttachments[2].texture = fb->wdepth;
    rp.colorAttachments[2].loadAction = MTLLoadActionClear;
    rp.colorAttachments[2].clearColor = MTLClearColorMake(1e9, 0, 0, 0);   /* nothing drawn: far away */
    rp.colorAttachments[2].storeAction = g_hd2d ? MTLStoreActionStore : MTLStoreActionDontCare;
    rp.depthAttachment.texture = fb->depth;
    rp.depthAttachment.loadAction = MTLLoadActionClear;
    rp.depthAttachment.clearDepth = cd;
    rp.depthAttachment.storeAction = MTLStoreActionStore;
    rp.stencilAttachment.texture = fb->depth;
    rp.stencilAttachment.loadAction = MTLLoadActionClear;
    rp.stencilAttachment.clearStencil = ca > 0.0f ? ST_SOLID : 0;
    rp.stencilAttachment.storeAction = MTLStoreActionDontCare;
    id<MTLRenderCommandEncoder> enc = [cur_cb() renderCommandEncoderWithDescriptor:rp];

    PolyU u; memset(&u, 0, sizeof u);
    { static float ew = -1; if (ew < 0) { const char* e = getenv("DQIX_EDGEW"); ew = e ? (float)atof(e) : 0; }
      /* outlines: one DS pixel wide at 1x; above that half of one (see gl_render.c) */
      u.edgew = ew > 0 ? ew : g_scale >= 2 ? (float)g_scale * 0.5f : 1.0f; }
    u.highlight = (disp3d >> 1) & 1;
    u.alphatest = (disp3d >> 2) & 1;
    u.aref = (float)(IO16_(0x340) & 31);
    u.wbuf = f->wbuf;
    for (int i = 0; i < 32; i++) { uint16_t c = IO16_(0x380 + i * 2); u.toon[i][0] = (float)(c & 31) / 31.0f; u.toon[i][1] = (float)((c >> 5) & 31) / 31.0f; u.toon[i][2] = (float)((c >> 10) & 31) / 31.0f; }
    u.part = 0;
    id<MTLBuffer> vb = nv > 0 ? [dev newBufferWithBytes:g_verts length:sizeof(float) * VTX_FLOATS * (size_t)nv options:MTLResourceStorageModeShared] : nil;
    [enc setVertexBuffer:vb offset:0 atIndex:0];
    int blend_on = (disp3d >> 3) & 1;
    /* The DS's rules for translucent pixels, kept per pixel in the stencil buffer
     * (ST_SOLID, ST_TRANS + id; shadows: ST_MASKBIT). See gl_render.c. */
    DrawState s = { 0, 1, 1, MTLCompareFunctionLess, 1, MTLCompareFunctionAlways, 0, 0xFF, 0xFF,
                    MTLStencilOperationKeep, MTLStencilOperationKeep, MTLStencilOperationKeep };
    int in_mask = 0;
    for (int i = 0; i < nb; i++) {
        Batch* b = &g_batches[i];
        [enc setFragmentTexture:b->tex atIndex:0];
        [enc setFragmentSamplerState:samplers[b->sampler] atIndex:0];
        float ts[2] = { b->tw, b->th };
        [enc setVertexBytes:ts length:sizeof ts atIndex:1];
        u.sharpa = g_smooth && b->alpha1;
        s.depth_func = b->depth_eq ? MTLCompareFunctionEqual : MTLCompareFunctionLess;
        s.depth_write = b->depth_write;
        MTLPrimitiveType prim = b->lines ? MTLPrimitiveTypeLine : MTLPrimitiveTypeTriangle;
        if (b->shadow == 1) {                                   /* shadow mask: no colour; mark where it fails the depth test */
            s.mask0 = 0; s.mask1 = 0;
            s.depth_write = 0;
            s.st_write = ST_MASKBIT;
            if (!in_mask) {                                    /* a new mask: the old marks go (a clear through the write mask) */
                DrawState c = s;
                c.depth_func = MTLCompareFunctionAlways; c.st_func = MTLCompareFunctionAlways; c.st_ref = 0;
                c.st_fail = c.st_zfail = c.st_pass = MTLStencilOperationReplace;
                [enc setRenderPipelineState:pl_stclear];
                [enc setDepthStencilState:depth_stencil(&c)];
                [enc setStencilReferenceValue:0];
                QuadU q = { { -1, -1, 2, 2 }, { 0, 0, 1, 1 }, -1, 0, 0, 0 };
                [enc setVertexBytes:&q length:sizeof q atIndex:0];
                [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:6];
                [enc setVertexBuffer:vb offset:0 atIndex:0];       /* the polygons again */
            }
            in_mask = 1;
            s.st_func = MTLCompareFunctionAlways; s.st_ref = ST_MASKBIT; s.st_read = ST_MASKBIT;
            s.st_fail = MTLStencilOperationKeep; s.st_zfail = MTLStencilOperationReplace; s.st_pass = MTLStencilOperationKeep;
            draw(enc, &s, &u, prim, b->first, b->count);
            s.mask0 = 1;
            continue;
        }
        in_mask = 0;
        if (!b->translucent && !b->shadow) {
            s.blend = 0;
            s.mask1 = 1;
            s.st_write = 0xFF;
            s.st_func = MTLCompareFunctionAlways; s.st_ref = ST_SOLID; s.st_read = 0xFF;
            s.st_fail = MTLStencilOperationKeep; s.st_zfail = MTLStencilOperationKeep; s.st_pass = MTLStencilOperationReplace;
            draw(enc, &s, &u, prim, b->first, b->count);
            continue;
        }
        /* a translucent polygon's solid pixels (alpha 31) are drawn like an opaque polygon's on the DS */
        if (b->shadow != 2) {
            u.part = 1;
            s.blend = 0;
            s.mask1 = 1;
            s.depth_write = 1;
            s.st_write = 0xFF;
            s.st_func = MTLCompareFunctionAlways; s.st_ref = ST_SOLID; s.st_read = 0xFF;
            s.st_fail = MTLStencilOperationKeep; s.st_zfail = MTLStencilOperationKeep; s.st_pass = MTLStencilOperationReplace;
            draw(enc, &s, &u, prim, b->first, b->count);
            s.depth_write = b->depth_write;
            u.part = 2;
        }
        uint32_t ref = ST_SOLID | ST_TRANS | (b->pid & ST_ID);
        /* translucent pixels keep the opaque polygon id; their fog flag is the pixel's AND the polygon's */
        s.mask1 = b->fog ? 0 : 2;
        if (b->shadow == 2) {                                   /* the shadow: only on marked pixels */
            s.blend = blend_on ? 1 : 0;
            s.st_write = 0xFF;
            s.st_func = MTLCompareFunctionEqual; s.st_ref = ref | ST_MASKBIT; s.st_read = ST_MASKBIT;   /* the marks stay for the next shadow, like the DS */
            s.st_fail = MTLStencilOperationKeep; s.st_zfail = MTLStencilOperationKeep; s.st_pass = MTLStencilOperationReplace;
            draw(enc, &s, &u, prim, b->first, b->count);
            u.part = 0;
            continue;
        }
        s.st_write = 0xFF;
        s.st_fail = MTLStencilOperationKeep; s.st_zfail = MTLStencilOperationKeep; s.st_pass = MTLStencilOperationReplace;
        /* 1: pixels with no colour yet: the polygon as it is */
        s.blend = 0;
        s.st_func = MTLCompareFunctionGreater; s.st_ref = ref; s.st_read = ST_SOLID;
        draw(enc, &s, &u, prim, b->first, b->count);
        /* 2: the others, except where this id already is (or 1 just drew) */
        s.blend = blend_on ? 1 : 2;                             /* blending off: the colour replaces, alpha still the larger */
        s.st_func = MTLCompareFunctionNotEqual; s.st_read = ST_SOLID | ST_TRANS | ST_ID;
        draw(enc, &s, &u, prim, b->first, b->count);
        u.part = 0;
    }
    [enc endEncoding];

    /* edge marking + fog -> post */
    rp = [MTLRenderPassDescriptor renderPassDescriptor];
    rp.colorAttachments[0].texture = fb->post;
    rp.colorAttachments[0].loadAction = MTLLoadActionDontCare;
    rp.colorAttachments[0].storeAction = MTLStoreActionStore;
    enc = [cur_cb() renderCommandEncoderWithDescriptor:rp];
    [enc setRenderPipelineState:pl_post];
    [enc setFragmentTexture:fb->color atIndex:0];
    [enc setFragmentTexture:fb->attr atIndex:1];
    [enc setFragmentTexture:fb->depth atIndex:2];
    PostU pu; memset(&pu, 0, sizeof pu);
    pu.edge = (disp3d >> 5) & 1;
    /* the DS's half-and-half outlines (anti-aliasing) only at 1x (see gl_render.c) */
    { static int eb = -2; if (eb == -2) eb = getenv("DQIX_EDGEBLEND") ? atoi(getenv("DQIX_EDGEBLEND")) : -1;
      pu.aa = (eb < 0 ? g_scale == 1 : eb) && ((disp3d >> 4) & 1); }
    pu.scale = g_scale;
    for (int i = 0; i < 8; i++) { uint16_t c = IO16_(0x330 + i * 2); pu.edgecol[i][0] = (float)(c & 31) / 31.0f; pu.edgecol[i][1] = (float)((c >> 5) & 31) / 31.0f; pu.edgecol[i][2] = (float)((c >> 10) & 31) / 31.0f; }
    pu.fog = (disp3d >> 7) & 1;
    pu.fog_alpha_only = (disp3d >> 6) & 1;
    uint16_t fcol = IO16_(0x358);
    pu.fogcol[0] = (float)(fcol & 31) / 31.0f; pu.fogcol[1] = (float)((fcol >> 5) & 31) / 31.0f;
    pu.fogcol[2] = (float)((fcol >> 10) & 31) / 31.0f; pu.fogcol[3] = (float)(g_io[0x35A] & 31) / 31.0f;
    for (int i = 0; i < 32; i++) pu.fogtab[i] = (float)(g_io[0x360 + i] & 127);
    pu.fogoff = IO16_(0x35C) & 0x7FFF;
    pu.fogshift = (disp3d >> 8) & 15;
    pu.wbuf = f->wbuf;
    [enc setFragmentBytes:&pu length:sizeof pu atIndex:0];
    QuadU q = { { -1, -1, 2, 2 }, { 0, 0, 1, 1 }, -1, 0, 0, 0 };
    [enc setVertexBytes:&q length:sizeof q atIndex:0];
    [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:6];
    [enc endEncoding];
    fb->hd2d_valid = 0;
    if (g_hd2d) { hd2d_passes(fb); fb->hd2d_valid = 1; }
    flush(0);                                                   /* the GPU starts on it now */
    fb->valid = 1;
    g_fb_latest = target;
}

static const Renderer3D g_gl_renderer = { gl_vblank, gl_render };
const Renderer3D* gl_renderer3d(void) { return &g_gl_renderer; }

/* ---------------------------------------------------------------- composition */
extern uint32_t g_engine_out[2][256 * 192];
extern uint32_t g_comp_c2[256 * 192];
extern uint8_t g_comp_info[256 * 192 * 4];
extern int g_comp_has3d;

static id<MTLTexture> upload_rgb(const uint32_t* src) {
    static uint8_t buf[256 * 192 * 4];
    for (int i = 0; i < 256 * 192; i++) { uint32_t c = src[i]; buf[i * 4] = (uint8_t)(c >> 16); buf[i * 4 + 1] = (uint8_t)(c >> 8); buf[i * 4 + 2] = (uint8_t)c; buf[i * 4 + 3] = (uint8_t)(c >> 24); }
    return upload(MTLPixelFormatRGBA8Unorm, 256, 192, buf, 4);
}

/* Build both screens into final_tex (top screen in the upper half). */
void gl_compose(int a_on_top, int lcd_on) {
    @autoreleasepool {
    id<MTLTexture> col[2] = { upload_rgb(g_engine_out[0]), upload_rgb(g_engine_out[1]) };
    int has3d = g_comp_has3d && g_fb_shown >= 0 && g_fb[g_fb_shown].valid;
    id<MTLTexture> c2 = white_tex, info = nil;
    if (has3d) c2 = upload_rgb(g_comp_c2);
    info = upload(MTLPixelFormatRGBA8Uint, 256, 192, g_comp_info, 4);
    MTLRenderPassDescriptor* rp = [MTLRenderPassDescriptor renderPassDescriptor];
    rp.colorAttachments[0].texture = final_tex;
    rp.colorAttachments[0].loadAction = MTLLoadActionClear;
    rp.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 1);
    rp.colorAttachments[0].storeAction = MTLStoreActionStore;
    id<MTLRenderCommandEncoder> enc = [cur_cb() renderCommandEncoderWithDescriptor:rp];
    g_wide_shown = -1;
    if (lcd_on) {
        float W = (float)(256 + 2 * g_wside);
        [enc setRenderPipelineState:pl_comp];
        for (int eng = 0; eng < 2; eng++) {
            int top = eng == 0 ? a_on_top : !a_on_top;
            [enc setFragmentTexture:col[eng] atIndex:0];
            [enc setFragmentTexture:c2 atIndex:1];
            [enc setFragmentTexture:info atIndex:2];
            Fb3D* sf = has3d ? &g_fb[g_fb_shown] : NULL;
            [enc setFragmentTexture:!sf ? white_tex : sf->hd2d_valid ? sf->hd2d : sf->post atIndex:3];
            int wide = eng == 0 && has3d && g_wside > 0;           /* the 3D screen, widened; the other in the middle */
            CompU cu = { eng == 0 && has3d, wide, W, (float)g_wside };
            [enc setFragmentBytes:&cu length:sizeof cu atIndex:0];
            if (wide) g_wide_shown = top ? 0 : 1;
            QuadU q = { { -1.0f, top ? 0.0f : -1.0f, 2.0f, 1.0f }, { 0, 0, 1, 1 }, -1, 0, 0, 0 };   /* top screen = upper half of NDC */
            if (!wide) { q.rect[0] = -1.0f + 2.0f * (float)g_wside / W; q.rect[2] = 2.0f * 256.0f / W; }
            [enc setVertexBytes:&q length:sizeof q atIndex:0];
            [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:6];
        }
    }
    [enc endEncoding];
    flush(0);
    }
}

/* Motion blur buffer: mix this frame into what the LCD shows (linear light). */
static id<MTLTexture> ghost_step(float keep) {
    if (ghost_w != g_fb3_w || ghost_h != g_fb_h * 2) {
        for (int i = 0; i < 2; i++) ghost_tex[i] = new_tex(MTLPixelFormatRGBA16Float, g_fb3_w, g_fb_h * 2, 1, 1);
        ghost_w = g_fb3_w; ghost_h = g_fb_h * 2; ghost_cur = -1;
    }
    int prev = ghost_cur, cur = ghost_cur < 0 ? 0 : ghost_cur ^ 1;
    MTLRenderPassDescriptor* rp = [MTLRenderPassDescriptor renderPassDescriptor];
    rp.colorAttachments[0].texture = ghost_tex[cur];
    rp.colorAttachments[0].loadAction = MTLLoadActionDontCare;
    rp.colorAttachments[0].storeAction = MTLStoreActionStore;
    id<MTLRenderCommandEncoder> enc = [cur_cb() renderCommandEncoderWithDescriptor:rp];
    [enc setRenderPipelineState:pl_ghost];
    [enc setFragmentTexture:final_tex atIndex:0];
    [enc setFragmentTexture:prev < 0 ? final_tex : ghost_tex[prev] atIndex:1];
    [enc setFragmentSamplerState:smp_nearest atIndex:0];
    GhostU gu = { prev < 0 ? 0.0f : keep, 0, 0, 0 };
    [enc setFragmentBytes:&gu length:sizeof gu atIndex:0];
    QuadU q = { { -1, -1, 2, 2 }, { 0, 0, 1, 1 }, -1, 0, 0, 0 };
    [enc setVertexBytes:&q length:sizeof q atIndex:0];
    [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:6];
    [enc endEncoding];
    ghost_cur = cur;
    return ghost_tex[cur];
}

/* Draw the screens (and the console, if chosen) into the window chosen with
 * gfx_begin(). Call once per emulated frame: the motion blur advances with every call. */
void gl_present(const GlPresent* v) {
    @autoreleasepool {
    MtlWin* mw = &g_wins[g_cur_win];
    if (!mw->layer) return;
    int ghost = v->lcd && v->ghost;
    id<MTLTexture> src = final_tex;
    if (ghost && v->again && ghost_cur >= 0) src = ghost_tex[ghost_cur];   /* a second window: this frame's blur as it is */
    else if (ghost) src = ghost_step(0.35f);
    else if (!v->again) ghost_cur = -1;
    /* Shown smaller than it was rendered: average with mipmaps; shown larger:
     * linear, and the shader keeps the pixels sharp (see gl_render.c). */
    float px_top = v->sc * v->top[2] / (v->topx[1] * (float)g_scale), px_bot = v->sc * v->bot[2] / (v->botx[1] * (float)g_scale);
    float px_per_texel = px_top > px_bot ? px_top : px_bot;
    int minify = px_per_texel < 0.99f;
    if (minify) { id<MTLBlitCommandEncoder> bl = [cur_cb() blitCommandEncoder]; [bl generateMipmapsForTexture:src]; [bl endEncoding]; }
    mw->layer.drawableSize = CGSizeMake(v->win_w, v->win_h);
    id<CAMetalDrawable> drawable = [mw->layer nextDrawable];
    if (!drawable) return;
    [win_keep addObject:drawable];
    mw->drawable = drawable;
    MTLRenderPassDescriptor* rp = [MTLRenderPassDescriptor renderPassDescriptor];
    rp.colorAttachments[0].texture = drawable.texture;
    rp.colorAttachments[0].loadAction = MTLLoadActionDontCare;
    rp.colorAttachments[0].storeAction = MTLStoreActionStore;
    id<MTLRenderCommandEncoder> enc = [cur_cb() renderCommandEncoderWithDescriptor:rp];
    [enc setRenderPipelineState:pl_screen];
    [enc setFragmentTexture:src atIndex:0];
    [enc setFragmentSamplerState:minify ? smp_mip : smp_linear atIndex:0];
    ScreenU su; memset(&su, 0, sizeof su);
    su.view[0] = v->ox; su.view[1] = v->oy; su.view[2] = v->sc;
    su.win_canvas[0] = (float)v->win_w; su.win_canvas[1] = (float)v->win_h; su.win_canvas[2] = v->cw; su.win_canvas[3] = v->ch;
    memcpy(su.top, v->top, sizeof su.top); memcpy(su.bot, v->bot, sizeof su.bot);
    {   /* the console's windows: upper and lower, wherever each screen is shown */
        const float* up = v->top[1] <= v->bot[1] ? v->top : v->bot, *lo = up == v->top ? v->bot : v->top;
        memcpy(su.ctop, up, sizeof su.ctop); memcpy(su.cbot, lo, sizeof su.cbot);
    }
    su.topx_botx[0] = v->topx[0]; su.topx_botx[1] = v->topx[1]; su.topx_botx[2] = v->botx[0]; su.topx_botx[3] = v->botx[1];
    memcpy(su.lid, v->lid, 3 * sizeof(float)); memcpy(su.base, v->base, 3 * sizeof(float));
    memcpy(su.btn, v->btn, 3 * sizeof(float)); memcpy(su.hinge, v->hinge, 3 * sizeof(float));
    int skin = v->shell && v->skin && skin_tex;
    if (skin) { su.skin_texw_alpha[0] = v->skin_k / (float)skin_tex.width; su.skin_texw_alpha[1] = v->skin_k / (float)skin_tex.height; }
    su.skin_texw_alpha[2] = (float)(256 + 2 * g_wside);
    su.skin_texw_alpha[3] = v->alpha > 0 ? v->alpha : 1.0f;
    su.f0[0] = ghost; su.f0[1] = v->front; su.f0[2] = v->lcd; su.f0[3] = v->lcd_color;
    su.f1[0] = v->grid; su.f1[1] = v->shell; su.f1[2] = skin;
    su.f2[0] = v->backlight;
    [enc setFragmentBytes:&su length:sizeof su atIndex:0];
    [enc setFragmentTexture:skin ? skin_tex : white_tex atIndex:1];
    [enc setFragmentSamplerState:smp_mip atIndex:1];
    QuadU q = { { -1, -1, 2, 2 }, { 0, 0, 1, 1 }, 1, 0, 0, 0 };   /* the window: not flipped */
    [enc setVertexBytes:&q length:sizeof q atIndex:0];
    [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:6];
    [enc endEncoding];
    }
}

/* A console photo (RGBA, rows top-down), drawn around the screens. */
void gl_set_skin(const uint8_t* rgba, int w, int h) {
    if (!dev) return;
    MTLTextureDescriptor* d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:(NSUInteger)w height:(NSUInteger)h mipmapped:YES];
    d.usage = MTLTextureUsageShaderRead;
    d.storageMode = MTLStorageModeShared;
    skin_tex = [dev newTextureWithDescriptor:d];
    [skin_tex replaceRegion:MTLRegionMake2D(0, 0, (NSUInteger)w, (NSUInteger)h) mipmapLevel:0 withBytes:rgba bytesPerRow:(NSUInteger)w * 4];
    id<MTLBlitCommandEncoder> bl = [cur_cb() blitCommandEncoder];
    [bl generateMipmapsForTexture:skin_tex];
    [bl endEncoding];
}

/* A texture's pixels as they are in memory (rows as under GL: see the top). */
static uint8_t* read_tex(id<MTLTexture> t, int w, int h) {
    size_t row = (size_t)w * 4;
    id<MTLBuffer> b = [dev newBufferWithLength:row * (size_t)h options:MTLResourceStorageModeShared];
    id<MTLBlitCommandEncoder> bl = [cur_cb() blitCommandEncoder];
    [bl copyFromTexture:t sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:MTLSizeMake((NSUInteger)w, (NSUInteger)h, 1)
               toBuffer:b destinationOffset:0 destinationBytesPerRow:row destinationBytesPerImage:row * (size_t)h];
    [bl endEncoding];
    flush(1);
    uint8_t* out = malloc(row * (size_t)h);
    memcpy(out, b.contents, row * (size_t)h);
    return out;
}

/* The window as shown (for tests): top-down RGB, caller frees. Call after
 * gl_present and before gfx_swap. */
uint8_t* gl_read_window(int w, int h) {
    @autoreleasepool {
    uint8_t* rgb = calloc((size_t)w * h, 3);
    MtlWin* mw = &g_wins[g_cur_win];
    if (!mw->drawable) return rgb;
    id<MTLTexture> t = mw->drawable.texture;
    int tw = (int)t.width < w ? (int)t.width : w, th = (int)t.height < h ? (int)t.height : h;
    uint8_t* buf = read_tex(t, tw, th);                        /* BGRA, top-down (the window isn't flipped) */
    for (int y = 0; y < th; y++) for (int x = 0; x < tw; x++) {
        const uint8_t* q = buf + ((size_t)y * tw + x) * 4; uint8_t* o = rgb + ((size_t)y * w + x) * 3;
        o[0] = q[2]; o[1] = q[1]; o[2] = q[0];
    }
    free(buf);
    return rgb;
    }
}

/* Both screens at the rendering resolution, top-down RGB (caller frees). */
uint8_t* gl_read_full(int* out_w, int* out_h) {
    @autoreleasepool {
    int w = g_fb3_w, h = g_fb_h * 2;
    uint8_t* buf = read_tex(final_tex, w, h);
    uint8_t* rgb = malloc((size_t)w * h * 3);
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) {
            const uint8_t* p = buf + ((size_t)(h - 1 - y) * w + x) * 4;
            uint8_t* q = rgb + ((size_t)y * w + x) * 3;
            q[0] = p[0]; q[1] = p[1]; q[2] = p[2];
        }
    free(buf);
    *out_w = w; *out_h = h;
    return rgb;
    }
}

void gl_read_screens(uint32_t out[2][256 * 192]) {
    @autoreleasepool {
    int w = g_fb3_w, h = g_fb_h * 2;
    uint8_t* buf = read_tex(final_tex, w, h);
    for (int sc = 0; sc < 2; sc++)
        for (int y = 0; y < 192; y++)
            for (int x = 0; x < 256; x++) {
                int gy = h - 1 - ((sc * 192 + y) * g_scale + g_scale / 2), gx = (x + g_wside) * g_scale + g_scale / 2;   /* the DS's picture */
                const uint8_t* p = buf + ((size_t)gy * w + gx) * 4;
                out[sc][y * 256 + x] = (uint32_t)p[0] << 16 | (uint32_t)p[1] << 8 | p[2];
            }
    free(buf);
    }
}

/* For tests: the 3D layer as rendered (RGBA8, bottom-up rows). */
int gl_read_3d(uint8_t* out, int* w, int* h) {
    if (g_fb_latest < 0) return 0;
    @autoreleasepool {
    *w = g_fb3_w; *h = g_fb_h;
    uint8_t* buf = read_tex(g_fb[g_fb_latest].post, g_fb3_w, g_fb_h);
    memcpy(out, buf, (size_t)g_fb3_w * g_fb_h * 4);
    free(buf);
    }
    return 1;
}

/* Widescreen: which screen was composed wide last (0 top, 1 bottom, -1
 * none) and the DS pixels added each side. */
int gl_wide_shown(int* side) { *side = g_wside; return g_wide_shown; }

/* ---------------------------------------------------------------- windows */
uint32_t gfx_window_flags(void) { return SDL_WINDOW_METAL; }

int gfx_attach(SDL_Window* win) {
    @autoreleasepool {
    if (!dev) dev = MTLCreateSystemDefaultDevice();
    if (!dev) { fprintf(stderr, "Metal not available: using the software renderer\n"); return 0; }
    int k = g_wins[0].win ? 1 : 0;
    if (g_wins[k].win) return 0;
    SDL_MetalView view = SDL_Metal_CreateView(win);
    if (!view) { fprintf(stderr, "Metal view: %s\n", SDL_GetError()); return 0; }
    CAMetalLayer* layer = (__bridge CAMetalLayer*)SDL_Metal_GetLayer(view);
    layer.device = dev;
    layer.pixelFormat = MTLPixelFormatBGRA8Unorm;
    layer.framebufferOnly = NO;                                /* read back for tests (gl_read_window) */
    layer.displaySyncEnabled = NO;                             /* pacing follows the audio clock */
    layer.maximumDrawableCount = 3;
    if (!win_keep) win_keep = [NSMutableArray array];
    [win_keep addObject:layer];
    g_wins[k] = (MtlWin){ win, view, layer, nil };
    return 1;
    }
}

void gfx_detach(SDL_Window* win) {
    for (int k = 0; k < 2; k++) {
        if (g_wins[k].win != win) continue;
        flush(1);
        [win_keep removeObject:g_wins[k].layer];
        if (g_wins[k].drawable) [win_keep removeObject:g_wins[k].drawable];
        SDL_Metal_DestroyView(g_wins[k].view);
        memset(&g_wins[k], 0, sizeof g_wins[k]);
        if (k == 0 && g_wins[1].win) { g_wins[0] = g_wins[1]; memset(&g_wins[1], 0, sizeof g_wins[1]); }
        if (g_cur_win == k) g_cur_win = 0;
    }
}

void gfx_drawable_size(SDL_Window* win, int* w, int* h) { SDL_Metal_GetDrawableSize(win, w, h); }

void gfx_begin(SDL_Window* win) {
    for (int k = 0; k < 2; k++) if (g_wins[k].win == win) g_cur_win = k;
}

void gfx_swap(SDL_Window* win) {
    @autoreleasepool {
    for (int k = 0; k < 2; k++) {
        MtlWin* mw = &g_wins[k];
        if (mw->win != win) continue;
        if (mw->drawable) {
            [cur_cb() presentDrawable:mw->drawable];
            [win_keep removeObject:mw->drawable];
            mw->drawable = nil;
        }
        flush(0);
    }
    }
}
