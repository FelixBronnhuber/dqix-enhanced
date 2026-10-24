/* OpenGL renderer (GL 3.3 core) for the DS 3D engine, plus the final
 * composition of 3D with the 2D layers and presentation.
 *
 *  - The geometry engine (gpu3d.c) hands over each frame's polygons, already
 *    clipped and in drawing order; they are drawn at `scale` x the DS
 *    resolution into a framebuffer with color (premultiplied alpha),
 *    attributes (polygon id, fog, edge flags) and depth.
 *  - Textures are decoded once and cached; a texture is re-decoded only when
 *    the bytes it is made from (texels, palette) change.
 *  - A post pass applies edge marking and fog.
 *  - The 2D engines still run on the CPU at 256x192. For engine A they mark
 *    where the 3D layer shows and how it blends (gpu2d.c); the composition
 *    pass samples the high-resolution 3D there.
 * Everything is loaded through SDL_GL_GetProcAddress (no link-time GL). */
#include "SDL.h"
#include "SDL_opengl.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "nds.h"
#include "gpu3d.h"
#include "gl_render.h"

/* ---------------------------------------------------------------- GL functions */
#define GL_FUNCS(X) \
    X(const GLubyte*, glGetString, (GLenum)) \
    X(void, glViewport, (GLint, GLint, GLsizei, GLsizei)) \
    X(void, glClearColor, (GLfloat, GLfloat, GLfloat, GLfloat)) \
    X(void, glClear, (GLbitfield)) \
    X(void, glClearBufferfv, (GLenum, GLint, const GLfloat*)) \
    X(void, glClearBufferfi, (GLenum, GLint, GLfloat, GLint)) \
    X(void, glStencilFunc, (GLenum, GLint, GLuint)) \
    X(void, glStencilOp, (GLenum, GLenum, GLenum)) \
    X(void, glStencilMask, (GLuint)) \
    X(void, glClearBufferiv, (GLenum, GLint, const GLint*)) \
    X(void, glGetIntegerv, (GLenum, GLint*)) \
    X(void, glEnable, (GLenum)) \
    X(void, glDisable, (GLenum)) \
    X(void, glEnablei, (GLenum, GLuint)) \
    X(void, glDisablei, (GLenum, GLuint)) \
    X(void, glBlendFuncSeparate, (GLenum, GLenum, GLenum, GLenum)) \
    X(void, glBlendEquationSeparate, (GLenum, GLenum)) \
    X(void, glDepthFunc, (GLenum)) \
    X(void, glDepthMask, (GLboolean)) \
    X(void, glColorMaski, (GLuint, GLboolean, GLboolean, GLboolean, GLboolean)) \
    X(void, glGenTextures, (GLsizei, GLuint*)) \
    X(void, glDeleteTextures, (GLsizei, const GLuint*)) \
    X(void, glBindTexture, (GLenum, GLuint)) \
    X(void, glTexImage2D, (GLenum, GLint, GLint, GLsizei, GLsizei, GLint, GLenum, GLenum, const void*)) \
    X(void, glTexSubImage2D, (GLenum, GLint, GLint, GLint, GLsizei, GLsizei, GLenum, GLenum, const void*)) \
    X(void, glTexParameteri, (GLenum, GLenum, GLint)) \
    X(void, glGenerateMipmap, (GLenum)) \
    X(void, glActiveTexture, (GLenum)) \
    X(void, glPixelStorei, (GLenum, GLint)) \
    X(void, glGenFramebuffers, (GLsizei, GLuint*)) \
    X(void, glDeleteFramebuffers, (GLsizei, const GLuint*)) \
    X(void, glBindFramebuffer, (GLenum, GLuint)) \
    X(void, glFramebufferTexture2D, (GLenum, GLenum, GLenum, GLuint, GLint)) \
    X(GLenum, glCheckFramebufferStatus, (GLenum)) \
    X(void, glDrawBuffers, (GLsizei, const GLenum*)) \
    X(void, glReadBuffer, (GLenum)) \
    X(void, glReadPixels, (GLint, GLint, GLsizei, GLsizei, GLenum, GLenum, void*)) \
    X(GLuint, glCreateShader, (GLenum)) \
    X(void, glShaderSource, (GLuint, GLsizei, const GLchar* const*, const GLint*)) \
    X(void, glCompileShader, (GLuint)) \
    X(void, glGetShaderiv, (GLuint, GLenum, GLint*)) \
    X(void, glGetShaderInfoLog, (GLuint, GLsizei, GLsizei*, GLchar*)) \
    X(GLuint, glCreateProgram, (void)) \
    X(void, glAttachShader, (GLuint, GLuint)) \
    X(void, glBindAttribLocation, (GLuint, GLuint, const GLchar*)) \
    X(void, glBindFragDataLocation, (GLuint, GLuint, const GLchar*)) \
    X(void, glLinkProgram, (GLuint)) \
    X(void, glGetProgramiv, (GLuint, GLenum, GLint*)) \
    X(void, glGetProgramInfoLog, (GLuint, GLsizei, GLsizei*, GLchar*)) \
    X(void, glUseProgram, (GLuint)) \
    X(GLint, glGetUniformLocation, (GLuint, const GLchar*)) \
    X(void, glUniform1i, (GLint, GLint)) \
    X(void, glUniform1f, (GLint, GLfloat)) \
    X(void, glUniform2f, (GLint, GLfloat, GLfloat)) \
    X(void, glUniform4f, (GLint, GLfloat, GLfloat, GLfloat, GLfloat)) \
    X(void, glUniform1fv, (GLint, GLsizei, const GLfloat*)) \
    X(void, glUniform3fv, (GLint, GLsizei, const GLfloat*)) \
    X(void, glUniform2fv, (GLint, GLsizei, const GLfloat*)) \
    X(void, glGenVertexArrays, (GLsizei, GLuint*)) \
    X(void, glBindVertexArray, (GLuint)) \
    X(void, glGenBuffers, (GLsizei, GLuint*)) \
    X(void, glBindBuffer, (GLenum, GLuint)) \
    X(void, glBufferData, (GLenum, GLsizeiptr, const void*, GLenum)) \
    X(void, glVertexAttribPointer, (GLuint, GLint, GLenum, GLboolean, GLsizei, const void*)) \
    X(void, glEnableVertexAttribArray, (GLuint)) \
    X(void, glDrawArrays, (GLenum, GLint, GLsizei)) \
    X(void, glGenSamplers, (GLsizei, GLuint*)) \
    X(void, glSamplerParameteri, (GLuint, GLenum, GLint)) \
    X(void, glBindSampler, (GLuint, GLuint)) \
    X(void, glLineWidth, (GLfloat)) \
    X(void, glTexBuffer, (GLenum, GLenum, GLuint)) \
    X(GLenum, glGetError, (void)) \
    X(void, glFinish, (void))

#define X(ret, name, args) static ret (APIENTRY* p_##name) args;
GL_FUNCS(X)
#undef X

static int load_gl(void) {
#define X(ret, name, args) p_##name = (ret (APIENTRY*) args)SDL_GL_GetProcAddress(#name); if (!p_##name) { fprintf(stderr, "GL: missing %s\n", #name); return 0; }
    GL_FUNCS(X)
#undef X
    return 1;
}

#define VTX_FLOATS 21                    /* floats per vertex sent to the GPU (the last 3: view-space position) */

/* ---------------------------------------------------------------- shaders */
static const char* VS_POLY =
    "#version 330 core\n"
    "in vec4 a_pos; in vec4 a_color; in vec2 a_tex; in vec4 a_attr; in vec4 a_edge; in vec3 a_vpos;\n"
    "out vec4 v_color; out vec2 v_tex; flat out vec4 v_attr; noperspective out vec3 v_bary; flat out int v_ext; out vec3 v_vpos;\n"
    "uniform vec2 u_texsize;\n"
    "void main() { gl_Position = a_pos; v_color = a_color; v_tex = a_tex / u_texsize; v_attr = a_attr;\n"
    "  v_bary = a_edge.xyz; v_ext = int(a_edge.w + 0.5); v_vpos = a_vpos; }\n";

/* DS pixel pipeline, in 5-bit color steps like the hardware. */
static const char* FS_POLY =
    "#version 330 core\n"
    "in vec4 v_color; in vec2 v_tex; flat in vec4 v_attr; noperspective in vec3 v_bary; flat in int v_ext; in vec3 v_vpos;\n"
    "layout(location = 0) out vec4 o_color;\n"
    "layout(location = 1) out vec4 o_attr;\n"
    "layout(location = 2) out vec4 o_g;\n"         /* only with the effects on (a third target): view position, distance */
    "uniform sampler2D u_tex;\n"
    "uniform int u_highlight; uniform int u_alphatest; uniform float u_aref; uniform int u_wbuf; uniform float u_edgew;\n"
    "uniform int u_part;\n"
    "uniform int u_sharpa;\n"                    /* smooth textures: a texture whose texels are see-through or not (no alpha levels): cut at half */                       /* 0 every pixel; 1 only solid ones (alpha 31); 2 only see-through ones */
    "uniform vec3 u_toon[32];\n"
    "void main() {\n"
    /* distance (in pixels) to each of the triangle's edges that is also an edge of
     * the DS polygon: pixels within one DS pixel of one are edge-marking candidates */
    "  vec3 gx = dFdx(v_bary), gy = dFdy(v_bary);\n"
    "  vec3 dist = v_bary / max(sqrt(gx * gx + gy * gy), vec3(1e-6));\n"
    "  float eflag = 0.0;\n"
    "  for (int k = 0; k < 3; k++) if (((v_ext >> k) & 1) != 0 && dist[k] < u_edgew) eflag = 1.0;\n"
    "  ivec3 vc = ivec3(clamp(floor(v_color.rgb + 0.5), 0.0, 31.0));\n"
    "  int pa = int(v_color.a + 0.5);\n"
    "  int mode = int(v_attr.x + 0.5); bool textured = v_attr.y > 0.5;\n"
    "  if (mode == 2) { ivec3 t = ivec3(u_toon[vc.r] * 31.0 + 0.5); if (u_highlight == 0) vc = t; else vc = ivec3(vc.r); }\n"
    "  ivec3 c = vc; int a = pa;\n"
    "  if (textured) {\n"
    "    vec4 t = texture(u_tex, v_tex);\n"
    "    if (u_sharpa != 0) t.a = t.a >= 0.5 ? 1.0 : 0.0;\n"
    "    ivec3 tc = ivec3(t.rgb * 31.0 + 0.5); int ta = int(t.a * 31.0 + 0.5);\n"
    "    if (mode == 1) { c = (tc * ta + vc * (31 - ta)) / 31; a = pa; }\n"
    "    else { c = ((tc + 1) * (vc + 1) - 1) >> 5; a = ((ta + 1) * (pa + 1) - 1) >> 5; }\n"
    "  }\n"
    "  if (mode == 2 && u_highlight != 0) { ivec3 t = ivec3(u_toon[int(v_color.r + 0.5)] * 31.0 + 0.5); c = min(c + t, ivec3(31)); }\n"
    "  if (a == 0) discard;\n"
    "  if (u_alphatest != 0 && float(a) <= u_aref) discard;\n"
    "  if ((u_part == 1 && a < 31) || (u_part == 2 && a >= 31)) discard;\n"
    "  float fa = a >= 31 ? 1.0 : float(a) / 31.0;\n"
    "  o_color = vec4(vec3(c) / 31.0, fa);\n"                  /* straight colour and alpha, like the DS's buffer */
    "  o_attr = vec4(v_attr.z, v_attr.w, eflag, 1.0);\n"
    "  o_g = vec4(v_vpos, 1.0 / gl_FragCoord.w);\n"
    "  gl_FragDepth = u_wbuf != 0 ? clamp((1.0 / gl_FragCoord.w) / 4096.0, 0.0, 1.0) : gl_FragCoord.z;\n"
    "}\n";

static const char* VS_QUAD =
    "#version 330 core\n"
    "in vec2 a_pos; out vec2 v_uv;\n"
    "uniform vec4 u_rect;\n"                         /* x, y, w, h in NDC */
    "uniform vec4 u_uv;\n"                           /* texture offset, scale */
    "void main() { v_uv = u_uv.xy + a_pos * u_uv.zw; gl_Position = vec4(u_rect.xy + a_pos * u_rect.zw, 0.0, 1.0); }\n";

/* Edge marking and fog on the finished 3D frame. */
static const char* FS_POST =
    "#version 330 core\n"
    "in vec2 v_uv; out vec4 o_color;\n"
    "uniform sampler2D u_color; uniform sampler2D u_attr; uniform sampler2D u_depth;\n"
    "uniform int u_edge; uniform vec3 u_edgecol[8]; uniform int u_scale; uniform int u_aa;\n"
    "uniform int u_fog; uniform int u_fog_alpha_only; uniform vec4 u_fogcol; uniform float u_fogtab[32];\n"
    "uniform int u_fogoff; uniform int u_fogshift; uniform int u_wbuf;\n"
    "uniform int u_rt; uniform sampler2D u_shade;\n"   /* ray tracing: 0 off, 1 on, 2 the shade alone (tests) */
    "void main() {\n"
    "  ivec2 p = ivec2(gl_FragCoord.xy);\n"
    "  vec4 c = texelFetch(u_color, p, 0); vec4 at = texelFetch(u_attr, p, 0);\n"
    "  if (u_rt != 0 && c.a > 0.0) { float sh = texture(u_shade, (vec2(p) + 0.5) / vec2(textureSize(u_color, 0))).r;\n"
    "    c.rgb = u_rt == 2 ? vec3(sh) : c.rgb * sh; }\n"   /* ray-traced shadows, occlusion */
    "  if (u_edge != 0 && at.b > 0.5 && at.a > 0.5) {\n"
    "    float id = at.r; float d = texelFetch(u_depth, p, 0).r;\n"
    "    ivec2 sz = textureSize(u_color, 0);\n"
    "    ivec2 o[4] = ivec2[4](ivec2(-u_scale, 0), ivec2(u_scale, 0), ivec2(0, -u_scale), ivec2(0, u_scale));\n"
    "    for (int k = 0; k < 4; k++) {\n"
    "      ivec2 q = p + o[k];\n"
    "      if (q.x < 0 || q.y < 0 || q.x >= sz.x || q.y >= sz.y) continue;\n"
    "      vec4 an = texelFetch(u_attr, q, 0);\n"
    "      if (abs(an.r - id) > 0.004 && d < texelFetch(u_depth, q, 0).r) { int i = int(id * 63.0 + 0.5) >> 3;\n"
    /* with anti-aliasing on, the DS blends a marked edge half and half with what is behind it
     * (the soft outlines of DQIX's characters); the farther neighbour stands in for that */
    "        if (u_aa != 0) { vec4 b = texelFetch(u_color, q, 0); c = vec4(b.a > 0.0 ? mix(b.rgb, u_edgecol[i], 17.0 / 32.0) : u_edgecol[i], b.a > 0.0 ? max(c.a, b.a) : c.a); }\n"
    "        else c = vec4(u_edgecol[i], c.a);\n"
    "        break; }\n"
    "    }\n"
    "  }\n"
    "  if (u_fog != 0 && at.g > 0.5) {\n"
    "    float d = texelFetch(u_depth, p, 0).r;\n"
    "    int z15 = u_wbuf != 0 ? int(d * 4096.0 * 4096.0) >> 9 : int(d * 16777215.0) >> 9;\n"
    "    int step = 0x400 >> u_fogshift; float dens;\n"
    "    if (step <= 0) dens = u_fogtab[31];\n"
    "    else { int i = (z15 - u_fogoff) / step;\n"
    /* like the DS: step i blends table entries i-1 and i (entry -1 = entry 0) */
    "      if (z15 - u_fogoff < 0) dens = u_fogtab[0]; else if (i >= 32) dens = u_fogtab[31];\n"
    "      else { float f = float((z15 - u_fogoff) - i * step) / float(step); dens = mix(u_fogtab[max(i - 1, 0)], u_fogtab[i], f); } }\n"
    "    if (dens >= 127.0) dens = 128.0; dens /= 128.0;\n"
    "    float a = c.a; vec3 rgb = c.rgb;\n"
    "    if (u_fog_alpha_only == 0) rgb = mix(rgb, u_fogcol.rgb, dens);\n"
    "    a = mix(a, u_fogcol.a, dens);\n"
    "    c = vec4(rgb, a);\n"
    "  }\n"
    "  o_color = c;\n"
    "}\n";

/* Final picture of one screen: 2D layers (CPU, 256x192) + the 3D layer. */
static const char* FS_COMP =
    "#version 330 core\n"
    "in vec2 v_uv; out vec4 o_color;\n"
    "uniform sampler2D u_col; uniform sampler2D u_c2; uniform usampler2D u_info; uniform sampler2D u_3d;\n"
    "uniform int u_has3d; uniform vec2 u_outsize;\n"
    "uniform int u_wide; uniform float u_wtot, u_wside;\n"   /* widescreen: DS pixels across, added each side */
    "vec3 q5(vec3 c) { return floor(c * 31.0 + 0.5); }\n"
    "void main() {\n"
    "  vec2 uv = vec2(v_uv.x, 1.0 - v_uv.y);\n"                  /* row 0 = top line */
    "  float dsx = u_wide != 0 ? uv.x * u_wtot - u_wside : uv.x * 256.0;\n"
    "  ivec2 p2 = ivec2(min(vec2(dsx, uv.y * 192.0), vec2(255.0, 191.0)));\n"
    "  bool side = u_wide != 0 && (dsx < 0.0 || dsx >= 256.0);\n"
    "  if (side) {\n"
    /* Beyond the DS's picture there are no 2D layers: show the 3D there on lines
     * where the edge of the DS's picture shows it (the field), with that
     * pixel's effects (fades); elsewhere (2D menus, text boxes at the edge) black. */
    "    int found = -1;\n"
    "    for (int k = 0; k < 8; k++) { int x = dsx < 0.0 ? k : 255 - k;\n"
    "      if ((texelFetch(u_info, ivec2(x, p2.y), 0).r & 3u) != 0u) { found = x; break; } }\n"
    "    if (found < 0) {\n"
    "      int mb0 = int(texelFetch(u_c2, ivec2(dsx < 0.0 ? 0 : 255, p2.y), 0).a * 255.0 + 0.5);\n"
    "      vec3 o0 = vec3(0.0); if ((mb0 >> 5) == 1) o0 = vec3(float(mb0 & 31) / 16.0);\n"
    "      o_color = vec4(o0, 1.0); return;\n"
    "    }\n"
    "    p2.x = found;\n"
    "  }\n"
    "  vec3 col = texelFetch(u_col, p2, 0).rgb;\n"
    "  if (u_has3d == 0) { o_color = vec4(col, 1.0); return; }\n"
    "  uvec4 info = texelFetch(u_info, p2, 0);\n"
    "  int mode = int(info.r & 3u);\n"

    "  if (mode == 0) { o_color = vec4(col, 1.0); return; }\n"
    "  vec4 c2t = texelFetch(u_c2, p2, 0);\n"
    "  vec3 c2 = c2t.rgb; int mb = int(c2t.a * 255.0 + 0.5);\n"
    "  ivec2 sz = textureSize(u_3d, 0);\n"
    "  vec4 s = texelFetch(u_3d, min(ivec2(v_uv * vec2(sz)), sz - 1), 0);\n"
    "  float A = s.a; vec3 c3 = s.rgb;\n"
    "  float eva = float(info.g) / 16.0, evb = float(info.b) / 16.0, evy = float(info.a) / 16.0;\n"
    "  vec3 o;\n"
    "  if (mode == 1) {\n"
    "    bool sec = (info.r & 4u) != 0u; int eff = int((info.r >> 3) & 3u);\n"
    /* The 3D layer over a 2nd blend target: the DS blends by the 3D pixel's own
     * alpha, never by EVA/EVB (solid models stay solid: the equipment screen
     * asks for 10/16 over its background), whatever the effect is set to */
    "    if (A <= 0.0) o = col;\n"
    "    else if (sec) o = A < 0.999 ? c3 * A + c2 * (1.0 - A) : c3;\n"
    "    else if (eff == 2) o = c3 + (1.0 - c3) * evy;\n"
    "    else if (eff == 3) o = c3 - c3 * evy;\n"
    "    else o = c3;\n"
    "  } else {\n"
    "    vec3 under = A > 0.0 ? c3 : c2;\n"
    "    o = min(col * eva + under * evb, vec3(1.0));\n"
    "  }\n"
    "  int mbm = mb >> 5; float mbf = float(mb & 31) / 16.0;\n"
    "  if (mbm == 1) o = o + (1.0 - o) * mbf; else if (mbm == 2) o = o - o * mbf;\n"
    "  o_color = vec4(o, 1.0);\n"
    "}\n";

static const char* FS_COPY =
    "#version 330 core\n"
    "in vec2 v_uv; out vec4 o_color; uniform sampler2D u_src;\n"
    "void main() { o_color = vec4(texture(u_src, v_uv).rgb, 1.0); }\n";

/* Showing the screens in the window: exact pixels, or the DS Lite screen
 * (its LCD colours, pixel grid, backlight), optionally inside a drawn DS Lite.
 * The colour profile is Pokefan531's DS Lite measurement (public domain,
 * libretro "dslite-color"). Canvas units are DS pixels. */
static const char* FS_SCREEN =
    "#version 330 core\n"
    "out vec4 o_color;\n"
    "uniform sampler2D u_src;\n"
    "uniform int u_src_linear;        // source holds linear light (motion-blur buffer)\n"
    "uniform vec2 u_win;              // drawable size\n"
    "uniform vec3 u_view;             // canvas -> window: window = xy + canvas * z\n"
    "uniform vec2 u_canvas;\n"
    "uniform vec4 u_top, u_bot;       // screen rects (canvas units = DS pixels)\n"
    "uniform vec2 u_topx, u_botx;     // each screen in the source texture: x offset, width (DS pixels)\n"
    "uniform float u_texw;            // the source texture's width (DS pixels)\n"
    "uniform int u_front;             // where the screens overlap: 1 = the bottom one in front\n"
    "uniform float u_alpha;           // ...and how solid the one in front is (1 = hides the other)\n"
    "uniform vec4 u_ctop, u_cbot;     // the console's upper and lower screen windows (the screens may be swapped)\n"
    "uniform int u_lcd, u_lcdcolor, u_grid;\n"
    "uniform float u_backlight;\n"
    "uniform int u_shell;\n"
    "uniform vec3 u_lid, u_base, u_btn, u_hinge;\n"
    "uniform int u_skinmode;               // 1: a console photo (u_skintex) instead of the drawn one\n"
    "uniform sampler2D u_skintex;\n"
    "uniform vec2 u_skinscale;             // canvas units -> skin texture coordinates\n"
    "\n"
    "const mat3 DSL = mat3(0.93, 0.025, 0.008,  0.14, 0.90, -0.03,  -0.07, 0.075, 1.022);\n"
    "const float DSL_LUM = 0.935;\n"
    "\n"
    "float px;                        // one window pixel in canvas units (for anti-aliasing)\n"
    "\n"
    "float cover(float d) { return clamp(0.5 - d / px, 0.0, 1.0); }      // signed distance -> coverage\n"
    "float sd_box(vec2 p, vec2 c, vec2 h, vec4 r) {                     // r: radii (tl, tr, br, bl)\n"
    "    p -= c;\n"
    "    float rr = p.x < 0.0 ? (p.y < 0.0 ? r.x : r.w) : (p.y < 0.0 ? r.y : r.z);\n"
    "    vec2 q = abs(p) - h + rr;\n"
    "    return min(max(q.x, q.y), 0.0) + length(max(q, 0.0)) - rr;\n"
    "}\n"
    "float sd_circle(vec2 p, vec2 c, float r) { return length(p - c) - r; }\n"
    "\n"
    "/* --- the screens ------------------------------------------------------- */\n"
    "vec3 to_lin(vec3 c) { return pow(max(c, 0.0), vec3(2.2)); }\n"
    "\n"
    /* Both screens are one texture (top screen in the upper half, and with
     * widescreen a screen may use only the middle). Keep each screen's samples
     * inside its own part, so filtering or a smaller mipmap never mixes in the
     * other screen (a line of the top screen along the bottom one's edge). */
    "vec2 u_lo, u_hi;                 // the current screen's part of the texture (set in main)\n"
    /* How far one window pixel moves in the texture (and in DS pixels), worked
     * out from the layout. Not from the GPU's derivatives: on a screen's edge
     * row they compare with pixels of the console around it, which picked the
     * blurriest mipmap (a line of one flat colour along the edge). */
    "vec2 g_duv; float g_dp;\n"
    "vec2 keep_in(vec2 uv, vec2 duv) {\n"
    "    vec2 tsz = vec2(textureSize(u_src, 0));\n"
    "    float f = max(abs(duv.x) * tsz.x, abs(duv.y) * tsz.y);      // texels per window pixel\n"
    "    vec2 m = 0.5 * (f <= 1.001 ? 1.0 : exp2(ceil(log2(f) - 0.001))) / tsz;   // half a texel of the mipmap used\n"
    "    return clamp(uv, u_lo + m, u_hi - m);\n"
    "}\n"
    "vec3 fetch(vec2 uv) {            // uv in texture space\n"
    "    vec3 c = textureGrad(u_src, keep_in(uv, g_duv), vec2(g_duv.x, 0.0), vec2(0.0, g_duv.y)).rgb;\n"
    "    return u_src_linear != 0 ? c : to_lin(c);\n"
    "}\n"
    "\n"
    "vec3 screen(vec2 p, int bottom) {                    // p: DS pixels inside the screen\n"
    "    vec2 tsz = vec2(textureSize(u_src, 0));\n"
    "    vec2 uv = vec2((p.x + (bottom != 0 ? u_botx.x : u_topx.x)) / u_texw, (bottom != 0 ? 0.5 : 1.0) - p.y / 384.0);\n"
    "    vec3 lin;\n"
    "    {\n"
    "        vec2 t = uv * tsz - 0.5;                     // sharp bilinear: crisp pixels, even at any size\n"
    "        vec2 i = floor(t), f = t - i;\n"
    "        vec2 s = 1.0 / max(g_duv * tsz, vec2(1e-4));\n"
    "        f = clamp((f - 0.5) * s + 0.5, 0.0, 1.0);\n"
    "        lin = fetch((i + 0.5 + f) / tsz);\n"
    "    }\n"
    "    if (u_lcdcolor != 0) lin = clamp(DSL * lin * DSL_LUM, 0.0, 1.0);\n"
    "    if (u_grid != 0) {\n"
    "        float strength = clamp((u_view.z - 1.5) / 2.0, 0.0, 1.0);   // fades in from 1.5x, full at 3.5x\n"
    "        if (strength > 0.0) {\n"
    "            vec2 fp = fract(p);\n"
    "            float w = clamp(g_dp * 0.75, 0.0, 0.3);\n"
    "            vec3 sub;\n"
    "            for (int k = 0; k < 3; k++) {            // red, green, blue stripes\n"
    "                float d = abs(fp.x - (float(k) + 0.5) / 3.0);\n"
    "                float m = 1.0 - smoothstep(1.0 / 6.0 - w, 1.0 / 6.0 + w, d);\n"
    "                sub[k] = mix(0.62, 1.0, m);\n"
    "            }\n"
    "            float gx = min(fp.x, 1.0 - fp.x), gy = min(fp.y, 1.0 - fp.y);\n"
    "            float line = mix(0.55, 1.0, smoothstep(0.035 - w, 0.035 + w, gx)) * mix(0.45, 1.0, smoothstep(0.06 - w, 0.06 + w, gy));\n"
    "            vec3 mask = sub * line;\n"
    "            mask /= 0.74;                            // keep the average brightness\n"
    "            lin *= mix(vec3(1.0), mask, strength);\n"
    "        }\n"
    "    }\n"
    "    lin = lin * u_backlight;\n"
    "    lin = lin * (1.0 - 0.002) + 0.002 * u_backlight;              // LCD black is never fully black\n"
    "    return lin;\n"
    "}\n"
    "\n"
    "/* --- the DS Lite ---------------------------------------------------------- */\n"
    "vec3 shade_plastic(vec3 col, float d, vec2 p, float glossy) {\n"
    "    // d: distance inside the part (negative inside); soft bevel near the edge\n"
    "    float in_ = -d;\n"
    "    float bevel = smoothstep(0.0, 7.0, in_);\n"
    "    vec3 c = col * mix(0.78, 1.0, bevel);\n"
    "    c += (1.0 - smoothstep(4.0, 9.0, abs(in_ - 5.0))) * 0.04;      // highlight along the bevel\n"
    "    float g = glossy * smoothstep(0.0, 1.0, 1.0 - abs((p.x - p.y * 0.6) / u_canvas.x - 0.2) * 2.5) * 0.06;\n"
    "    return c + g;\n"
    "}\n"
    "\n"
    "vec4 dslite(vec2 p) {\n"
    "    vec3 bg = vec3(0.055, 0.058, 0.065) * (1.0 - 0.35 * length((p - u_canvas * 0.5) / u_canvas));\n"
    "    vec3 col = bg;\n"
    "    float W = u_canvas.x;\n"
    "    float hingeY = u_ctop.y + u_ctop.w + 40.0;\n"
    "    // soft shadow of the console\n"
    "    float sh = sd_box(p, vec2(W * 0.5, u_canvas.y * 0.5 + 6.0), vec2(W * 0.5 - 4.0, u_canvas.y * 0.5 - 8.0), vec4(40.0));\n"
    "    col *= mix(0.55, 1.0, smoothstep(-4.0, 22.0, sh));\n"
    "    // base (bottom half)\n"
    "    float dbase = sd_box(p, vec2(W * 0.5, (hingeY + u_canvas.y) * 0.5), vec2(W * 0.5 - 2.0, (u_canvas.y - hingeY) * 0.5 - 2.0), vec4(10.0, 10.0, 38.0, 38.0));\n"
    "    col = mix(col, shade_plastic(u_base, dbase, p, 0.6), cover(dbase));\n"
    "    // lid (top half)\n"
    "    float dlid = sd_box(p, vec2(W * 0.5, hingeY * 0.5), vec2(W * 0.5 - 2.0, hingeY * 0.5 - 2.0), vec4(38.0, 38.0, 10.0, 10.0));\n"
    "    col = mix(col, shade_plastic(u_lid, dlid, p, 1.0), cover(dlid));\n"
    "    // hinge\n"
    "    float dh = sd_box(p, vec2(W * 0.5, hingeY), vec2(W * 0.5 - 18.0, 9.0), vec4(9.0));\n"
    "    float hy = clamp((p.y - hingeY) / 9.0, -1.0, 1.0);\n"
    "    vec3 hc = u_hinge * (0.75 + 0.35 * sqrt(max(1.0 - hy * hy, 0.0)));\n"
    "    col = mix(col, hc, cover(dh));\n"
    "    col = mix(col, u_hinge * 0.35, cover(sd_box(p, vec2(W * 0.5, hingeY), vec2(1.2, 7.0), vec4(1.0))) * 0.6);   // seam\n"
    "    col = mix(col, vec3(0.08), cover(sd_circle(p, vec2(W * 0.5 + 24.0, hingeY), 2.4)));                        // microphone\n"
    "    float led = sd_circle(p, vec2(W - 46.0, hingeY), 3.0);\n"
    "    col = mix(col, vec3(0.25, 0.95, 0.35), cover(led));\n"
    "    col += vec3(0.1, 0.5, 0.15) * exp(-max(led, 0.0) / 4.0) * 0.35;                                       // LED glow\n"
    "    // speakers beside the top screen\n"
    "    for (int s = 0; s < 2; s++) {\n"
    "        float cx = s == 0 ? u_ctop.x * 0.5 : W - u_ctop.x * 0.5;\n"
    "        float cy = u_ctop.y + u_ctop.w * 0.5;\n"
    "        for (int r = 0; r < 5; r++) for (int c = 0; c < 3; c++) {\n"
    "            vec2 hc2 = vec2(cx + (float(c) - 1.0) * 9.0, cy + (float(r) - 2.0) * 9.0);\n"
    "            col = mix(col, u_lid * 0.25, cover(sd_circle(p, hc2, 2.4)));\n"
    "        }\n"
    "    }\n"
    "    // screen windows (black glass around each LCD)\n"
    "    for (int s = 0; s < 2; s++) {\n"
    "        vec4 r = s == 0 ? u_ctop : u_cbot;\n"
    "        float m = s == 0 ? 13.0 : 9.0;\n"
    "        float dw = sd_box(p, r.xy + r.zw * 0.5, r.zw * 0.5 + m, vec4(s == 0 ? 7.0 : 4.0));\n"
    "        if (s == 1) {   // raised lip around the touch screen\n"
    "            float dl = sd_box(p, r.xy + r.zw * 0.5, r.zw * 0.5 + m + 5.0, vec4(8.0));\n"
    "            col = mix(col, u_base * 0.86, cover(dl));\n"
    "        }\n"
    "        vec3 glass = vec3(0.045, 0.047, 0.05) + 0.03 * smoothstep(0.0, 1.0, 1.0 - (p.y - r.y) / r.w);\n"
    "        col = mix(col, glass, cover(dw));\n"
    "    }\n"
    "    // D-pad\n"
    "    vec2 dp = vec2(u_cbot.x * 0.5, u_cbot.y + 58.0);\n"
    "    col = mix(col, u_base * 0.8, cover(min(sd_box(p, dp, vec2(45.0, 16.0), vec4(6.0)), sd_box(p, dp, vec2(16.0, 45.0), vec4(6.0)))) * 0.6);\n"
    "    float darm = min(sd_box(p, dp, vec2(42.0, 13.0), vec4(4.0)), sd_box(p, dp, vec2(13.0, 42.0), vec4(4.0)));\n"
    "    vec3 btn = shade_plastic(u_btn, darm, p, 0.5);\n"
    "    btn *= 1.0 - 0.18 * smoothstep(10.0, 0.0, length(p - dp));          // dimple in the middle\n"
    "    col = mix(col, btn, cover(darm));\n"
    "    // A B X Y\n"
    "    vec2 ab = vec2(W - u_cbot.x * 0.5, u_cbot.y + 58.0);\n"
    "    vec2 offs[4] = vec2[4](vec2(30.0, 0.0), vec2(0.0, 30.0), vec2(0.0, -30.0), vec2(-30.0, 0.0));\n"
    "    for (int k = 0; k < 4; k++) {\n"
    "        vec2 c = ab + offs[k];\n"
    "        col = mix(col, u_base * 0.75, cover(sd_circle(p, c, 15.5)) * 0.6);\n"
    "        float db = sd_circle(p, c, 13.0);\n"
    "        vec3 bc = u_btn * (1.0 + 0.12 * clamp(-(p.y - c.y + p.x - c.x) / 18.0, -1.0, 1.0));\n"
    "        col = mix(col, shade_plastic(bc, db, p, 0.3), cover(db));\n"
    "    }\n"
    "    // START / SELECT\n"
    "    for (int k = 0; k < 2; k++) {\n"
    "        vec2 c = ab + vec2(4.0, 78.0 + 26.0 * float(k));\n"
    "        col = mix(col, u_base * 0.75, cover(sd_circle(p, c, 8.5)) * 0.6);\n"
    "        float db = sd_circle(p, c, 6.5);\n"
    "        col = mix(col, shade_plastic(u_btn, db, p, 0.3), cover(db));\n"
    "    }\n"
    "    return vec4(col, 1.0);\n"
    "}\n"
    "\n"
    "vec3 shade(bool in_bot, vec2 p) {                 // one screen at canvas point p (gamma-encoded)\n"
    "    vec4 r = in_bot ? u_bot : u_top;\n"
    "    vec2 tx = in_bot ? u_botx : u_topx;\n"
    "    u_lo = vec2(tx.x / u_texw, in_bot ? 0.0 : 0.5); u_hi = vec2((tx.x + tx.y) / u_texw, in_bot ? 0.5 : 1.0);\n"
    "    g_duv = vec2(tx.y / (r.z * u_view.z * u_texw), 192.0 / (r.w * u_view.z * 384.0)); g_dp = tx.y / (r.z * u_view.z);\n"
    "    vec2 sp = (p - r.xy) * vec2(tx.y, 192.0) / r.zw;          // DS pixels\n"
    "    if (u_lcd == 0) {                             // sharp bilinear: exact pixels, even edges at any size\n"
    "        vec2 tsz = vec2(textureSize(u_src, 0));\n"
    "        vec2 t = vec2((sp.x + tx.x) / u_texw, (in_bot ? 0.5 : 1.0) - sp.y / 384.0) * tsz - 0.5;\n"
    "        vec2 i = floor(t), f = t - i;\n"
    "        vec2 s = 1.0 / max(g_duv * tsz, vec2(1e-4));\n"
    "        f = clamp((f - 0.5) * s + 0.5, 0.0, 1.0);\n"
    "        return textureGrad(u_src, keep_in((i + 0.5 + f) / tsz, g_duv), vec2(g_duv.x, 0.0), vec2(0.0, g_duv.y)).rgb;\n"
    "    }\n"
    "    return pow(clamp(screen(sp, in_bot ? 1 : 0), 0.0, 1.0), vec3(1.0 / 2.2));\n"
    "}\n"
    "\n"
    "void main() {\n"
    "    vec2 wp = vec2(gl_FragCoord.x, u_win.y - gl_FragCoord.y);\n"
    "    vec2 p = (wp - u_view.xy) / u_view.z;\n"
    "    px = 1.0 / u_view.z;\n"
    "    vec3 outc;                                       // gamma-encoded result\n"
    "    bool in_top = all(greaterThanEqual(p, u_top.xy)) && all(lessThan(p, u_top.xy + u_top.zw));\n"
    "    bool in_bot = all(greaterThanEqual(p, u_bot.xy)) && all(lessThan(p, u_bot.xy + u_bot.zw));\n"
    "    if (in_top || in_bot) {\n"
    "        bool both = in_top && in_bot;\n"
    "        bool fb = both ? u_front != 0 : in_bot;      // the screen in front\n"
    "        outc = shade(fb, p);\n"
    "        if (both && u_alpha < 0.999) outc = mix(shade(!fb, p), outc, u_alpha);   // see-through: the other one behind it\n"
    "    } else if (u_shell != 0 && u_skinmode != 0) {\n"
    "        vec3 bg = vec3(0.055, 0.058, 0.065) * (1.0 - 0.35 * length((p - u_canvas * 0.5) / u_canvas));\n"
    "        vec4 c = texture(u_skintex, p * u_skinscale);\n"
    "        bool inside = all(greaterThanEqual(p, vec2(0.0))) && all(lessThan(p, u_canvas));\n"
    "        outc = inside ? mix(bg, c.rgb, c.a) : bg;\n"
    "    } else if (u_shell != 0) {\n"
    "        outc = dslite(p).rgb;\n"
    "    } else {\n"
    "        outc = vec3(0.0);\n"
    "    }\n"
    "    o_color = vec4(outc, 1.0);\n"
    "}\n";

/* LCD motion blur: each frame is mixed with what the screen still shows
 * from the previous ones (kept in linear light). */
static const char* FS_GHOST =
    "#version 330 core\n"
    "in vec2 v_uv; out vec4 o_color; uniform sampler2D u_cur, u_prev; uniform float u_keep;\n"
    "void main() { vec3 c = pow(texture(u_cur, v_uv).rgb, vec3(2.2)); vec3 p = texture(u_prev, v_uv).rgb;\n"
    "  o_color = vec4(mix(c, p, u_keep), 1.0); }\n";

/* HD-2D: depth of field, bloom, colour grade, vignette on the 3D layer (as
 * metal_render.m). Row 0 of each texture is the bottom of the picture. */
/* the focus: the distance at the middle of the picture (where the hero or the
 * monsters are, in front of the background): the nearer third of 15 samples,
 * eased over frames */
static const char* FS_FOCUS =
    "#version 330 core\n"
    "out vec4 o_color; uniform sampler2D u_wd, u_prev; uniform float u_keep;\n"
    "void main() {\n"
    "  float v[15]; int n = 0;\n"
    "  vec2 sz = vec2(textureSize(u_wd, 0));\n"
    "  for (int j = 0; j < 3; j++) for (int i = 0; i < 5; i++) {\n"
    "    vec2 t = vec2(0.34 + 0.08 * float(i), 1.0 - (0.45 + 0.12 * float(j)));\n"   /* x across, y from the top */
    "    float w = texelFetch(u_wd, ivec2(t * sz), 0).a;\n"
    "    if (w < 1e8) v[n++] = w;\n"
    "  }\n"
    "  float old = texelFetch(u_prev, ivec2(0), 0).r;\n"
    "  if (n == 0) { o_color = vec4(old); return; }\n"
    "  for (int a = 1; a < n; a++) for (int b = a; b > 0 && v[b - 1] > v[b]; b--) { float t = v[b]; v[b] = v[b - 1]; v[b - 1] = t; }\n"
    "  float m = v[n / 3];\n"
    "  o_color = vec4(old <= 0.0 ? m : mix(m, old, u_keep)); }\n";

/* blur size in DS pixels: grows with the distance from the focus, more behind
 * it than in front; plus a little toward the top and bottom (tilt-shift) */
static const char* FS_DOF =
    "#version 330 core\n"
    "out vec4 o_color; uniform sampler2D u_col, u_wd, u_foc;\n"
    "uniform float u_scale, u_blur, u_focus, u_band, u_tilt;\n"
    "float coc(float w, float f, float ytd) {\n"
    "  if (w >= 1e8) return 3.0 * u_blur;\n"                      /* nothing drawn there (sky): far */
    "  float d = (w - f) / max(f, 1e-3);\n"
    "  float c = d > 0.0 ? d * 3.0 : -d * 2.5;\n"
    "  float tilt = ((1.0 - smoothstep(0.0, 0.42, ytd)) * 1.2 + smoothstep(0.8, 1.0, ytd) * 0.8) * u_tilt;\n"
    "  return clamp(max(c - u_band, 0.0) + tilt, 0.0, 2.5) * u_blur; }\n"
    "void main() {\n"
    "  ivec2 p = ivec2(gl_FragCoord.xy); ivec2 sz = textureSize(u_col, 0);\n"
    "  vec4 c0 = texelFetch(u_col, p, 0);\n"
    "  if (c0.a <= 0.0) { o_color = c0; return; }\n"
    "  float f = texelFetch(u_foc, ivec2(0), 0).r * u_focus;\n"
    "  float ytd = 1.0 - (float(p.y) + 0.5) / float(sz.y);\n"
    "  float r0 = coc(texelFetch(u_wd, p, 0).a, f, ytd) * u_scale;\n"   /* in pixels of this texture */
    "  if (r0 < 0.5) { o_color = c0; return; }\n"
    "  vec3 sum = c0.rgb; float ws = 1.0;\n"
    "  for (int i = 1; i < 24; i++) {\n"                          /* a golden-angle spiral */
    "    float rr = sqrt(float(i) / 24.0) * r0, a = float(i) * 2.39996;\n"
    "    ivec2 q = clamp(p + ivec2(round(vec2(cos(a), sin(a)) * rr)), ivec2(0), sz - 1);\n"
    "    vec4 c = texelFetch(u_col, q, 0);\n"
    "    if (c.a <= 0.0) continue;\n"
    "    float rq = coc(texelFetch(u_wd, q, 0).a, f, 1.0 - (float(q.y) + 0.5) / float(sz.y)) * u_scale;\n"
    "    float k = clamp(min(rq, r0) - rr + 1.0, 0.0, 1.0);\n"   /* a sharp pixel nearby doesn't spread over this one */
    "    sum += c.rgb * k; ws += k;\n"
    "  }\n"
    "  o_color = vec4(sum / ws, c0.a); }\n";

/* bloom: the bright parts, down to a few sizes and back up */
static const char* FS_BRIGHT =
    "#version 330 core\n"
    "in vec2 v_uv; out vec4 o_color; uniform sampler2D u_src; uniform float u_scale, u_threshold;\n"
    "void main() {\n"
    "  vec2 dsz = u_scale / vec2(textureSize(u_src, 0));\n"     /* one DS pixel (this texel) in the source */
    "  vec3 acc = vec3(0.0);\n"
    "  for (int j = 0; j < 4; j++) for (int i = 0; i < 4; i++) {\n"
    "    vec4 c = texture(u_src, v_uv + (vec2(i, j) - 1.5) / 4.0 * dsz);\n"
    "    float l = dot(c.rgb, vec3(0.299, 0.587, 0.114));\n"
    "    acc += c.rgb * smoothstep(u_threshold, 1.0, l) * step(0.001, c.a);\n"
    "  }\n"
    "  o_color = vec4(acc / 16.0, 1.0); }\n";
static const char* FS_DOWN =
    "#version 330 core\n"
    "in vec2 v_uv; out vec4 o_color; uniform sampler2D u_src;\n"
    "void main() {\n"
    "  vec2 d = 1.0 / vec2(textureSize(u_src, 0));\n"
    "  o_color = vec4((texture(u_src, v_uv + vec2(-d.x, -d.y)).rgb + texture(u_src, v_uv + vec2(d.x, -d.y)).rgb +\n"
    "                  texture(u_src, v_uv + vec2(-d.x, d.y)).rgb + texture(u_src, v_uv + vec2(d.x, d.y)).rgb) * 0.25, 1.0); }\n";
static const char* FS_UP =
    "#version 330 core\n"
    "in vec2 v_uv; out vec4 o_color; uniform sampler2D u_src;\n"
    "void main() {\n"
    "  vec2 d = 1.0 / vec2(textureSize(u_src, 0)), uv = v_uv;\n"
    "  vec3 c = texture(u_src, uv).rgb * 4.0;\n"                 /* a tent: smooth when it grows */
    "  c += (texture(u_src, uv + vec2(d.x, 0)).rgb + texture(u_src, uv - vec2(d.x, 0)).rgb + texture(u_src, uv + vec2(0, d.y)).rgb + texture(u_src, uv - vec2(0, d.y)).rgb) * 2.0;\n"
    "  c += texture(u_src, uv + d).rgb + texture(u_src, uv - d).rgb + texture(u_src, uv + vec2(d.x, -d.y)).rgb + texture(u_src, uv - vec2(d.x, -d.y)).rgb;\n"
    "  o_color = vec4(c / 16.0, 1.0); }\n";                      /* added to what is there */
/* the finish: bloom added, a warm grade, a vignette */
static const char* FS_HD2D =
    "#version 330 core\n"
    "out vec4 o_color; uniform sampler2D u_col, u_bloom; uniform float u_bloomk, u_warmth, u_vignette;\n"
    "void main() {\n"
    "  vec2 sz = vec2(textureSize(u_col, 0));\n"
    "  vec4 c0 = texelFetch(u_col, ivec2(gl_FragCoord.xy), 0);\n"
    "  if (c0.a <= 0.0) { o_color = c0; return; }\n"
    "  vec2 uv = gl_FragCoord.xy / sz;\n"
    "  vec3 c = c0.rgb + texture(u_bloom, uv).rgb * (u_bloomk / 5.0);\n"
    "  float l = dot(c, vec3(0.299, 0.587, 0.114));\n"
    "  c = mix(vec3(l), c, 1.0 + 0.12 * u_warmth);\n"                               /* a little more colour */
    "  c *= mix(vec3(1.0), mix(vec3(0.97, 0.99, 1.04), vec3(1.03, 1.0, 0.95), smoothstep(0.2, 0.9, l)), u_warmth);\n"   /* cool shadows, warm light */
    "  vec2 v = (uv - 0.5) * vec2(1.0, 0.8);\n"
    "  c *= 1.0 - u_vignette * smoothstep(0.3, 0.75, length(v));\n"
    "  o_color = vec4(clamp(c, 0.0, 1.0), c0.a); }\n";

/* Ray-traced shadows and occlusion: per pixel of the G-buffer (view position;
 * the normal from the neighbours), one ray toward the sun and a few short ones
 * around, against the frame's own triangles; then a depth-aware blur into a
 * shade factor. GL has no ray tracing of its own: the triangles go into a
 * bounding volume hierarchy (built on the CPU each frame, see rt_passes) that
 * this shader walks. Texture buffers:
 *   nodes: 2 texels each: (min, first), (max, count); count 0: children first, first + 1
 *   tris:  3 texels each: (corner, cut-out index or -1) x 3
 *   cuts:  3 texels each: texel coordinates s x 3, t x 3, then (mask offset, w, h, wrap)
 *   bits:  the cut-out textures' masks, 1 bit a texel (solid) */
static const char* FS_RT_TRACE =
    "#version 330 core\n"
    "out vec4 o_color;\n"
    "uniform sampler2D u_gpos; uniform samplerBuffer u_nodes, u_tris, u_cuts; uniform usamplerBuffer u_bits;\n"
    "uniform vec3 u_sun; uniform int u_frame, u_has_sun, u_ratio; uniform float u_ao_r;\n"
    "float wrap1(float x, float n, int m) {\n"                      /* 0 clamp, 1 repeat, 2 mirrored repeat */
    "  if (m == 0) return clamp(x, 0.0, n - 1.0);\n"
    "  if (m == 1) return x - n * floor(x / n);\n"
    "  float y = x - 2.0 * n * floor(x / (2.0 * n)); return y < n ? y : 2.0 * n - 1.0 - y; }\n"
    "bool solid_at(int cut, float bu, float bv) {\n"
    "  vec4 s = texelFetch(u_cuts, cut * 3), t = texelFetch(u_cuts, cut * 3 + 1), c = texelFetch(u_cuts, cut * 3 + 2);\n"
    "  float w0 = 1.0 - bu - bv;\n"
    "  float ss = s.x * w0 + s.y * bu + s.z * bv, tt = t.x * w0 + t.y * bu + t.z * bv;\n"
    "  int w = int(c.y), h = int(c.z), wrap = int(c.w);\n"
    "  int x = clamp(int(wrap1(floor(ss), float(w), wrap / 3)), 0, w - 1), y = clamp(int(wrap1(floor(tt), float(h), wrap % 3)), 0, h - 1);\n"
    "  int i = y * w + x;\n"
    "  return ((texelFetch(u_bits, int(c.x) + (i >> 3)).r >> uint(i & 7)) & 1u) != 0u; }\n"
    /* the nearest (or any, for shadows) hit beyond tmin, cut-outs by their texels; -1 if none */
    "float hit(vec3 o, vec3 d, float tmin, float tmax, bool any) {\n"
    "  vec3 inv = 1.0 / vec3(abs(d.x) > 1e-12 ? d.x : 1e-12, abs(d.y) > 1e-12 ? d.y : 1e-12, abs(d.z) > 1e-12 ? d.z : 1e-12);\n"
    "  float best = tmax; bool found = false;\n"
    "  int stack[64]; int sp = 0; stack[sp++] = 0;\n"
    "  while (sp > 0) {\n"
    "    int ni = stack[--sp];\n"
    "    vec4 a = texelFetch(u_nodes, ni * 2), b = texelFetch(u_nodes, ni * 2 + 1);\n"
    "    vec3 t0 = (a.xyz - o) * inv, t1 = (b.xyz - o) * inv, lo = min(t0, t1), hi = max(t0, t1);\n"
    "    if (max(max(lo.x, lo.y), max(lo.z, tmin)) > min(min(hi.x, hi.y), min(hi.z, best))) continue;\n"
    "    int first = int(a.w), cnt = int(b.w);\n"
    "    if (cnt == 0) { if (sp < 63) { stack[sp++] = first + 1; stack[sp++] = first; } continue; }\n"
    "    for (int k = first; k < first + cnt; k++) {\n"                /* Moller-Trumbore, both sides */
    "      vec4 p0 = texelFetch(u_tris, k * 3);\n"
    "      vec3 e1 = texelFetch(u_tris, k * 3 + 1).xyz - p0.xyz, e2 = texelFetch(u_tris, k * 3 + 2).xyz - p0.xyz, pv = cross(d, e2);\n"
    "      float det = dot(e1, pv);\n"
    "      if (det == 0.0) continue;\n"
    "      float id = 1.0 / det; vec3 tv = o - p0.xyz;\n"
    "      float u = dot(tv, pv) * id; if (u < 0.0 || u > 1.0) continue;\n"
    "      vec3 qv = cross(tv, e1); float v = dot(d, qv) * id; if (v < 0.0 || u + v > 1.0) continue;\n"
    "      float t = dot(e2, qv) * id; if (t < tmin || t >= best) continue;\n"
    "      if (p0.w >= 0.0 && !solid_at(int(p0.w), u, v)) continue;\n"
    "      best = t; found = true;\n"
    "      if (any) return t;\n"
    "    }\n"
    "  }\n"
    "  return found ? best : -1.0; }\n"
    "uint pcg(uint v) { uint s = v * 747796405u + 2891336453u; uint w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u; return (w >> 22u) ^ w; }\n"
    "float rnd(inout uint st) { st = pcg(st); return float(st) / 4294967296.0; }\n"
    "vec3 around(vec3 n, float a, float b) {\n"                     /* cosine-weighted around n */
    "  float r = sqrt(a), ph = 6.2831853 * b;\n"
    "  vec3 t = abs(n.x) > 0.5 ? vec3(0, 1, 0) : vec3(1, 0, 0);\n"
    "  vec3 x = normalize(cross(t, n)), y = cross(n, x);\n"
    "  return normalize(x * (r * cos(ph)) + y * (r * sin(ph)) + n * sqrt(max(1.0 - a, 0.0))); }\n"
    "void main() {\n"
    "  ivec2 gid = ivec2(gl_FragCoord.xy), gs = textureSize(u_gpos, 0);\n"
    "  ivec2 c = min(gid * u_ratio + u_ratio / 2, gs - 1);\n"
    "  vec4 g = texelFetch(u_gpos, c, 0);\n"
    "  if (g.w >= 1e8 || g.w <= 0.0) { o_color = vec4(1.0, 1.0, 0.0, 0.0); return; }\n"   /* nothing drawn */
    "  vec3 p = g.xyz;\n"
    "  vec3 xa = texelFetch(u_gpos, min(c + ivec2(1, 0), gs - 1), 0).xyz - p, xb = p - texelFetch(u_gpos, max(c - ivec2(1, 0), ivec2(0)), 0).xyz;\n"
    "  vec3 ya = texelFetch(u_gpos, min(c + ivec2(0, 1), gs - 1), 0).xyz - p, yb = p - texelFetch(u_gpos, max(c - ivec2(0, 1), ivec2(0)), 0).xyz;\n"
    "  vec3 n = cross(abs(xa.z) < abs(xb.z) ? xa : xb, abs(ya.z) < abs(yb.z) ? ya : yb);\n"   /* the face's normal */
    "  n = dot(n, n) > 0.0 ? normalize(n) : vec3(0.0, 0.0, 1.0);\n"
    "  if (dot(n, -p) < 0.0) n = -n;\n"
    "  float dist = max(length(p), 1e-3), eps = dist * 0.003;\n"
    "  uint st = pcg(uint(gid.x) * 1973u + uint(gid.y) * 9277u + uint(u_frame) * 26699u);\n"
    "  float vis = 1.0, ndl = 0.0;\n"
    "  if (u_has_sun != 0) {\n"
    "    ndl = dot(n, u_sun);\n"
    "    if (ndl > 0.0) {\n"
    "      vec3 d = normalize(u_sun + (vec3(rnd(st), rnd(st), rnd(st)) - 0.5) * 0.06);\n"   /* a sun of some size: soft edges */
    "      vis = hit(p + n * eps, d, eps, 1e7, true) < 0.0 ? 1.0 : 0.0;\n"
    "    }\n"
    "  }\n"
    "  float rad = u_ao_r * dist, occ = 0.0;\n"                     /* occlusion: as far as a fraction of the distance */
    "  for (int k = 0; k < 4; k++) {\n"
    "    float ra = rnd(st), rb = rnd(st);\n"
    "    float d = hit(p + n * eps, around(n, ra, rb), eps, rad, false);\n"
    "    if (d >= 0.0) occ += 1.0 - d / rad;\n"
    "  }\n"
    "  o_color = vec4(vis, 1.0 - occ / 4.0, max(ndl, 0.0), dist); }\n";
static const char* FS_RT_BLUR =
    "#version 330 core\n"
    "out vec4 o_color; uniform sampler2D u_raw; uniform float u_sun_k, u_ao_k;\n"
    "void main() {\n"
    "  ivec2 gid = ivec2(gl_FragCoord.xy), sz = textureSize(u_raw, 0);\n"
    "  vec4 c = texelFetch(u_raw, gid, 0);\n"
    "  if (c.w <= 0.0) { o_color = vec4(1.0); return; }\n"
    "  float vis = 0.0, ao = 0.0, ws = 0.0;\n"
    "  for (int j = -2; j <= 2; j++) for (int i = -2; i <= 2; i++) {\n"
    "    vec4 t = texelFetch(u_raw, clamp(gid + ivec2(i, j), ivec2(0), sz - 1), 0);\n"
    "    if (t.w <= 0.0) continue;\n"
    "    float w = exp(-abs(t.w - c.w) / (c.w * 0.03)) * exp(-float(i * i + j * j) / 6.0);\n"   /* same surface, nearby */
    "    vis += t.x * w; ao += t.y * w; ws += w;\n"
    "  }\n"
    "  vis /= ws; ao /= ws;\n"
    "  float shadow = (1.0 - vis) * smoothstep(0.0, 0.3, c.z) * u_sun_k;\n"   /* only where the sun would reach */
    "  o_color = vec4((1.0 - shadow) * mix(1.0, ao, u_ao_k)); }\n";

static GLuint compile(GLenum type, const char* src) {
    GLuint s = p_glCreateShader(type);
    p_glShaderSource(s, 1, &src, NULL);
    p_glCompileShader(s);
    GLint ok; p_glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) { char log[2048]; p_glGetShaderInfoLog(s, sizeof log, NULL, log); fprintf(stderr, "GL shader error:\n%s\n", log); return 0; }
    return s;
}

static GLuint program(const char* vs, const char* fs, int poly) {
    GLuint p = p_glCreateProgram();
    GLuint a = compile(GL_VERTEX_SHADER, vs), b = compile(GL_FRAGMENT_SHADER, fs);
    if (!a || !b) return 0;
    p_glAttachShader(p, a); p_glAttachShader(p, b);
    if (poly) {
        p_glBindAttribLocation(p, 0, "a_pos"); p_glBindAttribLocation(p, 1, "a_color");
        p_glBindAttribLocation(p, 2, "a_tex"); p_glBindAttribLocation(p, 3, "a_attr"); p_glBindAttribLocation(p, 4, "a_edge");
        p_glBindAttribLocation(p, 5, "a_vpos");
    } else p_glBindAttribLocation(p, 0, "a_pos");
    p_glLinkProgram(p);
    GLint ok; p_glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) { char log[2048]; p_glGetProgramInfoLog(p, sizeof log, NULL, log); fprintf(stderr, "GL link error:\n%s\n", log); return 0; }
    return p;
}

/* ---------------------------------------------------------------- state */
static int g_scale = 2;
static GLuint prog_poly, prog_post, prog_comp, prog_copy, prog_screen, prog_ghost;
static GLuint ghost_tex[2], ghost_fbo[2]; static int ghost_cur = -1, ghost_w, ghost_h;
static GLuint skin_tex; static int skin_w, skin_h;          /* console photo (View > Console colour) */
static GLuint vao_poly, vbo_poly, vao_quad, vbo_quad;
static GLuint samplers[9];                                    /* [s_mode * 3 + t_mode]: clamp, repeat, mirror */
static GLuint white_tex;

typedef struct {
    GLuint fbo, color, attr, depth;                            /* polygon pass */
    GLuint post_fbo, post;                                     /* after edge marking / fog */
    GLuint wdepth;                                             /* effects on: view position + distance (a third target of the pass) */
    GLuint scratch_fbo, hd2d_fbo, hd2d;                        /* ...the colour alone (depth of field), the HD-2D picture */
    int valid, hd2d_valid;
} Fb3D;
static Fb3D g_fb[2];
static int g_fb_latest = -1, g_fb_shown = -1;                  /* rendered last / visible this frame */
static int g_fb_w, g_fb_h;                                     /* one DS screen at scale */
static int g_wside, g_fb3_w;                                   /* widescreen: DS pixels added each side; 3D / final width at scale */
static int g_wide_shown = -1;                                  /* the screen composed wide last: 0 top, 1 bottom, -1 none */
static float g_half_w = 128.0f;                                /* the frame being drawn: half its width (DS pixels) */

static GLuint tex_col[2], tex_c2, tex_info;                   /* 2D inputs (per engine color; engine A extras) */
static GLuint final_fbo, final_tex;                            /* both screens, stacked, at scale */
static int g_ready;

/* The extra looks (Graphics menu), off at first. Off, nothing of them is made
 * or drawn: the frame is exactly as before. Their buffers (fx_*) exist only
 * while one is on. */
static int g_hd2d, g_rt;                                       /* the HD-2D look; ray-traced shadows and occlusion */
static int hd2d_ok, rt_ok;                                     /* their shaders compiled */
static GlLook g_look = GL_LOOK_DEFAULTS;                       /* the settings of both (Graphics > Effect settings) */
static int fx_on;                                              /* the buffers below exist, for this size */
static GLuint prog_focus, prog_dof, prog_bright, prog_down, prog_up, prog_hd2d, prog_trace, prog_rtblur;
static GLuint smp_linear;
#define BLOOM_LEVELS 5
static GLuint bloom_tex[BLOOM_LEVELS], bloom_fbo[BLOOM_LEVELS];   /* DS resolution, then halves */
static GLuint focus_tex[2], focus_fbo[2]; static int focus_cur;   /* the focus distance, eased over frames */
static GLuint rt_raw, rt_raw_fbo, rt_shade, rt_shade_fbo;       /* the rays' resolution (Effect settings) */
static int rt_w, rt_h, rt_ratio = 1;                           /* ...and 3D pixels per ray-traced pixel */
static GLuint rt_buf[4], rt_tbo[4];                            /* texture buffers: nodes, triangles, cut-outs, masks (FS_RT_TRACE) */
static GLint rt_tbo_max = 65536;                               /* texels a texture buffer can hold */

const int* g_gl_scale = &g_scale;

/* ---------------------------------------------------------------- textures */
typedef struct { uint32_t ti, pl; uint64_t hash; GLuint tex; int w, h; uint64_t checked, used; int live, holes; uint8_t* mask; } TexEntry;   /* holes: see-through texels, mask: 1 bit a texel, solid (ray tracing) */
#define TEXCACHE 4096
static TexEntry g_texcache[TEXCACHE];
static int g_smooth;                                           /* smooth textures (bilinear filtering) */
static uint64_t g_frameno;
static uint8_t* g_decode;

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
        p_glDeleteTextures(1, &lru->tex);
        free(lru->mask);
        lru->live = 0;
        free_slot = lru;
    }
    memset(free_slot, 0, sizeof *free_slot);
    free_slot->ti = key; free_slot->pl = pl; free_slot->live = 1; free_slot->checked = 0;
    p_glGenTextures(1, &free_slot->tex);
    return free_slot;
}

/* Performance counters, read and reset by the frontend each frame (bug reports). */
GlStats g_gl_stats;
static char g_gl_name[300];
const char* gl_renderer_name(void) { return g_gl_name; }
int gl_render_scale(void) { return g_scale; }
static double perf_ms(void) { return (double)SDL_GetPerformanceCounter() * 1000.0 / (double)SDL_GetPerformanceFrequency(); }

/* Smooth textures: a see-through texel's colour is never shown, but filtering
 * mixes it into its neighbours' (a dark fringe around every cut-out). Give
 * each one the average colour of the solid texels around it (two rounds, so
 * thin gaps fill too). Harmless without filtering: it stays see-through. */
static void bleed(uint8_t* px, int w, int h) {
    for (int round = 0; round < 2; round++) {
        int changed = 0;
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++) {
                uint8_t* p = px + ((size_t)y * w + x) * 4;
                if (p[3]) continue;                                   /* solid (or already filled: alpha 1) */
                int r = 0, g = 0, b = 0, n = 0;
                for (int dy = -1; dy <= 1; dy++)
                    for (int dx = -1; dx <= 1; dx++) {
                        int xx = (x + dx + w) % w, yy = (y + dy + h) % h;   /* textures usually repeat */
                        const uint8_t* q = px + ((size_t)yy * w + xx) * 4;
                        if (q[3]) { r += q[0]; g += q[1]; b += q[2]; n++; }
                    }
                if (n) { p[0] = (uint8_t)(r / n); p[1] = (uint8_t)(g / n); p[2] = (uint8_t)(b / n); p[3] = 1; changed = 1; }   /* alpha 1 for now: counts as solid next round */
            }
        if (!changed) break;
    }
    for (size_t i = 0; i < (size_t)w * h; i++) if (px[i * 4 + 3] == 1) px[i * 4 + 3] = 0;
}

static GLuint texture_for(uint32_t ti, uint32_t pl, int* w, int* h, int* holes, const uint8_t** mask) {
    TexEntry* e = tex_lookup(ti, pl);
    gpu3d_tex_size(ti, w, h);
    e->used = g_frameno;
    if (e->checked != g_frameno) {
        e->checked = g_frameno;
        uint64_t hs = gpu3d_texture_hash(ti, pl);
        if (hs != e->hash || e->w != *w || e->h != *h) {
            double t0 = perf_ms();
            e->hash = hs; e->w = *w; e->h = *h;
            gpu3d_decode_texture(ti, pl, g_decode);
            e->holes = 0;                                       /* ray tracing: which texels cast a shadow */
            for (size_t i = 0; i < (size_t)*w * *h && !e->holes; i++) e->holes = g_decode[i * 4 + 3] == 0;
            free(e->mask); e->mask = NULL;
            if (e->holes && (e->mask = calloc(((size_t)*w * *h + 7) / 8, 1)))
                for (size_t i = 0; i < (size_t)*w * *h; i++) if (g_decode[i * 4 + 3]) e->mask[i >> 3] |= (uint8_t)(1 << (i & 7));
            if (g_smooth) bleed(g_decode, *w, *h);
            p_glBindTexture(GL_TEXTURE_2D, e->tex);
            p_glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, *w, *h, 0, GL_RGBA, GL_UNSIGNED_BYTE, g_decode);
            g_gl_stats.uploads++; g_gl_stats.upload_texels += *w * *h; g_gl_stats.upload_ms += perf_ms() - t0;
        }
    }
    *holes = e->holes; *mask = e->mask;
    return e->tex;
}

/* ---------------------------------------------------------------- setup */
static GLuint new_tex(GLint ifmt, int w, int h, GLenum fmt, GLenum type, GLint filter) {
    GLuint t; p_glGenTextures(1, &t);
    p_glBindTexture(GL_TEXTURE_2D, t);
    p_glTexImage2D(GL_TEXTURE_2D, 0, ifmt, w, h, 0, fmt, type, NULL);
    p_glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, filter);
    p_glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filter);
    p_glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    p_glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    return t;
}

static void make_fbs(void) {
    g_fb_w = 256 * g_scale; g_fb_h = 192 * g_scale;
    g_fb3_w = (256 + 2 * g_wside) * g_scale;
    for (int i = 0; i < 2; i++) {
        Fb3D* f = &g_fb[i];
        f->color = new_tex(GL_RGBA8, g_fb3_w, g_fb_h, GL_RGBA, GL_UNSIGNED_BYTE, GL_NEAREST);
        f->attr  = new_tex(GL_RGBA8, g_fb3_w, g_fb_h, GL_RGBA, GL_UNSIGNED_BYTE, GL_NEAREST);
        f->depth = new_tex(GL_DEPTH24_STENCIL8, g_fb3_w, g_fb_h, GL_DEPTH_STENCIL, GL_UNSIGNED_INT_24_8, GL_NEAREST);   /* stencil: the DS's per-pixel flags (see the drawing) */
        p_glGenFramebuffers(1, &f->fbo);
        p_glBindFramebuffer(GL_FRAMEBUFFER, f->fbo);
        p_glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, f->color, 0);
        p_glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, f->attr, 0);
        p_glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_TEXTURE_2D, f->depth, 0);
        GLenum bufs[2] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1 };
        p_glDrawBuffers(2, bufs);
        if (p_glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) fprintf(stderr, "GL: 3D framebuffer incomplete\n");
        f->post = new_tex(GL_RGBA8, g_fb3_w, g_fb_h, GL_RGBA, GL_UNSIGNED_BYTE, GL_NEAREST);
        p_glGenFramebuffers(1, &f->post_fbo);
        p_glBindFramebuffer(GL_FRAMEBUFFER, f->post_fbo);
        p_glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, f->post, 0);
        f->valid = 0;
    }
    final_tex = new_tex(GL_RGBA8, g_fb3_w, g_fb_h * 2, GL_RGBA, GL_UNSIGNED_BYTE, GL_LINEAR);
    p_glGenFramebuffers(1, &final_fbo);
    p_glBindFramebuffer(GL_FRAMEBUFFER, final_fbo);
    p_glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, final_tex, 0);
    p_glBindFramebuffer(GL_FRAMEBUFFER, 0);
    g_fb_latest = g_fb_shown = -1;
}

/* a framebuffer drawing into tex alone */
static GLuint fbo_for(GLuint tex) {
    GLuint f; p_glGenFramebuffers(1, &f);
    p_glBindFramebuffer(GL_FRAMEBUFFER, f);
    p_glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
    return f;
}

/* The ray-traced shade's textures: rays for rt_res DS pixels across each DS
 * pixel (Effect settings: 0.5 to 2; they are blurred anyway). */
static void free_rt_tex(void) {
    if (!rt_raw) return;
    GLuint t[2] = { rt_raw, rt_shade }, b[2] = { rt_raw_fbo, rt_shade_fbo };
    p_glDeleteTextures(2, t); p_glDeleteFramebuffers(2, b);
    rt_raw = rt_shade = rt_raw_fbo = rt_shade_fbo = 0;
}
static void make_rt_tex(void) {
    free_rt_tex();
    float r = g_look.rt_res > 0 ? g_look.rt_res : 1.5f;
    rt_ratio = (int)lroundf((float)g_scale / r);
    if (rt_ratio < 1) rt_ratio = 1;
    if (!rt_ok) return;
    rt_w = (g_fb3_w + rt_ratio - 1) / rt_ratio; rt_h = (g_fb_h + rt_ratio - 1) / rt_ratio;
    rt_raw = new_tex(GL_RGBA16F, rt_w, rt_h, GL_RGBA, GL_FLOAT, GL_NEAREST);
    rt_shade = new_tex(GL_R16F, rt_w, rt_h, GL_RED, GL_FLOAT, GL_LINEAR);
    rt_raw_fbo = fbo_for(rt_raw); rt_shade_fbo = fbo_for(rt_shade);
    p_glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

/* The effects' buffers for the current size: a third target of the polygon
 * pass (view position, distance), the HD-2D picture, bloom, ray tracing. */
static void make_fx(void) {
    for (int i = 0; i < 2; i++) {
        Fb3D* f = &g_fb[i];
        f->wdepth = new_tex(GL_RGBA32F, g_fb3_w, g_fb_h, GL_RGBA, GL_FLOAT, GL_NEAREST);
        p_glBindFramebuffer(GL_FRAMEBUFFER, f->fbo);
        p_glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, GL_TEXTURE_2D, f->wdepth, 0);
        GLenum bufs[3] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2 };
        p_glDrawBuffers(3, bufs);
        if (p_glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) fprintf(stderr, "GL: 3D framebuffer with effects incomplete\n");
        f->hd2d = new_tex(GL_RGBA8, g_fb3_w, g_fb_h, GL_RGBA, GL_UNSIGNED_BYTE, GL_NEAREST);
        f->hd2d_fbo = fbo_for(f->hd2d);
        f->scratch_fbo = fbo_for(f->color);
        f->hd2d_valid = 0;
    }
    for (int i = 0; i < BLOOM_LEVELS; i++) {
        bloom_tex[i] = new_tex(GL_RGBA16F, (256 + 2 * g_wside) >> i, 192 >> i, GL_RGBA, GL_FLOAT, GL_LINEAR);
        bloom_fbo[i] = fbo_for(bloom_tex[i]);
    }
    make_rt_tex();
    p_glBindFramebuffer(GL_FRAMEBUFFER, 0);
    fx_on = 1;
}
static void free_fx(void) {
    if (!fx_on) return;
    for (int i = 0; i < 2; i++) {
        Fb3D* f = &g_fb[i];
        p_glBindFramebuffer(GL_FRAMEBUFFER, f->fbo);              /* the polygon pass as it was: two targets */
        p_glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, GL_TEXTURE_2D, 0, 0);
        GLenum bufs[2] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1 };
        p_glDrawBuffers(2, bufs);
        GLuint t[2] = { f->wdepth, f->hd2d }, b[2] = { f->hd2d_fbo, f->scratch_fbo };
        p_glDeleteTextures(2, t); p_glDeleteFramebuffers(2, b);
        f->wdepth = f->hd2d = f->hd2d_fbo = f->scratch_fbo = 0;
        f->hd2d_valid = 0;
    }
    p_glDeleteTextures(BLOOM_LEVELS, bloom_tex); p_glDeleteFramebuffers(BLOOM_LEVELS, bloom_fbo);
    free_rt_tex();
    p_glBindFramebuffer(GL_FRAMEBUFFER, 0);
    fx_on = 0;
}

static void free_fbs(void) {
    free_fx();
    for (int i = 0; i < 2; i++) {
        Fb3D* f = &g_fb[i];
        GLuint t[4] = { f->color, f->attr, f->depth, f->post }, b[2] = { f->fbo, f->post_fbo };
        p_glDeleteTextures(4, t); p_glDeleteFramebuffers(2, b);
    }
    p_glDeleteTextures(1, &final_tex); p_glDeleteFramebuffers(1, &final_fbo);
}

/* The largest 3D resolution this GPU can hold: widescreen up to 512 DS
 * pixels across, both screens 384 high in one texture. */
static int max_scale(void) {
    GLint m = 4096; p_glGetIntegerv(GL_MAX_TEXTURE_SIZE, &m);
    int s = (int)m / 512;
    return s < 1 ? 1 : s > GL_MAX_SCALE ? GL_MAX_SCALE : s;
}
int gl_render_max_scale(void) { return g_ready ? max_scale() : GL_MAX_SCALE; }

int gl_render_init(int scale) {
    if (!load_gl()) return 0;
    g_scale = scale < 1 ? 1 : scale > max_scale() ? max_scale() : scale;
    prog_poly = program(VS_POLY, FS_POLY, 1);
    prog_post = program(VS_QUAD, FS_POST, 0);
    prog_comp = program(VS_QUAD, FS_COMP, 0);
    prog_copy = program(VS_QUAD, FS_COPY, 0);
    prog_screen = program(VS_QUAD, FS_SCREEN, 0);
    prog_ghost = program(VS_QUAD, FS_GHOST, 0);
    if (!prog_poly || !prog_post || !prog_comp || !prog_copy || !prog_screen || !prog_ghost) return 0;
    p_glGenVertexArrays(1, &vao_poly); p_glBindVertexArray(vao_poly);
    p_glGenBuffers(1, &vbo_poly); p_glBindBuffer(GL_ARRAY_BUFFER, vbo_poly);
    GLsizei stride = VTX_FLOATS * sizeof(float);
    p_glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, stride, (void*)0);
    p_glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, stride, (void*)(4 * sizeof(float)));
    p_glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, stride, (void*)(8 * sizeof(float)));
    p_glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, stride, (void*)(10 * sizeof(float)));
    p_glVertexAttribPointer(4, 4, GL_FLOAT, GL_FALSE, stride, (void*)(14 * sizeof(float)));
    p_glVertexAttribPointer(5, 3, GL_FLOAT, GL_FALSE, stride, (void*)(18 * sizeof(float)));
    for (int i = 0; i < 6; i++) p_glEnableVertexAttribArray((GLuint)i);
    static const float quad[12] = { 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1 };
    p_glGenVertexArrays(1, &vao_quad); p_glBindVertexArray(vao_quad);
    p_glGenBuffers(1, &vbo_quad); p_glBindBuffer(GL_ARRAY_BUFFER, vbo_quad);
    p_glBufferData(GL_ARRAY_BUFFER, sizeof quad, quad, GL_STATIC_DRAW);
    p_glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
    p_glEnableVertexAttribArray(0);
    p_glGenSamplers(9, samplers);
    static const GLint wrap[3] = { GL_CLAMP_TO_EDGE, GL_REPEAT, GL_MIRRORED_REPEAT };
    for (int i = 0; i < 9; i++) {
        p_glSamplerParameteri(samplers[i], GL_TEXTURE_MIN_FILTER, g_smooth ? GL_LINEAR : GL_NEAREST);
        p_glSamplerParameteri(samplers[i], GL_TEXTURE_MAG_FILTER, g_smooth ? GL_LINEAR : GL_NEAREST);
        p_glSamplerParameteri(samplers[i], GL_TEXTURE_WRAP_S, wrap[i / 3]);
        p_glSamplerParameteri(samplers[i], GL_TEXTURE_WRAP_T, wrap[i % 3]);
    }
    static const uint8_t white[4] = { 255, 255, 255, 255 };
    white_tex = new_tex(GL_RGBA8, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, GL_NEAREST);
    p_glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, white);
    tex_col[0] = new_tex(GL_RGBA8, 256, 192, GL_RGBA, GL_UNSIGNED_BYTE, GL_NEAREST);
    tex_col[1] = new_tex(GL_RGBA8, 256, 192, GL_RGBA, GL_UNSIGNED_BYTE, GL_NEAREST);
    tex_c2 = new_tex(GL_RGBA8, 256, 192, GL_RGBA, GL_UNSIGNED_BYTE, GL_NEAREST);
    tex_info = new_tex(GL_RGBA8UI, 256, 192, GL_RGBA_INTEGER, GL_UNSIGNED_BYTE, GL_NEAREST);
    g_decode = malloc(1024 * 1024 * 4);
    p_glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    p_glPixelStorei(GL_PACK_ALIGNMENT, 1);
    /* the extra looks: without them (a shader this driver won't take) the game is as it is */
    prog_focus = program(VS_QUAD, FS_FOCUS, 0);
    prog_dof = program(VS_QUAD, FS_DOF, 0);
    prog_bright = program(VS_QUAD, FS_BRIGHT, 0);
    prog_down = program(VS_QUAD, FS_DOWN, 0);
    prog_up = program(VS_QUAD, FS_UP, 0);
    prog_hd2d = program(VS_QUAD, FS_HD2D, 0);
    hd2d_ok = prog_focus && prog_dof && prog_bright && prog_down && prog_up && prog_hd2d;
    if (!getenv("DQIX_NO_RT")) {
        prog_trace = program(VS_QUAD, FS_RT_TRACE, 0);
        prog_rtblur = program(VS_QUAD, FS_RT_BLUR, 0);
        rt_ok = prog_trace && prog_rtblur;
    }
    if (!hd2d_ok || (!rt_ok && !getenv("DQIX_NO_RT"))) fprintf(stderr, "GL: effects not available:%s%s\n", hd2d_ok ? "" : " HD-2D", rt_ok ? "" : " ray tracing");
    p_glGenSamplers(1, &smp_linear);
    p_glSamplerParameteri(smp_linear, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    p_glSamplerParameteri(smp_linear, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    p_glSamplerParameteri(smp_linear, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    p_glSamplerParameteri(smp_linear, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    if (hd2d_ok) {
        static const float zero = 0.0f;
        for (int i = 0; i < 2; i++) {
            focus_tex[i] = new_tex(GL_R32F, 1, 1, GL_RED, GL_FLOAT, GL_NEAREST);
            p_glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 1, 1, GL_RED, GL_FLOAT, &zero);   /* no focus yet */
            focus_fbo[i] = fbo_for(focus_tex[i]);
        }
        p_glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }
    if (rt_ok) {
        static const GLenum fmt[4] = { GL_RGBA32F, GL_RGBA32F, GL_RGBA32F, GL_R8UI };
        p_glGetIntegerv(GL_MAX_TEXTURE_BUFFER_SIZE, &rt_tbo_max);
        p_glGenBuffers(4, rt_buf); p_glGenTextures(4, rt_tbo);
        for (int i = 0; i < 4; i++) {
            p_glBindBuffer(GL_TEXTURE_BUFFER, rt_buf[i]);
            p_glBufferData(GL_TEXTURE_BUFFER, 16, NULL, GL_STREAM_DRAW);
            p_glBindTexture(GL_TEXTURE_BUFFER, rt_tbo[i]);
            p_glTexBuffer(GL_TEXTURE_BUFFER, fmt[i], rt_buf[i]);
        }
        p_glBindBuffer(GL_TEXTURE_BUFFER, 0); p_glBindTexture(GL_TEXTURE_BUFFER, 0);
    }
    make_fbs();
    g_ready = 1;
    snprintf(g_gl_name, sizeof g_gl_name, "%s / %s / OpenGL %s", (const char*)p_glGetString(GL_VENDOR),
             (const char*)p_glGetString(GL_RENDERER), (const char*)p_glGetString(GL_VERSION));
    fprintf(stderr, "GL renderer: %s, 3D at %dx\n", g_gl_name, g_scale);
    return 1;
}

/* Smooth textures: filter the DS's textures (the DS shows each texel as a
 * block). The textures are decoded again (see bleed). */
void gl_set_smooth(int on) {
    on = on != 0;
    if (!g_ready) { g_smooth = on; return; }
    if (on == g_smooth) return;
    g_smooth = on;
    for (int i = 0; i < 9; i++) {
        p_glSamplerParameteri(samplers[i], GL_TEXTURE_MIN_FILTER, on ? GL_LINEAR : GL_NEAREST);
        p_glSamplerParameteri(samplers[i], GL_TEXTURE_MAG_FILTER, on ? GL_LINEAR : GL_NEAREST);
    }
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
 * triangle edge opposite corner k is an edge of the DS polygon (not a diagonal
 * added to split the polygon into triangles). */
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
    o[14] = corner == 0; o[15] = corner == 1; o[16] = corner == 2;       /* lines: all 0 = always an edge */
    o[17] = (float)ext;
    o[18] = v->vx; o[19] = v->vy; o[20] = v->vz;
    (*nv)++;
}

#define ST_SOLID 0x80                    /* stencil: the pixel has a colour (alpha > 0) */
#define ST_TRANS 0x40                    /* a translucent polygon is on it (its id: bits 0-4, of the DS's 0-5) */
#define ST_MASKBIT 0x20                  /* marked by the latest shadow mask */
#define ST_ID 0x1F
typedef struct { int first, count; GLuint tex; int sampler; float tw, th; int translucent, depth_write, depth_eq, lines, pid, shadow, fog, alpha1, holes, mask_off; } Batch;
static Batch* g_batches; static int g_bcap;

/* The polygon pass's third target (effects on): written with the pixel's attributes. */
static void mask2(int on) { if (fx_on) p_glColorMaski(2, on, on, on, on); }

/* ---------------------------------------------------------------- ray tracing */
static uint8_t* rt_bits; static size_t rt_bits_n, rt_bits_cap;  /* this frame's cut-out masks */
static const uint8_t* rt_bits_src[256]; static uint32_t rt_bits_at[256]; static int rt_bits_k;
/* a cut-out texture's mask in this frame's buffer (once a frame each); -1 if it won't fit */
static int rt_mask(const uint8_t* m, int w, int h) {
    for (int i = 0; i < rt_bits_k; i++) if (rt_bits_src[i] == m) return (int)rt_bits_at[i];
    size_t n = ((size_t)w * h + 7) / 8;
    if (rt_bits_n + n > (size_t)rt_tbo_max || rt_bits_n + n > (1u << 24)) return -1;   /* (offsets are floats in the shader) */
    if (rt_bits_n + n > rt_bits_cap) { rt_bits_cap = (rt_bits_n + n) * 2; rt_bits = realloc(rt_bits, rt_bits_cap); }
    memcpy(rt_bits + rt_bits_n, m, n);
    int at = (int)rt_bits_n; rt_bits_n += n;
    if (rt_bits_k < 256) { rt_bits_src[rt_bits_k] = m; rt_bits_at[rt_bits_k++] = (uint32_t)at; }
    return at;
}

/* The frame's triangles (view space) and a bounding volume hierarchy over
 * them, built each frame: binned surface area heuristic, a few triangles a leaf. */
typedef struct { float v[9]; float cut; float lo[3], hi[3], c[3]; } RtTri;   /* cut: index into rt_cuts, -1 solid */
typedef struct { float s[3], t[3]; float off, w, h, wrap; } RtCut;
typedef struct { float lo[3], hi[3]; int first, count; } RtNode;              /* count 0: children at first, first + 1 */
static RtTri* rt_tri; static size_t rt_tri_n, rt_tri_cap;
static RtCut* rt_cuts; static size_t rt_cuts_n, rt_cuts_cap;
static RtNode* rt_node; static int rt_node_n;
static int* rt_idx;
static float* rt_up; static size_t rt_up_cap;                  /* what goes to the GPU */

static float area3(const float* lo, const float* hi) {
    float d[3] = { hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2] };
    return d[0] < 0 ? 0 : d[0] * d[1] + d[1] * d[2] + d[2] * d[0];
}
static void bvh_build(int ni, int start, int end, int depth) {
    RtNode* n = &rt_node[ni];
    float clo[3] = { 1e30f, 1e30f, 1e30f }, chi[3] = { -1e30f, -1e30f, -1e30f };
    for (int k = 0; k < 3; k++) { n->lo[k] = 1e30f; n->hi[k] = -1e30f; }
    for (int i = start; i < end; i++) {
        const RtTri* t = &rt_tri[rt_idx[i]];
        for (int k = 0; k < 3; k++) {
            if (t->lo[k] < n->lo[k]) n->lo[k] = t->lo[k];
            if (t->hi[k] > n->hi[k]) n->hi[k] = t->hi[k];
            if (t->c[k] < clo[k]) clo[k] = t->c[k];
            if (t->c[k] > chi[k]) chi[k] = t->c[k];
        }
    }
    int cnt = end - start;
    n->first = start; n->count = cnt;
    if (cnt <= 2 || depth >= 48) return;                       /* (the shader's stack holds 64) */
    int ax = 0;
    for (int k = 1; k < 3; k++) if (chi[k] - clo[k] > chi[ax] - clo[ax]) ax = k;
    float ext = chi[ax] - clo[ax];
    if (ext <= 1e-9f) return;                                  /* all in one place */
    enum { NB = 12 };
    int bn[NB] = { 0 }; float blo[NB][3], bhi[NB][3];
    for (int b = 0; b < NB; b++) for (int k = 0; k < 3; k++) { blo[b][k] = 1e30f; bhi[b][k] = -1e30f; }
    for (int i = start; i < end; i++) {
        const RtTri* t = &rt_tri[rt_idx[i]];
        int b = (int)((t->c[ax] - clo[ax]) / ext * NB); if (b >= NB) b = NB - 1;
        bn[b]++;
        for (int k = 0; k < 3; k++) { if (t->lo[k] < blo[b][k]) blo[b][k] = t->lo[k]; if (t->hi[k] > bhi[b][k]) bhi[b][k] = t->hi[k]; }
    }
    float rcost[NB]; int rn[NB];                               /* from the right: bins b.. */
    {   float lo[3] = { 1e30f, 1e30f, 1e30f }, hi[3] = { -1e30f, -1e30f, -1e30f }; int c = 0;
        for (int b = NB - 1; b > 0; b--) {
            c += bn[b];
            for (int k = 0; k < 3; k++) { if (blo[b][k] < lo[k]) lo[k] = blo[b][k]; if (bhi[b][k] > hi[k]) hi[k] = bhi[b][k]; }
            rcost[b] = c ? area3(lo, hi) * (float)c : 0; rn[b] = c;
        }
    }
    float best = 1e30f; int split = -1;
    {   float lo[3] = { 1e30f, 1e30f, 1e30f }, hi[3] = { -1e30f, -1e30f, -1e30f }; int c = 0;
        for (int b = 0; b < NB - 1; b++) {
            c += bn[b];
            for (int k = 0; k < 3; k++) { if (blo[b][k] < lo[k]) lo[k] = blo[b][k]; if (bhi[b][k] > hi[k]) hi[k] = bhi[b][k]; }
            if (!c || !rn[b + 1]) continue;
            float cost = area3(lo, hi) * (float)c + rcost[b + 1];
            if (cost < best) { best = cost; split = b; }
        }
    }
    if (split < 0 || (cnt <= 6 && best >= area3(n->lo, n->hi) * (float)cnt)) return;   /* a leaf is as good */
    int mid = start;
    for (int i = start; i < end; i++) {
        const RtTri* t = &rt_tri[rt_idx[i]];
        int b = (int)((t->c[ax] - clo[ax]) / ext * NB); if (b >= NB) b = NB - 1;
        if (b <= split) { int s = rt_idx[i]; rt_idx[i] = rt_idx[mid]; rt_idx[mid++] = s; }
    }
    if (mid == start || mid == end) return;
    int l = rt_node_n; rt_node_n += 2;
    n->first = l; n->count = 0;                                /* (n stays valid: the nodes were allocated up front) */
    bvh_build(l, start, mid, depth + 1);
    bvh_build(l + 1, mid, end, depth + 1);
}

/* one texture buffer's new contents */
static void tbo_fill(int i, const void* data, size_t bytes) {
    p_glBindBuffer(GL_TEXTURE_BUFFER, rt_buf[i]);
    p_glBufferData(GL_TEXTURE_BUFFER, (GLsizeiptr)(bytes ? bytes : 16), bytes ? data : NULL, GL_STREAM_DRAW);
}

static void bind_tex(int unit, GLenum target, GLuint tex, GLuint smp) {
    p_glActiveTexture(GL_TEXTURE0 + (GLenum)unit); p_glBindTexture(target, tex); p_glBindSampler((GLuint)unit, smp);
}
/* a full-target quad with the current program */
static void quad_draw(GLuint prog) {
    p_glUniform4f(p_glGetUniformLocation(prog, "u_rect"), -1, -1, 2, 2);
    p_glUniform4f(p_glGetUniformLocation(prog, "u_uv"), 0, 0, 1, 1);
    p_glBindVertexArray(vao_quad);
    p_glDrawArrays(GL_TRIANGLES, 0, 6);
}
static void pass_to(GLuint fbo, int w, int h, GLuint prog) {
    p_glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    p_glViewport(0, 0, w, h);
    p_glUseProgram(prog);
}
static void unbind_units(int n) {
    for (int i = n - 1; i >= 0; i--) { p_glActiveTexture(GL_TEXTURE0 + (GLenum)i); p_glBindSampler((GLuint)i, 0); p_glBindTexture(GL_TEXTURE_BUFFER, 0); }
}

/* Ray tracing on this frame's solid triangles (view space): shadows from the
 * sun, occlusion; the shade factor goes to rt_shade. 0 if nothing to trace. */
static int rt_passes(Fb3D* fb, const Frame3D* f, int nb) {
    rt_tri_n = rt_cuts_n = 0;
    for (int i = 0; i < nb; i++) {
        const Batch* b = &g_batches[i];
        if (b->translucent || b->lines || b->shadow) continue;
        if (b->holes && b->mask_off < 0) continue;              /* no mask: it would cast boxes */
        for (int k = 0; k + 2 < b->count; k += 3) {
            const float* v = g_verts + (size_t)(b->first + k) * VTX_FLOATS;
            if (rt_tri_n + 1 > rt_tri_cap) { rt_tri_cap = (rt_tri_n + 1) * 2; rt_tri = realloc(rt_tri, rt_tri_cap * sizeof *rt_tri); }
            RtTri* t = &rt_tri[rt_tri_n++];
            for (int j = 0; j < 3; j++) memcpy(t->v + j * 3, v + j * VTX_FLOATS + 18, 3 * sizeof(float));
            for (int c = 0; c < 3; c++) {
                float a = t->v[c], bb = t->v[3 + c], cc = t->v[6 + c];
                t->lo[c] = fminf(a, fminf(bb, cc)); t->hi[c] = fmaxf(a, fmaxf(bb, cc)); t->c[c] = (a + bb + cc) / 3.0f;
            }
            t->cut = -1;
            if (!b->holes) continue;
            if (rt_cuts_n + 1 > rt_cuts_cap) { rt_cuts_cap = (rt_cuts_n + 1) * 2; rt_cuts = realloc(rt_cuts, rt_cuts_cap * sizeof *rt_cuts); }
            RtCut* c = &rt_cuts[rt_cuts_n];
            for (int j = 0; j < 3; j++) { c->s[j] = v[j * VTX_FLOATS + 8]; c->t[j] = v[j * VTX_FLOATS + 9]; }
            c->off = (float)b->mask_off; c->w = b->tw; c->h = b->th; c->wrap = (float)b->sampler;
            t->cut = (float)rt_cuts_n++;
        }
    }
    {   static int lg = -1; if (lg < 0) lg = getenv("DQIX_RT_LOG") != NULL;   /* tests: what the rays see */
        if (lg && g_frame % 300 == 0)
            fprintf(stderr, "rt: frame %llu triangles %zu cut-out %zu (masks %zu bytes)\n", (unsigned long long)g_frame, rt_tri_n, rt_cuts_n, rt_bits_n);
    }
    if (!rt_tri_n) return 0;
    if (rt_tri_n * 4 > (size_t)rt_tbo_max) rt_tri_n = (size_t)rt_tbo_max / 4;   /* the nodes too must fit (2 a triangle, 2 texels each) */
    double t0 = perf_ms();
    rt_node = realloc(rt_node, sizeof *rt_node * rt_tri_n * 2);
    rt_idx = realloc(rt_idx, sizeof *rt_idx * rt_tri_n);
    for (size_t i = 0; i < rt_tri_n; i++) rt_idx[i] = (int)i;
    rt_node_n = 1;
    bvh_build(0, 0, (int)rt_tri_n, 0);
    size_t need = (size_t)rt_node_n * 8 > rt_tri_n * 12 ? (size_t)rt_node_n * 8 : rt_tri_n * 12;
    if (need > rt_up_cap) { rt_up_cap = need * 2; rt_up = realloc(rt_up, rt_up_cap * sizeof(float)); }
    for (int i = 0; i < rt_node_n; i++) {
        float* o = rt_up + (size_t)i * 8; const RtNode* n = &rt_node[i];
        o[0] = n->lo[0]; o[1] = n->lo[1]; o[2] = n->lo[2]; o[3] = (float)n->first;
        o[4] = n->hi[0]; o[5] = n->hi[1]; o[6] = n->hi[2]; o[7] = (float)n->count;
    }
    tbo_fill(0, rt_up, (size_t)rt_node_n * 8 * sizeof(float));
    for (size_t i = 0; i < rt_tri_n; i++) {                    /* in the hierarchy's order */
        float* o = rt_up + i * 12; const RtTri* t = &rt_tri[rt_idx[i]];
        for (int j = 0; j < 3; j++) { memcpy(o + j * 4, t->v + j * 3, 3 * sizeof(float)); o[j * 4 + 3] = j ? 0 : t->cut; }
    }
    tbo_fill(1, rt_up, rt_tri_n * 12 * sizeof(float));
    if (rt_cuts_n * 12 > rt_up_cap) { rt_up_cap = rt_cuts_n * 24; rt_up = realloc(rt_up, rt_up_cap * sizeof(float)); }
    for (size_t i = 0; i < rt_cuts_n; i++) {
        float* o = rt_up + i * 12; const RtCut* c = &rt_cuts[i];
        o[0] = c->s[0]; o[1] = c->s[1]; o[2] = c->s[2]; o[3] = 0;
        o[4] = c->t[0]; o[5] = c->t[1]; o[6] = c->t[2]; o[7] = 0;
        o[8] = c->off; o[9] = c->w; o[10] = c->h; o[11] = c->wrap;
    }
    tbo_fill(2, rt_up, rt_cuts_n * 12 * sizeof(float));
    tbo_fill(3, rt_bits, rt_bits_n);
    p_glBindBuffer(GL_TEXTURE_BUFFER, 0);
    {   static int lg = -1; if (lg < 0) lg = getenv("DQIX_RT_LOG") != NULL;
        if (lg && g_frame % 300 == 0) fprintf(stderr, "rt: frame %llu nodes %d, built and sent in %.2f ms\n", (unsigned long long)g_frame, rt_node_n, perf_ms() - t0);
    }
    /* The game's own light if it lights its models with the DS's lights; DQIX
     * doesn't (vertex colours, toon shading), so then a sun of our own: above,
     * a little to the left and behind the camera (see metal_render.m). */
    float az = g_look.sun_turn * 3.14159265f / 180.0f, el = g_look.sun_height * 3.14159265f / 180.0f;
    float sun[3] = { cosf(el) * sinf(az), sinf(el), cosf(el) * cosf(az) }, sl = 0;   /* turn 0 = from behind the camera */
    if (f->has_sun) memcpy(sun, f->sun, sizeof sun);
    for (int c = 0; c < 3; c++) sl += sun[c] * sun[c];
    sl = 1.0f / sqrtf(sl);
    for (int c = 0; c < 3; c++) sun[c] *= sl;
    p_glDisable(GL_DEPTH_TEST); p_glDisable(GL_STENCIL_TEST); p_glDisablei(GL_BLEND, 0);
    pass_to(rt_raw_fbo, rt_w, rt_h, prog_trace);
    GLuint P = prog_trace;
    bind_tex(0, GL_TEXTURE_2D, fb->wdepth, 0);
    for (int i = 0; i < 4; i++) bind_tex(1 + i, GL_TEXTURE_BUFFER, rt_tbo[i], 0);
    p_glUniform1i(p_glGetUniformLocation(P, "u_gpos"), 0);
    p_glUniform1i(p_glGetUniformLocation(P, "u_nodes"), 1);
    p_glUniform1i(p_glGetUniformLocation(P, "u_tris"), 2);
    p_glUniform1i(p_glGetUniformLocation(P, "u_cuts"), 3);
    p_glUniform1i(p_glGetUniformLocation(P, "u_bits"), 4);
    p_glUniform3fv(p_glGetUniformLocation(P, "u_sun"), 1, sun);
    p_glUniform1i(p_glGetUniformLocation(P, "u_frame"), (GLint)(g_frameno & 0x7FFFFFFF));
    p_glUniform1i(p_glGetUniformLocation(P, "u_has_sun"), 1);
    p_glUniform1i(p_glGetUniformLocation(P, "u_ratio"), rt_ratio);
    p_glUniform1f(p_glGetUniformLocation(P, "u_ao_r"), g_look.reach);
    quad_draw(P);
    unbind_units(5);
    pass_to(rt_shade_fbo, rt_w, rt_h, prog_rtblur);
    bind_tex(0, GL_TEXTURE_2D, rt_raw, 0);
    p_glUniform1i(p_glGetUniformLocation(prog_rtblur, "u_raw"), 0);
    p_glUniform1f(p_glGetUniformLocation(prog_rtblur, "u_sun_k"), g_look.shadow);
    p_glUniform1f(p_glGetUniformLocation(prog_rtblur, "u_ao_k"), g_look.shading);
    quad_draw(prog_rtblur);
    return 1;
}

/* The HD-2D look on the finished 3D layer: post -> depth of field -> + bloom, grade, vignette -> hd2d.
 * The colour goes through fb->color as scratch (the polygon pass is done with it). */
static void hd2d_passes(Fb3D* fb) {
    const GlLook* L = &g_look;
    int W = 256 + 2 * g_wside;
    p_glDisable(GL_DEPTH_TEST); p_glDisable(GL_STENCIL_TEST); p_glDisablei(GL_BLEND, 0);
    int prev = focus_cur, cur = focus_cur ^ 1;
    pass_to(focus_fbo[cur], 1, 1, prog_focus);
    bind_tex(0, GL_TEXTURE_2D, fb->wdepth, 0); bind_tex(1, GL_TEXTURE_2D, focus_tex[prev], 0);
    p_glUniform1i(p_glGetUniformLocation(prog_focus, "u_wd"), 0);
    p_glUniform1i(p_glGetUniformLocation(prog_focus, "u_prev"), 1);
    p_glUniform1f(p_glGetUniformLocation(prog_focus, "u_keep"), 0.85f);
    quad_draw(prog_focus);
    focus_cur = cur;
    pass_to(fb->scratch_fbo, g_fb3_w, g_fb_h, prog_dof);
    bind_tex(0, GL_TEXTURE_2D, fb->post, 0); bind_tex(1, GL_TEXTURE_2D, fb->wdepth, 0); bind_tex(2, GL_TEXTURE_2D, focus_tex[cur], 0);
    p_glUniform1i(p_glGetUniformLocation(prog_dof, "u_col"), 0);
    p_glUniform1i(p_glGetUniformLocation(prog_dof, "u_wd"), 1);
    p_glUniform1i(p_glGetUniformLocation(prog_dof, "u_foc"), 2);
    p_glUniform1f(p_glGetUniformLocation(prog_dof, "u_scale"), (float)g_scale);
    p_glUniform1f(p_glGetUniformLocation(prog_dof, "u_blur"), L->blur);
    p_glUniform1f(p_glGetUniformLocation(prog_dof, "u_focus"), L->focus);
    p_glUniform1f(p_glGetUniformLocation(prog_dof, "u_band"), L->focus_band);
    p_glUniform1f(p_glGetUniformLocation(prog_dof, "u_tilt"), L->tilt);
    quad_draw(prog_dof);
    unbind_units(3);
    pass_to(bloom_fbo[0], W, 192, prog_bright);
    bind_tex(0, GL_TEXTURE_2D, fb->color, smp_linear);
    p_glUniform1i(p_glGetUniformLocation(prog_bright, "u_src"), 0);
    p_glUniform1f(p_glGetUniformLocation(prog_bright, "u_scale"), (float)g_scale);
    p_glUniform1f(p_glGetUniformLocation(prog_bright, "u_threshold"), L->bloom_threshold);
    quad_draw(prog_bright);
    for (int i = 1; i < BLOOM_LEVELS; i++) {
        pass_to(bloom_fbo[i], W >> i, 192 >> i, prog_down);
        bind_tex(0, GL_TEXTURE_2D, bloom_tex[i - 1], smp_linear);
        p_glUniform1i(p_glGetUniformLocation(prog_down, "u_src"), 0);
        quad_draw(prog_down);
    }
    p_glEnablei(GL_BLEND, 0);                                  /* added to what is there */
    p_glBlendFuncSeparate(GL_ONE, GL_ONE, GL_ONE, GL_ONE);
    p_glBlendEquationSeparate(GL_FUNC_ADD, GL_FUNC_ADD);
    for (int i = BLOOM_LEVELS - 2; i >= 0; i--) {
        pass_to(bloom_fbo[i], W >> i, 192 >> i, prog_up);
        bind_tex(0, GL_TEXTURE_2D, bloom_tex[i + 1], smp_linear);
        p_glUniform1i(p_glGetUniformLocation(prog_up, "u_src"), 0);
        quad_draw(prog_up);
    }
    p_glDisablei(GL_BLEND, 0);
    pass_to(fb->hd2d_fbo, g_fb3_w, g_fb_h, prog_hd2d);
    bind_tex(0, GL_TEXTURE_2D, fb->color, 0); bind_tex(1, GL_TEXTURE_2D, bloom_tex[0], smp_linear);
    p_glUniform1i(p_glGetUniformLocation(prog_hd2d, "u_col"), 0);
    p_glUniform1i(p_glGetUniformLocation(prog_hd2d, "u_bloom"), 1);
    p_glUniform1f(p_glGetUniformLocation(prog_hd2d, "u_bloomk"), L->bloom);
    p_glUniform1f(p_glGetUniformLocation(prog_hd2d, "u_warmth"), L->warmth);
    p_glUniform1f(p_glGetUniformLocation(prog_hd2d, "u_vignette"), L->vignette);
    quad_draw(prog_hd2d);
    unbind_units(2);
}

static void gl_vblank(void) {
    g_fb_shown = g_fb_latest;                                  /* what was rendered last is visible now */
}

static void gl_render_(const Frame3D* f);
static void gl_render(const Frame3D* f) {
    double t0 = perf_ms();
    gl_render_(f);
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
    int hd2d = g_hd2d && hd2d_ok, rt = g_rt && rt_ok;
    if ((hd2d || rt) && !fx_on) make_fx();                     /* the effects' buffers only while one is on */
    else if (!hd2d && !rt && fx_on) free_fx();
    uint16_t disp3d = IO16_(0x060);
    int textures_on = disp3d & 1;
    /* vertices + batches */
    int nv = 0, nb = 0;
    rt_bits_n = 0; rt_bits_k = 0;
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
            const uint8_t* mask;
            b.tex = texture_for(p->teximg, p->pltt, &w, &h, &b.holes, &mask);
            b.mask_off = b.holes && mask && rt ? rt_mask(mask, w, h) : -1;
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
    p_glBindFramebuffer(GL_FRAMEBUFFER, fb->fbo);
    p_glViewport(0, 0, g_fb3_w, g_fb_h);

    /* clear: color (straight), attributes (clear polygon id, fog), depth and the per-pixel flags */
    uint32_t clear = IO32_(0x350);
    uint16_t cdepth = IO16_(0x354) & 0x7FFF;
    float ca = (float)((clear >> 16) & 31) / 31.0f;
    float cc[4] = { (float)(clear & 31) / 31.0f, (float)((clear >> 5) & 31) / 31.0f, (float)((clear >> 10) & 31) / 31.0f, ca };
    float cattr[4] = { (float)((clear >> 24) & 63) / 63.0f, (float)((clear >> 15) & 1), 0.0f, 0.0f };
    uint32_t d24 = (uint32_t)cdepth * 0x200 + ((cdepth + 1) / 0x8000) * 0x1FF;
    float cd = cdepth == 0x7FFF ? 1.0f : f->wbuf ? fminf((float)d24 / 4096.0f / 4096.0f, 1.0f) : (float)d24 / 16777215.0f;
    p_glColorMaski(0, 1, 1, 1, 1); p_glColorMaski(1, 1, 1, 1, 1);
    p_glDepthMask(GL_TRUE);
    p_glClearBufferfv(GL_COLOR, 0, cc);
    p_glClearBufferfv(GL_COLOR, 1, cattr);
    if (fx_on) { static const float far_[4] = { 0, 0, 0, 1e9f }; mask2(1); p_glClearBufferfv(GL_COLOR, 2, far_); }   /* nothing drawn: far away */
    p_glStencilMask(0xFF);
    p_glClearBufferfi(GL_DEPTH_STENCIL, 0, cd, ca > 0.0f ? ST_SOLID : 0);

    p_glBindVertexArray(vao_poly);
    p_glBindBuffer(GL_ARRAY_BUFFER, vbo_poly);
    p_glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)(sizeof(float) * VTX_FLOATS * (size_t)nv), g_verts, GL_STREAM_DRAW);
    p_glUseProgram(prog_poly);
    { static float ew = -1; if (ew < 0) { const char* e = getenv("DQIX_EDGEW"); ew = e ? (float)atof(e) : 0; }
      /* outlines: one DS pixel wide at 1x; above that half of one, so a thin leg or
       * arm keeps its colour inside the outline instead of becoming all outline */
      p_glUniform1f(p_glGetUniformLocation(prog_poly, "u_edgew"), ew > 0 ? ew : g_scale >= 2 ? (float)g_scale * 0.5f : 1.0f); }
    p_glUniform1i(p_glGetUniformLocation(prog_poly, "u_tex"), 0);
    p_glUniform1i(p_glGetUniformLocation(prog_poly, "u_highlight"), (disp3d >> 1) & 1);
    p_glUniform1i(p_glGetUniformLocation(prog_poly, "u_alphatest"), (disp3d >> 2) & 1);
    p_glUniform1f(p_glGetUniformLocation(prog_poly, "u_aref"), (float)(IO16_(0x340) & 31));
    p_glUniform1i(p_glGetUniformLocation(prog_poly, "u_wbuf"), f->wbuf);
    float toon[96];
    for (int i = 0; i < 32; i++) { uint16_t c = IO16_(0x380 + i * 2); toon[i * 3] = (float)(c & 31) / 31.0f; toon[i * 3 + 1] = (float)((c >> 5) & 31) / 31.0f; toon[i * 3 + 2] = (float)((c >> 10) & 31) / 31.0f; }
    p_glUniform3fv(p_glGetUniformLocation(prog_poly, "u_toon"), 32, toon);
    GLint loc_texsize = p_glGetUniformLocation(prog_poly, "u_texsize");
    GLint loc_sharpa = p_glGetUniformLocation(prog_poly, "u_sharpa");
    GLint loc_part = p_glGetUniformLocation(prog_poly, "u_part");
    p_glUniform1i(loc_part, 0);
    p_glEnable(GL_DEPTH_TEST);
    p_glActiveTexture(GL_TEXTURE0);
    int blend_on = (disp3d >> 3) & 1;
    /* The DS's rules for translucent pixels, kept per pixel in the stencil buffer:
     *   ST_SOLID  the pixel has a colour (alpha > 0): opaque, or the clear colour
     *   ST_TRANS  a translucent polygon is on it, with its id in the low 6 bits
     * - onto a pixel with no colour yet, a translucent polygon is written as it is
     *   (its own colour and alpha: nothing to blend with);
     * - onto any other pixel it is blended, and alpha becomes the larger of the two;
     * - never onto a pixel already covered by a translucent polygon with the same
     *   id (one mesh doesn't blend over itself: layered clouds, glows, water).
     * Shadows: the mask (id 0) marks where it is hidden (depth test fails); the
     * shadow then darkens only those pixels, and not the object of its own id. */
    p_glEnable(GL_STENCIL_TEST);
    int in_mask = 0;
    for (int i = 0; i < nb; i++) {
        Batch* b = &g_batches[i];
        p_glBindTexture(GL_TEXTURE_2D, b->tex);
        p_glBindSampler(0, samplers[b->sampler]);
        p_glUniform2f(loc_texsize, b->tw, b->th);
        p_glUniform1i(loc_sharpa, g_smooth && b->alpha1);
        p_glDepthFunc(b->depth_eq ? GL_EQUAL : GL_LESS);
        p_glDepthMask(b->depth_write ? GL_TRUE : GL_FALSE);
        GLenum prim = b->lines ? GL_LINES : GL_TRIANGLES;
        if (b->shadow == 1) {                                   /* shadow mask: no colour; mark where it fails the depth test */
            p_glColorMaski(0, 0, 0, 0, 0); p_glColorMaski(1, 0, 0, 0, 0); mask2(0);
            p_glDepthMask(GL_FALSE);
            p_glStencilMask(ST_MASKBIT);
            if (!in_mask) { GLint zero = 0; p_glClearBufferiv(GL_STENCIL, 0, &zero); }   /* a new mask: the old marks go */
            in_mask = 1;
            p_glStencilFunc(GL_ALWAYS, ST_MASKBIT, ST_MASKBIT);
            p_glStencilOp(GL_KEEP, GL_REPLACE, GL_KEEP);
            p_glDrawArrays(prim, b->first, b->count);
            p_glColorMaski(0, 1, 1, 1, 1);
            continue;
        }
        in_mask = 0;
        if (!b->translucent && !b->shadow) {
            p_glDisablei(GL_BLEND, 0);
            p_glColorMaski(1, 1, 1, 1, 1); mask2(1);
            p_glStencilMask(0xFF);
            p_glStencilFunc(GL_ALWAYS, ST_SOLID, 0xFF);
            p_glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE);
            p_glDrawArrays(prim, b->first, b->count);
            continue;
        }
        /* a translucent polygon's solid pixels (alpha 31: most of a textured model with
         * see-through edges) are drawn like an opaque polygon's on the DS: they hide
         * what's behind, write depth, and aren't see-through pixels for the id rule */
        if (b->shadow != 2) {
            p_glUniform1i(loc_part, 1);
            p_glDisablei(GL_BLEND, 0);
            p_glColorMaski(1, 1, 1, 1, 1); mask2(1);
            p_glDepthMask(GL_TRUE);
            p_glStencilMask(0xFF);
            p_glStencilFunc(GL_ALWAYS, ST_SOLID, 0xFF);
            p_glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE);
            p_glDrawArrays(prim, b->first, b->count);
            p_glDepthMask(b->depth_write ? GL_TRUE : GL_FALSE);
            p_glUniform1i(loc_part, 2);
        }
        GLint ref = ST_SOLID | ST_TRANS | (b->pid & ST_ID);
        /* translucent pixels keep the opaque polygon id; their fog flag is the pixel's AND the
         * polygon's (the DS): a polygon with fog leaves it as it was, one without clears it */
        p_glColorMaski(1, 0, b->fog ? 0 : 1, 0, 0); mask2(0);
        if (b->shadow == 2) {                                   /* the shadow: only on marked pixels; clears the marks */
            if (blend_on) { p_glEnablei(GL_BLEND, 0); p_glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE); p_glBlendEquationSeparate(GL_FUNC_ADD, GL_MAX); }
            else p_glDisablei(GL_BLEND, 0);
            p_glStencilMask(0xFF);
            p_glStencilFunc(GL_EQUAL, ref | ST_MASKBIT, ST_MASKBIT);   /* the marks stay for the next shadow, like the DS */
            p_glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE);
            p_glDrawArrays(prim, b->first, b->count);
            p_glUniform1i(loc_part, 0);
            continue;
        }
        p_glStencilMask(0xFF);
        p_glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE);
        /* 1: pixels with no colour yet: the polygon as it is */
        p_glDisablei(GL_BLEND, 0);
        p_glStencilFunc(GL_GREATER, ref, ST_SOLID);
        p_glDrawArrays(prim, b->first, b->count);
        /* 2: the others, except where this id already is (or 1 just drew) */
        if (blend_on) {
            p_glEnablei(GL_BLEND, 0);
            p_glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE);
            p_glBlendEquationSeparate(GL_FUNC_ADD, GL_MAX);
        } else {                                                /* blending off: the colour replaces, alpha still the larger */
            p_glEnablei(GL_BLEND, 0);
            p_glBlendFuncSeparate(GL_ONE, GL_ZERO, GL_ONE, GL_ONE);
            p_glBlendEquationSeparate(GL_FUNC_ADD, GL_MAX);
        }
        p_glStencilFunc(GL_NOTEQUAL, ref, ST_SOLID | ST_TRANS | ST_ID);
        p_glDrawArrays(prim, b->first, b->count);
        p_glUniform1i(loc_part, 0);
    }
    p_glDisable(GL_STENCIL_TEST);
    p_glStencilMask(0xFF);
    p_glDisablei(GL_BLEND, 0);

    p_glDisable(GL_DEPTH_TEST);
    p_glDepthMask(GL_TRUE);
    mask2(1);
    if (rt) rt = rt_passes(fb, f, nb);

    /* edge marking + fog -> post */
    p_glBindFramebuffer(GL_FRAMEBUFFER, fb->post_fbo);
    p_glViewport(0, 0, g_fb3_w, g_fb_h);
    p_glUseProgram(prog_post);
    p_glActiveTexture(GL_TEXTURE0); p_glBindTexture(GL_TEXTURE_2D, fb->color); p_glBindSampler(0, 0);
    p_glActiveTexture(GL_TEXTURE1); p_glBindTexture(GL_TEXTURE_2D, fb->attr);
    p_glActiveTexture(GL_TEXTURE2); p_glBindTexture(GL_TEXTURE_2D, fb->depth);
    p_glUniform1i(p_glGetUniformLocation(prog_post, "u_color"), 0);
    p_glUniform1i(p_glGetUniformLocation(prog_post, "u_attr"), 1);
    p_glUniform1i(p_glGetUniformLocation(prog_post, "u_depth"), 2);
    p_glUniform1i(p_glGetUniformLocation(prog_post, "u_edge"), (disp3d >> 5) & 1);
    /* the DS's half-and-half outlines (anti-aliasing) only at 1x: at higher resolutions
     * they turn into a see-through band around characters */
    { static int eb = -2; if (eb == -2) eb = getenv("DQIX_EDGEBLEND") ? atoi(getenv("DQIX_EDGEBLEND")) : -1;
      p_glUniform1i(p_glGetUniformLocation(prog_post, "u_aa"), (eb < 0 ? g_scale == 1 : eb) && ((disp3d >> 4) & 1)); }
    p_glUniform1i(p_glGetUniformLocation(prog_post, "u_scale"), g_scale);
    float ec[24];
    for (int i = 0; i < 8; i++) { uint16_t c = IO16_(0x330 + i * 2); ec[i * 3] = (float)(c & 31) / 31.0f; ec[i * 3 + 1] = (float)((c >> 5) & 31) / 31.0f; ec[i * 3 + 2] = (float)((c >> 10) & 31) / 31.0f; }
    p_glUniform3fv(p_glGetUniformLocation(prog_post, "u_edgecol"), 8, ec);
    p_glUniform1i(p_glGetUniformLocation(prog_post, "u_fog"), (disp3d >> 7) & 1);
    p_glUniform1i(p_glGetUniformLocation(prog_post, "u_fog_alpha_only"), (disp3d >> 6) & 1);
    uint16_t fcol = IO16_(0x358);
    p_glUniform4f(p_glGetUniformLocation(prog_post, "u_fogcol"), (float)(fcol & 31) / 31.0f, (float)((fcol >> 5) & 31) / 31.0f,
                  (float)((fcol >> 10) & 31) / 31.0f, (float)(g_io[0x35A] & 31) / 31.0f);
    float ft[32]; for (int i = 0; i < 32; i++) ft[i] = (float)(g_io[0x360 + i] & 127);
    p_glUniform1fv(p_glGetUniformLocation(prog_post, "u_fogtab"), 32, ft);
    p_glUniform1i(p_glGetUniformLocation(prog_post, "u_fogoff"), IO16_(0x35C) & 0x7FFF);
    p_glUniform1i(p_glGetUniformLocation(prog_post, "u_fogshift"), (disp3d >> 8) & 15);
    p_glUniform1i(p_glGetUniformLocation(prog_post, "u_wbuf"), f->wbuf);
    {   static int dbg = -1; if (dbg < 0) dbg = getenv("DQIX_RT_DEBUG") != NULL;   /* tests: the shade alone */
        p_glUniform1i(p_glGetUniformLocation(prog_post, "u_rt"), rt ? 1 + dbg : 0); }
    if (rt) {
        p_glActiveTexture(GL_TEXTURE3); p_glBindTexture(GL_TEXTURE_2D, rt_shade);
        p_glUniform1i(p_glGetUniformLocation(prog_post, "u_shade"), 3);
    }
    p_glUniform4f(p_glGetUniformLocation(prog_post, "u_rect"), -1, -1, 2, 2);
    p_glUniform4f(p_glGetUniformLocation(prog_post, "u_uv"), 0, 0, 1, 1);
    p_glBindVertexArray(vao_quad);
    p_glDrawArrays(GL_TRIANGLES, 0, 6);
    fb->hd2d_valid = 0;
    if (hd2d) { hd2d_passes(fb); fb->hd2d_valid = 1; }
    p_glActiveTexture(GL_TEXTURE0);
    p_glBindFramebuffer(GL_FRAMEBUFFER, 0);
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

static void upload_rgb(GLuint tex, const uint32_t* src) {
    static uint8_t buf[256 * 192 * 4];
    for (int i = 0; i < 256 * 192; i++) { uint32_t c = src[i]; buf[i * 4] = (uint8_t)(c >> 16); buf[i * 4 + 1] = (uint8_t)(c >> 8); buf[i * 4 + 2] = (uint8_t)c; buf[i * 4 + 3] = (uint8_t)(c >> 24); }
    p_glBindTexture(GL_TEXTURE_2D, tex);
    p_glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 256, 192, GL_RGBA, GL_UNSIGNED_BYTE, buf);
}

/* Build both screens into final_tex (top screen in the upper half). */
void gl_compose(int a_on_top, int lcd_on) {
    upload_rgb(tex_col[0], g_engine_out[0]);
    upload_rgb(tex_col[1], g_engine_out[1]);
    int has3d = g_comp_has3d && g_fb_shown >= 0 && g_fb[g_fb_shown].valid;
    if (has3d) {
        upload_rgb(tex_c2, g_comp_c2);
        p_glBindTexture(GL_TEXTURE_2D, tex_info);
        p_glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 256, 192, GL_RGBA_INTEGER, GL_UNSIGNED_BYTE, g_comp_info);
    }
    p_glBindFramebuffer(GL_FRAMEBUFFER, final_fbo);
    p_glViewport(0, 0, g_fb3_w, g_fb_h * 2);
    p_glDisable(GL_DEPTH_TEST); p_glDisable(GL_BLEND);
    p_glClearColor(0, 0, 0, 1); p_glClear(GL_COLOR_BUFFER_BIT);
    g_wide_shown = -1;
    if (!lcd_on) { p_glBindFramebuffer(GL_FRAMEBUFFER, 0); return; }
    float W = (float)(256 + 2 * g_wside);
    p_glUseProgram(prog_comp);
    p_glUniform1i(p_glGetUniformLocation(prog_comp, "u_col"), 0);
    p_glUniform1i(p_glGetUniformLocation(prog_comp, "u_c2"), 1);
    p_glUniform1i(p_glGetUniformLocation(prog_comp, "u_info"), 2);
    p_glUniform1i(p_glGetUniformLocation(prog_comp, "u_3d"), 3);
    GLint loc_rect = p_glGetUniformLocation(prog_comp, "u_rect"), loc_has3d = p_glGetUniformLocation(prog_comp, "u_has3d");
    GLint loc_wide = p_glGetUniformLocation(prog_comp, "u_wide");
    p_glUniform1f(p_glGetUniformLocation(prog_comp, "u_wtot"), W);
    p_glUniform1f(p_glGetUniformLocation(prog_comp, "u_wside"), (float)g_wside);
    p_glUniform4f(p_glGetUniformLocation(prog_comp, "u_uv"), 0, 0, 1, 1);
    p_glBindVertexArray(vao_quad);
    for (int eng = 0; eng < 2; eng++) {
        int top = eng == 0 ? a_on_top : !a_on_top;
        p_glActiveTexture(GL_TEXTURE0); p_glBindTexture(GL_TEXTURE_2D, tex_col[eng]);
        p_glActiveTexture(GL_TEXTURE1); p_glBindTexture(GL_TEXTURE_2D, tex_c2);
        p_glActiveTexture(GL_TEXTURE2); p_glBindTexture(GL_TEXTURE_2D, tex_info);
        const Fb3D* sf = has3d ? &g_fb[g_fb_shown] : NULL;
        p_glActiveTexture(GL_TEXTURE3); p_glBindTexture(GL_TEXTURE_2D, !sf ? white_tex : sf->hd2d_valid ? sf->hd2d : sf->post);
        p_glUniform1i(loc_has3d, eng == 0 && has3d);
        int wide = eng == 0 && has3d && g_wside > 0;                 /* the 3D screen, widened; the other in the middle */
        p_glUniform1i(loc_wide, wide);
        if (wide) g_wide_shown = top ? 0 : 1;
        if (wide) p_glUniform4f(loc_rect, -1.0f, top ? 0.0f : -1.0f, 2.0f, 1.0f);   /* top screen = upper half of NDC */
        else p_glUniform4f(loc_rect, -1.0f + 2.0f * (float)g_wside / W, top ? 0.0f : -1.0f, 2.0f * 256.0f / W, 1.0f);
        p_glDrawArrays(GL_TRIANGLES, 0, 6);
    }
    p_glActiveTexture(GL_TEXTURE0);
    p_glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

/* Motion blur buffer: mix this frame into what the LCD shows (linear light). */
static GLuint ghost_step(float keep) {
    if (ghost_w != g_fb3_w || ghost_h != g_fb_h * 2) {
        for (int i = 0; i < 2; i++) {
            if (ghost_tex[i]) { p_glDeleteTextures(1, &ghost_tex[i]); p_glDeleteFramebuffers(1, &ghost_fbo[i]); }
            ghost_tex[i] = new_tex(GL_RGBA16F, g_fb3_w, g_fb_h * 2, GL_RGBA, GL_FLOAT, GL_LINEAR);
            p_glGenFramebuffers(1, &ghost_fbo[i]);
            p_glBindFramebuffer(GL_FRAMEBUFFER, ghost_fbo[i]);
            p_glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, ghost_tex[i], 0);
        }
        ghost_w = g_fb3_w; ghost_h = g_fb_h * 2; ghost_cur = -1;
    }
    int prev = ghost_cur, cur = ghost_cur < 0 ? 0 : ghost_cur ^ 1;
    p_glBindFramebuffer(GL_FRAMEBUFFER, ghost_fbo[cur]);
    p_glViewport(0, 0, ghost_w, ghost_h);
    p_glDisable(GL_DEPTH_TEST); p_glDisable(GL_BLEND);
    p_glUseProgram(prog_ghost);
    p_glActiveTexture(GL_TEXTURE0); p_glBindTexture(GL_TEXTURE_2D, final_tex);
    p_glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST); p_glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    p_glActiveTexture(GL_TEXTURE1); p_glBindTexture(GL_TEXTURE_2D, prev < 0 ? final_tex : ghost_tex[prev]);
    p_glUniform1i(p_glGetUniformLocation(prog_ghost, "u_cur"), 0);
    p_glUniform1i(p_glGetUniformLocation(prog_ghost, "u_prev"), 1);
    p_glUniform1f(p_glGetUniformLocation(prog_ghost, "u_keep"), prev < 0 ? 0.0f : keep);
    p_glUniform4f(p_glGetUniformLocation(prog_ghost, "u_rect"), -1, -1, 2, 2);
    p_glUniform4f(p_glGetUniformLocation(prog_ghost, "u_uv"), 0, 0, 1, 1);
    p_glBindVertexArray(vao_quad);
    p_glDrawArrays(GL_TRIANGLES, 0, 6);
    p_glActiveTexture(GL_TEXTURE0);
    ghost_cur = cur;
    return ghost_tex[cur];
}

/* Draw the screens (and the console, if chosen) into the window. Call once
 * per emulated frame: the motion blur advances with every call. */
void gl_present(const GlPresent* v) {
    int ghost = v->lcd && v->ghost;
    GLuint src = final_tex;
    if (ghost && v->again && ghost_cur >= 0) src = ghost_tex[ghost_cur];   /* a second window: this frame's blur as it is */
    else if (ghost) src = ghost_step(0.35f);
    else if (!v->again) ghost_cur = -1;
    p_glBindFramebuffer(GL_FRAMEBUFFER, 0);
    p_glViewport(0, 0, v->win_w, v->win_h);
    p_glDisable(GL_DEPTH_TEST); p_glDisable(GL_BLEND);
    p_glUseProgram(prog_screen);
    p_glActiveTexture(GL_TEXTURE0); p_glBindTexture(GL_TEXTURE_2D, src); p_glBindSampler(0, 0);
    /* Shown smaller than it was rendered (3D resolution above the window size):
     * average with mipmaps so the extra detail becomes smooth edges, not shimmer.
     * Shown larger: linear, and the shader keeps the pixels sharp (sharp
     * bilinear: every pixel the same size even at an uneven scale, where
     * nearest made some rows and columns wider than others). */
    float px_top = v->sc * v->top[2] / (v->topx[1] * (float)g_scale), px_bot = v->sc * v->bot[2] / (v->botx[1] * (float)g_scale);
    float px_per_texel = px_top > px_bot ? px_top : px_bot;                 /* the bigger screen */
    int minify = px_per_texel < 0.99f;
    if (minify) p_glGenerateMipmap(GL_TEXTURE_2D);
    p_glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, minify ? GL_LINEAR_MIPMAP_LINEAR : GL_LINEAR);   /* the shader keeps pixels sharp */
    p_glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    GLuint P = prog_screen;
    p_glUniform1i(p_glGetUniformLocation(P, "u_src"), 0);
    p_glUniform1i(p_glGetUniformLocation(P, "u_src_linear"), ghost);
    p_glUniform2f(p_glGetUniformLocation(P, "u_win"), (float)v->win_w, (float)v->win_h);
    p_glUniform3fv(p_glGetUniformLocation(P, "u_view"), 1, (const float[3]){ v->ox, v->oy, v->sc });
    p_glUniform2f(p_glGetUniformLocation(P, "u_canvas"), v->cw, v->ch);
    p_glUniform4f(p_glGetUniformLocation(P, "u_top"), v->top[0], v->top[1], v->top[2], v->top[3]);
    p_glUniform4f(p_glGetUniformLocation(P, "u_bot"), v->bot[0], v->bot[1], v->bot[2], v->bot[3]);
    p_glUniform2f(p_glGetUniformLocation(P, "u_topx"), v->topx[0], v->topx[1]);
    p_glUniform2f(p_glGetUniformLocation(P, "u_botx"), v->botx[0], v->botx[1]);
    p_glUniform1f(p_glGetUniformLocation(P, "u_texw"), (float)(256 + 2 * g_wside));
    p_glUniform1i(p_glGetUniformLocation(P, "u_front"), v->front);
    p_glUniform1f(p_glGetUniformLocation(P, "u_alpha"), v->alpha > 0 ? v->alpha : 1.0f);
    {   /* the console's windows: upper and lower, wherever each screen is shown */
        const float* up = v->top[1] <= v->bot[1] ? v->top : v->bot, *lo = up == v->top ? v->bot : v->top;
        p_glUniform4f(p_glGetUniformLocation(P, "u_ctop"), up[0], up[1], up[2], up[3]);
        p_glUniform4f(p_glGetUniformLocation(P, "u_cbot"), lo[0], lo[1], lo[2], lo[3]);
    }
    p_glUniform1i(p_glGetUniformLocation(P, "u_lcd"), v->lcd);
    p_glUniform1i(p_glGetUniformLocation(P, "u_lcdcolor"), v->lcd_color);
    p_glUniform1i(p_glGetUniformLocation(P, "u_grid"), v->grid);
    p_glUniform1f(p_glGetUniformLocation(P, "u_backlight"), v->backlight);
    p_glUniform1i(p_glGetUniformLocation(P, "u_shell"), v->shell);
    p_glUniform3fv(p_glGetUniformLocation(P, "u_lid"), 1, v->lid);
    p_glUniform3fv(p_glGetUniformLocation(P, "u_base"), 1, v->base);
    p_glUniform3fv(p_glGetUniformLocation(P, "u_btn"), 1, v->btn);
    p_glUniform3fv(p_glGetUniformLocation(P, "u_hinge"), 1, v->hinge);
    int skin = v->shell && v->skin && skin_tex;
    p_glUniform1i(p_glGetUniformLocation(P, "u_skinmode"), skin);
    p_glUniform1i(p_glGetUniformLocation(P, "u_skintex"), 1);
    if (skin) {
        p_glActiveTexture(GL_TEXTURE1); p_glBindTexture(GL_TEXTURE_2D, skin_tex);
        p_glUniform2f(p_glGetUniformLocation(P, "u_skinscale"), v->skin_k / (float)skin_w, v->skin_k / (float)skin_h);
        p_glActiveTexture(GL_TEXTURE0);
    }
    p_glUniform4f(p_glGetUniformLocation(P, "u_rect"), -1, -1, 2, 2);
    p_glUniform4f(p_glGetUniformLocation(P, "u_uv"), 0, 0, 1, 1);
    p_glBindVertexArray(vao_quad);
    p_glDrawArrays(GL_TRIANGLES, 0, 6);
}

/* A console photo (RGBA, rows top-down), drawn around the screens. */
void gl_set_skin(const uint8_t* rgba, int w, int h) {
    if (!skin_tex) p_glGenTextures(1, &skin_tex);
    p_glBindTexture(GL_TEXTURE_2D, skin_tex);
    p_glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    p_glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, rgba);
    p_glGenerateMipmap(GL_TEXTURE_2D);
    p_glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    p_glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    p_glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    p_glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    skin_w = w; skin_h = h;
}

/* The window as shown (for tests): top-down RGB, caller frees. */
uint8_t* gl_read_window(int w, int h) {
    uint8_t* buf = malloc((size_t)w * h * 4), *rgb = malloc((size_t)w * h * 3);
    p_glBindFramebuffer(GL_FRAMEBUFFER, 0);
    p_glReadBuffer(GL_BACK);
    p_glReadPixels(0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, buf);
    for (int y = 0; y < h; y++) for (int x = 0; x < w; x++) {
        const uint8_t* q = buf + ((size_t)(h - 1 - y) * w + x) * 4; uint8_t* o = rgb + ((size_t)y * w + x) * 3;
        o[0] = q[0]; o[1] = q[1]; o[2] = q[2];
    }
    free(buf);
    return rgb;
}

/* For screenshots/reports: read the composed screens at 256x384 (nearest). */
/* Both screens at the rendering resolution, top-down RGB (caller frees). */
uint8_t* gl_read_full(int* out_w, int* out_h) {
    int w = g_fb3_w, h = g_fb_h * 2;
    uint8_t* buf = malloc((size_t)w * h * 4);
    uint8_t* rgb = malloc((size_t)w * h * 3);
    p_glBindFramebuffer(GL_FRAMEBUFFER, final_fbo);
    p_glReadBuffer(GL_COLOR_ATTACHMENT0);
    p_glReadPixels(0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, buf);
    p_glBindFramebuffer(GL_FRAMEBUFFER, 0);
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

void gl_read_screens(uint32_t out[2][256 * 192]) {
    int w = g_fb3_w, h = g_fb_h * 2;
    uint8_t* buf = malloc((size_t)w * h * 4);
    p_glBindFramebuffer(GL_FRAMEBUFFER, final_fbo);
    p_glReadBuffer(GL_COLOR_ATTACHMENT0);
    p_glReadPixels(0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, buf);
    p_glBindFramebuffer(GL_FRAMEBUFFER, 0);
    for (int sc = 0; sc < 2; sc++)
        for (int y = 0; y < 192; y++)
            for (int x = 0; x < 256; x++) {
                int gy = h - 1 - ((sc * 192 + y) * g_scale + g_scale / 2), gx = (x + g_wside) * g_scale + g_scale / 2;   /* the DS's picture */
                const uint8_t* p = buf + ((size_t)gy * w + gx) * 4;
                out[sc][y * 256 + x] = (uint32_t)p[0] << 16 | (uint32_t)p[1] << 8 | p[2];
            }
    free(buf);
}

/* For tests: the 3D layer as rendered (premultiplied RGBA8, bottom-up rows). */
int gl_read_3d(uint8_t* out, int* w, int* h) {
    if (g_fb_latest < 0) return 0;
    *w = g_fb3_w; *h = g_fb_h;
    p_glBindFramebuffer(GL_FRAMEBUFFER, g_fb[g_fb_latest].post_fbo);
    p_glReadBuffer(GL_COLOR_ATTACHMENT0);
    p_glReadPixels(0, 0, g_fb3_w, g_fb_h, GL_RGBA, GL_UNSIGNED_BYTE, out);
    p_glBindFramebuffer(GL_FRAMEBUFFER, 0);
    return 1;
}

/* Widescreen: which screen was composed wide last (0 top, 1 bottom, -1
 * none) and the DS pixels added each side. */
int gl_wide_shown(int* side) { *side = g_wside; return g_wide_shown; }

/* The extra looks (gl_render.h): what this driver can do, and switching them. */
int gl_effects(void) { return (hd2d_ok ? GL_EFFECT_HD2D : 0) | (rt_ok ? GL_EFFECT_RT : 0); }
void gl_set_hd2d(int on) { g_hd2d = on != 0; }
void gl_set_rt(int on) { g_rt = on != 0; }
void gl_set_look(const GlLook* l) {
    int res = l->rt_res != g_look.rt_res;
    g_look = *l;
    if (res && g_ready && fx_on) { make_rt_tex(); }             /* a new ray resolution: its textures */
}

/* ---------------------------------------------------------------- windows */
/* One context for every window: the first window's. */
static SDL_GLContext g_ctx;
static SDL_Window* g_ctx_win;

uint32_t gfx_window_flags(void) {
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    return SDL_WINDOW_OPENGL;
}

int gfx_attach(SDL_Window* win) {
    if (!g_ctx) {
        g_ctx = SDL_GL_CreateContext(win);
        if (!g_ctx) { fprintf(stderr, "OpenGL 3.3 not available (%s): using the software renderer\n", SDL_GetError()); return 0; }
        g_ctx_win = win;
        SDL_GL_SetSwapInterval(0);                               /* pacing follows the audio clock */
        return 1;
    }
    if (SDL_GL_MakeCurrent(win, g_ctx) == 0) SDL_GL_SetSwapInterval(0);
    SDL_GL_MakeCurrent(g_ctx_win, g_ctx);
    return 1;
}

void gfx_detach(SDL_Window* win) {
    if (!g_ctx) return;
    if (win == g_ctx_win) { SDL_GL_DeleteContext(g_ctx); g_ctx = NULL; g_ctx_win = NULL; g_ready = 0; fx_on = 0; return; }
    SDL_GL_MakeCurrent(g_ctx_win, g_ctx);
}

void gfx_drawable_size(SDL_Window* win, int* w, int* h) { SDL_GL_GetDrawableSize(win, w, h); }
void gfx_begin(SDL_Window* win) { if (g_ctx) SDL_GL_MakeCurrent(win, g_ctx); }
void gfx_swap(SDL_Window* win) {
    SDL_GL_SwapWindow(win);
    if (win != g_ctx_win && g_ctx) SDL_GL_MakeCurrent(g_ctx_win, g_ctx);
}
