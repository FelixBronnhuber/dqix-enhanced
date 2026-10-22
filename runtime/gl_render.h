/* GPU renderer interface: OpenGL (gl_render.c), or Metal on macOS
 * (metal_render.m). The SDL frontend opens its windows through the gfx_*
 * functions, then calls the others. */
#pragma once
#include <stdint.h>
#include "SDL.h"
#include "gpu3d.h"

/* The windows: SDL_CreateWindow flags (call before creating one; GL sets its
 * context attributes here), then gfx_attach() for each window (the first one
 * creates the context / device; 0 = not available). gfx_begin() picks the
 * window gl_present() draws into; gfx_swap() shows it. */
uint32_t gfx_window_flags(void);
int gfx_attach(SDL_Window* win);
void gfx_detach(SDL_Window* win);
void gfx_drawable_size(SDL_Window* win, int* w, int* h);
void gfx_begin(SDL_Window* win);
void gfx_swap(SDL_Window* win);

int gl_render_init(int scale);                        /* 0 on failure (use the software path) */
void gl_render_set_scale(int scale);
#define GL_MAX_SCALE 12                               /* the highest 3D resolution offered */
int gl_render_max_scale(void);                        /* ...that this GPU can hold */
void gl_set_smooth(int on);                           /* smooth textures (filtered) instead of the DS's blocks */
const Renderer3D* gl_renderer3d(void);
void gl_compose(int a_on_top, int lcd_on);           /* at VBlank: 2D + 3D -> both screens */
/* How to show the screens: canvas (DS pixels) placed in the window at
 * (ox, oy) with sc window pixels per DS pixel. */
typedef struct {
    int win_w, win_h;
    float ox, oy, sc, cw, ch;
    float top[4], bot[4];                             /* screen rects on the canvas: x, y, w, h */
    float topx[2], botx[2];                           /* each screen in the composed picture: x offset, width (DS pixels) */
    int front;                                        /* screens overlapping: 1 = the bottom one in front */
    float alpha;                                      /* ...how solid the one in front is (0 or 1 = solid) */
    int again;                                        /* another window, same frame: don't advance the motion blur */
    int lcd, lcd_color, grid, ghost;                  /* DS Lite screen filter and its parts */
    float backlight;                                  /* 0..1 */
    int shell;                                        /* draw the console around the screens */
    float lid[3], base[3], btn[3], hinge[3];          /* its colours (linear 0..1, as shown) */
    int skin;                                         /* draw the console photo (gl_set_skin) instead */
    float skin_k;                                     /* photo pixels per canvas unit */
} GlPresent;
void gl_set_skin(const uint8_t* rgba, int w, int h);
void gl_present(const GlPresent* v);
uint8_t* gl_read_window(int w, int h);
int gl_wide_shown(int* side);                         /* widescreen: the screen composed wide (0 top, 1 bottom, -1 none) */
uint8_t* gl_read_full(int* w, int* h);
void gl_read_screens(uint32_t out[2][256 * 192]);
int gl_read_3d(uint8_t* out, int* w, int* h);

/* Per-frame counters (the frontend reads and clears them). */
typedef struct { double render_ms, upload_ms; int uploads, polys; long upload_texels; } GlStats;
extern GlStats g_gl_stats;
const char* gl_renderer_name(void);

/* Extra looks beyond the DS's (Metal only for now): which this renderer has, and switching them. */
#define GL_EFFECT_HD2D 1                                      /* depth of field, bloom, a warm grade */
int gl_effects(void);
void gl_set_hd2d(int on);
int gl_render_scale(void);
