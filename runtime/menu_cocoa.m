/* Menu bar and windows for the game on macOS (see menu.h): the same menus
 * and dialogs as menu_win32.c, in Cocoa. macOS has one menu bar for the whole
 * program, so the game's menus go into it, after the program menu SDL makes
 * (About, Hide, Quit). A chosen item comes back to the game as an SDL event.
 * Items have no key equivalents: every key goes to the game, as on Windows;
 * the keys are shown in the items' names instead. */
#import <Cocoa/Cocoa.h>
#include "menu.h"
#include "controls.h"
#include <string.h>
#include <stdlib.h>

static Uint32 g_evt = (Uint32)-1;                              /* the SDL event a menu item sends */
static NSMenu* g_bar;
static NSMutableDictionary<NSNumber*, NSNumber*>* g_checked, *g_enabled;   /* kept for items made later */

static NSString* S(const char* utf8) { return [NSString stringWithUTF8String:utf8 ? utf8 : ""] ?: @""; }

@interface DqixTarget : NSObject
- (void)chosen:(NSMenuItem*)item;
@end
@implementation DqixTarget
- (void)chosen:(NSMenuItem*)item {
    if (g_evt == (Uint32)-1) return;
    SDL_Event e; memset(&e, 0, sizeof e);
    e.type = g_evt; e.user.code = (Sint32)item.tag;
    SDL_PushEvent(&e);
}
@end
static DqixTarget* g_target;

/* "&Pause\tP" -> "Pause (P)": no mnemonics, the key in brackets */
static NSString* label(const char* text) {
    NSMutableString* s = [NSMutableString string];
    NSString* t = S(text);
    NSRange tab = [t rangeOfString:@"\t"];
    NSString* name = tab.location == NSNotFound ? t : [t substringToIndex:tab.location];
    [s appendString:[name stringByReplacingOccurrencesOfString:@"&" withString:@""]];
    if (tab.location != NSNotFound) [s appendFormat:@"   (%@)", [t substringFromIndex:tab.location + 1]];
    return s;
}
static void item(NSMenu* m, int id, const char* text) {
    NSMenuItem* i = [[NSMenuItem alloc] initWithTitle:label(text) action:@selector(chosen:) keyEquivalent:@""];
    i.target = g_target; i.tag = id;
    NSNumber* c = g_checked[@(id)]; if (c) i.state = c.boolValue ? NSControlStateValueOn : NSControlStateValueOff;
    NSNumber* en = g_enabled[@(id)]; if (en) i.enabled = en.boolValue;
    [m addItem:i];
}
static void sep(NSMenu* m) { [m addItem:[NSMenuItem separatorItem]]; }
static NSMenu* sub(NSMenu* parent, const char* text) {
    NSMenu* m = [[NSMenu alloc] initWithTitle:label(text)];
    m.autoenablesItems = NO;
    NSMenuItem* i = [[NSMenuItem alloc] initWithTitle:label(text) action:NULL keyEquivalent:@""];
    i.submenu = m;
    [parent addItem:i];
    return m;
}
/* a top-level menu, before SDL's Window menu */
static NSMenu* top(const char* text) {
    NSMenu* bar = NSApp.mainMenu;
    NSMenu* m = [[NSMenu alloc] initWithTitle:label(text)];
    m.autoenablesItems = NO;
    NSMenuItem* i = [[NSMenuItem alloc] initWithTitle:label(text) action:NULL keyEquivalent:@""];
    i.submenu = m;
    NSInteger at = bar.numberOfItems;
    for (NSInteger k = 1; k < bar.numberOfItems; k++) if ([bar itemAtIndex:k].submenu == NSApp.windowsMenu) { at = k; break; }
    [bar insertItem:i atIndex:at];
    return m;
}

void menu_init(SDL_Window* win, const char* const* shell_names, int n_shells) {
    (void)win;
    if (g_bar || !NSApp.mainMenu) return;
    @autoreleasepool {
    if (g_evt == (Uint32)-1) g_evt = SDL_RegisterEvents(1);
    if (!g_target) g_target = [DqixTarget new];
    if (!g_checked) { g_checked = [NSMutableDictionary dictionary]; g_enabled = [NSMutableDictionary dictionary]; }
    g_bar = NSApp.mainMenu;

    NSMenu* game = top("&Game");
    item(game, MI_PAUSE, "&Pause\tP");
    item(game, MI_TURBO, "&Fast forward\tor hold Tab");
    item(game, MI_RESET, "&Reset...\tCtrl+R");
    sep(game);
    item(game, MI_VOLUME, "&Volume...\t- / =");
    item(game, MI_MUTE, "&Mute");
    item(game, MI_CONTROLS_EDIT, "&Controls...");
    sep(game);
    item(game, MI_SHOT, "Save &screenshot\tF12");
    item(game, MI_REPORT, "Save &bug report\tF10");
    sep(game);
    item(game, MI_EXIT, "E&xit\tEsc");

    NSMenu* gfx = top("G&raphics");
    item(gfx, MI_CLASSIC, "&Classic: exactly like the DS\tF5");
    item(gfx, MI_ENHANCED, "&Enhanced: sharper 3D\tF5");
    sep(gfx);
    NSMenu* res = sub(gfx, "3D &resolution (Enhanced)\tF4");
    item(res, MI_RES1, "1x (DS)");
    item(res, MI_RES2, "2x");
    item(res, MI_RES3, "3x");
    item(res, MI_RES4, "4x");
    item(res, MI_RES6, "6x");
    item(res, MI_RES8, "8x");
    sep(res);
    item(res, MI_RES_AUTO, "&Auto: as sharp as your screen (up to 12x)");
    item(gfx, MI_SMOOTH, "&Smooth textures (Enhanced)");
    sep(gfx);
    item(gfx, MI_HD2D, "&HD-2D look: depth of field, glow, warm colours (Enhanced)");
    item(gfx, MI_RT, "&Ray-traced shadows and shading (Enhanced)");

    NSMenu* scr = top("&Screen");
    item(scr, MI_LCD, "&DS Lite screen\tF6");
    sep(scr);
    item(scr, MI_LCDCOL, "LCD &colours");
    item(scr, MI_GRID, "&Pixel grid (at 3x and larger)");
    item(scr, MI_GHOST, "&Motion blur");
    NSMenu* bl = sub(scr, "&Backlight");
    item(bl, MI_BL1, "&Low");
    item(bl, MI_BL2, "&Medium");
    item(bl, MI_BL3, "&High");
    item(bl, MI_BL4, "Ma&x");

    NSMenu* view = top("&View");
    item(view, MI_SHELL, "Show the DS &Lite\tF7");
    NSMenu* col = sub(view, "Console &colour");
    for (int i = 0; i < n_shells; i++) item(col, MI_SHELLC0 + i, shell_names[i]);
    item(view, MI_GAP, "&Gap between the screens");
    item(view, MI_ACTUAL, "&Actual size (as big as a real DS Lite)\tF8");
    sep(view);
    item(view, MI_STACKED, "S&tacked\tF2");
    item(view, MI_SIDE, "&Side by side\tF2");
    item(view, MI_CORNER, "Top screen in the &corner\tF2");
    item(view, MI_WIDE, "&Widescreen (the action wide, the other screen in a corner)\tF2");
    item(view, MI_WIDE_DYN, "Widescreen, d&ynamic (experimental: makes room for menus)\tF2");
    item(view, MI_TWO, "Two wi&ndows (one screen in each, for two monitors)\tF2");
    sep(view);
    item(view, MI_SINGLE, "Single scr&een (widescreen: hide the small screen)\tU");
    item(view, MI_SWAP, "Swa&p the screens (widescreen: hold to see the other)\tF9");
    NSMenu* sm = sub(view, "Widescreen: the small scr&een");
    item(sm, MI_CORNER0, "Top left\tK");
    item(sm, MI_CORNER0 + 1, "Top right\tK");
    item(sm, MI_CORNER0 + 2, "Bottom right\tK");
    item(sm, MI_CORNER0 + 3, "Bottom left\tK");
    sep(sm);
    item(sm, MI_SSIZE0, "Small\tI");
    item(sm, MI_SSIZE0 + 1, "Medium\tI");
    item(sm, MI_SSIZE0 + 2, "Large\tI");
    sep(sm);
    item(sm, MI_OPACITY1, "100% (solid)\tO");
    item(sm, MI_OPACITY1 + 1, "75%\tO");
    item(sm, MI_OPACITY1 + 2, "50%\tO");
    item(sm, MI_OPACITY1 + 3, "25%\tO");
    sep(sm);
    item(sm, MI_POPOUT, "In its own &window (a clean field)");
    sep(view);
    NSMenu* size = sub(view, "&Window size\tF3");
    item(size, MI_SIZE1, "1x");
    item(size, MI_SIZE2, "2x");
    item(size, MI_SIZE3, "3x");
    item(size, MI_SIZE4, "4x");
    item(view, MI_INTEGER, "&Integer scaling");
    item(view, MI_FULL, "&Fullscreen\tF11");
    item(view, MI_FULL2, "Fullscreen: the &second window");

    NSMenu* wl = top("&Wireless");
    item(wl, MI_NET, "&Connect (Nintendo Wi-Fi Connection replacement)");
    NSMenu* srv = sub(wl, "&Server");
    item(srv, MI_SRV_BUILTIN, "&Built-in (offline: DQIX's preserved downloads)");
    sep(srv);
    item(srv, MI_SRV_ALTWFC, "&AltWFC (172.104.88.237)");
    item(srv, MI_SRV_WIIMMFI, "&Wiimmfi (164.132.44.106): not for DQIX");
    item(srv, MI_SRV_CUSTOM, "&Custom (wfc_server in settings.ini)");
    sep(wl);
    NSMenu* friends = sub(wl, "&Play with friends far away (tag mode, visiting)");
    item(friends, MI_SES_HOST, "&Host: your friends join you...");
    item(friends, MI_SES_JOIN, "&Join a friend...");
    item(friends, MI_SES_START, "&Start now (everyone is here)\tEnter");
    item(friends, MI_SES_LEAVE, "&Leave and play alone");
    sep(friends);
    item(friends, MI_SES_NAME, "Your &name...");
    item(friends, MI_SES_HELP, "&How does this work?");
    sep(wl);
    item(wl, MI_DLC, "&Unlock all downloadable content now (without Sellma)");
    sep(wl);
    item(wl, MI_NET_HELP, "&How does this work?");

    NSMenu* modern = top("&Modern");
    item(modern, MI_ANALOG, "Walk in any &direction with the left stick");
    item(modern, MI_WIDE, "&Widescreen: the action wide, the other screen in a corner");
    item(modern, MI_WIDE_DYN, "Widescreen, d&ynamic (experimental)");
    item(modern, MI_CORNER, "Top screen in the &corner (the field as on the DS)");
    item(modern, MI_POPOUT, "Widescreen: the small screen in its own &window (a clean field)");
    sep(modern);
    item(modern, MI_MODERN_HELP, "&What is this?");

    /* Help: SDL makes none; macOS gives the one named Help a search field */
    NSMenu* help = [[NSMenu alloc] initWithTitle:@"Help"];
    help.autoenablesItems = NO;
    NSMenuItem* hi = [[NSMenuItem alloc] initWithTitle:@"Help" action:NULL keyEquivalent:@""];
    hi.submenu = help;
    [g_bar addItem:hi];
    NSApp.helpMenu = help;
    item(help, MI_CONTROLS, "&Controls");
    item(help, MI_UPDATE, "Check for &updates...");
    item(help, MI_UPDATE_AUTO, "Check for updates at &start");
    sep(help);
    item(help, MI_SEND_REPORT, "&Send a bug report...");
    item(help, MI_ABOUT, "&About");
    menu_enable(MI_FULL2, 0);
    }
}

void menu_show(int on) { (void)on; }                           /* macOS hides the bar itself in fullscreen */

/* The second window (two windows): its Fullscreen is in the View menu. */
void menu_init_second(SDL_Window* win) { menu_enable(MI_FULL2, win != NULL); }
void menu_show_second(int on) { (void)on; }

static void each_item(NSMenu* m, int id, void (^f)(NSMenuItem*)) {
    for (NSMenuItem* i in m.itemArray) {
        if (i.submenu) each_item(i.submenu, id, f);
        else if (i.tag == id && i.action == @selector(chosen:)) f(i);
    }
}
void menu_check(int id, int on) {
    if (!g_checked) { g_checked = [NSMutableDictionary dictionary]; g_enabled = [NSMutableDictionary dictionary]; }
    g_checked[@(id)] = @(on != 0);
    if (g_bar) each_item(g_bar, id, ^(NSMenuItem* i) { i.state = on ? NSControlStateValueOn : NSControlStateValueOff; });
}
void menu_enable(int id, int on) {
    if (!g_checked) { g_checked = [NSMutableDictionary dictionary]; g_enabled = [NSMutableDictionary dictionary]; }
    g_enabled[@(id)] = @(on != 0);
    if (g_bar) each_item(g_bar, id, ^(NSMenuItem* i) { i.enabled = on != 0; });
}

int menu_command(const SDL_Event* e) {
    if (g_evt == (Uint32)-1 || e->type != g_evt) return 0;
    return e->user.code;
}

/* The display's size in millimetres (from its EDID, as macOS reports it)
 * over its width in drawn pixels (the game draws in Retina pixels). */
float display_mm_per_pixel(SDL_Window* win) {
    int d = SDL_GetWindowDisplayIndex(win);
    if (d < 0) return 0;
    @autoreleasepool {
    NSArray<NSScreen*>* screens = NSScreen.screens;
    SDL_Rect r; if (SDL_GetDisplayBounds(d, &r) != 0) return 0;
    for (NSScreen* s in screens) {
        CGDirectDisplayID id = [s.deviceDescription[@"NSScreenNumber"] unsignedIntValue];
        CGRect b = CGDisplayBounds(id);
        if ((int)b.origin.x != r.x || (int)b.origin.y != r.y) continue;
        CGSize mm = CGDisplayScreenSize(id);
        float px = (float)(s.frame.size.width * s.backingScaleFactor);
        if (mm.width < 100 || px <= 0) return 0;
        float v = (float)mm.width / px;
        return v > 0.02f && v < 1.5f ? v : 0;          /* Retina pixels are small (about 0.1 mm) */
    }
    }
    return 0;
}

/* ---------------------------------------------------------------- text box (bug descriptions) */
char* ask_text(SDL_Window* win, const char* title, const char* prompt) {
    (void)win;
    @autoreleasepool {
    NSAlert* a = [NSAlert new];
    a.messageText = S(title);
    a.informativeText = S(prompt);
    [a addButtonWithTitle:@"Save"];
    [a addButtonWithTitle:@"Skip"];
    NSScrollView* sv = [[NSScrollView alloc] initWithFrame:NSMakeRect(0, 0, 440, 220)];
    sv.hasVerticalScroller = YES; sv.borderType = NSBezelBorder;
    NSTextView* tv = [[NSTextView alloc] initWithFrame:sv.contentView.bounds];
    tv.richText = NO; tv.automaticQuoteSubstitutionEnabled = NO; tv.automaticDashSubstitutionEnabled = NO;
    tv.font = [NSFont systemFontOfSize:NSFont.systemFontSize];
    tv.autoresizingMask = NSViewWidthSizable;
    tv.verticallyResizable = YES; tv.textContainer.widthTracksTextView = YES;
    sv.documentView = tv;
    a.accessoryView = sv;
    [a layout];
    [a.window makeFirstResponder:tv];
    if ([a runModal] != NSAlertFirstButtonReturn) return NULL;
    return strdup(tv.string.UTF8String ?: "");
    }
}

/* ---------------------------------------------------------------- a message with buttons */
int choice_box(SDL_Window* win, const char* title, const char* text, const char* const* buttons, int n, int def, int cancel,
               const char* check, int* checked) {
    (void)win;
    if (n < 1 || n > 8) return -1;
    @autoreleasepool {
    NSAlert* a = [NSAlert new];
    a.messageText = S(title);
    a.informativeText = S(text);
    /* NSAlert puts the first button on the right and makes it the default:
     * add the default first, then the others in their order */
    int order[8], k = 0;
    order[k++] = def;
    for (int i = 0; i < n; i++) if (i != def) order[k++] = i;
    for (int i = 0; i < n; i++) {
        NSButton* b = [a addButtonWithTitle:S(buttons[order[i]])];
        if (order[i] == cancel && cancel != def) b.keyEquivalent = @"\033";   /* Esc */
    }
    if (check) {
        a.showsSuppressionButton = YES;
        a.suppressionButton.title = S(check);
        a.suppressionButton.state = checked && *checked ? NSControlStateValueOn : NSControlStateValueOff;
    }
    NSModalResponse r = [a runModal];
    if (check && checked) *checked = a.suppressionButton.state == NSControlStateValueOn;
    int i = (int)(r - NSAlertFirstButtonReturn);
    return i >= 0 && i < n ? order[i] : cancel;
    }
}

/* ---------------------------------------------------------------- small windows */
/* A panel above the game's window, its controls laid out top-down in points. */
@interface DqixPanel : NSPanel
@end
@implementation DqixPanel
- (BOOL)canBecomeKeyWindow { return YES; }
@end
static NSPanel* panel(NSString* title, CGFloat w, CGFloat h, id<NSWindowDelegate> delegate) {
    NSPanel* p = [[DqixPanel alloc] initWithContentRect:NSMakeRect(0, 0, w, h)
                                              styleMask:NSWindowStyleMaskTitled | NSWindowStyleMaskClosable
                                                backing:NSBackingStoreBuffered defer:NO];
    p.title = title;
    p.releasedWhenClosed = NO;
    p.hidesOnDeactivate = NO;
    p.floatingPanel = YES;
    p.delegate = delegate;
    [p center];
    return p;
}
static NSRect at(NSPanel* p, CGFloat x, CGFloat y, CGFloat w, CGFloat h) {   /* y from the top */
    return NSMakeRect(x, p.contentView.bounds.size.height - y - h, w, h);
}
static NSTextField* text_label(NSPanel* p, NSRect r, NSString* s) {
    NSTextField* t = [NSTextField wrappingLabelWithString:s];
    t.frame = r;
    [p.contentView addSubview:t];
    return t;
}
static NSButton* button(NSPanel* p, NSRect r, NSString* title, id target, SEL action) {
    NSButton* b = [NSButton buttonWithTitle:title target:target action:action];
    b.frame = r;
    [p.contentView addSubview:b];
    return b;
}

/* ---- the session window */
static NSPanel* g_ses;
static NSTextField *g_ses_head, *g_ses_addr, *g_ses_hint, *g_ses_list, *g_ses_status;
static NSButton *g_ses_copy, *g_ses_start, *g_ses_cancel;
static int g_ses_action;
static SessionWindow g_ses_shown;

@interface DqixSession : NSObject <NSWindowDelegate>
@end
@implementation DqixSession
- (void)copyAddress:(id)sender {
    NSPasteboard* pb = NSPasteboard.generalPasteboard;
    [pb clearContents];
    [pb setString:g_ses_addr.stringValue forType:NSPasteboardTypeString];
    g_ses_copy.title = @"Copied!";
    dispatch_after(dispatch_time(DISPATCH_TIME_NOW, 2 * NSEC_PER_SEC), dispatch_get_main_queue(), ^{ g_ses_copy.title = @"Copy"; });
}
- (void)start:(id)sender { if (g_ses_start.enabled) g_ses_action = 1; }
- (void)cancel:(id)sender { g_ses_action = 2; }
- (BOOL)windowShouldClose:(NSWindow*)w { g_ses_action = 2; return NO; }
@end
static DqixSession* g_ses_target;

int session_window_open(SDL_Window* win) {
    (void)win;
    if (g_ses) return 1;
    @autoreleasepool {
    if (!g_ses_target) g_ses_target = [DqixSession new];
    NSPanel* p = panel(@"Play with friends far away", 420, 440, g_ses_target);
    g_ses_head = text_label(p, at(p, 16, 14, 388, 18), @"");
    g_ses_addr = [NSTextField textFieldWithString:@""];
    g_ses_addr.frame = at(p, 16, 38, 300, 32);
    g_ses_addr.editable = NO; g_ses_addr.selectable = YES;
    g_ses_addr.alignment = NSTextAlignmentCenter;
    g_ses_addr.font = [NSFont boldSystemFontOfSize:20];
    [p.contentView addSubview:g_ses_addr];
    g_ses_copy = button(p, at(p, 324, 38, 80, 32), @"Copy", g_ses_target, @selector(copyAddress:));
    g_ses_hint = text_label(p, at(p, 16, 80, 388, 150), @"");
    text_label(p, at(p, 16, 236, 388, 18), @"Players:");
    g_ses_list = text_label(p, at(p, 16, 256, 388, 80), @"");
    g_ses_status = text_label(p, at(p, 16, 344, 388, 50), @"");
    g_ses_start = button(p, at(p, 186, 400, 130, 28), @"Start the game", g_ses_target, @selector(start:));
    g_ses_start.keyEquivalent = @"\r";
    g_ses_cancel = button(p, at(p, 324, 400, 80, 28), @"Cancel", g_ses_target, @selector(cancel:));
    g_ses = p;
    [p makeKeyAndOrderFront:nil];
    memset(&g_ses_shown, 0, sizeof g_ses_shown);
    g_ses_shown.can_start = -1;
    g_ses_action = 0;
    }
    return 1;
}

void session_window_update(const SessionWindow* w) {
    if (!g_ses) return;
    SessionWindow* o = &g_ses_shown;
    if (strcmp(o->heading, w->heading)) g_ses_head.stringValue = S(w->heading);
    if (strcmp(o->address, w->address)) g_ses_addr.stringValue = S(w->address);
    if (strcmp(o->hint, w->hint)) g_ses_hint.stringValue = S(w->hint);
    if (strcmp(o->status, w->status)) g_ses_status.stringValue = S(w->status);
    int list_changed = o->n_players != w->n_players;
    for (int i = 0; i < w->n_players && !list_changed; i++) list_changed = strcmp(o->players[i], w->players[i]) != 0;
    if (list_changed) {
        NSMutableArray* names = [NSMutableArray array];
        for (int i = 0; i < w->n_players; i++) [names addObject:S(w->players[i])];
        g_ses_list.stringValue = [names componentsJoinedByString:@"\n"];
    }
    int start = w->host && w->can_start && !w->failed;
    if (o->can_start != start || o->host != w->host || o->failed != w->failed) {
        g_ses_start.enabled = start != 0;
        g_ses_start.hidden = !(w->host && !w->failed);
        g_ses_cancel.title = w->failed ? @"Back" : @"Cancel";
        g_ses_copy.enabled = w->address[0] != 0;
    }
    *o = *w;
    o->can_start = start;
}

int session_window_action(void) { int a = g_ses_action; g_ses_action = 0; return a; }

void session_window_close(void) {
    if (!g_ses) return;
    [g_ses orderOut:nil];
    g_ses = nil;
}

/* ---- volume (Game > Volume...) */
static NSPanel* g_vol;
static NSSlider* g_vol_bar;
static NSTextField* g_vol_text;
static NSButton* g_vol_mute_box;
static int* g_vol_val; static int* g_vol_mute;

@interface DqixVolume : NSObject <NSWindowDelegate>
@end
@implementation DqixVolume
- (void)slid:(NSSlider*)s {
    int v = (int)lround(s.doubleValue);
    *g_vol_val = v; if (v > 0) *g_vol_mute = 0;
    volume_window_refresh();
}
- (void)muted:(NSButton*)b { *g_vol_mute = b.state == NSControlStateValueOn; }
- (void)close:(id)sender { [g_vol orderOut:nil]; }
@end
static DqixVolume* g_vol_target;

void volume_window_open(SDL_Window* win, int* volume, int* mute) {
    (void)win;
    g_vol_val = volume; g_vol_mute = mute;
    @autoreleasepool {
    if (!g_vol) {
        g_vol_target = [DqixVolume new];
        NSPanel* p = panel(@"Volume", 320, 112, g_vol_target);
        g_vol_bar = [NSSlider sliderWithValue:*volume minValue:0 maxValue:100 target:g_vol_target action:@selector(slid:)];
        g_vol_bar.frame = at(p, 16, 16, 230, 24);
        g_vol_bar.continuous = YES;
        [p.contentView addSubview:g_vol_bar];
        g_vol_text = text_label(p, at(p, 256, 19, 50, 18), @"");
        g_vol_mute_box = [NSButton checkboxWithTitle:@"Mute" target:g_vol_target action:@selector(muted:)];
        g_vol_mute_box.frame = at(p, 16, 50, 120, 20);
        [p.contentView addSubview:g_vol_mute_box];
        NSButton* c = button(p, at(p, 226, 76, 80, 28), @"Close", g_vol_target, @selector(close:));
        c.keyEquivalent = @"\r";
        g_vol = p;
    }
    volume_window_refresh();
    [g_vol makeKeyAndOrderFront:nil];
    }
}
void volume_window_refresh(void) {
    if (!g_vol) return;
    g_vol_bar.doubleValue = *g_vol_val;
    g_vol_text.stringValue = [NSString stringWithFormat:@"%d%%", *g_vol_val];
    g_vol_mute_box.state = *g_vol_mute ? NSControlStateValueOn : NSControlStateValueOff;
}

/* ---- controls (Game > Controls...) */
static NSPanel* g_ctlw;
static NSTableView* g_ctl_table;
static int g_ctl_req = -1, g_ctl_req_what;

@interface DqixControls : NSObject <NSWindowDelegate, NSTableViewDataSource, NSTableViewDelegate>
@end
@implementation DqixControls
- (NSInteger)numberOfRowsInTableView:(NSTableView*)t { return CA_N; }
- (id)tableView:(NSTableView*)t objectValueForTableColumn:(NSTableColumn*)c row:(NSInteger)row {
    char s[64];
    int a = (int)row;
    if ([c.identifier isEqualToString:@"action"]) return S(g_ctl[a].name);
    if ([c.identifier isEqualToString:@"key"]) controls_key_name(g_ctl_key[a][0], s, sizeof s);
    else if ([c.identifier isEqualToString:@"key2"]) controls_key_name(g_ctl_key[a][1], s, sizeof s);
    else controls_pad_name(g_ctl_pad[a], s, sizeof s);
    return S(s);
}
- (void)request:(int)what { NSInteger sel = g_ctl_table.selectedRow; if (sel >= 0) { g_ctl_req = (int)sel; g_ctl_req_what = what; } }
- (void)doubleClicked:(id)sender { if (g_ctl_table.clickedRow >= 0) { g_ctl_req = (int)g_ctl_table.clickedRow; g_ctl_req_what = 0; } }
- (void)key1:(id)sender { [self request:0]; }
- (void)key2:(id)sender { [self request:1]; }
- (void)pad:(id)sender { [self request:2]; }
- (void)clear:(id)sender {
    NSInteger sel = g_ctl_table.selectedRow;
    if (sel < 0) return;
    if (g_ctl_key[sel][1]) g_ctl_key[sel][1] = 0;
    else if (g_ctl_key[sel][0]) g_ctl_key[sel][0] = 0;
    else g_ctl_pad[sel] = CTL_NONE;
    controls_save(); [g_ctl_table reloadData];
}
- (void)defaults:(id)sender {
    NSAlert* a = [NSAlert new];
    a.messageText = @"Controls";
    a.informativeText = @"Put every key and button back to how the game came?";
    [a addButtonWithTitle:@"Yes"]; [a addButtonWithTitle:@"No"];
    if ([a runModal] == NSAlertFirstButtonReturn) { controls_defaults(); controls_save(); [g_ctl_table reloadData]; }
}
- (void)close:(id)sender { [g_ctlw orderOut:nil]; }
@end
static DqixControls* g_ctl_target;

void controls_window_open(SDL_Window* win) {
    (void)win;
    @autoreleasepool {
    if (!g_ctlw) {
        g_ctl_target = [DqixControls new];
        NSPanel* p = panel(@"Controls", 640, 500, g_ctl_target);
        text_label(p, at(p, 16, 12, 608, 36), @"Pick an action, then change one of its keys or its controller button. A key or button that "
                   "another action has moves over to it (they swap). Esc and Alt+Enter, and the sticks, stay as they are.");
        NSScrollView* sv = [[NSScrollView alloc] initWithFrame:at(p, 16, 56, 608, 340)];
        sv.hasVerticalScroller = YES; sv.borderType = NSBezelBorder;
        NSTableView* t = [[NSTableView alloc] initWithFrame:sv.contentView.bounds];
        NSString* ids[4] = { @"action", @"key", @"key2", @"pad" };
        NSString* heads[4] = { @"Action", @"Key", @"Second key", @"Controller" };
        CGFloat widths[4] = { 250, 100, 100, 140 };
        for (int i = 0; i < 4; i++) {
            NSTableColumn* c = [[NSTableColumn alloc] initWithIdentifier:ids[i]];
            c.title = heads[i]; c.width = widths[i]; c.editable = NO;
            [t addTableColumn:c];
        }
        t.dataSource = g_ctl_target; t.delegate = g_ctl_target;
        t.target = g_ctl_target; t.doubleAction = @selector(doubleClicked:);
        t.usesAlternatingRowBackgroundColors = YES;
        sv.documentView = t;
        [p.contentView addSubview:sv];
        g_ctl_table = t;
        button(p, at(p, 16, 404, 120, 28), @"Change key...", g_ctl_target, @selector(key1:));
        button(p, at(p, 140, 404, 120, 28), @"Second key...", g_ctl_target, @selector(key2:));
        button(p, at(p, 264, 404, 160, 28), @"Controller button...", g_ctl_target, @selector(pad:));
        button(p, at(p, 428, 404, 80, 28), @"Clear", g_ctl_target, @selector(clear:));
        button(p, at(p, 16, 456, 150, 28), @"Restore defaults", g_ctl_target, @selector(defaults:));
        NSButton* c = button(p, at(p, 544, 456, 80, 28), @"Close", g_ctl_target, @selector(close:));
        c.keyEquivalent = @"\r";
        g_ctlw = p;
        [t reloadData];
        [t selectRowIndexes:[NSIndexSet indexSetWithIndex:0] byExtendingSelection:NO];
    }
    [g_ctlw makeKeyAndOrderFront:nil];
    }
}
int controls_window_request(int* action, int* what) {
    if (g_ctl_req < 0) return 0;
    *action = g_ctl_req; *what = g_ctl_req_what; g_ctl_req = -1;
    return 1;
}
void controls_window_refresh(void) {
    if (!g_ctlw || !g_ctlw.visible) return;
    [g_ctl_table reloadData];
    [g_ctlw makeKeyAndOrderFront:nil];
}
