"""Dark mode, checked from the source of truth: the CSS tokens. Contrast is computed from the
token values themselves for BOTH themes, so a token edit that breaks a floor fails here."""

from __future__ import annotations

import colorsys
import json
import re
import shutil
import subprocess

import pytest

from sightline.server import FRONTEND_DIR

CSS = (FRONTEND_DIR / "style.css").read_text(encoding="utf-8")
INDEX = (FRONTEND_DIR / "index.html").read_text(encoding="utf-8")


# ------------------------------------------------------------------------------- tokens


def _block(selector: str) -> str:
    start = CSS.index(selector + " {")
    return CSS[start : CSS.index("\n}", start)]


def _tokens(block: str) -> dict[str, str]:
    return {m.group(1): m.group(2).strip() for m in re.finditer(r"^\s*--([\w-]+):\s*([^;]+);", block, re.M)}


LIGHT = _tokens(_block(":root"))
DARK = {**LIGHT, **_tokens(_block(':root[data-theme="dark"]'))}  # dark overrides light; the rest is shared
THEMES = {"light": LIGHT, "dark": DARK}


def _resolve(theme: dict[str, str], name: str) -> str:
    v = theme[name]
    while v.startswith("var("):
        v = theme[v[6:-1]]
    return v


def _rgba(value: str) -> tuple[float, float, float, float]:
    value = value.strip()
    if value.startswith("#"):
        h = value[1:]
        h = "".join(c * 2 for c in h) if len(h) == 3 else h
        return (*(int(h[i : i + 2], 16) for i in (0, 2, 4)), 1.0)
    m = re.fullmatch(r"rgba?\(([^)]+)\)", value)
    r, g, b, *a = (float(x) for x in m.group(1).split(","))
    return (r, g, b, a[0] if a else 1.0)


def _over(fg: tuple[float, ...], bg: tuple[float, ...]) -> tuple[float, float, float]:
    a = fg[3]
    return tuple(fg[i] * a + bg[i] * (1 - a) for i in range(3))


def _lum(rgb) -> float:
    def lin(c):
        c /= 255
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = (lin(c) for c in rgb[:3])
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(theme: str, fg: str, bg: str, over: str = "surface") -> float:
    """WCAG contrast of token `fg` on token `bg`. Translucent tokens are composited: the
    background over `over` (a page surface), and the foreground over that."""
    t = THEMES[theme]
    base = _rgba(_resolve(t, over))
    b = _over(_rgba(_resolve(t, bg)), base)
    f = _over(_rgba(_resolve(t, fg)), (*b, 1.0))
    hi, lo = sorted((_lum(f), _lum(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


SURFACES = ("page", "surface", "sunken")


# --------------------------------------------------------------- no hard-coded colours


COLOUR = re.compile(r"#[0-9a-fA-F]{3,8}\b|\brgba?\(|\bhsla?\(|\b(?:white|black)\b(?!-)")  # not white-space


def test_both_themes_define_the_minimum_token_set():
    needed = {"page", "surface", "sunken", "border", "ink", "ink-2", "muted", "accent", "c-novice", "c-peer", "c-expert",
              "chart-grid", "chart-axis", "chart-ref", "chart-shade-opacity", "tier-1-border", "tier-2-bg", "tier-3-bg", "tier-3-fg"}
    assert needed <= set(LIGHT), needed - set(LIGHT)
    changed = {k for k in LIGHT if DARK[k] != LIGHT[k]}
    # everything that is a colour and is not deliberately shared is redefined for dark
    assert {"page", "surface", "sunken", "ink", "ink-2", "muted", "c-novice", "c-peer", "c-expert", "accent"} <= changed


def test_no_component_hard_codes_a_colour():
    """Colour values may be written only on token lines inside the two theme blocks."""
    offenders = []
    # blank out comments (keeping their newlines so line numbers still match)
    code_only = re.sub(r"/\*.*?\*/", lambda m: "\n" * m.group(0).count("\n"), CSS, flags=re.S)
    in_tokens = False
    for n, line in enumerate(code_only.splitlines(), 1):
        if line.startswith(":root {") or line.startswith(':root[data-theme="dark"] {'):
            in_tokens = True
        elif in_tokens and line.startswith("}"):
            in_tokens = False
            continue
        if not in_tokens and COLOUR.search(line):
            offenders.append((n, line.strip()))
    assert offenders == []

    for path in FRONTEND_DIR.glob("js/*.js"):
        text = re.sub(r"//.*|/\*.*?\*/", "", path.read_text(encoding="utf-8"), flags=re.S)
        assert not COLOUR.search(text), f"{path.name} hard-codes a colour"
    head_script = INDEX[INDEX.index("<script>") : INDEX.index("</script>")]
    assert not COLOUR.search(head_script)


def test_body_gets_an_explicit_background_from_a_token():
    assert re.search(r"body\s*\{[^}]*background:\s*var\(--page\)", CSS)


# ------------------------------------------------------------------------ contrast floors


@pytest.mark.parametrize("theme", THEMES)
@pytest.mark.parametrize("fg", ["ink", "ink-2", "muted"])
@pytest.mark.parametrize("bg", SURFACES)
def test_body_text_holds_4_5_to_1_on_every_surface(theme, fg, bg):
    assert contrast(theme, fg, bg, over="page") >= 4.5, (theme, fg, bg)


@pytest.mark.parametrize("theme", THEMES)
@pytest.mark.parametrize("bg", ["wash", "sunken", "critical-wash", "tier-2-bg", "hl-bg"])
def test_text_stays_readable_on_tinted_fills(theme, bg):
    fg = "hl-fg" if bg == "hl-bg" else "critical" if bg == "critical-wash" else "ink"
    for base in ("page", "surface"):
        assert contrast(theme, fg, bg, over=base) >= 4.5, (theme, fg, bg, base)


@pytest.mark.parametrize("theme", THEMES)
def test_quoted_persona_text_and_provenance_panels_read_clearly(theme):
    """The content this product exists to show: quotes sit on surface/sunken in ink or ink-2."""
    for fg in ("ink", "ink-2"):
        for bg in ("surface", "sunken"):
            assert contrast(theme, fg, bg) >= 7.0, (theme, fg, bg)  # comfortably above the 4.5 floor


@pytest.mark.parametrize("theme", THEMES)
@pytest.mark.parametrize("stroke", ["stroke", "chart-axis", "chart-ref", "c-novice", "c-peer", "c-expert", "tier-1-border", "tier-2-border", "focus"])
@pytest.mark.parametrize("bg", ["page", "surface"])
def test_strokes_and_chart_marks_hold_3_to_1(theme, stroke, bg):
    assert contrast(theme, stroke, bg) >= 3.0, (theme, stroke, bg)


@pytest.mark.parametrize("theme", THEMES)
def test_tier_chips_are_readable_in_every_step(theme):
    assert contrast(theme, "ink", "tier-2-bg") >= 4.5
    assert contrast(theme, "tier-3-fg", "tier-3-bg") >= 4.5
    assert contrast(theme, "ink", "surface") >= 4.5  # the outline step has no fill


def test_tier_steps_differ_in_fill_not_only_in_colour():
    """Colour must not be the only carrier of the tier: the steps are outline, tint and solid."""
    chip = CSS[CSS.index(".tier-chip {") :]
    assert "background: transparent" in chip.split("}")[0]
    assert ".tier-chip.background_needed { background: var(--tier-2-bg)" in CSS
    assert ".tier-chip.expert_gated { background: var(--tier-3-bg)" in CSS
    assert 'h("button", {\n    class: `tier-chip' in (FRONTEND_DIR / "js" / "run-common.js").read_text(encoding="utf-8")  # label text is the button's content
    assert "}, t.label)" in (FRONTEND_DIR / "js" / "run-common.js").read_text(encoding="utf-8")


# ------------------------------------------------------------------------ chart + hues


def _hue(theme: str, name: str) -> float:
    r, g, b, _ = _rgba(_resolve(THEMES[theme], name))
    return colorsys.rgb_to_hls(r / 255, g / 255, b / 255)[0] * 360


@pytest.mark.parametrize("series", ["c-novice", "c-peer", "c-expert"])
def test_series_colours_keep_their_hue_across_themes(series):
    """A screenshot in one theme must be recognisable in the other: lightness moves, hue does not."""
    a, b = _hue("light", series), _hue("dark", series)
    assert abs(a - b) <= 8, (series, a, b)


def test_the_three_series_stay_distinguishable_from_each_other_in_both_themes():
    for theme in THEMES:
        hues = sorted(_hue(theme, s) for s in ("c-novice", "c-peer", "c-expert"))
        assert min(b - a for a, b in zip(hues, hues[1:])) >= 40


def test_shading_is_stronger_in_dark_so_it_reads_on_a_dark_page():
    assert float(DARK["chart-shade-opacity"]) > float(LIGHT["chart-shade-opacity"]) >= 0.1


def test_expert_series_is_told_apart_from_the_reference_by_stroke_width_and_dash():
    src = (FRONTEND_DIR / "js" / "charts.js").read_text(encoding="utf-8")
    assert 'definitional.has(k) ? 4 : 2' in src  # thicker than the measured series
    assert '"stroke-dasharray": "6 5"' in src  # the reference is dashed
    # ...and the dashed reference is laid over the expert's line, not hidden beneath it
    assert src.index('stroke-width": definitional.has(k) ? 4 : 2') < src.index('"inferred intent (reference)"')
    assert "var(--chart-shade-opacity)" in src and "var(--chart-ref)" in src and "var(--chart-axis)" in src


# ------------------------------------------------------------------------- slide images


def test_the_slide_card_is_light_and_identical_in_both_themes():
    assert DARK["slide-card-bg"] == LIGHT["slide-card-bg"]
    r, g, b, _ = _rgba(LIGHT["slide-card-bg"])
    assert _lum((r, g, b)) > 0.75
    # dark mode changes only its edge: the card carries no glow to change
    assert DARK["slide-card-border"] != LIGHT["slide-card-border"]
    assert re.search(r"rgba\(255, 255, 255, 0\.\d+\)", DARK["slide-card-border"]).group(0)  # a low-contrast light edge


def test_depth_is_a_rule_never_an_elevation_layer():
    """Editorial, not material: separation comes from 1px lines and surface contrast. A shadow
    anywhere means a component reached for elevation instead of a rule."""
    assert "box-shadow" not in CSS and "backdrop-filter" not in CSS
    assert "filter: blur" not in CSS


def test_nothing_is_pill_shaped_and_no_corner_is_rounder_than_4px():
    for radius in re.findall(r"border-radius:\s*([^;}]+)", CSS):
        radius = radius.strip()
        assert "%" not in radius and "999" not in radius, radius
        for px in re.findall(r"(\d+(?:\.\d+)?)px", radius):
            assert float(px) <= 4, radius


def test_slide_images_are_never_filtered_inverted_dimmed_or_blended():
    for m in re.finditer(r"([^{}]+)\{([^}]*)\}", CSS):
        selector, body = m.group(1), m.group(2)
        if not re.search(r"\bimg\b|\.thumb|\.slide-card", selector):
            continue
        for prop, value in re.findall(r"([\w-]+)\s*:\s*([^;]+);?", body):
            value = value.strip()
            if prop == "filter":
                assert value == "none", (selector, prop, value)
            if prop == "opacity":
                assert value == "1", (selector, prop, value)
            if prop == "mix-blend-mode":
                assert value == "normal", (selector, prop, value)
    assert "invert(" not in CSS and "grayscale(" not in CSS and "brightness(" not in CSS


def test_every_slide_image_in_the_ui_goes_through_the_slide_card():
    for path in FRONTEND_DIR.glob("js/*.js"):
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if 'h("img"' in line:
                assert "slide-card" in line, f"{path.name}:{n} makes an <img> outside the slide card"
    assert 'class: `slide-card' in (FRONTEND_DIR / "js" / "dom.js").read_text(encoding="utf-8")
    assert 'class: "thumb slide-card"' in (FRONTEND_DIR / "js" / "views-overview.js").read_text(encoding="utf-8")


# --------------------------------------------------------------------- toggle behaviour


def test_the_theme_is_set_before_first_paint_by_a_blocking_inline_script():
    head = INDEX[INDEX.index("<head>") : INDEX.index("</head>")]
    script, sheet = head.index("<script>"), head.index('rel="stylesheet"')
    assert script < sheet  # before the stylesheet, so first paint is already in the right theme
    inline = head[script : head.index("</script>")]
    assert "async" not in head[script - 1 : script + 20] and 'type="module"' not in head[script - 1 : script + 30]  # blocking
    assert "try {" in inline and "localStorage" in inline and "prefers-color-scheme" in inline
    assert 'setAttribute("data-theme"' in inline


def test_there_is_a_toggle_in_the_header_and_it_is_present_on_every_screen():
    header = INDEX[INDEX.index('<header class="topbar">') : INDEX.index("</header>")]
    assert 'id="theme-toggle"' in header and "aria-pressed" in header
    assert '<main id="app">' in INDEX and INDEX.index('id="theme-toggle"') < INDEX.index('<main id="app">')  # outside the routed content


def test_every_local_storage_access_is_guarded():
    for src in [FRONTEND_DIR / "js" / "theme.js", FRONTEND_DIR / "index.html"]:
        text = src.read_text(encoding="utf-8")
        for m in re.finditer(r"localStorage\.\w+", text):
            before = text[max(0, m.start() - 160) : m.start()]
            assert "try" in before, f"unguarded {m.group(0)} in {src.name}"


@pytest.mark.skipif(shutil.which("node") is None, reason="node is needed to run the theme module")
def test_the_toggle_persists_follows_the_system_and_survives_a_throwing_localstorage():
    p = subprocess.run(["node", str(FRONTEND_DIR / "smoke" / "theme.mjs")], capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert p.returncode == 0, p.stderr[-1500:]
    r = json.loads(p.stdout)
    assert r["system dark, nothing saved"]["after"] == "dark" and r["system light, nothing saved"]["after"] == "light"
    assert r["saved choice beats the system"]["after"] == "light"
    t = r["toggle overrides the system and persists"]
    assert (t["before"], t["after"], t["stored"]) == ("dark", "light", "light")
    priv = r["private window: storage throws, page still works"]
    assert (priv["before"], priv["after"], priv["stored"]) == ("light", "dark", None)  # no crash; lasts for the page
    assert r["head script already set it"]["after"] == "dark"
    assert r["follows the system until a choice is made"]["after"] == "dark"
    assert r["stops following once the user has chosen"]["after"] == "dark"
    assert all(v["pressed"] == str(v["after"] == "dark").lower() for v in r.values())  # aria-pressed tracks the theme
