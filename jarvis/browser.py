"""Browser automation: the web-page equivalent of executor.py's native-app
automation. Same philosophy -- target elements by visible role/text via
Playwright's accessibility-aware locators, never by guessing pixel
coordinates, for the same reason executor.py uses the Accessibility API
instead of screenshots: free/small models can't reliably click by pixel
position, but they can pick a name off a text list.

A single Chromium instance/page is kept alive across commands (module-level
singletons), the same way executor.py's target app stays open between
steps -- so a multi-step web task doesn't relaunch the browser each time.
Runs non-headless (a visible window) on purpose: this is voice-triggered
automation of your own browser, you should be able to see what it's doing.
"""

from playwright.sync_api import sync_playwright

MAX_SNAPSHOT_CHARS = 4000

_playwright = None
_browser = None
_page = None


def _ensure_page():
    global _playwright, _browser, _page
    if _page is not None and not _page.is_closed():
        return _page
    if _playwright is None:
        _playwright = sync_playwright().start()
    if _browser is None or not _browser.is_connected():
        _browser = _playwright.chromium.launch(headless=False)
    _page = _browser.new_page()
    return _page


def describe_page() -> str:
    """Text summary of the page's accessibility tree -- fed back to the
    model as its 'view' of the page, the same role executor.describe_ui
    plays for native apps. Uses Locator.aria_snapshot(), the current
    Playwright API for this (the older page.accessibility.snapshot() was
    removed from the installed version -- confirmed by testing, not just a
    docs check); it already returns a compact, readable role/name tree, so
    it's passed straight through rather than re-parsed into another format."""
    page = _ensure_page()
    snapshot = page.locator("body").aria_snapshot()
    if len(snapshot) > MAX_SNAPSHOT_CHARS:
        snapshot = snapshot[:MAX_SNAPSHOT_CHARS] + "\n...(truncated)"
    return f"Page: {page.title()!r} ({page.url})\n{snapshot}"


def browser_open(url: str) -> str:
    page = _ensure_page()
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    page.goto(url, wait_until="domcontentloaded", timeout=15000)
    return describe_page()


def browser_click(text: str) -> str:
    page = _ensure_page()
    for locator in (
        page.get_by_role("button", name=text, exact=False),
        page.get_by_role("link", name=text, exact=False),
        page.get_by_text(text, exact=False),
    ):
        try:
            if locator.count() > 0:
                locator.first.click(timeout=5000)
                return describe_page()
        except Exception:  # noqa: BLE001
            continue
    raise RuntimeError(f"No clickable element matching '{text}' found on the page.")


def browser_type(text: str) -> str:
    page = _ensure_page()
    page.keyboard.type(text)
    return "ok"


def browser_press_key(keys: str) -> str:
    page = _ensure_page()
    page.keyboard.press(keys)
    return describe_page()


def browser_go_back() -> str:
    page = _ensure_page()
    page.go_back(wait_until="domcontentloaded")
    return describe_page()


def browser_close() -> str:
    global _playwright, _browser, _page
    if _page:
        _page.close()
    if _browser:
        _browser.close()
    if _playwright:
        _playwright.stop()
    _playwright = _browser = _page = None
    return "Browser closed."


ACTIONS = {
    "browser_open": browser_open,
    "browser_click": browser_click,
    "browser_type": browser_type,
    "browser_press_key": browser_press_key,
    "browser_go_back": browser_go_back,
    "browser_close": browser_close,
}
