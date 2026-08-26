"""The app shell: four tabs, one status bar, and one way to call the service.

Everything slow — a run, a compile, an eval sweep, even reading the index —
goes through :meth:`AtsTailorApp.run_task`, which puts the call on a Textual
thread worker and hands the result back on the UI thread. Nothing in this
package calls the service on the event loop, because a two-minute solver call
made there would freeze the terminal with no way to say what it was doing.

The layout degrades rather than reflows: below 90 columns the right-hand pane
of the Tailor and Index screens is hidden (``narrow`` on the root), which is
what the plan asks for and what a three-pane layout can honestly do in 89
columns.
"""

from __future__ import annotations

import tempfile
import threading
from pathlib import Path
from typing import Any, Callable

from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.widgets import Footer, Header, Static, TabbedContent, TabPane

from ..service import Service
from .actions import Actions, Outcome, attempt
from .applications import ApplicationsScreen
from .evaluating import EvalScreen
from .indexing import IndexScreen
from .modals import ChoiceDialog, HelpDialog
from .tailoring import TailorScreen

#: Below this many columns the third pane is hidden.
NARROW_COLUMNS = 90

TABS = (
    ("tab-applications", "1 Applications"),
    ("tab-tailor", "2 Tailor"),
    ("tab-index", "3 Index"),
    ("tab-eval", "4 Eval"),
)

#: What a tab focuses when it opens. Every single-letter key on a screen is a
#: binding on that screen's container, and a binding is only reachable from a
#: widget inside it — so leaving focus on the tab bar would mean `s`, `p` and
#: `e` silently doing nothing on the screen they belong to.
FOCUS_TARGETS = {
    "tab-applications": "#applications-table",
    "tab-tailor": "#tailor-requirements",
    "tab-index": "#index-tree",
    "tab-eval": "#eval-table",
}


class AtsTailorApp(App):
    """One repository, four screens, one :class:`Service`."""

    CSS_PATH = "styles.tcss"
    TITLE = "ats-tailor"
    #: Textual binds ``ctrl+p`` to its command palette; here it is the provider
    #: picker (``docs/plans/phase-4b.md`` §1), and two things cannot have it.
    ENABLE_COMMAND_PALETTE = False

    BINDINGS = [
        Binding("1", "show_tab('tab-applications')", "applications"),
        Binding("2", "show_tab('tab-tailor')", "tailor"),
        Binding("3", "show_tab('tab-index')", "index"),
        Binding("4", "show_tab('tab-eval')", "eval"),
        Binding("question_mark", "help", "help", key_display="?"),
        Binding("ctrl+p", "provider", "provider"),
        Binding("ctrl+q", "quit", "quit"),
    ]

    def __init__(self, service: Service, scratch_dir: str | Path | None = None):
        super().__init__()
        self.service = service
        self.actions = Actions(service)
        self.scratch_dir = Path(
            scratch_dir or Path(tempfile.gettempdir()) / "ats-tailor-tui")
        self._busy: list[str] = []
        #: One service call at a time. This build of SQLite is in multi-thread
        #: mode (``sqlite3.threadsafety == 1``): the store's connection may move
        #: between threads but must never be used by two at once, and Textual
        #: runs every worker on its own thread. The MCP server holds the same
        #: lock for the same reason (``docs/v2-design.md`` §8, 4a). It also
        #: means two re-runs cannot race into one output directory.
        self._service_lock = threading.Lock()

    # ── layout ──
    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with TabbedContent(id="tabs"):
            with TabPane(TABS[0][1], id=TABS[0][0]):
                yield ApplicationsScreen(id="screen-applications")
            with TabPane(TABS[1][1], id=TABS[1][0]):
                yield TailorScreen(id="screen-tailor")
            with TabPane(TABS[2][1], id=TABS[2][0]):
                yield IndexScreen(id="screen-index")
            with TabPane(TABS[3][1], id=TABS[3][0]):
                yield EvalScreen(id="screen-eval")
        with Vertical(id="statusbar"):
            yield Static("", id="status-left")
            yield Static("", id="status-busy")
        yield Footer()

    def on_mount(self) -> None:
        self.sub_title = str(self.service.repo_root)
        self._apply_width(self.size.width)
        self.refresh_status()
        self.call_after_refresh(self._focus_tab, "tab-applications")

    @on(TabbedContent.TabActivated, "#tabs")
    def _tab_activated(self, event: TabbedContent.TabActivated) -> None:
        pane = getattr(event, "pane", None)
        self.call_after_refresh(self._focus_tab, getattr(pane, "id", "") or "")

    def _focus_tab(self, tab: str) -> None:
        selector = FOCUS_TARGETS.get(tab)
        if not selector:
            return
        try:
            self.query_one(selector).focus()
        except Exception:  # noqa: BLE001 - the screen may not have mounted yet
            pass

    def on_resize(self, event) -> None:
        self._apply_width(event.size.width)

    def _apply_width(self, width: int) -> None:
        self.set_class(width < NARROW_COLUMNS, "narrow")

    # ── the one way to call the service ──
    def run_task(
        self,
        label: str,
        call: Callable[[], Any],
        then: Callable[[Any], None] | None = None,
        on_error: Callable[[Outcome], bool] | None = None,
    ) -> None:
        """Run ``call`` off the UI thread; ``then`` gets the value on it.

        ``on_error`` may handle a failure and return ``False`` to suppress the
        toast — the Index screen shows validation errors beside the form, where
        a message that fades would be the wrong place for them.
        """
        self._worker(label, call, then, on_error)

    @work(thread=True, group="service")
    def _worker(self, label, call, then, on_error) -> None:
        self.call_from_thread(self._begin, label)
        with self._service_lock:
            outcome = attempt(call)
        self.call_from_thread(self._finish, label, outcome, then, on_error)

    def _begin(self, label: str) -> None:
        self._busy.append(label)
        self._render_busy()

    def _finish(self, label, outcome: Outcome, then, on_error) -> None:
        if label in self._busy:
            self._busy.remove(label)
        self._render_busy()
        if outcome.ok:
            if then is not None:
                then(outcome.value)
            return
        toast = True
        if on_error is not None:
            toast = bool(on_error(outcome))
        if toast:
            self.notify(outcome.toast(), title=label, severity="error", timeout=10)

    def _render_busy(self) -> None:
        try:
            widget = self.query_one("#status-busy", Static)
        except Exception:  # noqa: BLE001 - called during teardown
            return
        if self._busy:
            widget.update(f"[reverse] ⋯ {'; '.join(self._busy[-2:])} [/reverse]")
        else:
            widget.update("")

    @property
    def busy(self) -> bool:
        return bool(self._busy)

    # ── status bar ──
    def refresh_status(self) -> None:
        self.run_task("healthcheck", self.actions.health, then=self._show_status)

    def _show_status(self, health: dict) -> None:
        self.query_one("#status-left", Static).update(
            self.actions.status_line(health))

    # ── cross-screen ──
    def open_in_tailor(self, run_id: str) -> None:
        self.query_one("#tabs", TabbedContent).active = "tab-tailor"
        self.query_one("#screen-tailor", TailorScreen).open_run(run_id)

    def refresh_applications(self) -> None:
        self.query_one("#screen-applications", ApplicationsScreen).reload()

    # ── global actions ──
    def action_show_tab(self, tab: str) -> None:
        self.query_one("#tabs", TabbedContent).active = tab

    def action_help(self) -> None:
        self.push_screen(HelpDialog())

    def action_provider(self) -> None:
        def picked(name: str | None) -> None:
            if not name:
                return
            self.actions.set_provider(name)
            self.refresh_status()
            self.notify(f"provider is now {name}")

        providers = self.actions.providers
        current = self.actions.provider
        self.push_screen(
            ChoiceDialog(
                "LLM provider",
                [(name, name) for name in providers],
                subtitle=f"now: {current} — applies to the next run, not past ones",
                initial=max(0, list(providers).index(current)
                            if current in providers else 0),
            ),
            picked,
        )
