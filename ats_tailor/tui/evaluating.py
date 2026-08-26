"""Screen 4 — the measured baselines.

``eval/*.json`` is the only place in this repository that says whether a change
helped, so this screen is a reader first and a runner second: the table for
whatever is highlighted, and a diff between any two files, without spending the
minutes a sweep costs.

``e`` does spend them. It asks for the engine and the provider before it starts
because a sweep run against the wrong provider is a baseline that quietly is
not comparable with the one beside it.
"""

from __future__ import annotations

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import DataTable, Static

from .actions import format_percent
from .modals import ChoiceDialog, ComparePairDialog


class EvalScreen(Vertical):
    """Baselines on the left, the highlighted one's table on the right."""

    BINDINGS = [
        Binding("e", "run_eval", "run eval"),
        Binding("d", "diff", "diff two"),
        Binding("ctrl+r", "reload", "reload"),
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.baselines: list = []

    def compose(self) -> ComposeResult:
        with Horizontal(id="eval-columns"):
            with Vertical(id="eval-left", classes="pane"):
                yield DataTable(id="eval-table", cursor_type="row")
            with VerticalScroll(id="eval-right", classes="pane"):
                yield Static("", id="eval-detail")

    def on_mount(self) -> None:
        table = self.query_one("#eval-table", DataTable)
        for label, width in (("config", 14), ("when", 11), ("engine", 7),
                             ("provider", 9), ("runs", 5), ("must", 6),
                             ("nice", 6), ("ctx", 6)):
            table.add_column(label, width=width)
        self.query_one("#eval-left", Vertical).border_title = "baselines"
        self.query_one("#eval-right", VerticalScroll).border_title = "result"
        self.reload()

    def action_reload(self) -> None:
        self.reload()

    def reload(self) -> None:
        self.app.run_task("reading eval/", self.app.actions.baselines, then=self._show)

    def _show(self, baselines) -> None:
        self.baselines = list(baselines)
        table = self.query_one("#eval-table", DataTable)
        table.clear()
        for baseline in self.baselines:
            means = baseline.means or {}
            table.add_row(
                baseline.config_hash[:12] or "—",
                (baseline.created_at or "")[:10],
                baseline.engine or "—",
                baseline.provider or "—",
                str(baseline.runs),
                format_percent(means.get("must_coverage")),
                format_percent(means.get("nice_coverage")),
                format_percent(means.get("context_rate")),
            )
        detail = self.query_one("#eval-detail", Static)
        if not self.baselines:
            detail.update(
                "[dim]no baselines in eval/ yet — press [b]e[/b] to run the sweep.[/dim]")
        else:
            self._load_detail(0)

    def current(self):
        table = self.query_one("#eval-table", DataTable)
        index = table.cursor_row
        return self.baselines[index] if 0 <= index < len(self.baselines) else None

    @on(DataTable.RowHighlighted, "#eval-table")
    def _highlighted(self, event: DataTable.RowHighlighted) -> None:
        self._load_detail(event.cursor_row)

    def _load_detail(self, index: int) -> None:
        if not (0 <= index < len(self.baselines)):
            return
        path = self.baselines[index].path
        self.app.run_task(
            f"reading {path}",
            lambda: self.app.actions.eval_table(path),
            then=lambda table: self.query_one("#eval-detail", Static).update(
                f"[b]{path}[/b]\n\n{table}"),
        )

    # ── actions ──
    def action_diff(self) -> None:
        if len(self.baselines) < 2:
            self.app.notify("two baselines are needed to diff", severity="warning")
            return
        choices = [
            (f"{baseline.config_hash[:12]}  {baseline.engine}/{baseline.provider}"
             f"  {(baseline.created_at or '')[:10]}", baseline.path)
            for baseline in self.baselines
        ]

        def picked(pair) -> None:
            if not pair:
                return
            left, right = pair
            self.app.run_task(
                "diffing baselines",
                lambda: self.app.actions.eval_compare(left, right),
                then=lambda table: self.query_one("#eval-detail", Static).update(
                    f"[b]{left}  →  {right}[/b]\n\n{table}"),
            )

        self.app.push_screen(ComparePairDialog(choices), picked)

    def action_run_eval(self) -> None:
        def engine_picked(engine: str | None) -> None:
            if not engine:
                return

            def provider_picked(provider: str | None) -> None:
                if not provider:
                    return
                self.app.run_task(
                    f"eval sweep ({engine}/{provider}) — minutes, not seconds",
                    lambda: self.app.actions.run_eval(engine, provider),
                    then=self._finished,
                )

            self.app.push_screen(
                ChoiceDialog(
                    "Provider",
                    [(name, name) for name in self.app.actions.providers],
                    subtitle="who reads the postings during the sweep",
                ),
                provider_picked,
            )

        self.app.push_screen(
            ChoiceDialog("Engine", [("v2", "v2"), ("v1", "v1")],
                         subtitle="the selection engine to score"),
            engine_picked,
        )

    def _finished(self, message: str) -> None:
        self.app.notify(message, title="eval", timeout=12)
        self.reload()
