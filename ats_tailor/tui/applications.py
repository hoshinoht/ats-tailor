"""Screen 1 — the tracker.

Every run that was ever recorded, with the three numbers that say whether it is
worth sending and the one that says whether anybody has heard back. A ``⚠``
beside the status means the posting's gates flagged something (a clearance, a
citizenship requirement, a location) — a run that scores well and is barred is
the failure mode this column exists to catch.
"""

from __future__ import annotations

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.widgets import DataTable, Static

from .actions import RUN_STATUSES, ApplicationRow, format_percent
from .modals import AddPostingDialog, ChoiceDialog, PromptDialog

COLUMNS = (
    ("run", 14), ("created", 16), ("company", 18), ("role", 22),
    ("status", 14), ("pg", 3), ("must", 6), ("nice", 6), ("ctx", 6), ("quiet", 6),
)


class ApplicationsScreen(Vertical):
    """The runs table plus a detail line for whatever the cursor is on."""

    BINDINGS = [
        Binding("s", "set_status", "status"),
        Binding("n", "add_note", "note"),
        Binding("q", "toggle_quiet", "quiet only"),
        Binding("a", "add_posting", "add posting"),
        Binding("ctrl+r", "reload", "reload"),
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rows: list[ApplicationRow] = []
        self.quiet_only = False

    def compose(self) -> ComposeResult:
        table = DataTable(id="applications-table", cursor_type="row", zebra_stripes=True)
        yield table
        yield Static("", id="applications-detail")

    def on_mount(self) -> None:
        table = self.query_one("#applications-table", DataTable)
        for label, width in COLUMNS:
            table.add_column(label, width=width, key=label)
        self.reload()

    # ── loading ──
    def reload(self) -> None:
        self.app.run_task(
            "reading the tracker",
            lambda: self.app.actions.applications(self.quiet_only),
            then=self._show,
        )

    def action_reload(self) -> None:
        self.reload()

    def _show(self, rows: list[ApplicationRow]) -> None:
        table = self.query_one("#applications-table", DataTable)
        remembered = self.selected_run_id()
        table.clear()
        self.rows = list(rows)
        for row in self.rows:
            table.add_row(
                row.run_id[:12],
                row.created,
                row.company[:18],
                row.role[:22],
                _status_cell(row),
                "—" if row.pages is None else str(row.pages),
                format_percent(row.must),
                format_percent(row.nice),
                format_percent(row.context),
                "—" if row.days_quiet is None else f"{row.days_quiet}d",
                key=row.run_id,
            )
        if remembered:
            self._move_to(remembered)
        self._describe()
        scope = "quiet only" if self.quiet_only else "all"
        self.border_title = f"Applications — {len(self.rows)} run(s), {scope}"

    def _move_to(self, run_id: str) -> None:
        for position, row in enumerate(self.rows):
            if row.run_id == run_id:
                table = self.query_one("#applications-table", DataTable)
                table.move_cursor(row=position)
                return

    # ── the row under the cursor ──
    def current(self) -> ApplicationRow | None:
        table = self.query_one("#applications-table", DataTable)
        index = table.cursor_row
        if 0 <= index < len(self.rows):
            return self.rows[index]
        return None

    def selected_run_id(self) -> str | None:
        row = self.current()
        return row.run_id if row else None

    @on(DataTable.RowHighlighted, "#applications-table")
    def _highlighted(self) -> None:
        self._describe()

    @on(DataTable.RowSelected, "#applications-table")
    def _selected(self) -> None:
        row = self.current()
        if row is not None:
            self.app.open_in_tailor(row.run_id)

    def _describe(self) -> None:
        row = self.current()
        detail = self.query_one("#applications-detail", Static)
        if row is None:
            detail.update("no runs recorded yet — press [b]a[/b] to add a posting")
            return
        flags = "; ".join(f"{name}: {why}" for name, why in row.flags)
        detail.update(
            f"[b]{row.run_id}[/b]  {row.company} · {row.role}  [{row.status}]"
            + (f"\n[yellow]⚠ {flags}[/yellow]" if flags else "")
        )

    # ── actions ──
    def action_toggle_quiet(self) -> None:
        self.quiet_only = not self.quiet_only
        self.reload()

    def action_set_status(self) -> None:
        row = self.current()
        if row is None:
            self.app.notify("no run selected", severity="warning")
            return
        choices = [(status, status) for status in RUN_STATUSES]
        initial = next(
            (i for i, (label, _v) in enumerate(choices) if label == row.status), 0)

        def chosen(status: str | None) -> None:
            if not status:
                return
            self.app.run_task(
                f"{row.run_id} → {status}",
                lambda: self.app.actions.set_status(row.run_id, status),
                then=lambda _record: self.reload(),
            )

        self.app.push_screen(
            ChoiceDialog(f"Status of {row.company} · {row.role}", choices,
                         subtitle=f"now: {row.status}", initial=initial),
            chosen,
        )

    def action_add_note(self) -> None:
        row = self.current()
        if row is None:
            self.app.notify("no run selected", severity="warning")
            return

        def typed(note: str | None) -> None:
            if not note:
                return
            self.app.run_task(
                "adding a note",
                lambda: self.app.actions.add_note(row.run_id, note),
                then=lambda _record: self.app.notify(f"noted on {row.run_id}"),
            )

        self.app.push_screen(
            PromptDialog(f"Note on {row.company} · {row.role}",
                         placeholder="what happened"),
            typed,
        )

    def action_add_posting(self) -> None:
        def added(payload: dict | None) -> None:
            if not payload:
                return
            missing = [
                name for name in ("company", "role", "text") if not payload.get(name)
            ]
            if missing:
                self.app.notify(
                    f"missing: {', '.join(missing)}", severity="warning")
                return
            self.app.run_task(
                "sanitising and triaging the posting",
                lambda: self.app.actions.add_posting(
                    payload["text"], payload["company"], payload["role"],
                    payload.get("source_url") or None,
                ),
                then=self._posting_added,
            )

        self.app.push_screen(AddPostingDialog(), added)

    def _posting_added(self, result: dict) -> None:
        info, report = result["jd"], result["fit"]
        stripped = (
            "" if info.sanitised is None or info.sanitised.empty()
            else f"\nstripped: {info.sanitised.summary_line()}"
        )
        blocking = (
            f"\n⚠ gates: {', '.join(report.blocking_gates)}"
            if report.blocking_gates else ""
        )
        self.app.notify(
            f"{info.slug}: {report.requirements} requirement(s), "
            f"must ceiling {format_percent(report.must_upper_bound)}"
            f"{blocking}{stripped}",
            title="posting added",
            timeout=12,
        )
        self.reload()


def _status_cell(row: ApplicationRow) -> Text:
    if row.flagged:
        return Text(f"⚠ {row.status}", style="bold yellow")
    if row.status in {"offer", "hired"}:
        return Text(row.status, style="bold green")
    if row.status in {"rejected", "no_response", "withdrawn", "offer_declined"}:
        return Text(row.status, style="dim")
    return Text(row.status)
