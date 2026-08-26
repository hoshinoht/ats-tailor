"""Screen 3 — the index, which is the only thing in this repository that is a
fact rather than a derivation.

Left is the tree, centre is a form for whatever is selected, right is the one
question an editor actually has: *what does changing this affect?* — answered
by where the item ranked in every run on record. An item no run has ever
selected is an item to rewrite, and there is nowhere else that shows it.

Bullets are edited as ``<bullet id> :: <text>``, one per line. The id is
visible and load-bearing: ``index_put`` merges bullets by id, so keeping the id
in front of the text is what makes editing one bullet not a restatement of the
other five. Metrics are shown beside them and never edited here — they are
derived from the text, and a metric typed by hand is a claim with no source.
"""

from __future__ import annotations

from typing import Any, Mapping

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import DataTable, Input, Label, Static, TextArea, Tree
from textual.widgets.tree import TreeNode

#: kind → the fields the form shows, as ``(label, dotted path, widget kind)``.
#: ``list`` fields round-trip as a comma-separated line, which is how a person
#: reads a skills list and how the YAML stores it.
FORMS: dict[str, tuple[tuple[str, str, str], ...]] = {
    "project": (
        ("name", "name", "text"),
        ("type", "type", "text"),
        ("repo", "repo", "text"),
        ("start", "period.start", "text"),
        ("end", "period.end", "text"),
        ("summary", "summary", "text"),
        ("skills", "skills", "list"),
        ("keywords", "keywords", "list"),
    ),
    "role": (
        ("company", "company", "text"),
        ("title", "title", "text"),
        ("type", "type", "text"),
        ("start", "period.start", "text"),
        ("end", "period.end", "text"),
        ("skills", "skills", "list"),
        ("keywords", "keywords", "list"),
    ),
    "skill": (
        ("name", "name", "text"),
        ("proficiency", "proficiency", "text"),
        ("acronym", "acronym", "text"),
        ("aliases", "aliases", "list"),
    ),
    "certification": (
        ("name", "name", "text"),
        ("issuer", "issuer", "text"),
        ("date", "date", "text"),
        ("skills", "skills", "list"),
        ("keywords", "keywords", "list"),
        ("relevance", "relevance", "list"),
    ),
}

#: Which kinds carry bullets, and so get the TextArea and the impact panel.
BULLET_KINDS = ("project", "role")

BULLET_SEPARATOR = " :: "


class IndexScreen(Vertical):
    """Tree · form · impact."""

    BINDINGS = [
        Binding("ctrl+s", "save", "save"),
        Binding("ctrl+r", "reload", "reload"),
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.kind: str = ""
        self.entity_id: str = ""
        self.entity: dict[str, Any] = {}

    def compose(self) -> ComposeResult:
        with Horizontal(id="index-columns"):
            with Vertical(id="index-left", classes="pane"):
                yield Tree("index", id="index-tree")
            with VerticalScroll(id="index-centre", classes="pane"):
                yield Static("select an entity on the left", id="index-title")
                yield Vertical(id="index-fields")
                yield Label("bullets  (id :: text, one per line)", id="index-bullets-label")
                yield TextArea("", id="index-bullets", soft_wrap=True)
                yield Static("", id="index-metrics")
                yield Static("", id="index-errors")
            with Vertical(id="index-right", classes="pane"):
                yield DataTable(id="index-impact", cursor_type="row")

    def on_mount(self) -> None:
        self.query_one("#index-left", Vertical).border_title = "index"
        self.query_one("#index-centre", VerticalScroll).border_title = "entity"
        self.query_one("#index-right", Vertical).border_title = "impact"
        table = self.query_one("#index-impact", DataTable)
        for label, width in (("run", 14), ("when", 11), ("company", 14),
                             ("rank", 5), ("gain", 7)):
            table.add_column(label, width=width)
        self.query_one("#index-tree", Tree).show_root = False
        self.reload()

    # ── the tree ──
    def action_reload(self) -> None:
        self.reload()

    def reload(self) -> None:
        self.app.run_task(
            "reading the index",
            self.app.actions.index_tree,
            then=self._build_tree,
        )

    def _build_tree(self, groups) -> None:
        tree = self.query_one("#index-tree", Tree)
        tree.clear()
        for kind, label, entries in groups:
            branch = tree.root.add(f"{label}  ({len(entries)})", expand=False)
            for entity_id, entity_label in entries:
                branch.add_leaf(f"{entity_id}  ·  {entity_label}",
                                data=(kind, entity_id))
        tree.root.expand()

    @on(Tree.NodeHighlighted, "#index-tree")
    def _highlighted(self, event: Tree.NodeHighlighted) -> None:
        self._open(event.node)

    @on(Tree.NodeSelected, "#index-tree")
    def _selected(self, event: Tree.NodeSelected) -> None:
        self._open(event.node)

    def _open(self, node: TreeNode) -> None:
        data = node.data
        if not isinstance(data, tuple):
            return
        kind, entity_id = data
        if (kind, entity_id) == (self.kind, self.entity_id):
            return
        self.app.run_task(
            f"reading {kind} {entity_id}",
            lambda: self.app.actions.index_get(kind, entity_id),
            then=lambda payload: self._show(kind, entity_id, payload),
        )

    # ── the form ──
    def _show(self, kind: str, entity_id: str, payload: Mapping[str, Any]) -> None:
        self.kind, self.entity_id, self.entity = kind, entity_id, dict(payload)
        path = self.app.actions.entity_path(kind, entity_id)
        self.query_one("#index-title", Static).update(
            f"[b]{kind}[/b]  {entity_id}   [dim]{path}[/dim]")
        self.run_worker(self._rebuild_fields(kind), exclusive=True, group="form")

        bullets = self.query_one("#index-bullets", TextArea)
        label = self.query_one("#index-bullets-label", Label)
        metrics = self.query_one("#index-metrics", Static)
        rows = list(self.entity.get("bullets") or [])
        if kind in BULLET_KINDS:
            label.display = True
            bullets.display = True
            bullets.text = "\n".join(
                f"{row.get('id', '')}{BULLET_SEPARATOR}{row.get('text', '')}"
                for row in rows
            )
            metrics.update(_metrics_block(rows))
        else:
            label.display = False
            bullets.display = False
            bullets.text = ""
            metrics.update("")
        self.query_one("#index-errors", Static).update("")
        self._load_impact()

    async def _rebuild_fields(self, kind: str) -> None:
        """Swap the form's widgets for this kind's.

        Awaited rather than fired and forgotten: two entities of the same kind
        have the same field ids, so mounting the new inputs before the old ones
        are gone is a duplicate-id error, not a repaint.
        """
        fields = self.query_one("#index-fields", Vertical)
        await fields.remove_children()
        widgets: list = []
        for label, dotted, widget in FORMS.get(kind, ()):
            value = _get(self.entity, dotted)
            text = ", ".join(str(item) for item in value) if widget == "list" else (
                "" if value is None else str(value))
            widgets.append(Label(label, classes="field-label"))
            widgets.append(
                Input(value=text, id=_field_id(dotted), classes="field-input"))
        if widgets:
            await fields.mount_all(widgets)

    def _load_impact(self) -> None:
        table = self.query_one("#index-impact", DataTable)
        table.clear()
        if self.kind not in BULLET_KINDS or not self.entity_id:
            self.query_one("#index-right", Vertical).border_title = "impact"
            return
        entity_id = self.entity_id
        self.app.run_task(
            f"where {entity_id} ranked",
            lambda: self.app.actions.item_usage(entity_id),
            then=self._show_impact,
        )

    def _show_impact(self, usage) -> None:
        table = self.query_one("#index-impact", DataTable)
        table.clear()
        picked = 0
        for row in usage:
            if row.selected:
                picked += 1
            table.add_row(
                row.run_id[:12],
                (row.created_at or "")[:10],
                (row.company or "")[:14],
                "—" if row.rank is None else str(row.rank),
                "—" if row.gain is None else f"{row.gain:.3f}",
            )
        self.query_one("#index-right", Vertical).border_title = (
            f"impact — selected in {picked}/{len(usage)} run(s)")

    # ── saving ──
    def action_save(self) -> None:
        if not self.kind or not self.entity_id:
            self.app.notify("select an entity first", severity="warning")
            return
        try:
            patch = self._patch()
        except ValueError as exc:
            self._errors([str(exc)])
            return
        if not patch:
            self.app.notify("nothing changed")
            return
        kind, entity_id = self.kind, self.entity_id
        self.app.run_task(
            f"saving {kind} {entity_id}",
            lambda: self.app.actions.index_put(kind, entity_id, patch),
            then=self._saved,
            on_error=self._failed,
        )

    def _patch(self) -> dict[str, Any]:
        patch: dict[str, Any] = {}
        for _label, dotted, widget in FORMS.get(self.kind, ()):
            raw = self.query_one(f"#{_field_id(dotted)}", Input).value
            value: Any
            if widget == "list":
                value = [part.strip() for part in raw.split(",") if part.strip()]
            else:
                value = raw.strip() or None
            if value != _get(self.entity, dotted):
                _set(patch, dotted, value)
        if self.kind in BULLET_KINDS:
            bullets = _parse_bullets(
                self.query_one("#index-bullets", TextArea).text,
                self.entity.get("bullets") or [],
            )
            if bullets is not None:
                patch["bullets"] = bullets
        return patch

    def _saved(self, message: str) -> None:
        self.app.notify(message, title="index")
        self.query_one("#index-errors", Static).update("[green]saved[/green]")
        kind, entity_id = self.kind, self.entity_id
        self.kind = ""  # force _open to reload the entity it just wrote
        self.app.run_task(
            "re-reading the entity",
            lambda: self.app.actions.index_get(kind, entity_id),
            then=lambda payload: self._show(kind, entity_id, payload),
        )

    def _failed(self, outcome) -> bool:
        """Validation errors belong beside the form, not in a toast that fades."""
        self._errors([outcome.message, *outcome.details])
        return outcome.code != "validation_failed"  # let other codes toast too

    def _errors(self, lines) -> None:
        body = "\n".join(f"• {line}" for line in lines if line)
        self.query_one("#index-errors", Static).update(
            f"[red]{body}[/red]" if body else "")


# ── helpers ──────────────────────────────────────────────────────────────────
def _field_id(dotted: str) -> str:
    return "field-" + dotted.replace(".", "-")


def _get(payload: Mapping[str, Any], dotted: str) -> Any:
    current: Any = payload
    for part in dotted.split("."):
        if not isinstance(current, Mapping):
            return None
        current = current.get(part)
    return current


def _set(payload: dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    for part in parts[:-1]:
        payload = payload.setdefault(part, {})
    payload[parts[-1]] = value


def _metrics_block(rows) -> str:
    lines = [
        f"[dim]{row.get('id', '')}[/dim]  {', '.join(row.get('metrics') or []) or '—'}"
        for row in rows
    ]
    return "[dim]metrics (read-only)[/dim]\n" + ("\n".join(lines) or "—")


def _parse_bullets(text: str, current) -> list[dict[str, Any]] | None:
    """The TextArea back into a bullets patch, or ``None`` when nothing moved.

    A line without an id is a new bullet the writer has not named yet, and the
    service requires ``id``/``kind``/``text`` for one — so it is refused here,
    where the message can say which line, rather than as a schema error.
    """
    existing = {str(row.get("id")): row for row in current}
    parsed: list[dict[str, Any]] = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        bullet_id, separator, body = line.partition(BULLET_SEPARATOR.strip())
        bullet_id = bullet_id.strip()
        if not separator or not bullet_id:
            raise ValueError(
                f"line {number}: expected '<bullet id> :: text', got {line.strip()!r}")
        if bullet_id not in existing:
            raise ValueError(
                f"line {number}: {bullet_id!r} is not a bullet of this entity; "
                f"new bullets need a kind, so add them in the YAML")
        parsed.append({"id": bullet_id, "text": body.strip()})
    changed = [
        row for row in parsed
        if str(existing[row["id"]].get("text") or "") != row["text"]
    ]
    return changed or None
