"""Screen 2 — one run, in three panes.

Left is what the posting asked for and how well it was answered. Centre is what
an ATS actually reads: ``pdftotext -layout`` over the compiled PDF, never the
LaTeX source, because every failure this project exists to catch happens
between the two. Right is what the solver chose and what each choice bought.

Pins and exclusions are written to ``overrides.yaml`` the moment they are
pressed, and ``r`` is what pays for them. Splitting those two is the whole
reason this screen is usable: choosing is instant and re-running is a minute.
"""

from __future__ import annotations

from pathlib import Path

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import DataTable, Static, Tree
from textual.widgets.tree import TreeNode

from .actions import WEIGHTS, SelectionNode, TailorView
from .modals import ChoiceDialog, ImageDialog, VariantDialog

BAND_STYLES = {"covered": "green", "partial": "yellow", "missing": "red"}


class TailorScreen(Vertical):
    """Requirements · preview · selection, for whichever run is open."""

    BINDINGS = [
        Binding("p", "pin_item", "pin item"),
        Binding("x", "exclude_item", "exclude item"),
        Binding("P", "pin_bullet", "pin bullet"),
        Binding("X", "exclude_bullet", "exclude bullet"),
        Binding("w", "reweight", "reweight"),
        Binding("v", "variant", "variant"),
        Binding("r", "rerun", "re-run"),
        Binding("c", "check", "compile+check"),
        Binding("o", "open_pdf", "open PDF"),
        Binding("i", "inline_image", "raster"),
        Binding("l", "to_label", "to label"),
        Binding("ctrl+r", "reload", "reload"),
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.run_id: str | None = None
        self.view: TailorView | None = None

    def compose(self) -> ComposeResult:
        with Horizontal(id="tailor-columns"):
            with Vertical(id="tailor-left", classes="pane"):
                yield Static("", id="tailor-gates")
                yield DataTable(id="tailor-requirements", cursor_type="row")
            with Vertical(id="tailor-centre", classes="pane"):
                yield Static("no run open", id="tailor-header")
                with VerticalScroll(id="tailor-preview-scroll"):
                    yield Static("", id="tailor-preview")
            with Vertical(id="tailor-right", classes="pane"):
                yield Static("", id="tailor-overrides")
                yield Tree("selection", id="tailor-selection")

    def on_mount(self) -> None:
        table = self.query_one("#tailor-requirements", DataTable)
        for label, width in (("req", 10), ("k", 4), ("w", 5), ("cov", 6)):
            table.add_column(label, width=width)
        table.add_column("requirement")
        self.query_one("#tailor-left", Vertical).border_title = "requirements"
        self.query_one("#tailor-centre", Vertical).border_title = "what an ATS reads"
        self.query_one("#tailor-right", Vertical).border_title = "selected"
        self.query_one("#tailor-selection", Tree).show_root = False

    # ── loading ──
    def open_run(self, run_id: str) -> None:
        self.run_id = run_id
        self.reload()

    def action_reload(self) -> None:
        self.reload()

    def reload(self) -> None:
        run_id = self.run_id
        if not run_id:
            self.app.notify("open a run from Applications first", severity="warning")
            return
        self.app.run_task(
            f"loading {run_id}",
            lambda: self.app.actions.tailor_view(run_id),
            then=self._show,
        )

    def _show(self, view: TailorView) -> None:
        self.view = view
        self.query_one("#tailor-header", Static).update(
            f"[b]{view.company} · {view.role}[/b]  {view.header()}")
        gates = self.query_one("#tailor-gates", Static)
        lines = [view.gates_summary]
        lines += [f"[yellow]⚠ {name}[/yellow]: {why}" for name, why in view.gates]
        gates.update("\n".join(lines))

        table = self.query_one("#tailor-requirements", DataTable)
        table.clear()
        for row in view.requirements:
            style = BAND_STYLES[row.band]
            table.add_row(
                Text(row.requirement_id[:10], style=style),
                Text(row.kind[:4], style="dim"),
                Text(f"{row.weight:g}", style="dim"),
                Text(f"{row.coverage:.2f}", style=f"bold {style}"),
                Text(row.text, style=style),
                key=row.requirement_id,
            )

        preview = self.query_one("#tailor-preview", Static)
        if view.preview:
            preview.update(Text(view.preview))
        else:
            preview.update(
                f"[dim]no preview: {view.preview_error or 'nothing compiled yet'}"
                f"\npress [b]c[/b] to compile and check this run.[/dim]")

        self.query_one("#tailor-overrides", Static).update(
            f"[dim]overrides:[/dim] {view.overrides_summary}")
        tree = self.query_one("#tailor-selection", Tree)
        # Pinning re-reads the run, so the cursor has to survive the rebuild:
        # a user pinning an item and then excluding its neighbour would
        # otherwise be thrown back to the top of the tree between keystrokes.
        was_on = self.current_node()
        tree.clear()
        for section in view.sections:
            _attach(tree.root, section)
        tree.root.expand_all()
        # The root is hidden, so an untouched cursor sits on a node that is not
        # drawn and every key would report "put the cursor on an item".
        tree.cursor_line = 0
        if was_on is not None:
            _restore_cursor(tree, was_on)

    # ── what the cursor is on ──
    def current_requirement(self) -> str | None:
        table = self.query_one("#tailor-requirements", DataTable)
        rows = (self.view.requirements if self.view else [])
        index = table.cursor_row
        return rows[index].requirement_id if 0 <= index < len(rows) else None

    def current_node(self) -> SelectionNode | None:
        tree = self.query_one("#tailor-selection", Tree)
        node = tree.cursor_node
        return node.data if node is not None and isinstance(node.data, SelectionNode) else None

    def _item_under_cursor(self) -> str | None:
        node = self.current_node()
        if node is None:
            return None
        return node.item_id or (node.node_id if node.kind == "item" else None)

    def _bullet_under_cursor(self) -> str | None:
        node = self.current_node()
        return node.node_id if node is not None and node.kind == "bullet" else None

    # ── overrides ──
    def _require_run(self) -> str | None:
        if not self.run_id:
            self.app.notify("open a run from Applications first", severity="warning")
        return self.run_id

    def _override(self, what: str, call) -> None:
        run_id = self._require_run()
        if not run_id:
            return
        self.app.run_task(what, call, then=self._after_override)

    def _after_override(self, changed: str) -> None:
        self.app.notify(f"{changed} — press [b]r[/b] to re-run")
        run_id = self.run_id
        if run_id:
            self.app.run_task(
                "reading overrides",
                lambda: self.app.actions.tailor_view(run_id),
                then=self._show,
            )

    def action_pin_item(self) -> None:
        item_id, run_id = self._item_under_cursor(), self.run_id
        if not item_id:
            self.app.notify("put the cursor on an item in the right pane",
                            severity="warning")
            return
        self._override(f"pinning {item_id}",
                       lambda: self.app.actions.pin_item(run_id, item_id))

    def action_exclude_item(self) -> None:
        item_id, run_id = self._item_under_cursor(), self.run_id
        if not item_id:
            self.app.notify("put the cursor on an item in the right pane",
                            severity="warning")
            return
        self._override(f"excluding {item_id}",
                       lambda: self.app.actions.exclude_item(run_id, item_id))

    def action_pin_bullet(self) -> None:
        bullet_id, run_id = self._bullet_under_cursor(), self.run_id
        if not bullet_id:
            self.app.notify("put the cursor on a bullet in the right pane",
                            severity="warning")
            return
        self._override(f"pinning {bullet_id}",
                       lambda: self.app.actions.pin_bullet(run_id, bullet_id))

    def action_exclude_bullet(self) -> None:
        bullet_id, run_id = self._bullet_under_cursor(), self.run_id
        if not bullet_id:
            self.app.notify("put the cursor on a bullet in the right pane",
                            severity="warning")
            return
        self._override(f"excluding {bullet_id}",
                       lambda: self.app.actions.exclude_bullet(run_id, bullet_id))

    def action_reweight(self) -> None:
        requirement_id, run_id = self.current_requirement(), self.run_id
        if not requirement_id or not run_id:
            self.app.notify("put the cursor on a requirement", severity="warning")
            return

        def chosen(weight: float | None) -> None:
            if weight is None:
                return
            self._override(
                f"{requirement_id} → {weight:g}",
                lambda: self.app.actions.reweight(run_id, requirement_id, weight),
            )

        self.app.push_screen(
            ChoiceDialog(
                f"Weight for {requirement_id}",
                [(f"{value:g}", value) for value in WEIGHTS],
                subtitle="applies to this application only",
                initial=2,
            ),
            chosen,
        )

    # ── variants ──
    def action_variant(self) -> None:
        """Propose wording for the bullet under the cursor, for this archetype."""
        bullet_id, run_id = self._bullet_under_cursor(), self.run_id
        if not bullet_id or not run_id:
            self.app.notify("put the cursor on a bullet in the right pane",
                            severity="warning")
            return
        self._propose(run_id, bullet_id, None)

    def _propose(self, run_id: str, bullet_id: str, text: str | None) -> None:
        self.app.run_task(
            f"proposing wording for {bullet_id}",
            lambda: self.app.actions.propose_variant(run_id, bullet_id, text),
            then=lambda proposal: self._show_variant(run_id, bullet_id, proposal),
            on_error=lambda outcome: self._no_proposal(run_id, bullet_id, outcome),
        )

    def _no_proposal(self, run_id: str, bullet_id: str, outcome) -> bool:
        """A provider with nothing to say is an invitation to write it yourself.

        ``provider_unavailable`` is the ordinary case on a laptop with no model
        running, and refusing to open the panel would make the feature reachable
        only to people who have one. Every other error is a real error.
        """
        if outcome.code != "provider_unavailable":
            return True
        view = self.view
        source = ""
        for section in (view.sections if view else []):
            for item in section.children:
                for bullet in item.children:
                    if bullet.node_id == bullet_id:
                        source = bullet.label
        self.app.push_screen(
            VariantDialog(
                _blank_proposal(bullet_id, view.archetype if view else "", source),
                subtitle=f"{outcome.message} — press e to write one yourself",
            ),
            lambda answer: self._variant_answer(run_id, bullet_id, answer),
        )
        return False

    def _show_variant(self, run_id: str, bullet_id: str, proposal) -> None:
        self.app.push_screen(
            VariantDialog(proposal),
            lambda answer: self._variant_answer(run_id, bullet_id, answer),
        )

    def _variant_answer(self, run_id: str, bullet_id: str, answer) -> None:
        if not answer:
            return
        action, payload = answer
        if action == "recheck":
            if payload:
                self._propose(run_id, bullet_id, payload)
            return
        if action == "accept":
            self.app.run_task(
                f"accepting {payload}",
                lambda: self.app.actions.accept_proposal(payload),
                then=lambda message: self.app.notify(
                    f"{message} — press [b]r[/b] to re-run with it",
                    title="variant"),
            )
            return
        self.app.run_task(
            f"rejecting {payload}",
            lambda: self.app.actions.reject_proposal(payload),
            then=lambda message: self.app.notify(message, title="variant"),
        )

    # ── the slow three ──
    def action_rerun(self) -> None:
        run_id = self._require_run()
        if not run_id:
            return
        self.app.run_task(
            f"re-running {run_id}",
            lambda: self.app.actions.rerun(run_id),
            then=self._after_run,
        )

    def _after_run(self, summary: str) -> None:
        self.app.notify(summary, title="re-run")
        self.reload()
        self.app.refresh_applications()

    def action_check(self) -> None:
        run_id = self._require_run()
        if not run_id:
            return
        self.app.run_task(
            f"compiling and checking {run_id}",
            lambda: self.app.actions.check(run_id),
            then=self._after_run,
        )

    def action_open_pdf(self) -> None:
        run_id = self._require_run()
        if not run_id:
            return
        self.app.run_task(
            "opening the PDF",
            lambda: self.app.actions.open_pdf(run_id),
            then=lambda path: self.app.notify(f"opened {path}"),
        )

    def action_inline_image(self) -> None:
        run_id = self._require_run()
        if not run_id:
            return
        out_dir = Path(self.app.scratch_dir)
        self.app.run_task(
            "rendering page 1",
            lambda: self.app.actions.page_image(run_id, out_dir),
            then=self._show_image,
        )

    def _show_image(self, path: Path) -> None:
        view = self.view
        caption = f"{view.company} · {view.role}" if view else str(path.name)
        self.app.push_screen(ImageDialog(path, caption))

    def action_to_label(self) -> None:
        run_id = self._require_run()
        if not run_id:
            return
        self.app.run_task(
            "promoting pins into the label",
            lambda: self.app.actions.labels_from_run(run_id),
            then=lambda message: self.app.notify(message, title="labels"),
        )

    @on(Tree.NodeSelected, "#tailor-selection")
    def _node_selected(self, event: Tree.NodeSelected) -> None:
        node = event.node.data
        if isinstance(node, SelectionNode) and node.kind == "bullet":
            self.app.notify(node.label, title=node.item_id, timeout=6)


def _blank_proposal(bullet_id: str, archetype: str, source: str):
    """An empty proposal, so the panel can open with nothing in it.

    Not stored anywhere: it exists for the seconds between "no model answered"
    and the user typing wording of their own, which then goes through
    ``propose_variant`` and the guardrail like any other.
    """
    from ..service import ProposalInfo

    return ProposalInfo(
        id="", bullet_id=bullet_id, archetype=archetype or "?", text="",
        status="draft", source=source,
    )


def _restore_cursor(tree: Tree, wanted: SelectionNode) -> None:
    """Put the cursor back on the node with the same kind and id, if it is
    still there — after a re-run it may not be, and the top is then right."""
    line = 0
    while True:
        node = tree.get_node_at_line(line)
        if node is None:
            return
        data = node.data
        if (isinstance(data, SelectionNode)
                and (data.kind, data.node_id) == (wanted.kind, wanted.node_id)):
            tree.cursor_line = line
            return
        line += 1


def _attach(parent: TreeNode, node: SelectionNode) -> TreeNode:
    if node.children:
        branch = parent.add(node.label, data=node, expand=True)
        for child in node.children:
            _attach(branch, child)
        return branch
    return parent.add_leaf(node.label, data=node)
