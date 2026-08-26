"""The dialogs the four screens push: pick one of a list, type a line, paste a
posting, choose two baselines, read the help.

They are deliberately plain. A modal in this app exists to collect one answer
and dismiss with it; anything that needs a layout belongs on a screen.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, ListItem, ListView, Static, TextArea


class Dialog(ModalScreen[Any]):
    """A modal that always answers ``escape`` with ``None``."""

    BINDINGS = [Binding("escape", "dismiss_none", "cancel", show=True)]

    def action_dismiss_none(self) -> None:
        self.dismiss(None)


class ChoiceDialog(Dialog):
    """Pick one value from a short list. Enter accepts, escape cancels."""

    def __init__(self, title: str, choices: Iterable[tuple[str, Any]], *,
                 subtitle: str = "", initial: int = 0):
        super().__init__()
        self.title_text = title
        self.subtitle = subtitle
        self.choices = list(choices)
        self.initial = initial

    def compose(self) -> ComposeResult:
        with Vertical(classes="dialog", id="choice-dialog"):
            yield Label(self.title_text, classes="dialog-title")
            if self.subtitle:
                yield Static(self.subtitle, classes="dialog-subtitle")
            yield ListView(
                *[ListItem(Label(label)) for label, _value in self.choices],
                id="choice-list", initial_index=self.initial or 0,
            )

    def on_mount(self) -> None:
        self.query_one("#choice-list", ListView).focus()

    @on(ListView.Selected, "#choice-list")
    def _picked(self, event: ListView.Selected) -> None:
        index = event.list_view.index or 0
        if 0 <= index < len(self.choices):
            self.dismiss(self.choices[index][1])
        else:
            self.dismiss(None)


class PromptDialog(Dialog):
    """One line of text — a note, a company name, a filter."""

    def __init__(self, title: str, *, placeholder: str = "", value: str = ""):
        super().__init__()
        self.title_text = title
        self.placeholder = placeholder
        self.value = value

    def compose(self) -> ComposeResult:
        with Vertical(classes="dialog", id="prompt-dialog"):
            yield Label(self.title_text, classes="dialog-title")
            yield Input(value=self.value, placeholder=self.placeholder, id="prompt-input")

    def on_mount(self) -> None:
        self.query_one("#prompt-input", Input).focus()

    @on(Input.Submitted, "#prompt-input")
    def _submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        self.dismiss(text or None)


class AddPostingDialog(Dialog):
    """Company, role, an optional URL, and the pasted text of a posting.

    The URL is recorded and never fetched — the service is explicit about that
    and the label here says so, because a field that looks like a downloader is
    one a user will treat as a downloader.
    """

    BINDINGS = [
        Binding("escape", "dismiss_none", "cancel"),
        Binding("ctrl+s", "submit", "add"),
    ]

    def compose(self) -> ComposeResult:
        with Vertical(classes="dialog dialog-wide", id="add-posting-dialog"):
            yield Label("Add a posting", classes="dialog-title")
            yield Input(placeholder="company", id="posting-company")
            yield Input(placeholder="role", id="posting-role")
            yield Input(placeholder="source URL (recorded, never fetched)",
                        id="posting-url")
            yield TextArea("", id="posting-text", soft_wrap=True)
            with Horizontal(classes="dialog-buttons"):
                yield Button("Add  (ctrl+s)", variant="primary", id="posting-add")
                yield Button("Cancel  (esc)", id="posting-cancel")

    def on_mount(self) -> None:
        self.query_one("#posting-company", Input).focus()

    @on(Button.Pressed, "#posting-cancel")
    def _cancel(self) -> None:
        self.dismiss(None)

    @on(Button.Pressed, "#posting-add")
    def _add(self) -> None:
        self.action_submit()

    def action_submit(self) -> None:
        payload = {
            "company": self.query_one("#posting-company", Input).value.strip(),
            "role": self.query_one("#posting-role", Input).value.strip(),
            "source_url": self.query_one("#posting-url", Input).value.strip(),
            "text": self.query_one("#posting-text", TextArea).text,
        }
        self.dismiss(payload)


class ComparePairDialog(Dialog):
    """Two baselines to diff, chosen one after the other from the same list."""

    def __init__(self, baselines: Sequence[tuple[str, str]]):
        super().__init__()
        self.baselines = list(baselines)
        self.first: str | None = None

    def compose(self) -> ComposeResult:
        with Vertical(classes="dialog", id="compare-dialog"):
            yield Label("Compare baselines", classes="dialog-title")
            yield Static("Pick the baseline to measure *from*.", id="compare-hint")
            yield ListView(
                *[ListItem(Label(label)) for label, _path in self.baselines],
                id="compare-list",
            )

    def on_mount(self) -> None:
        self.query_one("#compare-list", ListView).focus()

    @on(ListView.Selected, "#compare-list")
    def _picked(self, event: ListView.Selected) -> None:
        index = event.list_view.index or 0
        if not (0 <= index < len(self.baselines)):
            return
        path = self.baselines[index][1]
        if self.first is None:
            self.first = path
            self.query_one("#compare-hint", Static).update(
                f"from {path} — now pick the baseline to measure *to*.")
            return
        self.dismiss((self.first, path))


class ImageDialog(Dialog):
    """Page 1 of the PDF, rastered inline where the terminal can show it."""

    def __init__(self, image_path, caption: str = ""):
        super().__init__()
        self.image_path = image_path
        self.caption = caption

    def compose(self) -> ComposeResult:
        with Vertical(classes="dialog dialog-wide", id="image-dialog"):
            yield Label(self.caption or "page 1", classes="dialog-title")
            try:
                from textual_image.widget import Image

                yield Image(str(self.image_path), id="page-image")
            except Exception as exc:  # noqa: BLE001 - degrade, never crash
                yield Static(
                    f"inline images are unavailable here ({exc}).\n"
                    f"The rendered page is at {self.image_path}.",
                    id="page-image-fallback",
                )
            yield Static("escape to close", classes="dialog-subtitle")


HELP_TEXT = """\
[b]ats-tailor[/b] — tab with [b]1[/b]–[b]4[/b], [b]?[/b] for this help,
[b]ctrl+p[/b] to change the LLM provider, [b]ctrl+q[/b] to quit.

[b]1 Applications[/b]  the tracker
  enter   open this run in Tailor
  s       set status        n  add a note
  q       quiet-only filter a  add a posting (paste, then fit)
  ctrl+r  reload

[b]2 Tailor[/b]  one run: requirements · what an ATS reads · what was chosen
  p / x   pin / exclude the item under the cursor
  P / X   pin / exclude the bullet under the cursor
  w       reweight the requirement under the cursor (0.25 · 0.5 · 1 · 1.5)
  r       re-run with the overrides as they now stand
  c       compile and check     o  open the PDF in the system viewer
  i       inline raster of page 1 (needs a terminal that can show images)
  l       write this run's pins into its label file
  ctrl+r  reload

[b]3 Index[/b]  the source of truth
  ctrl+s  validate the whole index and save this entity
  ctrl+r  reload from disk (discards unsaved edits)
  The impact panel is where the selected item ranked in every recorded run.

[b]4 Eval[/b]  the measured baselines
  e       run the sweep (engine and provider are asked for)
  d       diff two baselines
  ctrl+r  reload

Below 90 columns the right-hand pane is hidden; the keys still work on
whatever has focus.
"""


class HelpDialog(Dialog):
    def compose(self) -> ComposeResult:
        with Vertical(classes="dialog dialog-wide", id="help-dialog"):
            with VerticalScroll():
                yield Static(HELP_TEXT, id="help-text")
