"""
`LivePanel`: a fő ablak tetején lévő élő találat-panel – bal oldalt a
jelenleg feldolgozás alatt álló referencia kép, jobb oldalt az
eredmény (a talált forrás, vagy a legjobb – de elutasított – jelölt),
színes kerettel és jelvénnyel az állapot szerint. A tényleges
képösszeállítás a `live_panel_render.py`-ban van (widget-mentes, PIL-
alapú, unittest-tel tesztelhető) – ez a modul csak a Tkinter-wrapper:
`ImageTk.PhotoImage`-re konvertál és egy `tk.Label`-ben jeleníti meg.

Az `app.py` tölti fel adattal, a `main.ReferenceResult` eseményekből
(lásd ott) – ez a widget maga NEM tud semmit a keresési folyamatról,
csak már feloldott `Path`-eket (vagy `None`-t) fogad be.
"""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import ttk
from typing import Optional

from PIL import ImageTk

from ..i18n import t
from .live_panel_render import BORDER_IDLE, build_card, status_color

_BADGE_KEYS = {
    "found": "gui.live_panel.badge_found",
    "exists": "gui.live_panel.badge_exists",
    "not_found": "gui.live_panel.badge_not_found",
    "near_miss": "gui.live_panel.badge_near_miss",
    "error": "gui.live_panel.badge_error",
}


class LivePanel(tk.Frame):
    def __init__(self, master: tk.Misc):
        super().__init__(master)
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)

        self._reference_caption = ttk.Label(self)
        self._reference_caption.grid(row=0, column=0, sticky="w", padx=4, pady=(0, 2))
        self._left_img_label = tk.Label(self, bd=0)
        self._left_img_label.grid(row=1, column=0, padx=4, pady=(0, 6))

        self._result_caption = ttk.Label(self)
        self._result_caption.grid(row=0, column=1, sticky="w", padx=4, pady=(0, 2))
        self._right_img_label = tk.Label(self, bd=0)
        self._right_img_label.grid(row=1, column=1, padx=4, pady=(0, 6))

        # a Tkinter PhotoImage-et referencia nélkül a garbage collector
        # összeszedi, még ha a widget meg is jeleníti -- ezért a widget
        # élettartamáig itt tartjuk élve a legutóbbi képeket.
        self._left_photo: Optional[ImageTk.PhotoImage] = None
        self._right_photo: Optional[ImageTk.PhotoImage] = None
        self._right_status: Optional[str] = None
        self._right_image_path: Optional[Path] = None

        self.reset()

    def reset(self) -> None:
        self._left_photo = ImageTk.PhotoImage(build_card(None, border_color=BORDER_IDLE))
        self._left_img_label.config(image=self._left_photo)
        self._right_status = None
        self._right_image_path = None
        self._render_right()
        self.refresh_texts()

    def set_reference(self, image_path: Optional[Path]) -> None:
        self._left_photo = ImageTk.PhotoImage(build_card(image_path, border_color=BORDER_IDLE))
        self._left_img_label.config(image=self._left_photo)
        self._right_status = None
        self._right_image_path = None
        self._render_right()

    def set_result(self, status: str, image_path: Optional[Path]) -> None:
        self._right_status = status
        self._right_image_path = image_path
        self._render_right()

    def refresh_texts(self) -> None:
        self._reference_caption.config(text=t("gui.live_panel.reference_label"))
        self._result_caption.config(text=t("gui.live_panel.result_label"))
        self._render_right()

    def _render_right(self) -> None:
        status = self._right_status
        badge_key = _BADGE_KEYS.get(status or "")
        badge_text = t(badge_key) if badge_key else None
        border = status_color(status)
        composed = build_card(self._right_image_path, border_color=border,
                               badge_text=badge_text, badge_color=border)
        self._right_photo = ImageTk.PhotoImage(composed)
        self._right_img_label.config(image=self._right_photo)
