"""
Fő ablak: a jelenlegi CLI-kapcsolókat (--lang, --profile, útvonalak,
--limit/--workers/--top-k, --no-cache/--rebuild-cache/--dry-run) kezelő
Tkinter form, Futtatás/Szünet-Folytatás/Leállítás gombokkal.

A tényleges keresést a `worker.run_search()` indítja háttérszálon,
`main.main(argv)`-t hívva VÁLTOZATLANUL – ez a modul csak a widgeteket és
az eseménykezelőket tartalmazza, widget-et KIZÁRÓLAG a főszálról módosít
(`root.after`-alapú pollozás, lásd `_poll_queue`).
"""

from __future__ import annotations

import os
import queue
import re
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk
from typing import Any, Dict, Optional

from .. import __version__
from ..config import get_default_language, list_profiles, set_default_language
from ..i18n import init_translator, list_available_langs, t
from ..main import FOUND_DIR_NAME, ReferenceResult
from . import argv_builder, worker
from .live_panel import LivePanel
from .profile_editor import ProfileEditorWindow

_PROGRESS_RE = re.compile(r"\[(\d+)/(\d+)\]")

_NO_PROFILE = ""  # a profil-legördülő "nincs profil" opciójának belső értéke


class App:
    def __init__(self, root: tk.Tk):
        self.root = root

        self._current_lang = get_default_language()
        init_translator(self._current_lang)

        self._handle: Optional[worker.SearchHandle] = None
        self._cancel_event: Optional[threading.Event] = None
        self._pause_event: Optional[threading.Event] = None

        self._profile_descriptions: Dict[str, str] = {}

        self._build_widgets()
        self._refresh_texts()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------
    # Widget-felépítés
    # ------------------------------------------------------------------

    def _build_widgets(self) -> None:
        root = self.root
        root.geometry("760x870")
        main = ttk.Frame(root, padding=10)
        main.pack(fill="both", expand=True)
        main.columnconfigure(1, weight=1)

        row = 0

        # --- Élő találat-panel ---
        self.live_panel = LivePanel(main)
        self.live_panel.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(0, 8))
        row += 1

        # --- Nyelv ---
        self.lang_group = ttk.LabelFrame(main)
        self.lang_group.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(0, 6))
        self.lang_group.columnconfigure(1, weight=1)
        self.lang_label = ttk.Label(self.lang_group)
        self.lang_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.lang_var = tk.StringVar(value=self._current_lang)
        self.lang_combo = ttk.Combobox(self.lang_group, textvariable=self.lang_var,
                                        values=list_available_langs(), state="readonly", width=8)
        self.lang_combo.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        self.lang_combo.bind("<<ComboboxSelected>>", self._on_language_changed)
        self.make_default_button = ttk.Button(self.lang_group, command=self._on_make_default_clicked)
        self.make_default_button.grid(row=0, column=2, padx=5, pady=5, sticky="e")
        row += 1

        # --- Profil ---
        self.profile_group = ttk.LabelFrame(main)
        self.profile_group.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(0, 6))
        self.profile_group.columnconfigure(1, weight=1)
        self.profile_label = ttk.Label(self.profile_group)
        self.profile_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.profile_var = tk.StringVar(value=_NO_PROFILE)
        self.profile_combo = ttk.Combobox(self.profile_group, textvariable=self.profile_var,
                                           state="readonly", width=20)
        self.profile_combo.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        self.profile_combo.bind("<<ComboboxSelected>>", lambda _e: self._update_profile_description())
        self.manage_profiles_button = ttk.Button(self.profile_group, command=self._open_profile_editor)
        self.manage_profiles_button.grid(row=0, column=2, padx=5, pady=5, sticky="e")
        self.profile_desc_label = ttk.Label(self.profile_group, wraplength=520, foreground="#555555")
        self.profile_desc_label.grid(row=1, column=0, columnspan=3, padx=5, pady=(0, 5), sticky="w")
        row += 1

        # --- Útvonalak ---
        self.paths_group = ttk.LabelFrame(main)
        self.paths_group.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(0, 6))
        self.paths_group.columnconfigure(1, weight=1)

        self.reference_var = tk.StringVar()
        self.source_var = tk.StringVar()
        self.output_var = tk.StringVar()
        self.cache_var = tk.StringVar()

        self.reference_label = self._add_path_row(self.paths_group, 0, self.reference_var,
                                                    lambda: self._browse_into(self.reference_var))
        self.source_label = self._add_path_row(self.paths_group, 1, self.source_var,
                                                 lambda: self._browse_into(self.source_var))
        self.output_label = self._add_path_row(self.paths_group, 2, self.output_var,
                                                 lambda: self._browse_into(self.output_var))
        self.cache_label = self._add_path_row(self.paths_group, 3, self.cache_var,
                                               lambda: self._browse_into(self.cache_var))
        row += 1

        # --- Futtatásvezérlés ---
        self.run_control_group = ttk.LabelFrame(main)
        self.run_control_group.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(0, 6))

        self.limit_var = tk.StringVar()
        self.workers_var = tk.StringVar(value=str(max(1, os.cpu_count() or 4)))
        self.top_k_var = tk.StringVar()

        self.limit_label = ttk.Label(self.run_control_group)
        self.limit_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        ttk.Entry(self.run_control_group, textvariable=self.limit_var, width=8).grid(
            row=0, column=1, padx=5, pady=5, sticky="w")

        self.workers_label = ttk.Label(self.run_control_group)
        self.workers_label.grid(row=0, column=2, padx=5, pady=5, sticky="w")
        ttk.Entry(self.run_control_group, textvariable=self.workers_var, width=8).grid(
            row=0, column=3, padx=5, pady=5, sticky="w")

        self.top_k_label = ttk.Label(self.run_control_group)
        self.top_k_label.grid(row=0, column=4, padx=5, pady=5, sticky="w")
        ttk.Entry(self.run_control_group, textvariable=self.top_k_var, width=8).grid(
            row=0, column=5, padx=5, pady=5, sticky="w")
        row += 1

        # --- Cache ---
        self.cache_group = ttk.LabelFrame(main)
        self.cache_group.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(0, 6))
        self.no_cache_var = tk.BooleanVar(value=False)
        self.rebuild_cache_var = tk.BooleanVar(value=False)
        self.no_cache_check = ttk.Checkbutton(self.cache_group, variable=self.no_cache_var)
        self.no_cache_check.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.rebuild_cache_check = ttk.Checkbutton(self.cache_group, variable=self.rebuild_cache_var)
        self.rebuild_cache_check.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        row += 1

        # --- Futtatás ---
        self.run_group = ttk.LabelFrame(main)
        self.run_group.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(0, 6))
        self.dry_run_var = tk.BooleanVar(value=False)
        self.dry_run_check = ttk.Checkbutton(self.run_group, variable=self.dry_run_var)
        self.dry_run_check.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        self.run_button = ttk.Button(self.run_group, command=self._on_run_clicked)
        self.run_button.grid(row=0, column=1, padx=5, pady=5)
        self.pause_button = ttk.Button(self.run_group, command=self._on_pause_clicked, state="disabled")
        self.pause_button.grid(row=0, column=2, padx=5, pady=5)
        self.stop_button = ttk.Button(self.run_group, command=self._on_stop_clicked, state="disabled")
        self.stop_button.grid(row=0, column=3, padx=5, pady=5)

        self.version_label = ttk.Label(self.run_group, foreground="#888888")
        self.version_label.grid(row=0, column=4, padx=5, pady=5, sticky="e")
        self.run_group.columnconfigure(4, weight=1)
        row += 1

        # --- Élő kimenet ---
        self.progress_bar = ttk.Progressbar(main, mode="determinate")
        self.progress_bar.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(0, 4))
        row += 1

        self.log_group = ttk.LabelFrame(main)
        self.log_group.grid(row=row, column=0, columnspan=3, sticky="nsew")
        main.rowconfigure(row, weight=1)
        self.log_text = scrolledtext.ScrolledText(self.log_group, height=16, state="disabled", wrap="word")
        self.log_text.pack(fill="both", expand=True, padx=5, pady=5)

        self._populate_profiles()

    def _add_path_row(self, parent: ttk.LabelFrame, row: int, var: tk.StringVar, browse_cmd) -> ttk.Label:
        label = ttk.Label(parent)
        label.grid(row=row, column=0, padx=5, pady=4, sticky="w")
        ttk.Entry(parent, textvariable=var).grid(row=row, column=1, padx=5, pady=4, sticky="ew")
        ttk.Button(parent, text="...", width=4, command=browse_cmd).grid(
            row=row, column=2, padx=5, pady=4, sticky="e")
        return label

    def _populate_profiles(self) -> None:
        profiles = list_profiles()
        self._profile_descriptions = {name: desc for name, desc in profiles}
        self.profile_combo["values"] = [_NO_PROFILE] + [name for name, _ in profiles]
        self._update_profile_description()

    def _update_profile_description(self) -> None:
        name = self.profile_var.get()
        self.profile_desc_label.config(text=self._profile_descriptions.get(name, ""))

    # ------------------------------------------------------------------
    # Szöveg-frissítés (i18n)
    # ------------------------------------------------------------------

    def _refresh_texts(self) -> None:
        self.root.title(t("gui.window_title"))
        self.live_panel.refresh_texts()
        self.lang_group.config(text=t("gui.group_language"))
        self.lang_label.config(text=t("gui.label_language"))
        self.make_default_button.config(text=t("gui.button_make_default"))

        self.profile_group.config(text=t("gui.group_profile"))
        self.profile_label.config(text=t("gui.label_profile"))
        self.manage_profiles_button.config(text=t("gui.profile_editor.manage_button"))

        self.paths_group.config(text=t("gui.group_paths"))
        self.reference_label.config(text=t("gui.label_reference"))
        self.source_label.config(text=t("gui.label_source"))
        self.output_label.config(text=t("gui.label_output"))
        self.cache_label.config(text=t("gui.label_cache"))

        self.run_control_group.config(text=t("gui.group_run_control"))
        self.limit_label.config(text=t("gui.label_limit"))
        self.workers_label.config(text=t("gui.label_workers"))
        self.top_k_label.config(text=t("gui.label_top_k"))

        self.cache_group.config(text=t("gui.group_cache"))
        self.no_cache_check.config(text=t("gui.checkbox_no_cache"))
        self.rebuild_cache_check.config(text=t("gui.checkbox_rebuild_cache"))

        self.run_group.config(text=t("gui.group_run"))
        self.dry_run_check.config(text=t("gui.checkbox_dry_run"))
        self.run_button.config(text=t("gui.button_run"))
        self.pause_button.config(text=t("gui.button_pause"))
        self.stop_button.config(text=t("gui.button_stop"))
        self.version_label.config(text=f"v{__version__}")

        self.log_group.config(text=t("gui.group_log"))

        self._populate_profiles()

    def _on_language_changed(self, _event=None) -> None:
        self._current_lang = self.lang_var.get()
        init_translator(self._current_lang)
        self._refresh_texts()

    def _on_make_default_clicked(self) -> None:
        saved_path = set_default_language(self._current_lang)
        messagebox.showinfo(
            t("gui.window_title"),
            t("gui.language_default_set", lang=self._current_lang, path=saved_path),
        )

    def _open_profile_editor(self) -> None:
        ProfileEditorWindow(self.root, on_saved=self._populate_profiles)

    # ------------------------------------------------------------------
    # Futtatás / Szünet / Leállítás
    # ------------------------------------------------------------------

    def _browse_into(self, var: tk.StringVar) -> None:
        chosen = filedialog.askdirectory()
        if chosen:
            var.set(chosen)

    def _collect_form(self) -> Dict[str, Any]:
        return {
            "lang": self._current_lang,
            "profile": self.profile_var.get() or None,
            "reference": self.reference_var.get(),
            "source": self.source_var.get(),
            "output": self.output_var.get(),
            "cache": self.cache_var.get(),
            "limit": self.limit_var.get(),
            "workers": self.workers_var.get(),
            "top_k": self.top_k_var.get(),
            "no_cache": self.no_cache_var.get(),
            "rebuild_cache": self.rebuild_cache_var.get(),
            "dry_run": self.dry_run_var.get(),
        }

    def _on_run_clicked(self) -> None:
        if self._handle is not None and self._handle.is_alive():
            return
        form = self._collect_form()
        errors = argv_builder.validate_form(form)
        if errors:
            messagebox.showerror(t("gui.validation_error_title"), "\n".join(errors))
            return

        argv = argv_builder.build_argv(form)
        self._cancel_event = threading.Event()
        self._pause_event = threading.Event()
        self._set_running_state(True)
        self.live_panel.reset()
        self._append_log(f"\n$ image-matcher {' '.join(argv)}\n")
        self.progress_bar["value"] = 0
        self._handle = worker.run_search(argv, self._cancel_event, self._pause_event)
        self.root.after(80, self._poll_queue)

    def _on_pause_clicked(self) -> None:
        if self._pause_event is None:
            return
        if self._pause_event.is_set():
            self._pause_event.clear()
            self.pause_button.config(text=t("gui.button_pause"))
        else:
            self._pause_event.set()
            self.pause_button.config(text=t("gui.button_resume"))

    def _on_stop_clicked(self) -> None:
        if self._cancel_event is not None:
            self._cancel_event.set()
        self.stop_button.config(state="disabled")

    def _poll_queue(self) -> None:
        if self._handle is None:
            return
        try:
            while True:
                item = self._handle.queue.get_nowait()
                if isinstance(item, worker.Done):
                    self._on_search_done(item.exit_code, item.error)
                    return
                if isinstance(item, ReferenceResult):
                    self._handle_reference_result(item)
                    continue
                self._append_log(str(item))
        except queue.Empty:
            pass
        self.root.after(80, self._poll_queue)

    def _handle_reference_result(self, result: ReferenceResult) -> None:
        """A `LivePanel` frissítése egy `main.ReferenceResult` eseményből –
        a fájlnév -> `Path` feloldás itt, a GUI-ban történik (a widget
        maga semmit nem tud a keresési mappákról), mert a `matched`
        mezőt tartalmazó mappa a `status`-tól függ: "exists"-nél a
        kimeneti `found/`-ban van (a mentéskor átnevezve), minden más
        esetben a source-mappában, az eredeti fájlnévvel."""
        if result.status == "processing":
            ref_path = Path(self.reference_var.get()) / result.reference
            self.live_panel.set_reference(ref_path if ref_path.is_file() else None)
            return

        image_path: Optional[Path] = None
        if result.matched is not None:
            if result.status == "exists":
                image_path = Path(self.output_var.get()) / FOUND_DIR_NAME / result.matched
            else:
                image_path = Path(self.source_var.get()) / result.matched
            if not image_path.is_file():
                image_path = None
        self.live_panel.set_result(result.status, image_path)

    def _on_search_done(self, exit_code: int, error: Optional[BaseException]) -> None:
        self._set_running_state(False)
        self._handle = None
        self._cancel_event = None
        self._pause_event = None
        if error is not None:
            messagebox.showerror(t("gui.run_error_title"), str(error))

    def _set_running_state(self, running: bool) -> None:
        self.run_button.config(state="disabled" if running else "normal")
        self.pause_button.config(state="normal" if running else "disabled", text=t("gui.button_pause"))
        self.stop_button.config(state="normal" if running else "disabled")

    def _append_log(self, text: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", text)
        self.log_text.see("end")
        self.log_text.configure(state="disabled")
        match = _PROGRESS_RE.search(text)
        if match:
            idx, total = int(match.group(1)), int(match.group(2))
            self.progress_bar["maximum"] = max(total, 1)
            self.progress_bar["value"] = idx

    def _on_close(self) -> None:
        if self._handle is not None and self._handle.is_alive():
            if not messagebox.askyesno(t("gui.confirm_close_title"), t("gui.confirm_close_message")):
                return
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
