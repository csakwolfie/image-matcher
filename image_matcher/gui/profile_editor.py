"""
`ProfileEditorWindow`: másodlagos (`Toplevel`) ablak a profilok grafikus
szerkesztéséhez. Minden `config.TUNABLE_KEYS` mezőhöz egy felülírás-
checkbox + a `profile_fields.FIELD_SPECS` szerinti típusos widget + a
jelenlegi effektív érték + egy-mondatos leírás tartozik. A mentés a
`config.save_profile()`-t hívja (mindig `~/.image_matcher/profiles/`-ba).
"""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk
from typing import Any, Callable, Dict, List, Optional

from ..config import (
    TUNABLE_KEYS, load_default_config, load_profile_overrides, list_profiles, save_profile,
)
from ..i18n import t
from .profile_fields import BASIC_KEYS, FIELD_SPECS

_DETECTOR_CHOICES = ("SIFT", "AKAZE", "ORB", "BRISK")


class ProfileEditorWindow(tk.Toplevel):
    def __init__(self, master: tk.Misc, on_saved: Optional[Callable[[], None]] = None):
        super().__init__(master)
        self.title(t("gui.profile_editor.window_title"))
        self.geometry("820x700")
        self._on_saved = on_saved
        self._effective_cfg = load_default_config()

        self._override_vars: Dict[str, tk.BooleanVar] = {}
        self._value_vars: Dict[str, tk.StringVar] = {}
        self._bool_vars: Dict[str, tk.BooleanVar] = {}
        self._detector_order: List[str] = list(self._effective_cfg.detector_priority)
        self._row_frames: Dict[str, ttk.Frame] = {}

        self._build_header()
        self._build_field_list()
        self._build_footer()
        self._reset_to_effective()
        self._apply_advanced_visibility()

    # ------------------------------------------------------------------
    # Fejléc: meglévő profil betöltése / új profil / név / leírás
    # ------------------------------------------------------------------

    def _build_header(self) -> None:
        header = ttk.Frame(self, padding=10)
        header.pack(fill="x")
        header.columnconfigure(1, weight=1)

        ttk.Label(header, text=t("gui.profile_editor.label_load_existing")).grid(
            row=0, column=0, padx=5, pady=3, sticky="w")
        self._load_var = tk.StringVar()
        self._load_combo = ttk.Combobox(header, textvariable=self._load_var, state="readonly",
                                         values=[name for name, _ in list_profiles()])
        self._load_combo.grid(row=0, column=1, padx=5, pady=3, sticky="w")
        self._load_combo.bind("<<ComboboxSelected>>", lambda _e: self._load_profile_into_form(self._load_var.get()))
        ttk.Button(header, text=t("gui.profile_editor.button_new"), command=self._reset_to_effective).grid(
            row=0, column=2, padx=5, pady=3)

        ttk.Label(header, text=t("gui.profile_editor.label_name")).grid(
            row=1, column=0, padx=5, pady=3, sticky="w")
        self._name_var = tk.StringVar()
        ttk.Entry(header, textvariable=self._name_var).grid(row=1, column=1, padx=5, pady=3, sticky="ew")

        ttk.Label(header, text=t("gui.profile_editor.label_description")).grid(
            row=2, column=0, padx=5, pady=3, sticky="w")
        self._description_var = tk.StringVar()
        ttk.Entry(header, textvariable=self._description_var).grid(
            row=2, column=1, columnspan=2, padx=5, pady=3, sticky="ew")

        self._advanced_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(header, text=t("gui.profile_editor.toggle_advanced"), variable=self._advanced_var,
                         command=self._apply_advanced_visibility).grid(
            row=3, column=0, columnspan=3, padx=5, pady=(6, 0), sticky="w")

    # ------------------------------------------------------------------
    # Görgethető mezőlista
    # ------------------------------------------------------------------

    def _build_field_list(self) -> None:
        container = ttk.Frame(self)
        container.pack(fill="both", expand=True, padx=10, pady=5)

        canvas = tk.Canvas(container, borderwidth=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        inner = ttk.Frame(canvas)
        inner.bind("<Configure>", lambda _e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        for spec in FIELD_SPECS:
            self._build_field_row(inner, spec)

    def _build_field_row(self, parent: ttk.Frame, spec) -> None:
        row = ttk.Frame(parent, padding=(4, 4))
        row.pack(fill="x", anchor="w")
        self._row_frames[spec.key] = row

        override_var = tk.BooleanVar(value=False)
        self._override_vars[spec.key] = override_var
        ttk.Checkbutton(row, variable=override_var).grid(row=0, column=0, rowspan=2, padx=(0, 6))

        ttk.Label(row, text=spec.key, font=("TkDefaultFont", 9, "bold")).grid(
            row=0, column=1, sticky="w")

        widget_cell = ttk.Frame(row)
        widget_cell.grid(row=0, column=2, padx=(10, 10), sticky="w")
        self._build_value_widget(widget_cell, spec)

        effective_value = getattr(self._effective_cfg, spec.key)
        effective_label = ttk.Label(row, foreground="#777777")
        effective_label.grid(row=0, column=3, sticky="w")
        effective_label.config(text=t("gui.profile_editor.label_effective", value=effective_value))

        help_text = t(f"gui.profile_field.{spec.key}.help")
        ttk.Label(row, text=help_text, foreground="#555555", wraplength=700,
                  font=("TkDefaultFont", 8)).grid(row=1, column=1, columnspan=3, sticky="w", pady=(0, 4))

    def _build_value_widget(self, parent: ttk.Frame, spec) -> None:
        key = spec.key
        if spec.kind == "bool":
            var = tk.BooleanVar()
            self._bool_vars[key] = var
            ttk.Checkbutton(parent, variable=var).pack()
            return

        if spec.kind == "detector_list":
            self._detector_listbox = tk.Listbox(parent, height=4, width=10, exportselection=False)
            self._detector_listbox.pack(side="left")
            btns = ttk.Frame(parent)
            btns.pack(side="left", padx=(4, 0))
            ttk.Button(btns, text=t("gui.profile_editor.button_up"), width=5,
                       command=self._move_detector_up).pack()
            ttk.Button(btns, text=t("gui.profile_editor.button_down"), width=5,
                       command=self._move_detector_down).pack()
            return

        var = tk.StringVar()
        self._value_vars[key] = var
        if spec.kind == "slider":
            scale = ttk.Scale(parent, from_=spec.minimum, to=spec.maximum, orient="horizontal", length=180,
                               command=lambda v, k=key: self._on_slider_moved(k, v))
            scale.pack(side="left")
            self._scales = getattr(self, "_scales", {})
            self._scales[key] = scale
            entry = ttk.Entry(parent, textvariable=var, width=8)
            entry.pack(side="left", padx=(6, 0))
            entry.bind("<FocusOut>", lambda _e, k=key: self._on_entry_committed(k))
            entry.bind("<Return>", lambda _e, k=key: self._on_entry_committed(k))
        else:  # int_entry / float_entry
            ttk.Entry(parent, textvariable=var, width=12).pack(side="left")

    # ------------------------------------------------------------------
    # Csúszka <-> beviteli mező szinkron
    # ------------------------------------------------------------------

    def _on_slider_moved(self, key: str, raw_value: str) -> None:
        value = float(raw_value)
        if self._is_int_field(key):
            value = round(value)
        self._value_vars[key].set(str(value))

    def _on_entry_committed(self, key: str) -> None:
        spec = self._spec_by_key(key)
        try:
            value = float(self._value_vars[key].get())
        except ValueError:
            value = spec.minimum
        value = max(spec.minimum, min(spec.maximum, value))
        if self._is_int_field(key):
            value = round(value)
        self._value_vars[key].set(str(value))
        self._scales[key].set(value)

    def _is_int_field(self, key: str) -> bool:
        return isinstance(getattr(self._effective_cfg, key), int) and not isinstance(
            getattr(self._effective_cfg, key), bool)

    def _spec_by_key(self, key: str):
        return next(s for s in FIELD_SPECS if s.key == key)

    # ------------------------------------------------------------------
    # detector_priority sorrendezés
    # ------------------------------------------------------------------

    def _refresh_detector_listbox(self) -> None:
        self._detector_listbox.delete(0, "end")
        for name in self._detector_order:
            self._detector_listbox.insert("end", name)

    def _move_detector_up(self) -> None:
        self._move_detector(-1)

    def _move_detector_down(self) -> None:
        self._move_detector(1)

    def _move_detector(self, direction: int) -> None:
        sel = self._detector_listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        new_idx = idx + direction
        if not (0 <= new_idx < len(self._detector_order)):
            return
        self._detector_order[idx], self._detector_order[new_idx] = (
            self._detector_order[new_idx], self._detector_order[idx])
        self._refresh_detector_listbox()
        self._detector_listbox.selection_set(new_idx)

    # ------------------------------------------------------------------
    # Egyszerű / Haladó láthatóság
    # ------------------------------------------------------------------

    def _apply_advanced_visibility(self) -> None:
        show_advanced = self._advanced_var.get()
        for key, row in self._row_frames.items():
            if key in BASIC_KEYS or show_advanced:
                row.pack(fill="x", anchor="w")
            else:
                row.pack_forget()

    # ------------------------------------------------------------------
    # Form állapot: reset / betöltés
    # ------------------------------------------------------------------

    def _reset_to_effective(self) -> None:
        self._name_var.set("")
        self._description_var.set("")
        self._detector_order = list(self._effective_cfg.detector_priority)
        self._refresh_detector_listbox()
        for key in TUNABLE_KEYS:
            if key not in self._override_vars:
                continue
            self._override_vars[key].set(False)
            value = getattr(self._effective_cfg, key)
            if key in self._bool_vars:
                self._bool_vars[key].set(bool(value))
            elif key in self._value_vars:
                self._value_vars[key].set(str(value))
                if key in getattr(self, "_scales", {}):
                    self._scales[key].set(value)

    def _load_profile_into_form(self, name: str) -> None:
        if not name:
            return
        overrides, description = load_profile_overrides(name)
        self._reset_to_effective()
        self._name_var.set(name)
        self._description_var.set(description)
        for key, value in overrides.items():
            if key not in self._override_vars:
                continue
            self._override_vars[key].set(True)
            if key == "detector_priority":
                self._detector_order = list(value)
                self._refresh_detector_listbox()
            elif key in self._bool_vars:
                self._bool_vars[key].set(bool(value))
            elif key in self._value_vars:
                self._value_vars[key].set(str(value))
                if key in getattr(self, "_scales", {}):
                    self._scales[key].set(value)

    # ------------------------------------------------------------------
    # Értékek kiolvasása
    # ------------------------------------------------------------------

    def _collect_overrides(self) -> Dict[str, Any]:
        """Felold minden bejelölt kulcsot a form aktuális értékére. Hibás
        (nem szám) `int_entry`/`float_entry` tartalomnál `ValueError`-t dob
        a kulcs nevével – a hívók (`_on_test_clicked`/`_on_save_clicked`)
        ezt elkapva `messagebox`-ban jelzik, nem hagyják csendben elszállni
        a Tkinter-eseménykezelőt."""
        overrides: Dict[str, Any] = {}
        for key in TUNABLE_KEYS:
            if key not in self._override_vars or not self._override_vars[key].get():
                continue
            if key == "detector_priority":
                overrides[key] = list(self._detector_order)
            elif key in self._bool_vars:
                overrides[key] = self._bool_vars[key].get()
            else:
                target_type = type(getattr(self._effective_cfg, key))
                raw = self._value_vars[key].get()
                try:
                    overrides[key] = target_type(float(raw))
                except ValueError:
                    raise ValueError(t("gui.profile_editor.error_invalid_value", field=key, value=raw)) from None
        return overrides

    # ------------------------------------------------------------------
    # Footer: Teszt / Mentés / Mégse
    # ------------------------------------------------------------------

    def _build_footer(self) -> None:
        footer = ttk.Frame(self, padding=10)
        footer.pack(fill="x")
        ttk.Button(footer, text=t("gui.profile_editor.button_test"), command=self._on_test_clicked).pack(
            side="left")
        ttk.Button(footer, text=t("gui.profile_editor.button_save"), command=self._on_save_clicked).pack(
            side="left", padx=6)
        ttk.Button(footer, text=t("gui.profile_editor.button_cancel"), command=self.destroy).pack(side="left")

    def _on_test_clicked(self) -> None:
        try:
            overrides = self._collect_overrides()
        except ValueError as e:
            messagebox.showerror(t("gui.profile_editor.save_error_title"), str(e))
            return
        preview_cfg = self._effective_cfg.replace(**overrides)
        self._show_preview(preview_cfg)

    def _show_preview(self, preview_cfg) -> None:
        win = tk.Toplevel(self)
        win.title(t("gui.profile_editor.preview_window_title"))
        win.geometry("480x600")
        text = tk.Text(win, wrap="word")
        text.pack(fill="both", expand=True, padx=8, pady=8)
        for key in TUNABLE_KEYS:
            text.insert("end", f"{key:35s}: {getattr(preview_cfg, key)}\n")
        text.configure(state="disabled")
        ttk.Button(win, text=t("gui.profile_editor.button_close"), command=win.destroy).pack(pady=(0, 8))

    def _on_save_clicked(self) -> None:
        name = self._name_var.get().strip()
        if not name:
            messagebox.showerror(t("gui.profile_editor.save_error_title"),
                                  t("gui.profile_editor.error_name_required"))
            return
        try:
            overrides = self._collect_overrides()
            saved_path = save_profile(name, overrides, self._description_var.get().strip())
        except ValueError as e:
            messagebox.showerror(t("gui.profile_editor.save_error_title"), str(e))
            return
        messagebox.showinfo(t("gui.profile_editor.window_title"),
                             t("gui.profile_editor.save_confirmation", name=name, path=saved_path))
        self._load_combo["values"] = [n for n, _ in list_profiles()]
        if self._on_saved is not None:
            self._on_saved()
