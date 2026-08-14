"""
Háttérszálon futó keresés + a `main.main(argv)` kimenetének (print/`_Tee`)
élő streamelése a GUI felé egy `queue.Queue`-n keresztül.

Tkinter-widgetet KIZÁRÓLAG a főszálról szabad módosítani, ezért ez a modul
nem hív semmilyen widget-metódust – csak egy szál-biztos `queue.Queue`-t
tölt, amit a GUI főszála `root.after(...)`-fel pollozva ürít (lásd
`app.py`).
"""

from __future__ import annotations

import queue
import sys
import threading
import traceback
from typing import List, Optional

from ..main import main as run_main


class QueueWriter:
    """
    Írható stream-objektum (`write`/`flush`), ami minden beírt szöveget
    egy `queue.Queue`-ba told. A `main.main()` ezt "látja" konzolnak (a
    `_Tee`-je egyszerre ide ÉS a saját `log_*.txt` fájljába ír,
    változatlanul) – így a GUI log-panelje és a lemezre írt naplófájl
    tartalma garantáltan megegyezik a CLI-vel.

    FIGYELEM: a `sys.stdout` a Python-folyamat EGÉSZÉRE globális, NEM
    szálankénti állapot – amíg egy keresés fut (és `run_search` lecserélte
    a `sys.stdout`-ot erre az objektumra), a folyamat SEMMILYEN más
    szálának SEM szabad `print()`-et hívnia, mert az is ebbe a queue-ba
    kerülne. A `app.py` ezt betartja (a `_poll_queue`/`_append_log` csak
    Tkinter-widget-hívásokat használ, sosem `print()`-et) – de pl. egy
    diagnosztikai szkript, ami `print()`-tel írja ki a queue tartalmát
    MIKÖZBEN a keresés még fut, végtelen visszacsatolási hurkot okoz
    (a saját kiírása visszakerül a queue-ba, amit aztán újra kiír...).
    """

    def __init__(self, line_queue: "queue.Queue[object]"):
        self._queue = line_queue

    def write(self, data: str) -> None:
        if data:
            self._queue.put(data)

    def flush(self) -> None:
        pass


class Done:
    """Sentinel a queue-ban: jelzi, hogy a háttérszál végzett."""

    __slots__ = ("exit_code", "error")

    def __init__(self, exit_code: int, error: Optional[BaseException]):
        self.exit_code = exit_code
        self.error = error


class SearchHandle:
    """A GUI-nak visszaadott fogantyú egy elindított kereséshez."""

    def __init__(self, thread: threading.Thread, line_queue: "queue.Queue[object]"):
        self.thread = thread
        self.queue = line_queue

    def is_alive(self) -> bool:
        return self.thread.is_alive()


def run_search(
    argv: List[str],
    cancel_event: threading.Event,
    pause_event: threading.Event,
) -> SearchHandle:
    """
    Elindítja a keresést egy háttérszálon (daemon, hogy egy bezárt ablak
    ne akassza meg a folyamat kilépését). A szál a `main.main(argv)` hívása
    körül átmenetileg lecseréli a `sys.stdout`-ot egy `QueueWriter`-re –
    ez FOLYAMAT-szintű állapot, tehát egyszerre csak EGY keresés futhat a
    GUI-ból (ezt a Futtatás gomb widget-szinten is kikényszeríti).

    A visszaadott `SearchHandle.queue`-ba szöveges sorok (a `main.main()`
    kimenete), `main.ReferenceResult` objektumok (az élő találat-panelhez,
    lásd `app.py`/`live_panel.py`) és egy záró `Done`-sentinel kerül – a
    hívónak (`app.py`) saját, `root.after`-alapú pollozással kell
    kiürítenie, Tkinter-widgetet csak onnan szabad módosítani. A
    `ReferenceResult` egyszerű, immutable dataclass – a queue-ba tétele
    (a keresési háttérszálról) biztonságos, csak a FELDOLGOZÁSA (widget-
    frissítés) van a főszálhoz kötve.
    """
    line_queue: "queue.Queue[object]" = queue.Queue()

    def _worker() -> None:
        real_stdout = sys.stdout
        sys.stdout = QueueWriter(line_queue)
        exit_code = 1
        error: Optional[BaseException] = None
        try:
            exit_code = run_main(argv, cancel_event=cancel_event, pause_event=pause_event,
                                  on_result=line_queue.put)
        except BaseException as e:  # a háttérszál sose fagyassza le csendben a GUI-t
            error = e
            line_queue.put(f"[HIBA] {e}\n{traceback.format_exc()}")
        finally:
            sys.stdout = real_stdout
            line_queue.put(Done(exit_code, error))

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    return SearchHandle(thread, line_queue)
