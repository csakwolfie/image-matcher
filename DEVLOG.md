# Fejlesztői napló

## v1.1.0 — 2026-08-14 — Élő találat-panel a GUI tetején

A fő ablak tetején új, folyamatosan frissülő panel: bal oldalt a
jelenleg feldolgozás alatt álló referencia kép, jobb oldalt az eredmény
(a talált forrás, vagy a legjobb – de elutasított – jelölt), lekerekített
sarkú kártyaként, színes kerettel és jelvénnyel az állapot szerint (zöld
= találat, piros = nincs találat/hiba, sárga = közel volt, szürke =
feldolgozás alatt). A jelvény szövege a GUI aktuális nyelvén jelenik meg.

**Architektúra**: a keresés háttérszálon fut, és eddig csak szöveges
konzol-sorokat küldött a GUI-nak. Ahelyett, hogy a panel a (fordított,
töredezetten streamelt) naplószöveget próbálná visszafejteni, egy új,
strukturált csatorna épült be:
- `main.py`: új `ReferenceResult` (immutable dataclass) + opcionális
  `on_result` callback-paraméter, ugyanabban a mintában, mint a meglévő
  `cancel_event`/`pause_event` – a `_search()` ciklus minden ágánál
  (feldolgozás kezdete, találat, nincs találat, közel volt, már megvolt,
  hiba) meghívja, ha meg van adva. Alapból `None`, tehát a CLI
  viselkedése bitre pontosan változatlan marad.
- `gui/worker.py`: az `on_result`-ot közvetlenül a meglévő szál-biztos
  `queue.Queue`-ba köti (`on_result=line_queue.put`) – a `ReferenceResult`
  objektumok ugyanazon a csatornán utaznak, mint a szöveges log-sorok és
  a záró `Done`-sentinel.
- `gui/live_panel_render.py` (ÚJ, `tkinter`-mentes): a kártyák tiszta
  PIL-alapú összeállítása (lekerekített keret, kép behelyezése/körbe-
  vágása, jelvény) – unittest-tel, valódi ablak/display nélkül
  tesztelhető (ugyanaz a "tiszta logika külön fájlban" minta, mint
  `argv_builder.py`-nál). Betűtípus: néhány gyakori TrueType nevet
  próbál, de a végső tartalék a Pillow-ba csomagolt skálázható
  alapértelmezett betűtípus – nincs OS-specifikus feltételezés.
- `gui/live_panel.py` (ÚJ): `LivePanel(tk.Frame)` – a fenti tiszta
  logikát `ImageTk.PhotoImage`-re konvertálja és két `tk.Label`-ben
  jeleníti meg; az `app.py` a `_poll_queue`-ban dolgozza fel a
  `ReferenceResult` eseményeket (fájlnév → `Path` feloldás a `status`-tól
  függő mappában: "exists"-nél a kimeneti `found/`-ban, minden más
  esetben a source-mappában).

**Tesztek** (12 új, 141/141 összesen zöld): `test_main.py`
`OnResultCallbackTest` (találat/nincs találat/már megvolt esetek,
valódi szintetikus képekkel), `test_live_panel_render.py` (a tiszta
kártya-összeállítás). A `LivePanel` widget maga SZÁNDÉKOSAN nincs
automatizált tesztben – egy valódi `tk.Tk()` példányosítása megtörné a
headless Ubuntu CI-t (nincs X display) –, helyette élesben, valódi
képernyőmentésekkel ellenőrizve (feldolgozás → találat átmenet,
mindkét nyelven).

## v1.0.0 — 2026-08-14 — MVP kiadás

Az `image-matcher` első önálló, kiadásra szánt verziója: a klasszikus
(nem neurális) OpenCV feature-matching keresőmotor (SIFT/AKAZE/ORB/BRISK
+ RANSAC, kétlépcsős gyors-előszűrés + pontos-ellenőrzés stratégiával) a
hozzá tartozó Tkinter GUI-val (`image-matcher-gui`) és a vele egyenértékű
CLI-vel (`image-matcher`).

**Tartalmazza:**
- Kétlépcsős keresés (előszűrés + RANSAC-homográfia-ellenőrzés), A/B/C
  kompenzációs elfogadási ág a near-miss esetek kezelésére.
- Konfigurációs rendszer: `config.yaml` gyári alapértékek + beépített
  profilok (`balanced`/`high_recall`/`diagnostic`) + felhasználói egyedi
  profilok, világos precedenciával (CLI > profil > config.yaml).
- Grafikus profil-szerkesztő (`ProfileEditorWindow`) minden hangolható
  paraméterhez, csúszkákkal/beviteli mezőkkel és élő előnézettel.
- Kétnyelvű felület (magyar/angol), futásidőben váltható.
- Descriptor-cache (lemezre perzisztált, fingerprint-alapú
  érvénytelenítéssel) és leskálázott kép-cache a nagy forráskészletek
  kezelhető futásidejéhez.
- Tartalék kör kis natív felbontású referenciákra (downscale retry).
- Szünet/Folytatás/Leállítás vezérlés mind CLI-n (jel), mind GUI-n
  (élő log-streameléssel, háttérszálon futó kereséssel).

**Fejlesztési előzmény**: ez a repó egy hosszabb, iteratív fejlesztési
folyamat (Fix 1/2/3 küszöb-finomítások, i18n, GUI, profil-szerkesztő)
desztillált, tiszta kivágata. A fejlesztés során felmerült két
kiegészítő irány — egy emberi-jóváhagyású vizuális találat-ellenőrző
("Visual-Control-Module") és egy ground-truth-tal tanított neurális háló
alapú profil-ajánló ("Tanuló profil-ajánló", `image-matcher-train`) —
külön fejlesztési ágon élnek tovább, NEM része ennek az MVP kiadásnak,
hogy a base telepítés egyszerű és függőség-mentes (nincs GPU/`torch`-
igény) maradjon.

129 unittest teszt zöld (`python -m unittest discover -s tests -t .`).
