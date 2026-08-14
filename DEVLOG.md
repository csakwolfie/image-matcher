# Fejlesztői napló

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
