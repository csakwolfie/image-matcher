# High-Accuracy Image Matcher

[![Tests](https://github.com/csakwolfie/image-matcher/actions/workflows/tests.yml/badge.svg)](https://github.com/csakwolfie/image-matcher/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

🇭🇺 Magyar | 🇬🇧 [English](README.md)

Klasszikus (nem neurális) OpenCV feature-matching alapú képkereső: adott **referencia**
(körbevágott) képekhez megkeresi a hozzájuk tartozó **eredeti, teljes** képet egy nagy
forrásmappában. SIFT / AKAZE / ORB / BRISK detektorokat és RANSAC-homográfiát használ,
kétlépcsős (gyors előszűrés + pontos ellenőrzés) stratégiával, hogy nagy (több ezer
képes) forráskészleteken is kezelhető idő alatt fusson.

> Verzió: **1.0.0**. A fejlesztés részletes története a [DEVLOG.md](DEVLOG.md)-ben.
> Ingyenes, MIT licenc alatt ([LICENSE](LICENSE)) — szabadon használható, módosítható,
> terjeszthető, kereskedelmi célra is.

---

## Tartalom

- [Hogyan működik](#hogyan-működik)
- [Telepítés](#telepítés)
- [Gyors kezdés](#gyors-kezdés)
- [GUI](#gui)
- [Mappastruktúra](#mappastruktúra)
- [Architektúra / modulok](#architektúra--modulok)
- [CLI paraméterek](#cli-paraméterek)
- [Konfiguráció: config.yaml és profilok](#konfiguráció-configyaml-és-profilok)
- [Kimenetek](#kimenetek)
- [Cache-rendszer](#cache-rendszer)
- [Tesztelés](#tesztelés)
- [Hibaelhárítás](#hibaelhárítás)
- [Licenc](#licenc)

---

## Hogyan működik

**Kétlépcsős keresés:**

1. **Előfeldolgozás** — minden kép (referencia + forrás) leskálázva egy közepes
   méretre (`cache_long_side`, alapból 1600px), szürkeárnyalatos, 8-bit JPEG-ként
   cache-elve. Inkrementális: csak az új képeket dolgozza fel újra.
2. **1. kör (gyors jelöltkeresés)** — a kis cache-képeken, **csak** az első
   (legnagyobb prioritású, alapból SIFT) detektorral, **minden** forrást pontoz
   (nincs kizáró abszolút küszöb, csak egy technikai minimum), és a legjobb `top_k`
   (alapból 8) jelöltet viszi tovább.
3. **2. kör (pontos döntés)** — a jelölteket az **eredeti, teljes felbontású**
   fájlokból tölti be újra, és mind a 4 detektorral (prioritási sorrendben,
   early-accept-tel) szigorú küszöbökkel dönt.
4. **Találat esetén** a forrásfájl a `found/` mappába kerül, a referencia
   fájlnevét kapva (a forrás saját kiterjesztésével).

**Miért két kör?** Egy nagy forráskészletet minden detektorral, teljes felbontáson
összehasonlítani minden referenciával kezelhetetlenül lassú lenne. A gyors 1. kör (kis
kép, 1 detektor) szűkíti a jelölteket egy kis listára, amin aztán a drága, pontos
2. kör fut.

**Miért nincs kizáró küszöb az 1. körben?** Egy korábbi verzióban volt egy magasabb
technikai minimum, ami valódi találatokat zárt ki a jelöltlistából, mielőtt azok esélyt
kaptak volna a pontos 2. körre — ez volt az egyik legnagyobb hibaforrás a projekt
történetében. Az 1. kör ezért csak **rangsorol**, sosem zár ki abszolút küszöb alapján;
a tényleges döntést mindig a 2. kör hozza meg.

**Homográfia-plauzibilitás ellenőrzés** — védelem "mágnes-képek" (periodikus/repetitív
mintázatú, pl. rácsvonalas, perforált forrásképek) ellen, amik véletlenül geometriailag
"tisztának tűnő", de tartalmilag hibás RANSAC-illesztést kaphatnak. Egy valódi
fotó-kivágat/forrás párnál a becsült homográfia affin része (forgatás+skálázás+enyhe
nyírás) ésszerű tartományban van; egy degenerált illesztés (tükrözés, extrém nyírás,
irreális skálázás) erős jele a hamis geometriai egyezésnek — ezt a szűrő elutasítja.

**Konzol-kimenet** — minden referencia egy rövid blokkot kap:

```
[021/363] img-020.jpg  ✓ FOUND
  → _MES1234.jpg  (det=SIFT)
  good=287 | inliers=241 | score=0.934

[022/363] img-021.jpg  ~ NEAR MISS
  → _MES5678.jpg
  good=182 | inliers=96 | score=0.641
  reason: inlier_ratio 0.436 < MIN_INLIER_RATIO 0.93
```

`✓ EXISTS` (már megvolt), `✓ FOUND` (találat), `~ NEAR MISS` (volt
geometriailag plauzibilis jelölt, csak a küszöb alatt maradt —
`REJECT_INLIER_RATIO`/`REJECT_SCORE`), `✗ NOT FOUND` (minden más elutasítási
ok). Alul egy folyamatosan frissülő állapotsor mutatja az összképet:

```
Progress: 19/363 (5.2%) | 00:14 elapsed | ETA 04:44 | FOUND 12 | NOT FOUND 7
```

(Az állapotsor csak valódi terminálban jelenik meg — átirányított/pipe-olt
kimenetnél automatikusan kikapcsol. A naplófájlba mindig a blokk-üzenetek
kerülnek, az állapotsor nem.)

---

## Telepítés

Python 3.10+ (fejlesztve 3.14-en). Két lehetőség:

**1. Csomagként telepítve** (ajánlott — ekkor egy valódi `image-matcher`
parancs is elérhető lesz a `python run.py` mellett):

```bash
git clone <repó-URL>
cd image-search
pip install -e .
```

**2. Csak a függőségek, forrásból futtatva:**

```bash
pip install -r requirements.txt
```

> Az AKAZE és BRISK detektorokhoz teljes `opencv-contrib-python` build szükséges — ez
> már a projekt alapértelmezett függősége (nem a szűkebb `opencv-python`). Ha mégis
> hiányoznak, a program figyelmeztetéssel jelzi, és a maradék elérhető detektorokkal
> fut tovább (lásd [Hibaelhárítás](#hibaelhárítás)).

> **Minimál/headless Linux szervereken** (pl. Docker alapimage-eken, desktop
> környezet nélküli gépeken) az `opencv-contrib-python` importálás közben
> `libGL.so.1` hiányára panaszkodhat, mert egy ott nem telepített grafikus
> könyvtárhoz linkel. Vagy telepítsd a rendszer-libet (Debian/Ubuntu: `sudo
> apt install libgl1`), vagy cseréld a függőséget `opencv-contrib-python-
> headless`-re, aminek nincs szüksége rá — csak akkor releváns, ha azon a
> gépen a GUI-t nem is használod.

---

## Gyors kezdés

```bash
python run.py --reference "D:\referencia-kepek" --source "E:\forras-kepek" --output "runs\2026-08-11"
```

(Csomagként telepítve ugyanez `image-matcher --reference ... --source ... --output ...`
formában is elérhető, `python run.py` nélkül.)

Csak validálás, keresés nélkül (gyors ellenőrzés, mielőtt egy több órás futást
indítanál):

```bash
python run.py -r "D:\referencia-kepek" -s "E:\forras-kepek" -o "runs\teszt" --dry-run
```

Egy másik profillal, korlátozott referencia-számmal (gyors teszteléshez):

```bash
python run.py -r ref_mappa -s src_mappa -o out --profile high_recall --limit 20
```

Elérhető profilok listázása:

```bash
python run.py --list-profiles
```

---

## GUI

A CLI-kapcsolók (nyelv, profil, útvonalak, `--limit`/`--workers`/`--top-k`,
`--no-cache`/`--rebuild-cache`/`--dry-run`) egy Tkinter grafikus felületen
is elérhetők — nincs hozzá extra függőség (a Tkinter a Python
standard library része). Indítás:

```bash
python run_gui.py
# vagy
python -m image_matcher.gui
# csomagként telepítve:
image-matcher-gui
```

A Futtatás gomb mellett Szünet/Folytatás és Leállítás gomb is elérhető —
ezek a KÖVETKEZŐ referencia feldolgozása előtt lépnek életbe (egy már
folyamatban lévő, egyetlen referencián futó 2-köri keresést nem lehet
félbeszakítani, csak a referenciák közötti pontokon), megszakításnál/
szünetnél az addig összegyűjtött részleges eredmény is elmentődik a
`results.csv`/`results_candidates.csv`-be. A nyelv-legördülő melletti
"Legyen alapértelmezett" gomb ugyanazt csinálja, mint a `--lang <kód>`
CLI-kapcsoló önmagában kiadva (lásd [Nyelv](#nyelv)).

> Néhány Linux disztribúción a Tkinter nem alapból települ a Python
> mellé — ott előtte telepítsd az OS-csomagot (pl. Debian/Ubuntu:
> `sudo apt install python3-tk`; Fedora: `sudo dnf install
> python3-tkinter`). Ha csak a CLI-t (`image-matcher`) használod, erre
> nincs szükség.

---

## Mappastruktúra

```
image-search/
  run.py                    ← belépési pont (forrásból futtatáshoz, CLI)
  run_gui.py                 ← belépési pont a Tkinter GUI-hoz
  pyproject.toml            ← csomag-metaadatok, függőségek, "image-matcher"/"image-matcher-gui" parancsok
  LICENSE                    ← MIT licenc
  image_matcher/              ← a program forráskódja (lásd lejjebb)
    gui/                        ← Tkinter GUI (app.py, worker.py, argv_builder.py)
    data/
      config.yaml               ← gyári alapértékek (finomhangolási konstansok)
      profiles/
        balanced.yaml             ← szigorú küszöbök (alapértelmezett választás)
        high_recall.yaml           ← lazább küszöbök, jobb recall
        diagnostic.yaml             ← nagyon laza, soha nincs early-accept (finomhangoláshoz)
      lang/
        hu.lang.json                ← a program összes felhasználó felé megjelenő szövege, magyarul
        en.lang.json                 ← ugyanaz, angolul
  tests/                      ← unittest smoke/integrációs tesztek
```

A `config.yaml`/`profiles/` a csomagba ágyazott gyári alapértékek — nem kell
közvetlenül szerkeszteni őket. Saját felülírás létrehozásához helyezz egy
`config.yaml`-t és/vagy `profiles/` mappát a **futtatás munkakönyvtárába**
(elsőbbséget élvez), vagy a `~/.image_matcher/` mappába (felhasználói szintű,
munkakönyvtártól független felülírás). Lásd
[Konfiguráció](#konfiguráció-configyaml-és-profilok).

A `lang/` mappa a program összes konzol-/hiba-/súgó-szövegét tartalmazza
kulcs→szöveg JSON-fájlokban (`<nyelvkód>.lang.json`), ugyanazzal a
felfedezési precedenciával, mint a `config.yaml`/`profiles/`. A nyelvet a
[`--lang` kapcsoló](#nyelv) választja ki (alapértelmezett: `hu`); új nyelv
hozzáadásához elég egy új `<kód>.lang.json` fájlt elhelyezni itt (vagy a
`~/.image_matcher/lang/`/munkakönyvtár-beli felülírásban) — kódmódosítás
nem kell hozzá.

Egy futtatás után a **kimeneti mappa** (`--output DIR`) tartalma:

```
<output>/
  found/                     ← a megtalált, átnevezett forrásfájlok
  results.csv                ← referenciánkénti összegzés
  results_candidates.csv     ← jelölt×detektor-szintű részletes napló
  log_20260811_225412.txt    ← teljes konzol-kimenet időbélyeggel
  cache/                     ← kép- és descriptor-cache (ha nincs --no-cache)
```

---

## Architektúra / modulok

| Modul | Felelősség |
|---|---|
| `image_matcher/config.py` | `config.yaml` + `profiles/*.yaml` betöltése, precedencia-feloldás (`Config` — immutable dataclass) |
| `image_matcher/image_io.py` | Unicode-biztos képbetöltés Windows-on, skálázás, CLAHE |
| `image_matcher/detectors.py` | Detektor-gyár (SIFT/AKAZE/ORB/BRISK/KAZE) + szálankénti (thread-local) detector-példányok |
| `image_matcher/cache_disk.py` | `DescriptorCache` — memóriában és opcionálisan lemezen (perzisztensen) tárolt feature-descriptorok, fingerprint-alapú invalidációval |
| `image_matcher/preprocessing.py` | A kis, 8-bit JPEG cache inkrementális felépítése (1. körhöz) |
| `image_matcher/matching.py` | Descriptor matching, RANSAC geometriai ellenőrzés, homográfia-plauzibilitás, score-számítás, DecisionReason kategorizálás |
| `image_matcher/search.py` | 1. kör (`stage1_rank_candidates`) és 2. kör (`find_best_match_for_reference`) |
| `image_matcher/reporting.py` | `results.csv` és `results_candidates.csv` írása |
| `image_matcher/cli.py` | Parancssori felület (csoportosított `argparse`) |
| `image_matcher/main.py` | A teljes folyamat összefűzése (orchestráció) |

Minden függvény explicit `Config`-példányt kap paraméterként (nincs mutálható
globális állapot) — egy futtatás beállításai a futtatás elejétől a végéig
garantáltan konzisztensek.

---

## CLI paraméterek

```
python run.py [kapcsolók...]
```

### Nyelv

| Kapcsoló | Leírás |
|---|---|
| `--lang NYELV` | A program nyelve — jelenleg `hu` és `en`. A `--help` szövegek is a választott nyelven jelennek meg (a nyelv már a súgó felépítése előtt eldől). Ismeretlen nyelvnél a program hibaüzenettel (elérhető nyelvek felsorolásával) kilép. Új nyelv hozzáadása csak egy új `<kód>.lang.json` fájlt igényel a `data/lang/` mappában (lásd [Konfiguráció](#konfiguráció-configyaml-és-profilok)) — kódmódosítás nélkül. |

**Alapértelmezett nyelv beállítása:** ha a `--lang NYELV`-et **önmagában**
(más kapcsoló nélkül) adod ki, a program NEM próbál keresést futtatni,
hanem tartósan elmenti a megadott nyelvet a `config.yaml`
`default_language` kulcsába, és attól kezdve minden `--lang` nélküli
futtatás ezt használja:

```
python run.py --lang en
# 'en' is now the default language, saved to: ...\.image_matcher\config.yaml
```

Ha még nincs saját `config.yaml`-felülírásod (sem a munkakönyvtáradban,
sem a `~/.image_matcher/` mappában), ez a művelet létrehoz egyet a
`~/.image_matcher/config.yaml` helyen — a gyári alapértékek TELJES
másolataként, csak a `default_language` kulccsal felülírva (a
finomhangolási beállítások innentől ebből a fájlból töltődnek be, nem
automatikusan a csomag frissítéseiből — lásd [Konfiguráció](#konfiguráció-configyaml-és-profilok)).
Ha már van felülírásod (cwd vagy felhasználói szintű), azt frissíti
helyben, a többi beállításod megtartásával. Egy `--reference`/`--source`/
`--output`/`--list-profiles`/`--dry-run` melletti `--lang` csak az adott
futtatásra vonatkozik, NEM módosítja a tartós alapértelmezettet.

### Profilkezelés

| Kapcsoló | Leírás |
|---|---|
| `--profile NÉV` | Névvel ellátott profil betöltése a `profiles/` mappából (pl. `--profile diagnostic`). Ha nincs megadva, csak a `config.yaml` alapértékei érvényesek. |

### Útvonalak

| Kapcsoló | Leírás |
|---|---|
| `--reference DIR`, `-r` | Referencia (körbevágott) képek mappája. **Kötelező.** |
| `--source DIR`, `-s` | Eredeti, teljes képek mappája, ahol a referenciákat keressük. **Kötelező.** |
| `--output DIR`, `-o` | Kimeneti mappa (lásd [Mappastruktúra](#mappastruktúra)). **Kötelező.** |

### Futtatásvezérlés

| Kapcsoló | Leírás |
|---|---|
| `--limit N` | Csak az első N referenciát dolgozza fel (ábécésorrendben) — gyors teszteléshez. |
| `--workers N`, `-w` | Párhuzamos szálak száma. Alapból a CPU-magok száma. Javasolt érték: a fizikai magok száma — több szál csak felesleges hőt/throttlingot okoz. |
| `--top-k N` | Hány jelölt megy tovább az 1. körből a 2. körbe. Alapérték: a `config.yaml`/profil `stage1_top_k` értéke (8). Javasolt: 6–10. |

### Cache

| Kapcsoló | Leírás |
|---|---|
| `--cache DIR` | A cache mappa helye. Alapból `<output>/cache`. |
| `--no-cache` | Teljesen kikapcsolja a cache-elést (se kép-, se descriptor-cache) — minden újraszámolódik minden futtatásnál. Lassabb, de a futás után nem marad cache-mappa a lemezen. |
| `--rebuild-cache` | A meglévő cache-tartalmat figyelmen kívül hagyja és felülírja (kényszerített újragenerálás), utána továbbra is ír cache-t. |

### Futtatás

| Kapcsoló | Leírás |
|---|---|
| `--dry-run` | Validálja az útvonalakat és a feloldott (CLI+profil+config.yaml utáni) beállításokat, megszámolja a feldolgozandó képeket — de nem futtat tényleges keresést, nem ír `found/`-ot, CSV-t vagy cache-t. |
| `--version` | Kiírja a program verzióját, és kilép. |
| `--list-profiles` | Felsorolja a `profiles/` mappában elérhető profilokat (egysoros leírással), és kilép. |

### Precedencia

**Explicit CLI kapcsoló > `--profile` fájl > `config.yaml` alapérték.**

Vagyis: ha egy érték a `config.yaml`-ban van, a kiválasztott profil felülírhatja,
egy explicit CLI kapcsoló (ahol van ilyen — jelenleg csak a `--top-k`) pedig
mindkettőt felülírja.

---

## Konfiguráció: config.yaml és profilok

A finomhangolási konstansok (küszöbök, detektor-paraméterek, CLAHE,
homográfia-ellenőrzés, cache-méret, top-k stb.) a
**[image_matcher/data/config.yaml](image_matcher/data/config.yaml)**-ban élnek,
kommentekkel/leírással ellátva — ez a gyári alapbeállítás, ami mindig betöltődik,
ha nincs felülírva. Ezeket az értékeket **valós near-miss adatok (hibaelemzés)
alapján** hangoltuk, nem találgatással — módosítás előtt érdemes megnézni a
`results_candidates.csv`-t.

### Hol keresi a config.yaml-t és a profiles/-t

Precedencia (az első találat nyer):

1. `./config.yaml` / `./profiles/` a **jelenlegi munkakönyvtárban** — ha ide
   teszel egy saját `config.yaml`-t/`profiles/` mappát, azt használja a
   csomagba ágyazott gyári alapértékek helyett.
2. `~/.image_matcher/config.yaml` / `~/.image_matcher/profiles/` —
   felhasználói szintű felülírás, munkakönyvtártól függetlenül mindig aktív.
3. A csomagba ágyazott gyári alapértékek (`image_matcher/data/`) — ez mindig
   létezik, végső biztonsági háló, telepített csomagnál is működik.

### Legfontosabb kulcsok

| Kulcs | Alapérték | Jelentés |
|---|---|---|
| `min_good_matches` | 200 | Minimum "good" match a geometriai ellenőrzés (RANSAC) előtt |
| `min_inliers` | 220 | Minimum RANSAC inlier a végső elfogadáshoz |
| `min_inlier_ratio` | 0.93 | Inlier / good-match arány minimuma |
| `score_uncertain` | 0.97 | A tényleges elfogadási score-küszöb |
| `score_accept` | 0.75 | E felett nem próbálunk gyengébb prioritású detektort |
| `early_accept_score` | 1.00 | E felett (sikeres találatnál) a többi detektort ki se próbáljuk |
| `ratio_test_sift` / `ratio_test_bin` | 0.60 / 0.65 | Lowe ratio test küszöb (float ill. bináris descriptoroknál) |
| `cache_long_side` | 1600 | Az 1. köri (gyors) cache-képek hosszabb oldala pixelben |
| `stage1_top_k` | 8 | Hány jelölt megy az 1. körből a 2. körbe |
| `stage1_min_good_matches` | 8 | **Csak technikai minimum** az 1. körben — NEM kizáró küszöb |
| `use_clahe` | true | Adaptív kontraszt-kiegyenlítés a feature-detektálás előtt |
| `use_homography_check` | true | "Mágnes-kép" védelem (lásd fent) |
| `detector_priority` | `[SIFT, AKAZE, ORB, BRISK]` | A 2. kör detektor-sorrendje |
| `max_process_size` / `min_process_size` | 2200 / 400 | A pontos (2. köri) feldolgozás méret-korlátai |
| `default_language` | `hu` | A `--lang` kapcsoló nélküli alapértelmezett nyelv — **nem feature-matching finomhangolás**, hanem a CLI/i18n rendszeré (lásd [Nyelv](#nyelv)); a `--profile`-ok nem írhatják felül |

A teljes lista, minden kommenttel: lásd közvetlenül az
[image_matcher/data/config.yaml](image_matcher/data/config.yaml) fájlt.

### Profilok

A `profiles/` mappában, fájlonként egy profil — a fájlnév (kiterjesztés nélkül)
adja a profil nevét. Egy profil-fájl **csak azokat a kulcsokat** tartalmazza, amiket
a `config.yaml`-hoz képest felülír (nem kell duplikálni mindent), plusz egy opcionális
`description:` mezőt a `--list-profiles` kimenethez.

| Profil | Mikor használd |
|---|---|
| `balanced` | **Alapértelmezett választás.** Szigorú küszöbök — egy valós teszt-halmazon 0 hibás találatot (100% pontosság) mértünk vele. Megegyezik a `config.yaml` gyári értékeivel. |
| `high_recall` | Ha a cél a minél kevesebb kihagyott (false negative) találat, és elfogadható a nagyobb hibás-találat kockázat. **Nem lett formálisan összevetve a pontossággal** — éles használat előtt érdemes ellenőrizni a `results_candidates.csv`-vel a saját adathalmazodon. |
| `diagnostic` | **Nem éles használatra.** Nagyon laza küszöbök + az early-accept ki van kapcsolva (minden elérhető detektor mindig lefut minden jelöltre) — a lehető legrészletesebb `results_candidates.csv` naplót adja küszöb-finomhangoláshoz. |

Saját profil létrehozása: hozz létre egy `profiles/sajat_profil.yaml` fájlt a
munkakönyvtáradban (vagy a `~/.image_matcher/profiles/` mappában), amiben csak a
felülírni kívánt kulcsokat sorold fel, majd `--profile sajat_profil`.

---

## Kimenetek

### `results.csv` — referenciánkénti összegzés

| Oszlop | Jelentés |
|---|---|
| `Reference` | A referencia fájl neve |
| `MatchedFile` / `SavedAs` | A megtalált forrásfájl neve / a `found/`-ba mentett fájl neve (`NOT_FOUND` / `SKIPPED_ALREADY_IN_FOUND`, ha nincs találat / már megvolt) |
| `GoodMatches` / `Inliers` / `Score` | A győztes jelölt mérőszámai |
| `NearMissFile` / `NearMissGood` / `NearMissInliers` / `NearMissScore` | NOT_FOUND esetén: a legjobb (de küszöb alatti) jelölt — diagnosztikához |
| `Stage1Diag` | Diagnosztikai szöveg, ha már az 1. körben nem volt jelölt |
| `Stage1Candidates` | Hány jelölt ment tovább az 1. körből |
| `WinningDetector` | Melyik detektor találta meg (SIFT/AKAZE/ORB/BRISK) |
| `DecisionReason` | Tömör, gépileg szűrhető kategória — lásd lent |
| `RejectReason` | Részletes, szöveges bukási ok |

### `results_candidates.csv` — jelölt×detektor-szintű részletes napló

A 2. körben ténylegesen kipróbált **minden** (jelölt, detektor) kombináció saját
sorban: `Stage1Rank`, `Stage1Score`, `GoodMatches`, `Inliers`, `InlierRatio`,
`Stage2Score`, `Success`, `IsWinner`, `DecisionReason`, `RejectReason`. Ez a fő eszköz
a küszöbök finomhangolásához — innen látszik pontosan, melyik jelölt melyik konkrét
kapunál (good_matches / inliers / inlier_ratio / score / homográfia-plauzibilitás)
bukott el.

### `DecisionReason` kategóriák

| Kategória | Jelentés |
|---|---|
| `ACCEPT_STRONG_GEOMETRY` | Elfogadva, magas (`decision_strong_ratio` feletti) inlier-aránnyal |
| `ACCEPT_INLIER` | Elfogadva, de szerényebb inlier-arány mellett |
| `REJECT_NO_INLIERS` | Nem volt elég good match / a RANSAC nem talált homográfiát |
| `REJECT_SCALE` | Homográfia-plauzibilitás elutasítás: irreális skálázás |
| `REJECT_HOMOGRAPHY` | Homográfia-plauzibilitás elutasítás: tükrözés / extrém nyírás |
| `REJECT_INLIER_RATIO` | Az inlier-szám vagy -arány a küszöb alatt |
| `REJECT_SCORE` | Az összesített score a küszöb alatt |

---

## Cache-rendszer

Két, egymástól független cache réteg van:

1. **Kép-cache** (`<cache>/reference/<méret>/` és `<cache>/source/<méret>/`) — a
   leskálázott, szürkeárnyalatos, 8-bit JPEG képek az 1. körhöz. Inkrementális: a
   már meglévő fájlokat nem generálja újra.
2. **Descriptor cache** (`<cache>/descriptors/`) — a kiszámolt feature-leírók
   lemezre mentve, hogy **ismételt futtatásoknál** (pl. küszöb-hangolásnál) ne kelljen
   újraszámolni a drága feature-detektálást.

A descriptor cache kulcsa tartalmazza a feldolgozási beállítások (CLAHE, detektor-
paraméterek, feldolgozási méret) hash-ét ("fingerprint") is — ha ezek közül bármelyik
megváltozik (pl. profilváltás miatt), a cache automatikusan érvénytelenné válik, **nem**
ad hallgatólagosan elavult eredményt.

- `--rebuild-cache` — mindkét réteget kényszerítve újragenerálja, utána továbbra is ír cache-t.
- `--no-cache` — egyik réteget sem használja perzisztensen; a kép-cache egy ideiglenes
  mappában épül fel (mert a kétlépcsős algoritmus szerkezetileg igényli), amit a
  program a futás végén töröl.

---

## Tesztelés

A projekt `unittest`-alapú smoke- és integrációs teszteket tartalmaz (`tests/`),
pytest nélkül futtathatók:

```bash
python -m unittest discover -s tests -t .
```

Amit lefednek: profil-precedencia, homográfia-plauzibilitás (tükrözés/nyírás/skálázás
elutasítása), CLAHE kontraszt-hatás, descriptor cache lemezes hit/miss és
fingerprint-invalidáció, inkrementális kép-cache építés, és egy **végponttól-végpontig
teszt**: egy szintetikus, texturált kép kivágata a `balanced` profil szigorú
küszöbeivel helyesen megtalálja az eredeti forrást több jelölt közül, illetve
NOT_FOUND-ot ad, ha csak "idegen" jelöltek vannak.

---

## Hibaelhárítás

**`[FIGYELMEZTETÉS] AKAZE/BRISK nem elérhető ebben az OpenCV buildben`** — a
telepített OpenCV build nem tartalmazza ezeket az algoritmusokat (előfordulhat még
`opencv-contrib-python` mellett is, build-től függően). A program figyelmeztetéssel
jelzi, és a maradék elérhető detektorokkal fut tovább — ez nem hiba, csak
környezetfüggő korlát.

**Ékezetes (pl. `Céges képek`) mappanevek Windows-on** — az `image_io.py`
Unicode-biztos betöltést használ (`np.fromfile` + `cv2.imdecode`, Pillow fallback-kel),
ez nem igényel külön beállítást.

**Egy referenciára `NEM TALÁLHATÓ`, pedig szerinted meg kellene lennie** — nézd meg a
`results_candidates.csv`-t: minden ténylegesen kipróbált jelöltre megtalálod a pontos
bukási okot (`RejectReason` / `DecisionReason`). Ha sok jelölt `REJECT_INLIER_RATIO`
vagy `REJECT_SCORE` közelében bukik el, érdemes lehet kipróbálni a `high_recall`
profilt, vagy a `diagnostic` profillal részletesebb naplót gyűjteni.

**Lassú futás nagyon nagy forráskészleten** — növeld a `--workers` értéket a fizikai
CPU-magok számáig (ennél többnek nincs értelme), és/vagy csökkentsd a `--top-k`
értékét (kevesebb jelölt megy a drága 2. körbe).

---

## Licenc

MIT — lásd a [LICENSE](LICENSE) fájlt. Szabadon használható, módosítható és
terjeszthető, kereskedelmi célra is.
