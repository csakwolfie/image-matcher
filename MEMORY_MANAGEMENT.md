# Memóriakezelés — fejlesztői leírás

Ez a dokumentum a `image_matcher` csomag hat modulját elemzi memóriakezelési
szempontból: **[image_io.py](image_matcher/image_io.py)**,
**[cache_disk.py](image_matcher/cache_disk.py)**,
**[preprocessing.py](image_matcher/preprocessing.py)**,
**[search.py](image_matcher/search.py)**,
**[matching.py](image_matcher/matching.py)**,
**[detectors.py](image_matcher/detectors.py)**.

Cél: pontosan leírni, *mi* kerül memóriába, *mikor*, *ki* (melyik objektum) tartja
életben, és *mikor* szabadul fel — külön kiemelve azt az egyetlen helyet a
kódbázisban, ahol a memóriahasználat ténylegesen a teljes forráskészlet méretével
skálázódik, korlát nélkül.

---

## Közös alapelv: CPython refcounting

Egyik modulban sincs explicit `del` vagy `gc.collect()` hívás — ez **szándékosan
nincs szükség rá**. A CPython referenciaszámlálásos memóriakezelése miatt egy nagy
`numpy`/OpenCV tömb (pl. egy dekódolt kép) abban a pillanatban felszabadul, amikor
az utolsó rá mutató lokális változó kikerül a scope-ból (pl. egy függvény visszatér)
— ez **szinkron**, nem kell várni a generációs (ciklusokat kereső) garbage
collectorra. Ez a hat modul egyike sem hoz létre referenciaciklust (nincs
kereszthivatkozó objektum-gráf), ezért a "straight-line" kód (betöltés → transzformáció
→ visszaadás) mindig azonnal felszabadítja a közbenső puffereket, amint a hívó
függvény visszatér.

Emiatt a memóriakezelés szempontjából releváns kérdés szinte minden modulnál nem az,
hogy *"mikor szabadul fel"* (ez automatikus), hanem az, hogy ***meddig él* egy
objektum, mert valaki explicit módon egy dict-ben/listában megtartja** — ez csak
egyetlen helyen, a `cache_disk.py`-ban fordul elő tudatosan, run-időtartamra.

### Kockázati összefoglaló

| Modul | Memória-mintázat | Skálázódik a forráskészlet méretével? | Kockázat |
|---|---|---|---|
| `cache_disk.py` | **Run-időtartamra felhalmozó** (2 db soha nem ürülő dict) | **Igen — ez az egyetlen ilyen hely** | Magas nagy korpusznál |
| `detectors.py` | Szálankénti (thread-local) cache, de a pool-ok élettartamához kötve | Nem (detektoronként/szálanként O(1)) | Alacsony (hatékonysági, nem memória-kockázat) |
| `image_io.py` | Tisztán tranzens, hívásonként | Nem (egy hívás = egy kép munkakészlete) | Alacsony |
| `preprocessing.py` | Tisztán tranzens, worker-enkénti | Nem (max_workers × egy kép) | Alacsony |
| `search.py` | Tisztán tranzens + kis metaadat-listák | Metaadat igen, képadat nem | Alacsony |
| `matching.py` | Tisztán tranzens, állapotmentes | Nem | Elhanyagolható |

---

## `image_io.py`

Ez a modul **nem tart meg semmit** — minden hívása egy önálló, tranzens munkakészletet
allokál, amit a hívó tulajdonol, és ami a függvény visszatérésekor (a helyi
változók scope-ból kilépésekor) azonnal felszabadul.

**`_imread_unicode`** — 3 lépcsős fallback-lánc, és minden lépcső **más
memóriaköltségű**:

1. `np.fromfile` + `cv2.imdecode` — egy bájttömb (a fájl teljes nyers tartalma) +
   egy dekódolt kép egyszerre van életben, amíg a függvény vissza nem tér.
2. `cv2.imread` — közvetlenül dekódol, nincs külön bájttömb-köztes lépés.
3. **Pillow fallback** — ez a legdrágább út: egyszerre él a PIL belső puffere, a
   `np.array(pil_img)` **másolata**, és (ha színes) egy `cv2.cvtColor`
   **további másolata** — egy nagy, egzotikus TIFF-nél ez pillanatnyilag akár
   3× a kép méretének megfelelő memóriát is lefoglalhat. Ez csak akkor fut le,
   ha az első két (gyorsabb) út hibázott, tehát a gyakori eset nem ezt az utat
   járja be.

**`load_image_smart`** — a betöltött kép élettartama a függvényen belül: eredeti
dekódolt kép → szürkeárnyalatos konverzió (**új** tömb, `cv2.cvtColor` nem in-place)
→ opcionális skálázás (**új** tömb, `cv2.resize` sem in-place) → opcionális CLAHE
(**új** tömb). Ez azt jelenti, hogy a függvény **csúcs-memóriahasználata** a
legnagyobb pillanatban akár 2 teljes méretű kép egyidejű létét is jelentheti (az
eredeti + az első derivált), mielőtt a korábbi változat refcountja nullára esne. A
függvény végül **csak a legvégső** (skálázott + CLAHE) tömböt adja vissza — minden
közbenső változat a visszatéréskor felszabadul.

A CLAHE-objektumot **minden hívás új példányt hoz létre** (nem cache-elt, nem
singleton). Ez tudatos döntés: a `cv2.CLAHE` objektum nem garantáltan szálbiztos, egy
megosztott példány több worker-szálból egyszerre hívva versenyhelyzetet
okozhatna — egy új, olcsó, állapotmentes példány létrehozása hívásonként ezt a
kockázatot nullázza, minimális (néhány mikroszekundumos) többletköltségért cserébe.

**Nincs semmilyen szinten cache** ebben a modulban — a memorizálás felelőssége
teljes egészében a hívóra hárul (lásd `cache_disk.py`).

---

## `detectors.py`

**`create_detector`** — egy `cv2.Feature2D` detektor-objektum maga **könnyű**: csak
paramétereket tárol (nfeatures, threshold, stb.), nem tartalmaz kép- vagy
descriptor-adatot. A memóriaköltség nem a detektor *létezésében*, hanem abban van,
amit a `detectAndCompute()` hívás *ideiglenesen* allokál (belső piramis-puffereket
stb.) — ezt az OpenCV C++ oldala kezeli és szabadítja fel automatikusan, Python
szintről nem látható/nem befolyásolható.

**`ThreadLocalDetectors`** — szálankénti (`threading.local()`) detector-cache, hogy
egy adott (szál, detektor-név) párra a detektor-objektumot csak egyszer kelljen
létrehozni. **Fontos, valós korlátja ennek az optimalizációnak**, amit érdemes
tudni: a kódbázisban **minden** `ThreadPoolExecutor` egy `with ThreadPoolExecutor(...)
as pool:` blokkban jön létre — a `preprocess_directory`-ban (kétszer, referencia és
source mappánként), a `stage1_rank_candidates`-ben (referenciánként egyszer), a
`find_best_match_for_reference`-ben (referenciánként × detektoronként), és a
descriptor-cache "előmelegítésnél" `main.py`-ban. **Minden ilyen `with` blokk saját,
friss OS-szálakat indít, amik a blokk végén (`pool.shutdown()`) leállnak** — a Python
`ThreadPoolExecutor` nem ad vissza szálakat egy globális, újrahasznosított
szálkészletbe a példányok között.

Ennek a következménye: a `ThreadLocalDetectors` szálankénti cache-e valójában csak
**egyetlen pool élettartamán belül** ér valamit (pl. az összes `match_pair` hívás
egy adott `stage1_rank_candidates` híváson belül ugyanazokat a szálakat, tehát
ugyanazt a cache-elt detektort használja). Amint a `with` blokk lezárul és a szálak
megszűnnek, a bennük cache-elt detektor-objektumok is elvesznek (a haldokló szál
`threading.local()` tárolójával együtt) — a **következő** referencia/detektor
körének **új** szálai újra `create_detector()`-t hívnak. Ez **nem memóriaszivárgás**
(minden korrekten felszabadul a szál megszűnésekor), csak azt jelenti, hogy a
detektor-objektum újrafelhasználása a gyakorlatban csak egy-egy kereséskör
belsejére korlátozódik, nem a teljes futásra. Mivel maga a `cv2.SIFT_create(...)`
stb. hívás olcsó (nem a drága `detectAndCompute`), ez elhanyagolható
CPU-többletköltség — de érdemes tudni, hogy ez **nem** egy futás-szintű, tartós
detector-pool.

A ténylegesen drága munka (`detectAndCompute` eredménye) memorizálása **nem** ebben
a modulban történik, hanem a `DescriptorCache`-ben (lásd lent), aminek a `_cache`
dict-je **nem** szálankénti, hanem egyetlen, közösen zárolt (`self._lock`) dict —
ezért a detektor-objektumok újra-létrehozása a fenti pool-váltásoknál **nem** vezet a
drága feature-számítás megismétléséhez, csak magának a paraméter-objektumnak az
újra-instanciálásához.

---

## `cache_disk.py`

**Ez az egyetlen modul, ahol a memóriahasználat ténylegesen, korlát nélkül nő a
forráskészlet méretével a teljes futás időtartamára.** A `DescriptorCache` egy
példánya (`main.py`-ban egyszer jön létre egy futtatáshoz) két dict-et tart a
memóriában, mindkettőt `self._lock`-kal védve:

### 1. `self._cache` — a descriptor-memoizáció (mindig nő)

Kulcs: `(forrásfájl_útvonal, detektor_név)` → érték: `(keypoints_lista,
descriptor_tömb, skála)`. **Ez a dict soha nem ürül a futás alatt**, és **mindig**
feltöltődik egy adott (path, detektor) párra, akár frissen számolt, akár lemezes
cache-ből betöltött adatról van szó (a `get_descriptors` mindkét ágon beírja
`self._cache[key]`-t).

Konkrét méretbecslés (a projekt saját léptékén, ~5100 forráskép, alapértelmezett
`sift_nfeatures=5000`): egy SIFT descriptor-tömb elméleti felső korlátja
5000 × 128 dim × 4 bájt (float32) ≈ **2,4 MiB/kép**, plusz a keypoint-lista
(akár 5000 `cv2.KeyPoint` objektum, Python-objektum overhead-del) ≈ további
1–1,5 MiB/kép. **Valós fotóknál a ténylegesen kinyert keypoint-szám jellemzően
jóval a névleges felső korlát alatt marad** (a `sift_contrast=0.045` viszonylag
szigorú, és az 1. köri cache-képek is csak 1600px hosszú oldalúak) — de nagy
korpusznál (több ezer kép) ez így is **több gigabájtos, csak az 1. köri
(stage-1) detektorra vonatkozó** állandó memóriafoglalást jelenthet, mert a
`main.py`-beli "1. köri descriptor cache építése" lépés **explicit módon
végigmegy a teljes forráskészleten**, nem csak a jelölteken.

A 2. körben (`find_best_match_for_reference`) ezzel szemben csak a `top_k`
(alapból 8) jelöltre számolódnak descriptorok, méghozzá **legfeljebb 4
detektorra** — ez jelöltenként/detektoronként elhanyagolható a stage-1
korpusz-szintű költséghez képest.

### 2. `self._images_loaded` — a dekódolt nyers pixelek memoizációja (feltételesen nő)

Kulcs: fájlútvonal → érték: `(szürkeárnyalatos tömb, skála)`. Ezt a dictet
**kizárólag** a `get_image()` tölti, amit **kizárólag** a `get_descriptors()` hív,
**csak akkor**, ha a descriptor-lekérés mind a memóriában, mind (ha van
`persist_dir`) a lemezen **cache-miss** volt.

**Ez a legfontosabb, nem magától értetődő nuansz:** egy **meleg (perzisztens)
cache-sel futó ismételt futtatásnál** a `_load_from_disk()` sikeresen visszaadja a
descriptorokat, a függvény **visszatér, mielőtt egyáltalán elérné a
`self.get_image(path)` hívást** — vagyis egy teljesen bemelegített lemezes
cache mellett `_images_loaded` **gyakorlatilag üresen marad** a futás végéig,
függetlenül a korpusz méretétől. A nyers pixel-memória-kockázat tehát
kifejezetten **hidegindításos futásokra** (első futtatás, vagy `--rebuild-cache`)
korlátozódik — melegindításnál ez a réteg nem jelent terhelést.

### A zár (`self._lock`) hatóköre

A lock **csak** a dict-műveletek (get/set, számlálók növelése) körül van — a drága
munka (`load_image_smart`, `detectAndCompute`) **a lock-on kívül** fut. Ez tudatos:
a lock tartási ideje mikroszekundumos nagyságrendű, így nem lesz szűk keresztmetszet
sok worker-szál mellett sem — a valódi párhuzamosítás (amit a `ThreadPoolExecutor`
ígér) ténylegesen érvényesül, nem csak látszólagos.

### Lemezes perzisztencia — memóriaszempontok

`_save_to_disk`: a `np.savez_compressed` a tömörített buffert **memóriában építi
fel**, mielőtt lemezre írná — ez a descriptor-tömb méretével arányos, de **tranzens**
(a hívás visszatérésekor felszabadul), és egy `.tmp.npz` fájlba írva, atomi
`os.replace`-szel kerül a végleges helyére (megszakadt futásnál nem marad sérült
cache-fájl).

`_load_from_disk`: a `with np.load(pf, allow_pickle=False) as data:` **szándékosan**
context manager-ként van használva. Az `np.load` egy `.npz`-n **lusta** betöltő — a
zip-fájl file-handle-jét nyitva tartja, és csak a ténylegesen indexelt tömböket
(`data["keypoints"]`, `data["descriptors"]`) dekompresszálja. A `with` blokk nélkül
ez a file-handle nyitva maradna — több ezer cache-találat esetén ez elméletileg OS
file-handle-kimerüléshez vezethetne. A jelenlegi kód ezt helyesen kezeli; ha valaki
"egyszerűsítené" a `with`-et egy sima hívásra, az egy valós, nehezen észrevehető
regressziót vezetne be.

### `clear_images()` — jelenleg kihasználatlan felszabadító mechanizmus

A `DescriptorCache` osztály tartalmaz egy `clear_images()` metódust, ami **csak** a
nyers pixel-dict-et (`_images_loaded`) üríti, a descriptor-dict-et (`_cache`) —
a drágábban újraszámolható, de kisebb memóriaigényű adatot — **megtartja**. Ez a
v6.7-es kódból öröklött, célzott "olcsó felszabadítás" mechanizmus, de a jelenlegi
`main.py` orchestráció **sehol nem hívja meg**.

**Konkrét, biztonságos javaslat:** a `main.py`-beli "1. köri descriptor cache
építése" előmelegítő ciklus után (miután az összes forrás stage-1 descriptora már
a `_cache`-ben van) egy `cache.clear_images()` hívás gigabájtos nagyságrendben
szabadíthatna fel memóriát nagy, hidegindításos futásoknál — a 2. kör (jelöltek
teljes felbontású újratöltése) ettől nem sérülne, mert az más fájlútvonalakat
(eredeti, nem cache-kép) tölt be, amikre ekkor még nem is volt bejegyzés ebben a
dict-ben.

---

## `preprocessing.py`

Ez a modul memóriakezelés szempontjából a **legkevésbé kockázatos** — semmit nem
tart meg a hívások között.

**`_preprocess_one`** — egy kép teljes életciklusa **egyetlen függvényhíváson belül**
zajlik: betöltés (`load_image_smart`, lásd fent) → JPEG-kódolás (`cv2.imencode`,
**új** tömör buffer) → lemezre írás (`buf.tofile`) → a helyi `gray` és `buf`
változók a függvény visszatérésekor azonnal felszabadulnak. **Semmi nem kerül
megosztott állapotba** — ez a függvény tisztán funkcionális (bemenet → mellékhatás a
lemezen, nincs visszaadott adat a hívónak a sikeren/bukáson kívül).

Ennek közvetlen következménye: **a `preprocess_directory` csúcs-memóriahasználata
`max_workers × (egy kép munkakészlete)` — függetlenül attól, hogy a mappában 10 vagy
10 000 kép van.** Ez éles ellentétben áll a `cache_disk.py` mintázatával, ahol a
memóriahasználat a **teljes korpusz méretével** skálázódik.

**`preprocess_directory`** — a `mapping` (cache_path → eredeti_path) és `todo`
listák csak `Path`-objektumokat/stringeket tárolnak, nem kép- vagy
descriptor-adatot — akár 100 000 fájlnál is legfeljebb néhányszor tíz MB. A
`futures` dict (egy `Future` objektum feladatonként) ugyanígy csak metaadatot
tart, és a `with ThreadPoolExecutor(...) as pool:` blokk végén, a `futures`
lokális változóval együtt, teljes egészében felszabadul.

---

## `search.py`

Mindkét függvény **állapotmentes** — a modul maga semmit nem tart meg hívások
között; minden memorizálást a paraméterként kapott `DescriptorCache`-re delegál.

**`stage1_rank_candidates`**:

- A referencia cache-képét **közvetlenül** `load_image_smart`-tal tölti be, **nem**
  a `cache.get_image()`-en keresztül — ez azt jelenti, hogy a referencia-kép
  dekódolása **nincs memoizálva** sehol (ellentétben a forrásképekkel). Mivel egy
  adott referenciára ez a függvény a teljes futás alatt **pontosan egyszer** fut le
  (a `main.py` fő ciklusában), ez nem pazarlás a gyakorlatban — csak érdemes tudni,
  hogy ez egy **tudatosan aszimmetrikus** memorizációs politika: forrásképek
  cache-elve, referenciaképek nem.
- A `scored: List[Tuple[Path, float]]` lista **a teljes forráskorpusz méretével
  arányos** (minden forrást pontoz, kizárás nélkül) — de csak `Path` + `float`
  párokat tartalmaz, kép- vagy descriptor-adat nélkül, ezért még több ezer forrásnál
  is elhanyagolható méretű (néhány MB nagyságrend). A függvény visszatérése előtt
  `top_k`-ra vágva adja vissza — a teljes lista a függvény végén felszabadul.
- A `futures` dict itt is **a teljes forráskorpusz méretével** arányos méretű
  (minden forrásra egy `Future`), ami a legnagyobb metaadat-szerkezet ebben a
  modulban — de mivel csak apró, skalár eredményeket (`match_pair` visszatérési
  dict-jei) tartalmaz, ez is elhanyagolható a tényleges kép/descriptor-adatokhoz
  képest.

**`find_best_match_for_reference`**:

- **Figyelemre méltó, öröklött (v6.7-ből változatlanul átvett) mintázat:** a
  detektorok feletti `for det_name in cfg.detector_priority:` ciklus **minden egyes
  iterációban újra betölti és újra CLAHE-zi a referenciaképet**
  (`load_image_smart(ref_path, ...)`), pedig a nyers pixeladat detektoronként nem
  változik — csak a rajta futtatott `detectAndCompute` eredménye tér el. Bizonytalan
  esetben (amikor mind a 4 detektor lefut) ez a referenciakép **akár 4-szeri**
  dekódolását/skálázását/CLAHE-zését jelenti egyetlen `find_best_match_for_reference`
  híváson belül. Ez **nem memóriaszivárgás** (minden korábbi `ref_gray` példány a
  következő iteráció elején felszabadul, mielőtt az új betöltődne — a régi és az új
  nincs egyszerre életben), hanem egy redundáns **újraszámítás** — CPU-időt és
  tranzens allokációs/felszabadítási churn-t okoz feleslegesen. Egy jövőbeli
  optimalizáció a ciklus elé emelhetné a betöltést (egyszeri `ref_gray`, minden
  detektor újrahasznosítja).
- `all_results` — egy lista, aminek mérete `jelöltek_száma × ténylegesen kipróbált
  detektorok_száma` (early-accept miatt gyakran kevesebb, mint 4) — tipikusan
  8×4=32 bejegyzés alatt, mindegyik csak skalár mezőkkel (int/float/str) + egy
  `Path`. Ez a lista a hívóhoz (`main.py`) kerül vissza, ahol a teljes futás
  `candidates_detail` listájába gyűlik — ez már a `main.py` felelőssége, nem ezé a
  hat modulé, de érdemes tudni, hogy a `results_candidates.csv` végső mérete
  (és az addig memóriában tartott sorok száma) lineárisan nő a feldolgozott
  referenciák számával, egészen a futás végi CSV-kiírásig.
- `candidates` (a detektor-ciklus belsejében) — jelöltenkénti szűrt lista,
  detektor-iterációnként **újra létrejön**, nem hordozódik át a következő
  detektorra — elhanyagolható méret.

---

## `matching.py`

Ez a modul a hat közül a **legtisztábban funkcionális** — nincs benne osztály, nincs
semmilyen cache vagy megosztott állapot; minden függvény a bemeneteiből számol, és
egy kis eredményt ad vissza.

- **`match_descriptors`** minden híváskor **új** `cv2.BFMatcher` vagy
  `cv2.FlannBasedMatcher` objektumot hoz létre — ez **nem csak** hatékonysági
  kompromisszum, hanem **korrektségi követelmény**: a FLANN-matcher belső
  KD-fa-indexét a `knnMatch` hívás építi fel a konkrét `des1` (referencia)
  descriptor-halmazhoz; egy újrafelhasznált matcher-példány más referenciára
  hibás/elavult indexet használna. A `knn` köztes eredmény (egy `cv2.DMatch`
  pár-lista, akár `sift_nfeatures` hosszú) a legnagyobb tranzens allokáció ebben a
  függvényben — SIFT-nél elméleti felső korlátja néhány MB, a gyakorlatban ennél
  jóval kevesebb, és a függvény visszatérésekor (miután csak a szűrt `good` listát
  adta vissza) azonnal felszabadul.
- **`geometric_verification`** — `src_pts`/`dst_pts` a `good_matches` hosszával
  arányos, de mivel a `good_matches` maga is csak néhány száz/ezer elemű, ez
  kilobájtos nagyságrend. A `cv2.findHomography` belső RANSAC-munkamemóriáját az
  OpenCV C++ oldala kezeli, Python-oldalról nem látható és nem befolyásolható —
  bounded a `maxIters`/pontszám által.
- **`match_pair`** — ez a függvény van tömegesen `ThreadPoolExecutor`-ra
  beküldve (forrásonként az 1. körben, jelöltenként/detektoronként a 2. körben).
  **Semmit nem tart meg hívások között.** A `kp1`/`des1` (referencia keypoints/
  descriptors) paraméterek **objektumreferenciaként**, nem másolatként érkeznek —
  N konkurens worker-szál, ami ugyanarra a referenciára/detektorra dolgozik,
  **ugyanazt** a `des1` tömböt olvassa a memóriában (biztonságos, mert ezen az
  úton semmi nem módosítja), nincs szálankénti duplikáció. Az egyetlen memorizáló
  mellékhatás a `cache.get_descriptors(src_path, detector_name)` hívás, ami a
  `cache_disk.py`-ba delegálja a memóriakezelést (lásd fent).

---

## Összefoglaló ajánlások

1. **Legmagasabb prioritású, konkrét javaslat:** hívjuk meg a
   `DescriptorCache.clear_images()`-t a `main.py`-ban közvetlenül az "1. köri
   descriptor cache építése" előmelegítő ciklus után — ez gigabájtos nagyságrendű
   memóriát szabadíthat fel nagy, hidegindításos (vagy `--rebuild-cache`)
   futásoknál, a descriptorok (a drágán újraszámolható adat) elvesztése nélkül.
2. Ha a forráskorpusz mérete jelentősen (nagyságrendekkel) megnő a jelenlegi
   ~5100 képes léptékhez képest, érdemes megfontolni egy **korlátozott méretű
   (LRU-szerű) evikciót** a `DescriptorCache._cache`-hez — jelenleg ez a dict
   szándékosan korlátlanul nő a futás időtartamára.
3. A referenciakép ismételt betöltése detektoronként
   (`find_best_match_for_reference`) egy CPU/allokációs hatékonysági
   finomítási lehetőség (nem memóriakockázat) — a ciklus elé emelt, egyszeri
   `ref_gray` betöltés kiküszöbölné.
4. A `ThreadLocalDetectors` jelenlegi haszna pool-élettartamra korlátozódik a sok
   helyen újranyitott `ThreadPoolExecutor` miatt — ha a detektor-létrehozás
   valaha mérhető CPU-költséggé válna (jelenleg nem az), egy futás-szintű, közösen
   megosztott `ThreadPoolExecutor` nagyobb haszonnal járna, mint a jelenlegi,
   kereskörönként újrainduló pool-ok.
