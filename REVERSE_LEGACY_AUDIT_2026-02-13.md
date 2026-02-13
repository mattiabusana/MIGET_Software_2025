# Audit Reverse MIGET vs Legacy Fortran

Data audit: 2026-02-13

## 1) Ambito

Confronto tra pipeline reverse attuale (`miget/ui/inverse.py`, `miget/core.py`, `miget/io.py`, `miget/bohr.py`) e software legacy Fortran (`SHORT.FOR`, `VQBOHR.FOR`) su:

- preprocessing R/E e pesi;
- inversione distribuzione V/Q;
- calcolo gas ematici/Bohr;
- robustezza su dataset legacy.

## 2) Cosa e uguale (allineato al legacy)

1. Correzione temperatura partizioni (specie-specifica) e `PCFACT`.
   - Python: `miget/core.py:95`, `miget/core.py:100`, `miget/core.py:101`
   - Fortran: `File originali/SHORT.FOR:361`, `File originali/SHORT.FOR:366`, `File originali/SHORT.FOR:367`

2. Correzioni headspace su PAC/PVC e acetone correction (vincolo su Ether/Acetone).
   - Python: `miget/core.py:149`, `miget/core.py:163`, `miget/core.py:165`
   - Fortran: `File originali/SHORT.FOR:374`, `File originali/SHORT.FOR:377`, `File originali/SHORT.FOR:391`

3. Formule di derivazione quando manca PA/PE/PV.
   - Python: `miget/core.py:361`, `miget/core.py:370`
   - Fortran: `File originali/SHORT.FOR:510`, `File originali/SHORT.FOR:539`, `File originali/SHORT.FOR:570`

4. Struttura generale inversione con 50 compartimenti e asse log V/Q.
   - Python: `miget/core.py:453`, `miget/core.py:474`, `miget/core.py:487`
   - Fortran: `File originali/VQBOHR.FOR:315`, `File originali/VQBOHR.FOR:320`, `File originali/VQBOHR.FOR:321`

5. Verifica puntuale su caso MISSISSIPPI per R/E: match numerico completo.
   - Script: `tests/verify_full_match.py`
   - Risultato locale: tutte le 6 retention e 6 excretion in tolleranza zero (stampato `SUCCESS`).

## 3) Cosa e diverso (e impatto)

1. Solver inversione cambiato: `SMOOTH` legacy -> `scipy.optimize.nnls` con regolarizzazione differenze seconde.
   - Python: `miget/core.py:576`, `miget/core.py:557`
   - Fortran: `File originali/VQBOHR.FOR:601`, `File originali/VQBOHR.FOR:646`, `File originali/VQBOHR.FOR:762`
   - Impatto: in diversi casi migliora RSS, in altri peggiora molto.

2. Ultimo bin V/Q: legacy impone `VAQ(N)=10000` (deadspace infinito), Python lo ha rimosso.
   - Python: `miget/core.py:488`
   - Fortran: `File originali/VQBOHR.FOR:322`
   - Impatto: stima deadspace e coda alta V/Q non equivalente al modello originale.

3. Pesi non equivalenti al legacy; in Python solo retention weights popolati, excretion weights non calcolati.
   - Python: `miget/core.py:394`, `miget/core.py:417`, `miget/core.py:539`
   - Fortran: `File originali/SHORT.FOR:456`, `File originali/SHORT.FOR:463`, `File originali/SHORT.FOR:477`
   - Impatto: modalità `Excretion` in UI (`miget/ui/inverse.py:27`) è numericamente fragile/non affidabile.

4. Correzioni legacy mancanti o semplificate:
   - `PAPV` periferico (0.95) non gestito (Python fissato a 1.0): `miget/core.py:127` vs `File originali/SHORT.FOR:372`, `File originali/SHORT.FOR:373`
   - `EXPFAC` fissato a 1.0: `miget/core.py:133` vs `File originali/SHORT.FOR:370`, `File originali/SHORT.FOR:653`
   - correzioni VE (`CORR`, `COR1`, `VESTAR`) assenti nel port.

5. Parser legacy usa `split()` e non fixed-width robusto.
   - Python: `miget/io.py:48`, `miget/io.py:70`, `miget/io.py:126`
   - Impatto osservato: file `degner_a1.RAW` e `degner_a2.RAW` non parsati per token concatenati (`105.219439.8`, `115.619569.3`).

6. Bohr/gas exchange implementato in forma semplificata rispetto a VQBOHR originale.
   - Python: `miget/bohr.py:23`, `miget/bohr.py:39`, `miget/bohr.py:73`
   - Fortran: `File originali/VQBOHR.FOR:1188`, `File originali/VQBOHR.FOR:1219`, `File originali/VQBOHR.FOR:1325`
   - Impatto: la parte inert gas reverse può essere buona anche quando la parte emogas non è legacy-equivalente.

## 4) Evidenza quantitativa locale

Comando eseguito:

`python3 tests/run_bulk_verification.py`

Output sintetico:

- 20 input trovati
- 18 processati (2 falliti per parsing `degner_*`)
- modern migliore in 11/18 casi
- modern peggiore in 7/18 casi (alcuni molto peggiori)
- media miglioramento RSS: negativa (`-2.48`, dominata da regressioni grandi)

Esempi:

- Miglioramenti forti: `caloro1` 130.0 -> 3.84, `caloro2` 18.0 -> 0.36
- Regressioni forti: `hariri1` 23.2 -> 195.38, `hariri2` 6.06 -> 51.09

CSV aggiornato:

`tests/bulk_results_rss.csv`

## 5) Valutazione: la soluzione attuale e la migliore?

Valutazione breve: **parzialmente**.

- Per alcuni pattern clinici e migliore del legacy (RSS molto piu basso).
- Non e globalmente migliore o legacy-equivalente su tutto il dataset.
- Ha gap funzionali importanti (excretion weighting, bin deadspace, parsing fixed-width, parti Bohr semplificate).

Quindi, allo stato attuale, la soluzione e una **buona base moderna**, ma **non ancora la migliore** se l'obiettivo e:

1. compatibilita legacy affidabile run-by-run;
2. robustezza su tutti i file storici;
3. equivalenza fisiologica completa anche lato Bohr.

## 6) Priorita tecniche consigliate

1. Ripristinare il bin deadspace legacy (`VAQ[-1]=10000`) con flag di compatibilita.
2. Implementare davvero i pesi excretion (e validare `Weighting Mode = Excretion`).
3. Porting completo formule `WT` di `SHORT.FOR` (non approssimazione unica retention).
4. Parser fixed-width robusto per file RAW legacy malformati/spaziati in modo irregolare.
5. Test di regressione obbligatori su tutto `Data study VentPerf` con soglie per RSS/shunt/deadspace.
6. Separare chiaramente modalita "Legacy-compatible" da modalita "Modern-optimized".
