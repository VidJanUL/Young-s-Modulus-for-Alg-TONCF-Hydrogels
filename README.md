# Young-s-Modulus-for-Alg-TONCF-Hydrogels

# Young's modulus model for alginate–TONCF composite hydrogels

Reference implementation for the paper *"A mathematical model for the Young's
modulus of ionically crosslinked alginate–anionic nanocellulose composite
hydrogels"*.

The model describes the non-monotonic (`M-shaped`) dependence of the Young's
modulus of ionically crosslinked alginate / TEMPO-oxidised cellulose nanofibre
(TONCF) composite hydrogels on the TONCF mixing fraction, with the alginate
concentration, the TONCF stock concentration and the Ca²⁺ concentration as
independent inputs.

## How it works

Parameters fall into four groups:

- **(A) Fixed from theory** — the reinforcement amplitudes `b1`, `b3` and the
  disruption rate `beta`, from the ionic-crosslinking framework.
- **(B) Fixed from the reference curve geometry** — the percolation and geometric
  thresholds, the power-law exponents, and the agglomeration centre and width.
- **(C) The single fitted parameter** — the fibre reinforcement efficiency `q`,
  obtained by a one-parameter least-squares fit to the 7.5 % alginate reference
  curve. **`q` is the only quantity ever passed to the optimiser.**
- **(D) Computed, not fitted** — `C_sat`, the alginate concentration at which the
  agglomeration amplitude `B` vanishes. `B` is read off as the depth of the
  modulus dip at each composition, and since `B = b3·(C_sat − C_Alg)` with the
  slope `b3` fixed from theory, `C_sat = C_Alg + B/b3` follows by arithmetic.
  (`B` depends on `q`, so (C) and (D) are iterated to self-consistency.)

With all of that settled, the model **predicts** — with no further fitting — the
modulus at the other alginate concentrations (5, 6.25, 8.75, 10 %) and at a higher
TONCF stock concentration (4 %). It prints the parameters by group, the fitted
`q`, the computed `C_sat`, and the prediction quality (R² and RMSE), and writes the
figures to `figures/`.

## Files

| File | Description |
|------|-------------|
| `model.py` | Model definition, calibration, prediction and figures |
| `data.csv` | Experimental moduli (one row per measured point) |
| `figures/` | Output figures (created on run) |
| `requirements.txt` | Python dependencies |

## Data format

`data.csv` has one row per measurement:

| Column | Meaning |
|--------|---------|
| `Ctoncf_pct` | TONCF stock concentration (wt %) |
| `Calg_pct` | alginate concentration (wt %) |
| `Ca_pct` | Ca²⁺ crosslinker concentration (wt %) |
| `f` | TONCF mixing fraction |
| `E_kPa` | measured Young's modulus (kPa) |

To use your own data, keep the same columns. The reference set used for
calibration is selected by the `REF_CALG` and `REF_CTONCF` constants at the top
of `model.py`.

## Running

```bash
pip install -r requirements.txt
python model.py
```

Tested with Python 3.10+.
