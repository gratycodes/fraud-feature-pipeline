# Model card: synthetic card-fraud scorer

## Summary

A LightGBM binary classifier that estimates the probability that a card transaction is fraudulent. An expected-cost rule turns that probability into an approve/block decision. It was trained and evaluated **only on synthetic data** produced by `src/ffp/generator.py` (seed 7). This is a demonstration artefact for the feature pipeline in this repository. **It is not fit for real payment decisions.**

## Intended use

- To demonstrate point-in-time-correct training, online/offline feature parity, calibration, and cost-based decisions.
- Not intended for: scoring real transactions, benchmarking fraud models, or drawing conclusions about real fraud rates or patterns.

## Data

- 641,118 synthetic transactions over 150 days, from 3,000 cards at 500 merchants.
- 3,024 are fraudulent (0.47%): card testing 1,662, account takeover 842, merchant-compromise ring 520.
- Labels arrive with a delay (median 12 days, long tail). 8% of fraud is never reported, so those events are treated as legitimate in training.
- Legitimate look-alikes are included on purpose: transit tap bursts look like card testing, and travel looks like foreign account takeover.
- From day 138 there's a holiday-season shift: amounts ×1.35 and more foreign online shopping.
- No real people, cards, merchants or payment systems are involved.

## Timeline

| days | use |
|---|---|
| 0–30 | feature warm-up only (30-day merchant rate needs history) |
| 30–100 | training (299,763 rows, 1,225 reported frauds) |
| 100–108 | early stopping (best iteration 176) |
| 108–115 | calibration-method selection |
| 115–130 | label maturity gap; training labels are those known on day 130 |
| 130–150 | held-out test (85,188 rows, 406 frauds reported by the end of the data) |

## Features

There are 21 features, all defined in `src/ffp/features.py` and computed the same way offline (DuckDB) and online (Redis):

- Card velocity: counts and amount sums over 1 minute, 1 hour and 24 hours; small-amount count; distinct merchants and countries.
- Merchant velocity: counts over 1 hour and 24 hours, small-amount count, distinct cards.
- Seconds since the card's previous transaction, and amount relative to the card's 24-hour total.
- The merchant's 30-day smoothed chargeback rate, using only labels known at the previous daily snapshot.
- Request attributes: amount, foreign/online flags, hour (UTC), category.

## Performance on the held-out period (synthetic)

| metric | value |
|---|---|
| PR-AUC | 0.593 (bootstrap 95% CI 0.544–0.639) |
| ROC-AUC | 0.967 |
| Brier score | 0.00276 |
| ECE (quantile bins) | 0.0016 |

Calibration was chosen on a later window by Brier score. Raw scores (0.00277) beat Platt scaling (0.00287). Isotonic regression is only considered with at least 500 positives, and the calibration window has 171.

### Decision rule

Block if `p × (amount + $15) > (1 − p) × $5`, where $15 is the handling cost of a missed fraud and $5 is the friction cost of a blocked good customer. On the test period, this blocks 2,005 transactions with 15.0% precision and 74.1% recall. It costs $33,402 in total, against $111,226 for the rules baseline and $140,392 for approving everything.

### Recall by fraud pattern (including unreported fraud)

| pattern | model | rules baseline |
|---|---:|---:|
| card testing | 97% | 56% |
| account takeover | 93% | 15% |
| merchant ring (new in test period) | 22% | 0% |

The model is weak on a newly compromised merchant, because its main signal (the merchant's chargeback rate) only arrives weeks after the fraud.

## Limitations and risks

- **Synthetic-only.** The fraud patterns were designed by hand, and the model rediscovers them. Real fraudsters adapt to the model, and none of these numbers transfer.
- **Proxy features.** `is_foreign`, country and category can correlate with nationality, travel or income. On real data the model would need error rates checked per segment (for example, the false-block rate for foreign-resident cardholders) before any use.
- **Label bias.** Unreported fraud is labelled legitimate. Real chargeback data has the same problem, and it's worse for small amounts.
- **Drift.** The test period includes a seasonal shift. Drift is measured by PSI (see `reports/drift.md`), and there's no automated retraining.
- **Thresholds depend on the costs.** The $15 and $5 figures are assumptions. Change them and the decision rule changes.

## Artefacts

- `data/model/model.txt` holds the LightGBM text model. `metadata.json` holds the features, threshold, cost model and metrics, `calibration.json` holds the calibrator, and `reference.json` holds the PSI reference bins. These are plain files with no pickle.
- Reproduce everything with `ffp demo` (a minute or two on 8 vCPUs).
