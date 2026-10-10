# Feature drift (PSI)

**Training reference vs. the last 7 days (offline features)**

| feature | PSI | status |
|---|---:|---|
| is_foreign | 0.230 | moderate |
| merchant_fraud_rate_30d | 0.165 | moderate |
| is_online | 0.086 | stable |
| merchant_small_txn_count_1h | 0.076 | stable |
| category_code | 0.046 | stable |
| amount | 0.038 | stable |
| merchant_txn_count_24h | 0.033 | stable |
| card_amount_sum_24h | 0.020 | stable |
| card_amount_ratio_24h | 0.007 | stable |
| merchant_distinct_cards_1h | 0.007 | stable |
| card_distinct_merchants_24h | 0.005 | stable |
| merchant_txn_count_1h | 0.005 | stable |
| card_seconds_since_last_txn | 0.001 | stable |
| card_txn_count_24h | 0.001 | stable |
| card_txn_count_1h | 0.001 | stable |
| card_distinct_merchants_1h | 0.001 | stable |
| hour_of_day | 0.000 | stable |
| card_amount_sum_1h | 0.000 | stable |
| card_distinct_countries_24h | 0.000 | stable |
| card_txn_count_1m | 0.000 | stable |
| card_small_txn_count_1h | 0.000 | stable |

**Training reference vs. one replayed day of online traffic (served features)**

| feature | PSI | status |
|---|---:|---|
| merchant_fraud_rate_30d | 0.814 | major |
| is_foreign | 0.259 | major |
| is_online | 0.094 | stable |
| merchant_small_txn_count_1h | 0.082 | stable |
| merchant_txn_count_24h | 0.064 | stable |
| category_code | 0.054 | stable |
| amount | 0.048 | stable |
| card_amount_sum_24h | 0.017 | stable |
| merchant_distinct_cards_1h | 0.017 | stable |
| card_seconds_since_last_txn | 0.015 | stable |
| merchant_txn_count_1h | 0.014 | stable |
| card_txn_count_1h | 0.010 | stable |
| card_distinct_merchants_1h | 0.010 | stable |
| card_amount_ratio_24h | 0.008 | stable |
| card_txn_count_24h | 0.003 | stable |
| card_distinct_merchants_24h | 0.003 | stable |
| hour_of_day | 0.002 | stable |
| card_amount_sum_1h | 0.002 | stable |
| card_distinct_countries_24h | 0.000 | stable |
| card_txn_count_1m | 0.000 | stable |
| card_small_txn_count_1h | 0.000 | stable |
