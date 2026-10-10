# Train/serve skew

## Processor in sync with the scorer

Compared **3958** served events against offline recomputation (0 not found offline). Status: **OK** (threshold 0.10% mismatches per feature).

| feature | mismatches | rate | max abs diff | examples |
|---|---:|---:|---:|---|
| card_txn_count_1m | 0 | 0.00% | 0 |  |
| card_txn_count_1h | 0 | 0.00% | 0 |  |
| card_txn_count_24h | 0 | 0.00% | 0 |  |
| card_small_txn_count_1h | 0 | 0.00% | 0 |  |
| card_amount_sum_1h | 0 | 0.00% | 0 |  |
| card_amount_sum_24h | 0 | 0.00% | 0 |  |
| card_distinct_merchants_1h | 0 | 0.00% | 0 |  |
| card_distinct_merchants_24h | 0 | 0.00% | 0 |  |
| card_distinct_countries_24h | 0 | 0.00% | 0 |  |
| merchant_txn_count_1h | 0 | 0.00% | 0 |  |
| merchant_txn_count_24h | 0 | 0.00% | 0 |  |
| merchant_small_txn_count_1h | 0 | 0.00% | 0 |  |
| merchant_distinct_cards_1h | 0 | 0.00% | 0 |  |
| card_seconds_since_last_txn | 0 | 0.00% | 0 |  |
| merchant_fraud_rate_30d | 0 | 0.00% | 0 |  |
| amount | 0 | 0.00% | 0 |  |
| is_foreign | 0 | 0.00% | 0 |  |
| is_online | 0 | 0.00% | 0 |  |
| hour_of_day | 0 | 0.00% | 0 |  |
| category_code | 0 | 0.00% | 0 |  |
| card_amount_ratio_24h | 0 | 0.00% | 0 |  |

## Store started without bootstrapping last-seen

Compared **3958** served events against offline recomputation (0 not found offline). Status: **SKEW DETECTED** (threshold 0.10% mismatches per feature).

| feature | mismatches | rate | max abs diff | examples |
|---|---:|---:|---:|---|
| card_seconds_since_last_txn | 493 | 12.46% | inf | t00599398, t00599403, t00599413 |
| card_txn_count_1m | 0 | 0.00% | 0 |  |
| card_txn_count_1h | 0 | 0.00% | 0 |  |
| card_txn_count_24h | 0 | 0.00% | 0 |  |
| card_small_txn_count_1h | 0 | 0.00% | 0 |  |
| card_amount_sum_1h | 0 | 0.00% | 0 |  |
| card_amount_sum_24h | 0 | 0.00% | 0 |  |
| card_distinct_merchants_1h | 0 | 0.00% | 0 |  |
| card_distinct_merchants_24h | 0 | 0.00% | 0 |  |
| card_distinct_countries_24h | 0 | 0.00% | 0 |  |
| merchant_txn_count_1h | 0 | 0.00% | 0 |  |
| merchant_txn_count_24h | 0 | 0.00% | 0 |  |
| merchant_small_txn_count_1h | 0 | 0.00% | 0 |  |
| merchant_distinct_cards_1h | 0 | 0.00% | 0 |  |
| merchant_fraud_rate_30d | 0 | 0.00% | 0 |  |
| amount | 0 | 0.00% | 0 |  |
| is_foreign | 0 | 0.00% | 0 |  |
| is_online | 0 | 0.00% | 0 |  |
| hour_of_day | 0 | 0.00% | 0 |  |
| category_code | 0 | 0.00% | 0 |  |
| card_amount_ratio_24h | 0 | 0.00% | 0 |  |

## Processor lagging 50 events behind

Compared **3958** served events against offline recomputation (0 not found offline). Status: **SKEW DETECTED** (threshold 0.10% mismatches per feature).

| feature | mismatches | rate | max abs diff | examples |
|---|---:|---:|---:|---|
| merchant_txn_count_24h | 1858 | 46.94% | 15 | t00599400, t00599402, t00599404 |
| merchant_txn_count_1h | 1843 | 46.56% | 15 | t00599400, t00599402, t00599404 |
| merchant_distinct_cards_1h | 1804 | 45.58% | 15 | t00599400, t00599402, t00599404 |
| merchant_small_txn_count_1h | 381 | 9.63% | 4 | t00599421, t00599439, t00599441 |
| card_txn_count_1h | 140 | 3.54% | 3 | t00599421, t00599582, t00599586 |
| card_txn_count_24h | 140 | 3.54% | 3 | t00599421, t00599582, t00599586 |
| card_amount_sum_1h | 140 | 3.54% | 737.9 | t00599421, t00599582, t00599586 |
| card_amount_sum_24h | 140 | 3.54% | 737.9 | t00599421, t00599582, t00599586 |
| card_seconds_since_last_txn | 140 | 3.54% | 4.439e+05 | t00599421, t00599582, t00599586 |
| card_amount_ratio_24h | 140 | 3.54% | inf | t00599421, t00599582, t00599586 |
| card_distinct_merchants_1h | 117 | 2.96% | 2 | t00599421, t00599582, t00599586 |
| card_distinct_merchants_24h | 98 | 2.48% | 2 | t00599582, t00599586, t00599609 |
| card_small_txn_count_1h | 55 | 1.39% | 3 | t00599421, t00599586, t00599610 |
| card_distinct_countries_24h | 48 | 1.21% | 1 | t00599586, t00599609, t00599610 |
| card_txn_count_1m | 10 | 0.25% | 1 | t00600501, t00600574, t00601404 |
| merchant_fraud_rate_30d | 0 | 0.00% | 0 |  |
| amount | 0 | 0.00% | 0 |  |
| is_foreign | 0 | 0.00% | 0 |  |
| is_online | 0 | 0.00% | 0 |  |
| hour_of_day | 0 | 0.00% | 0 |  |
| category_code | 0 | 0.00% | 0 |  |
