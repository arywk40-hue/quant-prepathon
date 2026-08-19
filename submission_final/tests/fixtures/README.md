# Test fixtures

Synthetic edge cases are defined inline in the unit tests so that no raw
dataset or generated result is duplicated in the repository. The existing
phase tests cover malformed timestamps, gaps, invalid prices, structural NaNs,
validity masks, and Parquet round-trips.
