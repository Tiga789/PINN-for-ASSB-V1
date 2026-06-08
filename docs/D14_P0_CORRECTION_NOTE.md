# Correction note

The earlier `D14_P0_XJTU_preparation_scripts.zip` was wrong because it contained placeholder P2D-preparation files and did not implement D14-P0 as defined in the project flow.

The corrected D14-P0 scope is:

```text
Freeze accepted baselines
+ audit local source files
+ audit XJTU D10/D12 scorecards
+ keep battery-8 flagged/excluded
+ generate fingerprints for no-regression
```

This package implements that scope.
