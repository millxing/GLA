# 2025-26 Opponent-115 Split Model

This is an in-sample explanatory model for the current season only.
Season definition matches League Summary: include `regular_season` and `nba_cup_semi`, exclude `nba_cup_final`.

Method:
- Build each team's base season ORtg/DRtg/Net and split ratings in the two opponent buckets.
- Naive baseline: `base rating + league-average split delta` within the same split bucket.
- Model: weighted linear regression using base ORtg, base DRtg, bucket indicator, split games/share, and mean opponent season ORtg/DRtg inside the bucket.
- Sample weights are split games, so 40-game splits count more than 32-game splits.

## Fit Summary

```text
split_type     | target           | naive_weighted_mae | model_weighted_mae | naive_weighted_r2 | model_weighted_r2
---------------|------------------|--------------------|--------------------|-------------------|------------------
opp_def_rating | split_def_rating | 1.039              | 0.913              | 0.892             | 0.929            
opp_def_rating | split_off_rating | 1.064              | 0.988              | 0.905             | 0.922            
opp_off_rating | split_def_rating | 0.839              | 0.825              | 0.946             | 0.952            
opp_off_rating | split_off_rating | 1.239              | 1.062              | 0.853             | 0.878            
```

## Split by opponent season ORtg, bucket > 115

Most positive net residuals:

```text
team | actual_net | expected_net | residual_net
-----|------------|--------------|-------------
IND  | -9.19      | -12.74       | 3.56        
MIA  | -0.65      | -3.29        | 2.64        
CLE  | 1.12       | -1.32        | 2.44        
CHI  | -7.69      | -10.12       | 2.43        
SAS  | 5.17       | 3.19         | 1.98        
```

Most negative net residuals:

```text
team | actual_net | expected_net | residual_net
-----|------------|--------------|-------------
PHI  | -7.69      | -5.32        | -2.37       
LAL  | -6.72      | -4.42        | -2.30       
ATL  | -5.55      | -3.30        | -2.25       
CHA  | -2.11      | -0.23        | -1.88       
BKN  | -16.91     | -15.03       | -1.88       
```

## Split by opponent season ORtg, bucket <= 115

Most positive net residuals:

```text
team | actual_net | expected_net | residual_net
-----|------------|--------------|-------------
ATL  | 8.37       | 6.16         | 2.21        
PHI  | 7.04       | 4.89         | 2.16        
BKN  | -3.89      | -6.03        | 2.15        
LAL  | 7.68       | 5.76         | 1.92        
PHX  | 7.94       | 6.21         | 1.73        
```

Most negative net residuals:

```text
team | actual_net | expected_net | residual_net
-----|------------|--------------|-------------
IND  | -6.56      | -3.66        | -2.90       
CHI  | -3.05      | -0.67        | -2.38       
CLE  | 6.50       | 8.51         | -2.01       
DAL  | -2.28      | -0.40        | -1.89       
DET  | 10.87      | 12.63        | -1.76       
```

## Split by opponent season DRtg, bucket > 115

Most positive net residuals:

```text
team | actual_net | expected_net | residual_net
-----|------------|--------------|-------------
LAL  | 9.24       | 6.49         | 2.75        
PHI  | 7.96       | 5.73         | 2.23        
MEM  | 0.66       | -1.56        | 2.22        
BKN  | -1.91      | -4.01        | 2.10        
UTA  | -1.78      | -3.82        | 2.04        
```

Most negative net residuals:

```text
team | actual_net | expected_net | residual_net
-----|------------|--------------|-------------
DEN  | 7.18       | 10.57        | -3.39       
IND  | -4.29      | -1.46        | -2.83       
DET  | 11.64      | 13.81        | -2.17       
CHI  | -1.04      | 1.08         | -2.12       
HOU  | 7.03       | 8.99         | -1.96       
```

## Split by opponent season DRtg, bucket <= 115

Most positive net residuals:

```text
team | actual_net | expected_net | residual_net
-----|------------|--------------|-------------
DEN  | 3.70       | 0.90         | 2.80        
IND  | -10.04     | -12.00       | 1.96        
HOU  | 3.69       | 2.13         | 1.56        
SAS  | 6.07       | 4.66         | 1.41        
CHI  | -8.04      | -9.35        | 1.31        
```

Most negative net residuals:

```text
team | actual_net | expected_net | residual_net
-----|------------|--------------|-------------
MEM  | -11.56     | -9.46        | -2.10       
LAL  | -4.52      | -2.47        | -2.05       
PHI  | -6.19      | -4.39        | -1.81       
UTA  | -13.22     | -11.67       | -1.55       
POR  | -5.34      | -4.02        | -1.32       
```
