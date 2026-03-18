---
title: "Why has the 2-point FG% increased so much in the last seven years? (follow-up post)"
date: 2026-03-18
label: Statistical Analysis
summary: "Using investment performance attribution to explain the rise in NBA 2-point shooting percentage from 2017-18 to 2023-24."
tags: shooting, attribution, historical, reddit
visible: true
---

*[This post first appeared on Reddit in 2024, where I applied investment performance attribution to explain the rise in 2-point shooting percentages from 2017-18 to 2023-24.]*

This was a follow-up to [a previous Reddit post from the day before](https://www.reddit.com/r/nbadiscussion/comments/1ap6ss8/) where I argued that the improvement in offensive rating from 2017-18 to 2023-24 was driven, statistically at least, by the increase in 2-point shooting over that span.

Based on the comments on that post, I should be careful not to think about the rise in 2-point percentage in isolation from the rise in 3-point volume. More spacing should create better looks near the basket. My own instinct is to attack questions like this quantitatively, but the qualitative explanations people offered were useful because they suggested better hypotheses to test with the data. Sometimes the numbers can tell you a lot. Sometimes they can't tell you the whole story.

The table below shows 2-point shot data for 2017-18 and 2023-24. A few things jump out:

- There was a significant decrease (-8.4%) in the share of 2-point shots taken from 16 feet to the 3-point line, which is intuitively the least efficient 2-point area.
- There was a significant increase (+11.0%) in the share of 2-point shots taken from 3 to 10 feet.
- There was a mild decrease (-2.3%) in the share of 2-point shots taken at the rim.
- Efficiency improved at every range, but especially from 3 to 10 feet.

| Shot type | 2017-18 FG% | 2023-24 FG% | Difference | 2017-18<br>% taken | 2023-24<br>% taken | Difference |
| --- | --- | --- | --- | --- | --- | --- |
| All 2-pt | 51.0% | 54.6% | +3.6% | 66.3% | 60.9% | -5.4% |
| 0-3 feet | 65.8% | 69.6% | +3.8% | 42.4% | 40.1% | -2.3% |
| 3-10 feet | 39.4% | 45.7% | +6.3% | 23.5% | 34.5% | +11.0% |
| 10-16 feet | 41.5% | 44.8% | +3.3% | 16.0% | 15.8% | -0.2% |
| 16 feet-3PT | 40.0% | 40.7% | +0.7% | 18.1% | 9.7% | -8.4% |

For the "All 2-pt" row, the `% taken` columns are the share of all field-goal attempts that were 2-pointers. For the other rows, `% taken` is the share of all 2-point attempts that came from that range.

One thing I found interesting here: in 2017-18, the relationship between distance and efficiency was not monotonic. The worst 2-point shots were from 3 to 10 feet, not from long 2. By 2023-24 that had changed. Shooting percentage had become monotonic with distance, and 3-10 foot shots had turned into the second-best 2-point shots after attempts at the rim. I'd still love a really good qualitative explanation for that shift.

I wanted to explain the +3.6 percentage-point increase in 2-point shooting by splitting it into two parts:

- The change in 2-point shot mix, like taking fewer long 2s.
- The improvement in shooting percentage within each range.

To do that, I borrowed a technique from asset management called [performance attribution](https://en.wikipedia.org/wiki/Performance_attribution). In portfolio management, you decompose active return into three effects:

- Allocation: the impact of changing the weights across asset classes.
- Selection: the impact of doing better or worse within each asset class.
- Interaction: the leftover piece after accounting for allocation and selection.

The analogy here is straightforward enough. Think of the 2023-24 NBA season as the portfolio, 2017-18 as the benchmark, field-goal percentage at each range as the return, and the mix of 2-point attempts as the portfolio weights. Then the allocation effect measures the impact of changing the shot mix, and the selection effect measures the impact of better shooting at each distance.

The terminology is slightly awkward because in basketball you might naturally want to call shot mix "selection." That's not what selection means in attribution analysis, so I'm sticking with the finance vocabulary.

I'll skip the algebra and go straight to the results:

| Shot type | Shot mix (allocation) | Shot efficiency (selection) | Interaction | Total |
| --- | --- | --- | --- | --- |
| 0-3 feet | -0.34% | +1.61% | -0.09% | +1.18% |
| 3-10 feet | -1.27% | +1.48% | +0.69% | +0.90% |
| 10-16 feet | +0.02% | +0.53% | -0.01% | +0.54% |
| 16 feet-3PT | +0.93% | +0.13% | -0.06% | +1.00% |
| Totals | -0.67% | +3.75% | +0.54% | +3.62% |

Here's what I take from that breakdown:

- The decomposition matches the full +3.62% increase exactly, both by summing the rows and by summing the columns.
- The change in shot mix by itself was actually slightly harmful (-0.67%).
- That negative allocation result came mostly from the increase in 3-10 foot attempts, which used to be the least efficient 2-point shots.
- The improvement in shot efficiency explains more than 100% of the overall gain in 2-point percentage (+3.75% versus +3.62%).
- The biggest positive contribution came from shots at the rim. Players took a slightly smaller share of their 2-point attempts there, but they finished much better when they got there.
- The next biggest contribution came from taking fewer long 2s. That change in mix helped a lot even though shooting efficiency from that zone barely changed.
- The 3-10 foot range is the most interesting part of the story. Teams took many more shots there, which hurt the allocation term, but they also got dramatically better from that distance, which more than offset the shot-mix penalty.

So the short version is this: the jump in league-wide 2-point shooting from 2017-18 to 2023-24 was not mainly about taking a smarter mix of 2-point shots. If anything, shot mix was a slight drag. The story was that players simply shot better from every 2-point zone, especially from 3 to 10 feet.

That may sound obvious in hindsight, but I don't think it was obvious going in. It would have been perfectly plausible for better shot distribution to explain a large share of the increase. Instead, the data says the efficiency gains did almost all the work.
