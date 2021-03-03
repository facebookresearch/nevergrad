
Hi I noticed the contest for benchmarking. I'm not sure where you want the PR to go but here are some ideas:

- I've been using Elo ratings for bb-optimizers to try to get around a few issues. Not sure if that's such a great idea but [here they are](https://github.com/microprediction/optimizer-elo-ratings/tree/main/results/leaderboards)
- Two objectives functions that might be good tests. The first is [markowitz](https://github.com/microprediction/humpday/blob/main/humpday/objectives/portfolio.py) portfolio optimization and generalizations. The second is the "horse race problem", whose [objective function](https://github.com/microprediction/humpday/blob/main/humpday/objectives/horse.py) sits most naturally in projective space. The problem is well motivated and there is a [paper](https://github.com/microprediction/winning/blob/main/docs/Horse_Race_Problem__SIAM_.pdf) that has been accepted by SIAM. Both of these problems are convenient for benchmarking as the dimension is arbitrary, and the solution is either known analytically or can be computed very quickly using the [winning](https://github.com/microprediction/winning) package.    



