# Large-scale global optimization

## Technical report

```bibtex
@techreport{lsgo2013,
    author      = {Li, Xiaodong and Tang, Ke and Omidvar, Mohammmad Nabi and Yang, Zhenyu and Qin, Kai},
    title       = {Benchmark Functions for the CEC'2013 Special Session and Competition on Large-Scale Global Optimization},
    institution = {RMIT University},
    year        = {2013}
}
```

## Differences

Here are a few notable differences from the paper:

- `F3`, `F6`, `F10`: in the Octave version of the paper, the Ackley function lacks ` Tosz`, `Tasy` and `Lambda`. This implementation reproduces the CPP implementation, which has them activated.
- `F7`: in both the Octave version and the CPP version, `Tasy` and `Tosz` are missing for the side sphere loss. Since this implementation aims at reproducing the CPP results, it is missing here too.
- `F14` optimum is not specified since it is not straightforward to compute.

## API

More to come?
