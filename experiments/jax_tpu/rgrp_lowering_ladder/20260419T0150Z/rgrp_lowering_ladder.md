| Case | Variant | State | Projection | Trig | Unroll | Params | Compile s | Tok/s | Loss | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mlp-control | mlp |  |  |  |  | 14,822,912 | 3.78 | 168,830.62 | 9.1238 | ok |
| rgrp-default | rgrp | block-diagonal-4-masked-dense | sequence | precompute | 1 | 12,987,392 | 4.31 | 128,326.71 | 9.1116 | ok |
| state-dense | rgrp | dense | sequence | precompute | 1 | 12,987,392 | 4.43 | 124,388.77 | 9.1245 | ok |
| state-block4-grouped | rgrp | block-diagonal-4 | sequence | precompute | 1 | 12,594,176 | 4.45 | 86,666.02 | 9.1171 | ok |
| trig-inside-scan | rgrp | block-diagonal-4-masked-dense | sequence | scan | 1 | 12,987,392 | 4.26 | 109,420.45 | 9.1117 | ok |
| projection-inside-scan | rgrp | block-diagonal-4-masked-dense | scan | scan | 1 | 12,987,392 | 4.24 | 108,438.16 | 9.1116 | ok |
| unroll-2 | rgrp | block-diagonal-4-masked-dense | sequence | precompute | 2 | 12,987,392 | 4.77 | 121,108.70 | 9.1116 | ok |
| unroll-4 | rgrp | block-diagonal-4-masked-dense | sequence | precompute | 4 | 12,987,392 | 5.37 | 142,740.37 | 9.1116 | ok |
| unroll-8 | rgrp | block-diagonal-4-masked-dense | sequence | precompute | 8 | 12,987,392 | 7.14 | 129,227.31 | 9.1117 | ok |
