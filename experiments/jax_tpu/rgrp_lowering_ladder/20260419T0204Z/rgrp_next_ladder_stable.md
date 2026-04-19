| Case | Variant | Batch | Seq | D | State | Projection | Trig | Unroll | Params | Compile s | Tok/s | Loss | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mlp-control | mlp | 4 | 256 | 512 |  |  |  |  | 14,822,912 | 3.74 | 143,926.97 | 9.1238 | ok |
| unroll-1 | rgrp | 4 | 256 | 512 | block-diagonal-4-masked-dense | sequence | precompute | 1 | 12,987,392 | 4.34 | 109,946.34 | 9.1116 | ok |
| unroll-3 | rgrp | 4 | 256 | 512 | block-diagonal-4-masked-dense | sequence | precompute | 3 | 12,987,392 | 5.60 | 136,459.83 | 9.1116 | ok |
| unroll-4 | rgrp | 4 | 256 | 512 | block-diagonal-4-masked-dense | sequence | precompute | 4 | 12,987,392 | 5.50 | 129,154.11 | 9.1116 | ok |
| unroll-5 | rgrp | 4 | 256 | 512 | block-diagonal-4-masked-dense | sequence | precompute | 5 | 12,987,392 | 6.35 | 126,510.05 | 9.1116 | ok |
| unroll-6 | rgrp | 4 | 256 | 512 | block-diagonal-4-masked-dense | sequence | precompute | 6 | 12,987,392 | 8.70 | 135,402.27 | 9.1116 | ok |
| seq512-mlp | mlp | 4 | 512 | 512 |  |  |  |  | 14,953,984 | 5.61 | 297,221.64 | 9.1168 | ok |
| seq512-rgrp-unroll4 | rgrp | 4 | 512 | 512 | block-diagonal-4-masked-dense | sequence | precompute | 4 | 13,118,464 | 6.96 | 163,925.33 | 9.1121 | ok |
| seq1024-mlp | mlp | 4 | 1,024 | 512 |  |  |  |  | 15,216,128 | 5.88 | 427,136.38 | 9.1092 | ok |
| seq1024-rgrp-unroll4 | rgrp | 4 | 1,024 | 512 | block-diagonal-4-masked-dense | sequence | precompute | 4 | 13,380,608 | 8.27 | 183,856.39 | 9.1110 | ok |
