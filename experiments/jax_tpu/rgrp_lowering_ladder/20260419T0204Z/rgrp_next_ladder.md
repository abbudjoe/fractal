| Case | Variant | Batch | Seq | D | State | Projection | Trig | Unroll | Params | Compile s | Tok/s | Loss | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mlp-control | mlp | 4 | 256 | 512 |  |  |  |  | 14,822,912 | 3.67 | 132,543.01 | 9.1238 | ok |
| unroll-1 | rgrp | 4 | 256 | 512 | block-diagonal-4-masked-dense | sequence | precompute | 1 | 12,987,392 | 4.42 | 118,457.19 | 9.1116 | ok |
| unroll-3 | rgrp | 4 | 256 | 512 | block-diagonal-4-masked-dense | sequence | precompute | 3 | 12,987,392 | 5.60 | 139,555.31 | 9.1116 | ok |
| unroll-4 | rgrp | 4 | 256 | 512 | block-diagonal-4-masked-dense | sequence | precompute | 4 | 12,987,392 | 5.38 | 130,412.28 | 9.1116 | ok |
| unroll-5 | rgrp | 4 | 256 | 512 | block-diagonal-4-masked-dense | sequence | precompute | 5 | 12,987,392 | 6.39 | 118,823.84 | 9.1116 | ok |
| unroll-6 | rgrp | 4 | 256 | 512 | block-diagonal-4-masked-dense | sequence | precompute | 6 | 12,987,392 | 8.79 | 131,084.96 | 9.1116 | ok |
| seq512-mlp | mlp | 4 | 512 | 512 |  |  |  |  | 14,953,984 | 5.52 | 329,947.85 | 9.1168 | ok |
| seq512-rgrp-unroll4 | rgrp | 4 | 512 | 512 | block-diagonal-4-masked-dense | sequence | precompute | 4 | 13,118,464 | 7.15 | 159,885.14 | 9.1121 | ok |
| seq1024-mlp | mlp | 4 | 1,024 | 512 |  |  |  |  | 15,216,128 | 5.88 | 419,082.35 | 9.1092 | ok |
| seq1024-rgrp-unroll4 | rgrp | 4 | 1,024 | 512 | block-diagonal-4-masked-dense | sequence | precompute | 4 | 13,380,608 | 8.33 | 181,737.22 | 9.1110 | ok |
