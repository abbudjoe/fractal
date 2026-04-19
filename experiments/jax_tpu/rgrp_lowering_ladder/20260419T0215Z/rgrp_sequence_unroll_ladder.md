| Case | Variant | Batch | Seq | D | State | Projection | Trig | Unroll | Params | Compile s | Tok/s | Loss | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| seq512-mlp | mlp | 4 | 512 | 512 |  |  |  |  | 14,953,984 | 5.62 | 338,917.91 | 9.1168 | ok |
| seq512-rgrp-unroll3 | rgrp | 4 | 512 | 512 | block-diagonal-4-masked-dense | sequence | precompute | 3 | 13,118,464 | 8.62 | 187,648.49 | 9.1121 | ok |
| seq512-rgrp-unroll4 | rgrp | 4 | 512 | 512 | block-diagonal-4-masked-dense | sequence | precompute | 4 | 13,118,464 | 6.95 | 167,513.35 | 9.1121 | ok |
| seq1024-mlp | mlp | 4 | 1,024 | 512 |  |  |  |  | 15,216,128 | 5.98 | 426,399.74 | 9.1092 | ok |
| seq1024-rgrp-unroll3 | rgrp | 4 | 1,024 | 512 | block-diagonal-4-masked-dense | sequence | precompute | 3 | 13,380,608 | 8.24 | 201,841.16 | 9.1110 | ok |
| seq1024-rgrp-unroll4 | rgrp | 4 | 1,024 | 512 | block-diagonal-4-masked-dense | sequence | precompute | 4 | 13,380,608 | 8.17 | 188,017.98 | 9.1110 | ok |
