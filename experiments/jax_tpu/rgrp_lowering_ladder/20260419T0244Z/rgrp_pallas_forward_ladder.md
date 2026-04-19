| Case | Variant | Mode | Batch | Seq | D | State | Projection | Trig | Unroll | Params | Compile s | Tok/s | Loss | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| seq256-mlp | mlp |  | 4 | 256 | 512 |  |  |  |  | 14,822,912 | 1.75 | 3,140,937.29 | 9.1238 | ok |
| seq256-rgrp-scan-unroll3 | rgrp | scan | 4 | 256 | 512 | block-diagonal-4-masked-dense | sequence | precompute | 3 | 12,987,392 | 1.51 | 751,699.23 | 9.1115 | ok |
| seq256-rgrp-pallas-forward | rgrp | pallas-forward | 4 | 256 | 512 | block-diagonal-4-masked-dense | sequence | precompute | 1 | 12,987,392 | 13.48 | 160,315.16 | 9.1116 | ok |
| seq512-mlp | mlp |  | 4 | 512 | 512 |  |  |  |  | 14,953,984 | 1.45 | 3,745,489.59 | 9.1168 | ok |
| seq512-rgrp-scan-unroll3 | rgrp | scan | 4 | 512 | 512 | block-diagonal-4-masked-dense | sequence | precompute | 3 | 13,118,464 | 3.20 | 772,461.72 | 9.1120 | ok |
| seq512-rgrp-pallas-forward | rgrp | pallas-forward | 4 | 512 | 512 | block-diagonal-4-masked-dense | sequence | precompute | 1 | 13,118,464 | 30.10 | 290,697.78 | 9.1120 | ok |
| seq1024-mlp | mlp |  | 4 | 1,024 | 512 |  |  |  |  | 15,216,128 | 1.74 | 2,320,680.68 | 9.1092 | ok |
| seq1024-rgrp-scan-unroll3 | rgrp | scan | 4 | 1,024 | 512 | block-diagonal-4-masked-dense | sequence | precompute | 3 | 13,380,608 | 3.02 | 688,022.58 | 9.1109 | ok |
| seq1024-rgrp-pallas-forward | rgrp | pallas-forward | 4 | 1,024 | 512 | block-diagonal-4-masked-dense | sequence | precompute | 1 | 13,380,608 | 87.55 | 448,687.72 | 9.1110 | ok |
