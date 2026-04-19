| Case | Variant | Mode | Chunk | Batch | Seq | D | State | Projection | Trig | Unroll | Params | Compile s | Tok/s | Loss | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| seq256-mlp | mlp |  |  | 4 | 256 | 512 |  |  |  |  | 14,822,912 | 1.78 | 3,233,228.50 | 9.1238 | ok |
| seq256-rgrp-scan-unroll3 | rgrp | scan | 256 | 4 | 256 | 512 | block-diagonal-4-masked-dense | sequence | precompute | 3 | 12,987,392 | 1.53 | 758,862.01 | 9.1115 | ok |
| seq256-rgrp-pallas-forward | rgrp | pallas-forward | 256 | 4 | 256 | 512 | block-diagonal-4-masked-dense | sequence | precompute | 1 | 12,987,392 | 13.68 | 160,728.95 | 9.1116 | ok |
| seq256-rgrp-pallas-block-tiled-c128 | rgrp | pallas-block-tiled-forward | 128 | 4 | 256 | 512 | block-diagonal-4-masked-dense | sequence | precompute | 1 | 12,987,392 | 21.14 | 164,190.69 | 9.1116 | ok |
| seq512-mlp | mlp |  |  | 4 | 512 | 512 |  |  |  |  | 14,953,984 | 1.53 | 3,801,110.36 | 9.1168 | ok |
| seq512-rgrp-scan-unroll3 | rgrp | scan | 256 | 4 | 512 | 512 | block-diagonal-4-masked-dense | sequence | precompute | 3 | 13,118,464 | 3.21 | 770,767.22 | 9.1120 | ok |
| seq512-rgrp-pallas-forward | rgrp | pallas-forward | 256 | 4 | 512 | 512 | block-diagonal-4-masked-dense | sequence | precompute | 1 | 13,118,464 | 30.54 | 290,730.28 | 9.1120 | ok |
| seq512-rgrp-pallas-block-tiled-c128 | rgrp | pallas-block-tiled-forward | 128 | 4 | 512 | 512 | block-diagonal-4-masked-dense | sequence | precompute | 1 | 13,118,464 | 21.30 | 306,078.45 | 9.1120 | ok |
| seq1024-mlp | mlp |  |  | 4 | 1,024 | 512 |  |  |  |  | 15,216,128 | 1.78 | 2,317,385.92 | 9.1092 | ok |
| seq1024-rgrp-scan-unroll3 | rgrp | scan | 256 | 4 | 1,024 | 512 | block-diagonal-4-masked-dense | sequence | precompute | 3 | 13,380,608 | 3.03 | 686,487.27 | 9.1109 | ok |
| seq1024-rgrp-pallas-forward | rgrp | pallas-forward | 256 | 4 | 1,024 | 512 | block-diagonal-4-masked-dense | sequence | precompute | 1 | 13,380,608 | 87.50 | 448,113.33 | 9.1110 | ok |
| seq1024-rgrp-pallas-block-tiled-c128 | rgrp | pallas-block-tiled-forward | 128 | 4 | 1,024 | 512 | block-diagonal-4-masked-dense | sequence | precompute | 1 | 13,380,608 | 21.28 | 483,600.05 | 9.1110 | ok |
