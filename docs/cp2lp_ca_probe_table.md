## 1. Dose response: chain damage against edit size

| run | g | sigma_c nm | incoherence | dCA-CA nm | in native sd | CA-CA sd nm | outside window | chain_intact |
|---|---|---|---|---|---|---|---|---|
| real_reference | 1300 | 0.0000 | n/a | 0.0000 | 0.0 | 0.0065 | 3.1% | 0.9609 |
| onepass_contactonly__g005300 | 5300 | 0.0313 | 0.332 | 0.0147 | 3.1 | 0.0150 | 43.4% | 0.7188 |
| onepass_contactonly__g001300 | 1300 | 0.0332 | 0.414 | 0.0194 | 4.0 | 0.0191 | 50.0% | 0.5156 |
| onepass_gan__g001300 | 1300 | 0.1094 | 0.392 | 0.0607 | 12.6 | 0.0547 | 87.7% | 0.0000 |
| onepass_gan__g005300 | 5300 | 0.1705 | 0.296 | 0.0713 | 14.9 | 0.0704 | 80.6% | 0.0156 |

incoherence across 4 edited runs: min 0.296 max 0.414 spread 0.118

## 2. Apportionment: which decoder input carries the damage

**onepass_contactonly__g001300** (g=1300)

| decode | chain_intact | cn_dev median nm | cn_dev p99 nm | cost vs ceiling |
|---|---|---|---|---|
| source CA  + source z   (AE round trip = ceiling) | 0.9844 | 0.0029 | 0.0141 | 0.0% |
| source CA  + EDITED z   (latents alone) | 0.7031 | 0.0036 | 0.0283 | 28.6% |
| EDITED CA  + source z   (CA alone) | 0.2812 | 0.0057 | 0.0438 | 71.4% |
| EDITED CA  + EDITED z   (the actual output) | 0.5156 | 0.0050 | 0.0340 | 47.6% |
| projected CA + EDITED z (bond lengths repaired) | 0.6719 | 0.0043 | 0.0442 | 31.7% |
| projected CA + source z | 0.7188 | 0.0043 | 0.0307 | 27.0% |

**onepass_contactonly__g005300** (g=5300)

| decode | chain_intact | cn_dev median nm | cn_dev p99 nm | cost vs ceiling |
|---|---|---|---|---|
| source CA  + source z   (AE round trip = ceiling) | 0.9844 | 0.0029 | 0.0141 | 0.0% |
| source CA  + EDITED z   (latents alone) | 0.7656 | 0.0037 | 0.0272 | 22.2% |
| EDITED CA  + source z   (CA alone) | 0.5156 | 0.0053 | 0.0332 | 47.6% |
| EDITED CA  + EDITED z   (the actual output) | 0.7188 | 0.0045 | 0.0284 | 27.0% |
| projected CA + EDITED z (bond lengths repaired) | 0.6875 | 0.0042 | 0.0347 | 30.2% |
| projected CA + source z | 0.7812 | 0.0038 | 0.0264 | 20.6% |

**onepass_gan__g001300** (g=1300)

| decode | chain_intact | cn_dev median nm | cn_dev p99 nm | cost vs ceiling |
|---|---|---|---|---|
| source CA  + source z   (AE round trip = ceiling) | 0.9844 | 0.0029 | 0.0141 | 0.0% |
| source CA  + EDITED z   (latents alone) | 0.6250 | 0.0058 | 0.0257 | 36.5% |
| EDITED CA  + source z   (CA alone) | 0.0469 | 0.0182 | 0.3140 | 95.2% |
| EDITED CA  + EDITED z   (the actual output) | 0.0000 | 0.0188 | 0.2828 | 100.0% |
| projected CA + EDITED z (bond lengths repaired) | 0.1406 | 0.0108 | 0.1716 | 85.7% |
| projected CA + source z | 0.2031 | 0.0095 | 0.1782 | 79.4% |

**onepass_gan__g005300** (g=5300)

| decode | chain_intact | cn_dev median nm | cn_dev p99 nm | cost vs ceiling |
|---|---|---|---|---|
| source CA  + source z   (AE round trip = ceiling) | 0.9844 | 0.0029 | 0.0141 | 0.0% |
| source CA  + EDITED z   (latents alone) | 0.4219 | 0.0069 | 0.0898 | 57.1% |
| EDITED CA  + source z   (CA alone) | 0.0156 | 0.0156 | 0.3372 | 98.4% |
| EDITED CA  + EDITED z   (the actual output) | 0.0156 | 0.0150 | 0.2482 | 98.4% |
| projected CA + EDITED z (bond lengths repaired) | 0.0781 | 0.0112 | 0.1572 | 92.1% |
| projected CA + source z | 0.1250 | 0.0114 | 0.2255 | 87.3% |

**real_reference** (g=1300)

| decode | chain_intact | cn_dev median nm | cn_dev p99 nm | cost vs ceiling |
|---|---|---|---|---|
| source CA  + source z   (AE round trip = ceiling) | 0.9609 | 0.0028 | 0.0157 | 0.0% |
| source CA  + EDITED z   (latents alone) | 0.9609 | 0.0028 | 0.0157 | 0.0% |
| EDITED CA  + source z   (CA alone) | 0.9609 | 0.0028 | 0.0157 | 0.0% |
| EDITED CA  + EDITED z   (the actual output) | 0.9609 | 0.0028 | 0.0157 | 0.0% |
| projected CA + EDITED z (bond lengths repaired) | 0.9453 | 0.0027 | 0.0184 | 1.6% |
| projected CA + source z | 0.9453 | 0.0027 | 0.0184 | 1.6% |

## 3. The CA trace against its two native constraints

native (measured): CA-CA 3.838 +- 0.054 A, none above 3.98; bend 97.3 deg, none below 74.5

**onepass_contactonly__g001300**

| track | CA-CA med nm | CA-CA sd nm | bend min | bend p1 | bend med | bend p99 | bend max |
|---|---|---|---|---|---|---|---|
| source | 0.3835 | 0.00589 | 74.5 | 83.8 | 98.2 | 148.0 | 160.4 |
| edited | 0.3827 | 0.01905 | 74.9 | 81.8 | 97.2 | 147.9 | 153.6 |
| projected | 0.3809 | 0.00000 | 72.1 | 79.4 | 97.7 | 148.3 | 151.6 |

**onepass_contactonly__g005300**

| track | CA-CA med nm | CA-CA sd nm | bend min | bend p1 | bend med | bend p99 | bend max |
|---|---|---|---|---|---|---|---|
| source | 0.3835 | 0.00589 | 74.5 | 83.8 | 98.2 | 148.0 | 160.4 |
| edited | 0.3825 | 0.01502 | 74.4 | 80.8 | 97.2 | 146.9 | 153.5 |
| projected | 0.3809 | 0.00000 | 73.1 | 80.0 | 97.4 | 146.2 | 151.6 |

**onepass_gan__g001300**

| track | CA-CA med nm | CA-CA sd nm | bend min | bend p1 | bend med | bend p99 | bend max |
|---|---|---|---|---|---|---|---|
| source | 0.3835 | 0.00589 | 74.5 | 83.8 | 98.2 | 148.0 | 160.4 |
| edited | 0.3570 | 0.05471 | 25.6 | 73.1 | 102.4 | 161.7 | 173.5 |
| projected | 0.3809 | 0.00000 | 21.8 | 70.2 | 98.1 | 170.0 | 176.0 |

**onepass_gan__g005300**

| track | CA-CA med nm | CA-CA sd nm | bend min | bend p1 | bend med | bend p99 | bend max |
|---|---|---|---|---|---|---|---|
| source | 0.3835 | 0.00589 | 74.5 | 83.8 | 98.2 | 148.0 | 160.4 |
| edited | 0.3724 | 0.07037 | 61.3 | 72.4 | 102.4 | 156.6 | 162.8 |
| projected | 0.3809 | 0.00000 | 56.8 | 63.8 | 100.8 | 159.7 | 178.2 |

**real_reference**

| track | CA-CA med nm | CA-CA sd nm | bend min | bend p1 | bend med | bend p99 | bend max |
|---|---|---|---|---|---|---|---|
| source | 0.3807 | 0.00654 | 71.8 | 86.4 | 115.0 | 145.9 | 153.5 |
| edited | 0.3807 | 0.00654 | 71.8 | 86.4 | 115.0 | 145.9 | 153.5 |
| projected | 0.3809 | 0.00000 | 67.9 | 85.1 | 114.9 | 145.7 | 153.8 |

## All-bonds amplification

`chain_intact` requires EVERY bond inside tolerance, so it is the per-bond pass rate
raised to the bond count. At ~9.25 bonds per peptide:

| per-bond pass | peptides intact |
|---|---|
| 0.520 | 0.0026 |
| 0.700 | 0.0388 |
| 0.900 | 0.3830 |
| 0.950 | 0.6267 |
| 0.990 | 0.9125 |
| 0.999 | 0.9909 |

reaching 0.95 of peptides needs a per-bond pass rate of 0.9944.
