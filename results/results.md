# Results

Three seeds per setting; each seed draws its own split. Overall accuracy (OA) in %, mean ± standard deviation.
HSI: all hyperspectral bands; S2: Sentinel-2; L8: Landsat-8; RGBN: RGB plus near infrared.

## OA of every input, random and block split (50 labelled pixels per class)

| Scene | Input (bands) | SVM random | SVM block | FCN random | FCN block |
|---|---|---|---|---|---|
| Indian Pines | HSI (200) | 69.1 ± 2.9 | 57.4 ± 2.1 | 95.1 ± 1.4 | 63.1 ± 10.7 |
| | S2 (10) | 70.9 ± 2.6 | 59.3 ± 5.3 | 95.8 ± 0.7 | 60.5 ± 6.5 |
| | L8 (7) | 70.6 ± 1.2 | 59.1 ± 5.9 | 96.2 ± 0.9 | 58.3 ± 3.4 |
| | RGBN (4) | 58.3 ± 1.4 | 48.3 ± 3.1 | 95.0 ± 2.1 | 60.3 ± 12.2 |
| | RGB (3) | 50.1 ± 2.8 | 41.8 ± 4.1 | 95.9 ± 0.7 | 61.7 ± 8.5 |
| Pavia University | HSI (103) | 84.1 ± 0.9 | 75.5 ± 3.5 | 98.9 ± 0.3 | 92.5 ± 8.4 |
| | S2 (7) | 77.3 ± 2.0 | 64.9 ± 8.2 | 96.1 ± 2.9 | 88.5 ± 6.4 |
| | L8 (4) | 67.5 ± 3.3 | 58.4 ± 1.2 | 91.5 ± 1.7 | 77.5 ± 5.4 |
| | RGBN (4) | 75.1 ± 2.2 | 60.9 ± 7.5 | 97.0 ± 0.9 | 91.0 ± 7.6 |
| | RGB (3) | 64.9 ± 7.8 | 56.1 ± 9.0 | 95.0 ± 0.7 | 81.8 ± 1.6 |
| Salinas | HSI (204) | 87.3 ± 1.4 | 85.7 ± 0.9 | 93.5 ± 1.1 | 90.1 ± 3.3 |
| | S2 (10) | 88.2 ± 1.5 | 85.7 ± 1.0 | 93.5 ± 4.1 | 90.8 ± 3.0 |
| | L8 (7) | 87.2 ± 1.7 | 83.5 ± 1.6 | 94.1 ± 1.9 | 89.5 ± 2.3 |
| | RGBN (4) | 84.7 ± 2.0 | 80.1 ± 0.9 | 91.6 ± 0.3 | 87.9 ± 6.3 |
| | RGB (3) | 82.2 ± 1.9 | 77.2 ± 1.6 | 90.5 ± 1.5 | 84.4 ± 1.6 |

The MLP (spectra only, random split) reaches 68.1 ± 2.3, 78.5 ± 2.8 and 86.7 ± 1.0 on the three scenes.

## Spectral gain near and far from the training pixels

HSI minus RGB OA (points) for test pixels within 5 px of a training pixel (near) and more than 10 px away (far),
three seeds pooled, with 95% cluster-bootstrap intervals. Seeds: seeds with a larger gain far than near.

| Scene | Split | FCN near | FCN far | FCN seeds | SVM near | SVM far | SVM seeds |
|---|---|---|---|---|---|---|---|
| Indian Pines | random | -0.3 (-0.8, 0.2) | too few far pixels | | 19.0 (16.1, 21.9) | too few far pixels | |
| Indian Pines | block | -0.1 (-1.5, 1.4) | 11.7 (4.0, 20.5) | 3/3 | 18.6 (13.4, 24.0) | 14.1 (8.1, 20.9) | 1/3 |
| Pavia University | random | 1.2 (0.5, 2.1) | 5.7 (2.8, 9.3) | 3/3 | 14.3 (11.2, 17.6) | 22.7 (17.6, 27.5) | 3/3 |
| Pavia University | block | 7.9 (2.5, 15.9) | 10.7 (4.6, 17.5) | 2/3 | 15.6 (9.7, 22.4) | 19.4 (13.8, 25.2) | 1/3 |
| Salinas | random | 2.4 (0.9, 4.0) | 3.2 (-2.4, 9.1) | 2/3 | 5.1 (3.9, 6.5) | 2.8 (-0.7, 6.5) | 1/3 |
| Salinas | block | 3.6 (0.9, 7.1) | 6.5 (1.3, 12.1) | 3/3 | 5.6 (3.5, 8.3) | 9.1 (5.2, 13.1) | 3/3 |

FCN: far gain larger in 13 of 15 seed comparisons (one-sided sign test, p = 0.004). SVM: 9 of 15 (p = 0.30).
Random splits put 57 to 100% of test pixels within 10 px of a training pixel; block splits 13 to 66%.

## OA against labelled pixels per class (random split, all bands)

| Scene | Pixels per class | SVM | FCN |
|---|---|---|---|
| Indian Pines | 10 | 49.0 ± 4.7 | 72.2 ± 1.8 |
| | 25 | 58.6 ± 0.5 | 89.7 ± 0.8 |
| | 50 | 69.1 ± 2.9 | 95.1 ± 1.4 |
| | 100 | 75.7 ± 2.2 | 98.2 ± 0.7 |
| Pavia University | 10 | 61.8 ± 12.6 | 83.4 ± 4.1 |
| | 25 | 75.9 ± 3.2 | 93.7 ± 3.6 |
| | 50 | 84.1 ± 0.9 | 98.9 ± 0.3 |
| | 100 | 86.2 ± 0.7 | 99.2 ± 0.4 |
| Salinas | 10 | 82.7 ± 2.3 | 86.4 ± 6.8 |
| | 25 | 85.0 ± 1.2 | 89.6 ± 2.9 |
| | 50 | 87.3 ± 1.4 | 93.5 ± 1.1 |
| | 100 | 89.1 ± 1.2 | 92.8 ± 0.4 |

Per-seed values: [label_budget.csv](label_budget.csv).

## Figures

| File | Content |
|---|---|
| [random_vs_block.png](figures/random_vs_block.png) | OA of every input under random and block splits |
| [distance_maps_PU.png](figures/distance_maps_PU.png) | distance of each test pixel to the nearest training pixel, Pavia University |
| [distance_gain.png](figures/distance_gain.png) | spectral gain against that distance |
| [maps_PaviaUniversity.png](figures/maps_PaviaUniversity.png) | SVM, MLP and FCN maps and errors |
| [split_IndianPines.png](figures/split_IndianPines.png) | random and block split of Indian Pines, seed 1 |
| [label_budget.png](figures/label_budget.png) | OA against labelled pixels per class |
