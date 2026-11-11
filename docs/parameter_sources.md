# Parameter provenance

Every number used by the model lives in `sutong_fe/params.py` and maps to one
of the sources below. Where two documented sources disagree, the choice and
the reason are stated here.

## Geometry

| parameter | value | source |
|---|---|---|
| span arrangement | 100+100+300+1088+300+100+100 = 2088 m | Jiangsu DOT project record [S1]; Rao & Deng [S2]; Zhang [S3] |
| pylon positions | x = 500 m, 1588 m | derived: 100+100+300 from either end |
| pylon structural height | 300.4 m (top at +306 m ASL) | S1, S3 (tower-top elevation 306 m; base on the pile cap at ~+5.6 m) |
| pylon segment heights | lower 53.2 / middle 155.8 / upper 91.4 m | S3 (sums exactly to 300.4 m) |
| deck centreline | +70 m ASL (y = 64.4 m above the pylon base) | deck ~70 m above the river (S1); reproduces the as-built shortest stay (see below) |
| navigation clearance | 62 m | S1 |
| auxiliary / transition piers | x = 0, 100, 200, 1888, 1988, 2088 m | S1 span arrangement |

## Stay cables

| parameter | value | source |
|---|---|---|
| number of stays | 272 = 2 pylons x 2 sides x 2 planes x 34 levels | S2 ("4 x 34 x 2 = 272"); S3 |
| deck anchorage spacing | 16 m main span and near-tower side span; 12 m in the auxiliary spans | S2; S3 (deck segments: 16 m standard, 12 m in the tail region) |
| pylon anchorage zone | levels at y = 216.4-291.2 m (74.8 m zone, 2.267 m spacing) | zone height and segment heights from S3 (30 steel anchor-box segments, 73.6 m total, 2.3-2.9 m each); zone position anchored by reproducing the as-built shortest stay of 152.853 m [S4] |
| shortest / longest stay | as-built 152.853 m / 576.767 m | S4 (construction records); model chords 152.84 m / 589.4 m; the +2.2 % on the longest chord is attributed to anchor sockets, anchorage-face offsets and deck camber |
| wire / stay specs | 7 mm wires, fu = 1770 MPa, PES7-139 ... PES7-313 | S2, S4 |
| specification ladder | graded by anchor level (force demand) | follows the design-stage area table of [S5]; see calibration note below |
| cable modulus / density | E = 1.95e11 Pa; 9000 kg/m^3 incl. sheath | S5 |
| documented maximum stay force | 5.93 MN per plane | S4; the model reaches 6.0 MN per plane under dead load |

## Girder (streamlined flat steel box, 41 m wide incl. fairings, 4.0 m deep)

| parameter | value | source |
|---|---|---|
| E | 2.10e11 Pa | S5 |
| area / vertical bending inertia | 1.719 m^2 / 4.698 m^4 (standard); 2.394 m^2 / 6.834 m^4 (thickened side-span segments) | S5 section table; consistent with the plate layout of [S3] (14 mm top plate, 10 mm bottom, 4 m diaphragm spacing) |
| line mass | 26.20 t/m standard; 32.27 t/m side regions; +7163.2 t ballast | S5 (DL1 equivalent density 11.17 t/m^3, DL2 7.0 t/m, ballast stated separately); total superstructure ~68 kt matches the documented tonnage |
| cross-section geometry | 41 x 4 m, Q370qE | S1, S3, S4 |

## Pylons (reinforced concrete inverted-Y)

| parameter | value | source |
|---|---|---|
| E / density | 3.50e10 Pa / 2600 kg/m^3 | S5 |
| section properties per segment | lower A 59.19-54.37 m^2, I 1471.8-1154.4 m^4; middle 43.99-32.21 m^2, 941.2-395.2 m^4; upper 64.99-45.17 m^2, 939.2-418.6 m^4 | S5 section table (in-plane values; per-leg values doubled below the leg junction, as both legs share the same longitudinal position) |

## Boundary conditions

| item | model | source |
|---|---|---|
| pylon-deck connection | no vertical bearing (cables only); longitudinal dampers without elastic stiffness | S3, S5 and the SHM-system description of [S6] ("no vertical bearings between pylon and girder; wind bearings transversely; viscous dampers longitudinally") |
| auxiliary / transition piers | one-way sliding steel bearings: vertical support only | S6, S5 |
| pylon / pier bases | fixed | S5 |

## Measured modal targets

| mode | frequency | damping | source |
|---|---|---|---|
| VS1 / VAS1 | 0.1849 / 0.2176 Hz | 0.82 / 0.80 % | S6 (2010 SHM statistics) |
| VS2 / VAS2 | 0.3336 / 0.3887 Hz | 2.27 / 2.01 % | S6 |
| VS3 / VS4 / VS5 | 0.4471 / 0.5274 / 0.6641 Hz | 1.18 / 1.45 / 2.06 % | S6 |
| lateral LS1, torsion T1 / T2 | 0.1084 / 0.5718 / 0.6200 Hz | - | S6 (outside the 2-D vertical plane; not calibration targets) |
| independent cross-check | 0.185 / 0.218 / 0.334 / 0.396 Hz | - | S7 (Bayesian FFT identification) |

## Calibration scales (bounded updating)

| factor | bounds | result | justification |
|---|---|---|---|
| deck mass | 0.85-1.15 | 0.994 | superimposed dead-load allowance (0.6 % adjustment) |
| cable area | 0.90-1.18 | 0.994 | as-built vs design-stage area table differences, socket / anchorage compliance |
| pylon bending inertia | 0.70-1.30 | 1.105 | 2-D representation of the inverted-Y tower and foundation flexibility |
| girder bending inertia | 0.70-1.40 | 1.180 | effective stiffness with U-rib and diaphragm participation |

The side-span ballast is contained in the equivalent line mass of the
thickened segments, so no separate ballast factor is updated.  Additionally,
the shape finding caps every stay at the documented maximum stay force
(5.93 MN per cable plane, +5 % modelling allowance), which reproduces the
as-built force level of the flattest back-stays (delivered: 6.18 MN per
plane).

## Damage scenarios (stage 2)

| item | value | source |
|---|---|---|
| families | F1..F12 as in the study design (single stay, symmetric pair, fan-end set, three girder bands, pylon base, two bearing families, pylon tops, two combined families) | study design of this work |
| severity levels | standard families 5 / 15 / 35 % property loss; combined families 3 / 10 / 25 % | study design (light / moderate / severe) |
| F3 fan loss | 8 outermost anchor levels per fan (~25 % of each fan) | study design |
| F7 band | pylon base segment, 0-12 m (the pylon mesh is 4 m, so three elements cover the documented 0-10 m band) | study design, discretisation |
| F10 band | both upper columns above 270.4 m (above the anchor box, 291.2 m top) | study design |
| F5 location | low-side quarter point of the main span, x = 772 m | study design |

Two families of the original study act on restraints that do not exist in a
vertical plane (transverse bearing restraint, pylon-top rotational
restraint).  They are reinterpreted on the in-plane properties those
substructures actually provide here - the vertical stiffness of the
one-way sliding bearings and the flexural stiffness of the upper columns -
and this reinterpretation is stated wherever the scenarios are documented.

## Monitoring subset and ambient excitation (stage 2)

| item | value | source |
|---|---|---|
| channels | 64 = 43 deck-vertical (48 m spacing, symmetric about mid-span) + 5 deck-longitudinal + 8 pylon bidirectional + 8 stay-axial | monitoring-system subset of the study design |
| record length / rate | 60 s at 200 Hz, three records per scenario | study design (3 windows of 20 s per record) |
| modal superposition cutoff | 12 Hz, uniform 2 % modal damping | modal band of the damage features; damping from the measured statistics (0.8-2.3 %, S6), rounded to the design value |
| mean wind | 8 m/s at deck level, Iu 0.12 / Iw 0.08, Kaimal / Panofsky shapes, exponential coherence | moderate-breeze statistics of the site climate; spectral shapes are the standard wind-engineering forms |
| aerodynamic coefficients | Cd 0.7 deck / 1.2 pylon, lift-curve slope 2.0 | streamlined-box typical values |
| traffic | Poisson stream, 0.18 veh/s, 35 % trucks (550 kN gross, 3 axles), cars 30 kN, speeds 21 / 27 m/s | ordinary mixed traffic of the corridor |
| sensor noise | 1.5e-4 m/s^2 white per channel | typical accelerometer resolution class |
| seeds | (2024, scenario, record): one generator per record, wind + traffic + noise drawn in sequence | regeneration convention of this work |

All excitation amplitudes are engineering-typical values of the site and
its traffic (the site's published SHM papers report the operational
response statistics rather than the load histories); they are selected so
that the simulated ambient response lands inside the documented rms band
(~1-2 mg at the deck), which the driver verifies automatically.

## Feature extraction (stage 3)

| item | value | source |
|---|---|---|
| windowing | 3 x 20 s windows per 60 s record | study design |
| CWT wavelet | complex Morlet, w0 = 6 | standard admissible carrier (the smallest value with negligible DC leakage); the value of the study design |
| CWT scales | 50, log-spaced over 0.05-10 Hz | covers the modal band of the damage features with ~3 scales per half-octave |
| energy bands | 0.05-0.2 / 0.2-0.5 / 0.5-1 / 1-2 / 2-10 Hz | study design (the seven measured verticals fall in bands 1-3; bands 4-5 carry the higher local modes) |
| Hilbert decay | LS slope of log-envelope per channel per window | study design |
| SVD entropy | Shannon entropy of normalised squared singular values of the 64 x 4000 window matrix | study design |
| phase deviation | std-dev of the cross-spectral phase over 0.1-2 Hz, mean over the 42 adjacent deck-vertical pairs (Hann, 2000-sample segments, 50 % overlap) | study design |
| modal-amplitude ratio | FDD mode-1 (0.04-0.12 Hz, floating) / mode-2 (0.13-0.21 Hz, first vertical) amplitude at mid-span, 4x zero-padded PSD | study design; bands isolate the two modes unambiguously |

The scale-frequency mapping follows the standard Morlet pseudo-frequency
relation; the CWT is evaluated by FFT convolution with symmetric
reflection padding of at least the largest half-support, and the band
energies use the L2-normalised wavelet convention.

## Learning protocol (stage 4)

| item | value | source |
|---|---|---|
| architecture | per-channel CNN 1-16-32 (kernel 3), two message-passing GNN layers (hidden 16), evidential head (hidden 64) | study design (~73 k parameters) |
| sensor kernel | W = exp(-d/sigma_d), sigma_d = 624 m (median pairwise sensor distance) | study design |
| evidential loss | Sensoy type-II risk with the alpha-tilde = y + (1-y) alpha true-class correction, KL weight annealed min(1, t/10), cap 1.0 | Sensoy et al. annealing schedule; the correction is the fix of the study's earlier constant-lambda variant |
| optimisation | Adam lr 5e-4, weight decay 1e-5, batch 16, StepLR (40, 0.5), 120 epochs fixed budget | study design |
| cross-validation | GroupKFold (5 folds) over the 13 damage groups, seeds 0/1/2, 15 runs; fold standardisation fitted on training windows only | study design (leave-family-out) |
| baselines | SVM (RBF, C = 1), Random Forest (200 trees), 1D-CNN, LSTM (64 hidden), Transformer (2 layers, 64 dim, 4 heads) - same folds, same features | study design |
| ablations | no-GNN / no-CWT / no-scalars / no-evidential by actual module removal | study design |

## Sources

- [S1] Jiangsu Provincial Department of Transportation, "Overview of the
  Sutong Yangtze River Bridge project" (2008) - official project record.
- [S2] Rao Huarong, Deng Huibin, "Long-cable erection technology of a
  kilometer-scale cable-stayed bridge", Strategic Study of CAE / Engineering,
  2009, 11(3): 71-74.
- [S3] Zhang Xiongwen, "Key technology and innovation of the Sutong Bridge",
  proceedings paper of the Sutong Bridge construction command (2006).
- [S4] Sutong Bridge steel-box-girder and stay-cable construction documents
  (segment and cable schedule: shortest stay 152.853 m, longest 576.767 m,
  maximum stay force 604.29 t, plate thicknesses).
- [S5] Tongji University, State Key Laboratory for Disaster Reduction in
  Engineering, "Structural seismic research report of the Sutong Yangtze
  River Bridge project" (design-stage FE properties, boundary conditions,
  cable area table, concept-model modal results).
- [S6] Mao Jianxiao, Gui Gui, Wang Hao, Yang Zhaoyong, "Long-term tracking
  and variability analysis of the modal parameters of a long-span
  cable-stayed bridge", Chinese Journal of Vibration Engineering, 2024,
  37(1): 52-59. DOI: 10.16385/j.cnki.issn.1004-4523.2024.01.006.
- [S7] Yang Zhaoyong, Mao Jianxiao, Wang Hao, Zhang Yiming, "Application of
  the Bayesian method in modal parameter identification of a long-span
  cable-stayed bridge", Chinese Journal of Vibration Engineering, 2022,
  35(3). DOI: 10.16385/j.cnki.issn.1004-4523.2022.03.019.
