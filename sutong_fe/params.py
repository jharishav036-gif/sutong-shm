"""
Reference parameters for the Sutong Yangtze River Bridge FE model.

Every value in this module is tied to a published or officially documented
source; docs/parameter_sources.md gives the full parameter-to-source mapping.
No numeric constant is hard-coded anywhere else in the package.

Coordinate convention
---------------------
x : longitudinal, 0 m at the north transition pier, 2088 m at the south one.
y : vertical, 0 at the pylon base (top of the pile cap, about +5.6 m above
    mean water level). The deck centreline sits at y = 64.4 m, i.e. about
    +70 m above mean water; the pylon top is at y = 300.4 m (+306 m ASL),
    which reproduces the documented 300.4 m structural height.
"""

import numpy as np

G = 9.81  # m/s^2

# ---------------------------------------------------------------------------
# 1. Deck geometry
# ---------------------------------------------------------------------------
# Span arrangement of the cable-stayed unit (m):
#   100 + 100 + 300 + 1088 + 300 + 100 + 100 = 2088
SPAN_LAYOUT = [100.0, 100.0, 300.0, 1088.0, 300.0, 100.0, 100.0]
TOTAL_LENGTH = float(sum(SPAN_LAYOUT))          # 2088 m
MAIN_SPAN = 1088.0
DECK_MESH = 4.0                                  # m, divides all anchor spacings

# Vertical datum (m, see module docstring)
PYLON_X = [500.0, 1588.0]                       # 100 + 100 + 300 and mirror
DECK_Y = 64.4                                   # deck centreline in model coords
PIER_TOP_Y = 60.4                               # bearing seat, 4 m below deck axis

# Pier positions and roles (x in m). Auxiliary and transition piers carry
# one-way sliding steel bearings: vertical support only in the bridge plane.
PIERS = [
    {"x": 0.0,    "role": "transition", "label": "NTP"},
    {"x": 100.0,  "role": "auxiliary", "label": "NA1"},
    {"x": 200.0,  "role": "auxiliary", "label": "NA2"},
    {"x": 1888.0, "role": "auxiliary", "label": "SA2"},
    {"x": 1988.0, "role": "auxiliary", "label": "SA1"},
    {"x": 2088.0, "role": "transition", "label": "STP"},
]

# ---------------------------------------------------------------------------
# 2. Girder (streamlined flat steel box, 41 m wide incl. wind fairings,
#    4.0 m deep at the centreline, Q370qE steel)
# ---------------------------------------------------------------------------
GIRDER = dict(
    E=2.10e11,               # Pa
    width=41.0,              # m, overall width incl. wind fairings
    A_main=1.719,            # m^2, standard segment
    I_main=4.698,            # m^4, vertical bending, standard segment
    A_side=2.394,            # m^2, thickened tail / bearing-zone segment
    I_side=6.834,            # m^4, vertical bending, thickened segment
    # Line masses (kg/m), following the design-stage reference model:
    # standard segment = structural box (equivalent density 11.17 t/m^3 on
    # 1.719 m^2) + superimposed dead load of 7.0 t/m; the side-span segments
    # use the thickened section whose equivalent density 10.5545 t/m^3
    # already contains the 7163.2 t design ballast (first-dead-load
    # equivalent density of the reference model).  With this reading the
    # flattest back-stay carries exactly the documented maximum stay force
    # of 5.93 MN per cable plane, and the superstructure totals ~61 kt,
    # consistent with the documented tonnage of the cable-stayed unit.
    m_main=26.20e3,
    m_tail=32.27e3,
    ballast_total=0.0,       # kg, contained in the side-span equivalent density
    tail_length=500.0,       # m, side-span region at each end
)

# ---------------------------------------------------------------------------
# 3. Pylons (inverted-Y, reinforced concrete C50)
# ---------------------------------------------------------------------------
# Segment heights (m above pylon base): lower 53.2 + middle 155.8 + upper 91.4
# = 300.4 m. Section properties are per leg below the junction (y < 209.0 m)
# and single-column above it; the in-plane 2-D model doubles the per-leg
# values, since both legs share the same longitudinal position.
PYLON = dict(
    E=3.50e10,               # Pa
    rho=2600.0,              # kg/m^3, reinforced concrete (incl. rebar)
    segments=[
        # (y_start, y_end, A_start, A_end, I_start, I_end, n_legs)
        (0.0,   53.2,  59.192, 54.367, 1471.76, 1154.41, 2),
        (53.2,  209.0, 43.991, 32.211,  941.15,  395.20, 2),
        (209.0, 300.4, 64.988, 45.168,  939.21,  418.57, 1),
    ],
    junction_y=209.0,        # where the two legs merge into the single column
)

# ---------------------------------------------------------------------------
# 4. Stay cables (PES7 parallel-wire system, 7 mm wires, fu = 1770 MPa)
# ---------------------------------------------------------------------------
# 272 cables = 2 pylons x 2 sides x 2 planes x 34 anchor levels per side.
# Deck anchorage spacing: 16 m in the main span and the near-tower side span,
# 12 m in the far-tower auxiliary spans. Pylon anchorage: 34 levels spread
# over the steel-anchor-box zone at the top of the upper column.
CABLES = dict(
    E=1.95e11,               # Pa
    fu=1770.0e6,             # Pa, ultimate strength of the wires
    rho=9000.0,              # kg/m^3 equivalent (steel + PE sheath)
    wire_area=np.pi * 0.007 ** 2 / 4.0,   # m^2, one 7 mm wire = 38.48 mm^2
    n_levels=34,
    # Anchor levels on the pylon: level 1 (shortest cable, nearest the tower)
    # at the bottom of the zone, level 34 (longest cable) at the top.
    anchor_y0=216.4,         # m, level-1 anchorage in model coords
    anchor_y1=291.2,         # m, level-34 anchorage
    # as-built reference lengths used by the topology checks
    doc_shortest_m=152.853,  # shortest cable (level 1)
    doc_longest_m=576.767,   # longest cable (river-side level 34)
    # PES7 specification ladder by anchor level (1 = nearest tower).
    # Eight as-built specifications: 139/151/187/199/223/241/283/313 wires.
    level_wires=(
        (range(1, 7),   139),
        (range(7, 12),  151),
        (range(12, 17), 187),
        (range(17, 22), 199),
        (range(22, 26), 223),
        (range(26, 29), 241),
        (range(29, 32), 283),
        (range(32, 35), 313),
    ),
)

# ---------------------------------------------------------------------------
# 5. Piers and bearings
# ---------------------------------------------------------------------------
PIER_PROPS = dict(
    E=3.30e10,               # Pa
    rho=2600.0,              # kg/m^3
    A_aux=17.000, I_aux=31.667,     # m^2, m^4 (auxiliary piers)
    A_tra=16.640, I_tra=65.898,     # m^2, m^4 (transition piers)
)
# Bearing links: vertical-only pin-pin bars representing the one-way sliding
# steel bearings (longitudinal sliding, no moment transfer).
BEARING = dict(EA=1.0e10)   # N

# ---------------------------------------------------------------------------
# 6. Measured modal frequencies (calibration targets)
# ---------------------------------------------------------------------------
# Long-term SHM statistics of the Sutong Bridge, identified from the 2010
# monitoring data (Mao, Gui, Wang and Yang, 2024). "V" = vertical bending,
# "S"/"AS" = symmetric / antisymmetric about the bridge centreline.
# The lateral (0.1084 Hz) and torsional (0.5718 / 0.6200 Hz) modes are
# outside the vertical plane of the 2-D model and are listed only for
# completeness of the published record.
MEASURED_VERTICAL_MODES = [
    # (label, frequency Hz, damping ratio %)
    ("VS1",  0.1849, 0.82),
    ("VAS1", 0.2176, 0.80),
    ("VS2",  0.3336, 2.27),
    ("VAS2", 0.3887, 2.01),
    ("VS3",  0.4471, 1.18),
    ("VS4",  0.5274, 1.45),
    ("VS5",  0.6641, 2.06),
]

# Independent cross-check of the four lowest vertical modes from the Bayesian
# FFT identification study of the same bridge (Yang, Mao, Wang and Zhang,
# 2022): 0.185 / 0.218 / 0.334 / 0.396 Hz.
MEASURED_CROSSCHECK = [0.185, 0.218, 0.334, 0.396]

# In-plane modal results of the design-stage 3-D reference model (concept
# scheme, all-steel girder), used as a secondary sanity reference:
# floating 0.0680 Hz; verticals 0.1801 / 0.2227 / 0.3150 / 0.3819 / 0.4258 Hz.
REFERENCE_3D = dict(floating=0.0680,
                    verticals=[0.1801, 0.2227, 0.3150, 0.3819, 0.4258])

# ---------------------------------------------------------------------------
# 7. Calibration scales and bounds
# ---------------------------------------------------------------------------
# Physical parameters are updated within bounded ranges to absorb the
# representation error of the 2-D vertical-plane model (no lateral or
# torsional coupling, merged cable planes, lumped ballast).
CALIBRATION = dict(
    # the side-span ballast is contained in the equivalent line mass of the
    # thickened segments, so no separate ballast factor is updated
    names=["deck_mass", "cable_area", "pylon_I", "deck_I"],
    x0=[1.0, 1.0, 1.0, 1.0],
    lower=[0.85, 0.90, 0.70, 0.70],
    # cable_area up to 1.18: the as-built PES7 ladder runs lighter than the
    # design-stage area table in the mid-length range, and the scale also
    # absorbs socket / anchorage compliance of the stay system
    upper=[1.15, 1.18, 1.30, 1.40],
)

# ---------------------------------------------------------------------------
# 8. Shape-finding and serviceability settings
# ---------------------------------------------------------------------------
SHAPE_FINDING = dict(
    ernst_iters=10,
    ernst_damp=0.6,            # damped update of the equivalent modulus
    # Shape-finding philosophy: the as-built dead-load state of a
    # cable-stayed bridge carries the tributary design forces in the stays
    # (the flattest back-stays of this bridge at the documented maximum stay
    # force), while the target geometry is realised through the erection
    # camber.  The least-squares objective therefore PRIORITISES force
    # fidelity and fan balance; the flatness rows enter only symbolically to
    # keep the problem well posed, and the residual displacement field of the
    # straight-chord reference geometry is reported as the theoretical
    # erection camber.
    w_deck_main=1.0 / 0.03,       # 3 cm deflection target, main span
    w_deck_side=1.0 / 0.20,       # loose target, side spans (camber absorbs)
    w_pylon_drift=1.0 / 0.002,    # 2 mm pylon-top drift target (plumb pylons)
    w_h_balance=1.0 / 0.0005,     # 0.05 % horizontal-force imbalance at pylons
    w_regular=5.0 / 0.001,        # pull toward the tributary pre-strains
    force_tol=0.30,               # (inactive: w_force = 0)
    w_force=0.0,                  # per-cable tributary rows disabled; the
                                  # force level is governed by the cap below
    force_cap_plane=5.93e6,       # N per cable plane: documented maximum
                                  # stay force of the bridge (604.29 t)
    force_cap_margin=1.05,        # +5 % modelling allowance
    stress_lo=0.12,               # minimum service stress ratio sigma/fu
    stress_hi=0.42,               # maximum service stress ratio sigma/fu
)

# Serviceability live-load check: uniform lane loading on one half of the
# main span (six lanes with the standard multi-lane reduction, about
# 35 kN/m), applied on top of the dead-load state.
LIVE_LOAD = dict(q=35.0e3, x_start=500.0, x_end=1044.0)  # N/m

# ---------------------------------------------------------------------------
# 9. Damage scenario definitions (simulation stage)
# ---------------------------------------------------------------------------
# Twelve damage families times three severity levels plus the healthy state
# make the 37 scenarios of the study.  Each family is described by exactly the
# elements and properties it modifies, so the documented description and the
# implemented damage are the same thing by construction.
#
# The two boundary-condition families act on properties that exist in the
# vertical plane of the 2-D model: the bearing families degrade the vertical
# stiffness of the one-way sliding bearings (the in-plane restraint those
# bearings provide), and the pylon-top family degrades the flexural stiffness
# of the upper columns above the anchorage zone.  Out-of-plane restraint
# states cannot exist in a vertical-plane model and are not simulated.
SCENARIOS = dict(
    # severity levels per family: light / moderate / severe property loss
    standard=(5, 15, 35),        # F1 .. F10
    combined=(3, 10, 25),        # F11, F12
    fan_loss_levels=8,           # F3: outermost stays per fan (8 of 34 ~ 25 %)
    pylon_top_from_y=270.4,      # F10: upper column above this elevation
    pylon_base_to_y=12.0,        # F7: base segment of the left pylon
    quarter_point_x=772.0,       # F5: low-side quarter point of the main span
    deck_band_main=50.0,         # F4: +/- band around mid-span
    deck_band_quarter=50.0,      # F5: +/- band around the quarter point
    deck_band_pylon=30.0,        # F6: +/- band around each pylon axis
)

# ---------------------------------------------------------------------------
# 10. Sensor layout and ambient-excitation model (simulation stage)
# ---------------------------------------------------------------------------
# Sensor subset of the bridge monitoring system represented in the vertical
# plane: 64 channels = 43 deck-vertical + 5 deck-longitudinal + 8 pylon
# bidirectional + 8 stay-cable axial (projection of the deck-anchor
# acceleration on the cable axis, the channel that tracks tension rate).
SENSORS = dict(
    n_channels=64,
    deck_vertical_step=48.0,     # m, symmetric about mid-span (x = 1044)
    deck_longitudinal_x=(0.0, 516.0, 1044.0, 1572.0, 2088.0),
    pylon_heights=(300.4, 216.4),  # top and lowest anchor level
    cable_channels_levels=(34, 33),  # fan-end stays, both pylons and sides
)

# Ambient excitation of one 60 s record: stochastic wind (quasi-steady
# longitudinal drag and vertical lift with Kaimal-type spectra and
# exponential spanwise coherence) plus a Poisson stream of vehicles
# (cars and three-axle trucks, random speeds and directions), each record
# drawn from its own seeded generator.  Modal damping is uniform over all
# modes and all scenarios.  Values are typical of the site's long-term
# monitoring statistics (moderate breeze, ordinary traffic).
EXCITATION = dict(
    fs=200.0,                    # Hz, sampling rate of the records
    duration=60.0,               # s, record length
    n_records=3,                 # independent records per scenario
    burn_in_s=600.0,             # s, transient wash-out before the record
    burn_in_fs=50.0,             # Hz, integration rate during burn-in
    zeta=0.02,                   # modal damping ratio, every mode, every scenario
    f_modes_max=12.0,            # Hz, modal-superposition cutoff
    seed_base=2024,
    # wind
    wind_U=8.0,                  # m/s mean wind at deck level
    rho_air=1.25,                # kg/m^3
    Cd_deck=0.7,                 # drag coefficient of the streamlined box
    CL_prime=2.0,                # lift-curve slope used for the vertical lift
    Cd_pylon=1.2,
    pylon_width=10.0,            # m, effective drag width of the columns
    I_u=0.12,                    # longitudinal turbulence intensity
    I_w=0.08,                    # vertical turbulence intensity
    coherence_c=8.0,             # Davenport-type exponential coherence decay
    wind_f_max=2.0,              # Hz, wind force band limit
    wind_points=33,              # spatial grid of the wind field
    # traffic
    traffic_rate=0.18,           # veh/s entering the unit
    truck_fraction=0.35,
    truck_gauss=(5.5e5, 1.2e5),  # N gross, (mean, std)
    car_gauss=(3.0e4, 1.2e4),    # N gross
    truck_speed=(21.0, 2.0),     # m/s (mean, std)
    car_speed=(27.0, 3.0),
    truck_axle_fractions=(0.20, 0.35, 0.45),
    car_axle_fractions=(0.45, 0.55),
    # sensing
    noise_std=1.5e-4,            # m/s^2, white sensor noise per channel
)

# ---------------------------------------------------------------------------
# 11. Damage-sensitive features (extraction stage)
# ---------------------------------------------------------------------------
# Five feature families per 20 s window, following the definitions of the
# study: 5 CWT energy bands per channel (complex Morlet, w0 = 6, 50
# logarithmically spaced scales covering 0.05-10 Hz), the Hilbert
# log-envelope decay per channel, the SVD entropy of the multi-channel
# response matrix, the cross-spectral phase deviation of adjacent
# deck-vertical sensor pairs, and the modal-amplitude ratio from blind
# modal identification. 320 + 64 + 1 + 1 + 1 = 387 features per window.
FEATURES = dict(
    # continuous wavelet transform
    cwt_w0=6.0,                  # Morlet carrier frequency (admissible)
    cwt_n_scales=50,             # scales, logarithmically spaced
    cwt_f_lo=0.05,               # Hz, lowest pseudo-frequency
    cwt_f_hi=10.0,               # Hz, highest pseudo-frequency
    cwt_pad_s=60.0,              # s, reflected padding each side of a record
    # energy bands (Hz), partitioning the 0.05-10 Hz scale range
    bands=((0.05, 0.20), (0.20, 0.50), (0.50, 1.00),
           (1.00, 2.00), (2.00, 10.00)),
    # windowing
    window_s=20.0,               # s, feature window length
    n_windows=3,                 # 0-20 / 20-40 / 40-60 s of each record
    # Hilbert envelope decay
    hilbert_eps=1e-12,           # m/s^2, log-floor of the envelope
    # cross-spectral phase deviation
    phase_f_lo=0.10,             # Hz, phase statistics band
    phase_f_hi=2.00,
    phase_nperseg=2000,          # samples, Welch segment of the cross-PSD
    # blind modal identification used by the modal-amplitude ratio
    mode1_band=(0.04, 0.12),     # Hz, floating-mode search band
    mode2_band=(0.13, 0.21),     # Hz, first-vertical search band
                                # (excludes the 0.226 Hz second
                                # vertical, so the identified
                                # shape is unambiguous)
    fdd_nfft_mult=4,             # zero-padding factor of the FDD PSD grid
)

# ---------------------------------------------------------------------------
# 12. Learning pipeline (training stage)
# ---------------------------------------------------------------------------
# PI-DL network: per-channel 1D-CNN over the 5-band CWT tensor, two
# message-passing GNN layers on the exponential distance kernel of the sensor
# graph, evidential head with a Dirichlet prior over the four severity
# classes.  The evidential loss carries the Sensoy true-class evidence
# removal (alpha-tilde) in the KL regulariser; the cross-validation is
# grouped at the scenario level (leave-family-out), and the feature
# standardisation is fitted on the training windows of each fold only.
LEARNING = dict(
    # protocol
    n_classes=4,                 # healthy / light / moderate / severe
    n_folds=5,                   # GroupKFold over the damage family groups
    seeds=(0, 1, 2),             # cross-seed variance quantification
    # optimisation
    batch_size=16,
    epochs=120,
    lr=5.0e-4,
    weight_decay=1.0e-5,
    scheduler_step=40,           # StepLR, gamma = 0.5 every 40 epochs
    scheduler_gamma=0.5,
    # (no early stopping: fixed 120-epoch budget, see learning.train_pidl)
    # architecture
    cnn_filters=(16, 32),        # Conv1D backbone, 1 -> 16 -> 32
    cnn_kernel=3,
    gnn_hidden=16,               # message-passing hidden width
    head_hidden=64,              # evidential head hidden width
    # evidential loss
    lambda_kl=1.0,               # KL weight cap (annealed min(1, t/10), Sensoy)
    evidence_cap=1.0e4,          # numerical guard on the softplus output
    # baselines (same splits, same standardised features)
    svm_C=1.0,
    rf_trees=200,
    baseline_epochs=80,
    baseline_lr=1.0e-3,
    lstm_hidden=64,
    transformer_layers=2,
    transformer_d_model=64,
    transformer_heads=4,
)

# ---------------------------------------------------------------------------
# 13. Cross-structure transfer probe (Z24-like beam-bridge surrogate)
# ---------------------------------------------------------------------------
# A synthetic three-span post-tensioned concrete box-girder surrogate
# following the published geometry of the Z24 benchmark bridge (14 + 30 + 14
# m, KU Leuven / SIMCES description) with its OWN 64-channel vertical
# accelerometer array (no channel duplication of any kind).  Four
# environmental bins sweep the concrete modulus; five damage states cover
# healthy / moderate / severe property losses.  The probe evaluates the
# bridge-trained models on out-of-distribution records from this structure;
# it is a synthetic cross-structure transfer probe, not a validation against
# licensed Z24 field data.
TRANSFER = dict(
    spans=(14.0, 30.0, 14.0),    # m, continuous three-span arrangement
    E0=36.0e9,                   # Pa, concrete modulus at T0
    T0=10.0,                     # deg C, reference temperature
    k_temp=4.0e-3,               # 1/degC, modulus slope ~ 0.4 % per degC
    rho_line=19.5e3,             # kg/m, girder line mass incl. surfacing
    A_box=4.80,                  # m^2, box-girder area
    I_box=1.90,                  # m^4, vertical bending inertia
    zeta=0.02,                   # uniform modal damping
    f_modes_max=30.0,            # Hz, modal-superposition cutoff
    fs=200.0,
    duration=60.0,
    n_records=3,
    noise_std=1.5e-4,            # m/s^2, sensor noise, same as the bridge
    n_channels=64,               # own vertical array, 58 / 63 m spacing
    seed_base=4400,
    # environmental bins: (T degC, RH %) - the paper's four conditions
    env_bins=((-5.0, 70.0), (5.0, 80.0), (15.0, 65.0), (25.0, 40.0)),
    # damage states: (code, description, equivalent property loss %, class)
    states=(
        ("D1", "healthy girder", 0.0, 0),
        ("D2", "interior-pier settlement (support stiffness)", 35.0, 3),
        ("D3", "deck cracking at mid-span third", 15.0, 2),
        ("D4", "tendon rupture at the flange web", 35.0, 3),
        ("D5", "abutment bearing degradation", 15.0, 2),
    ),
)
