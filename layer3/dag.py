"""
TreBle Respire — Layer 3: Bayesian Inference Engine
DAG structure: all nodes and edges. NO CPTs in this file.

Architecture: Noisy-OR parameterised BN, pgmpy 1.1.0
References:
  - Heckerman 1995 (MS-TR-95-06)
  - Pearl 1988
  - Lauritzen & Spiegelhalter JRSS-B 1988
  - Wu et al. PLoS Comp Biol 2023 (LRTI precedent)

"""

from pgmpy.models import DiscreteBayesianNetwork
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# Root nodes — background patient characteristics
ROOT_NODES = [
    "Age_Comorbidity",       # age + comorbidity combined (child/adult/elderly_healthy/elderly_comorbid)
    "Obstructive_Dx",        # known obstructive diagnosis (COPD/asthma) — bool
    "Biomass_Exposure",      # biomass fuel exposure — bool (70% Indian homes)
    "Prior_Antibiotic_Use",  # antibiotics in past 3 months — bool (multiplies acoustic q_i by 0.70)
    "Season",                # monsoon / winter / other
]

# Clinical input nodes — collected directly by Layer 3 (NOT from Layer 2)
CLINICAL_NODES = [
    "Symptom_Duration",      # <3 days / 3-7 days / >7 days
    "Fever",                 # bool (temp > 37.8°C)
    "Cough_Character",       # dry / productive_mucoid / productive_purulent
    "Sputum_Purulence",      # bool — Anthonisen criterion 1
    "Dyspnea_Increase",      # bool — Anthonisen criterion 2
    "Sputum_Volume_Increase",# bool — Anthonisen criterion 3
    "SpO2",                  # <88 / 88-92 / 92-95 / >95 (discretized from float)
    "FEV1_FVC_Known",        # float or None — passed as evidence if known
]

# Acoustic input nodes — from Layer 2 interface contract
# IMPORTANT: field names must match Layer 2 output exactly
ACOUSTIC_NODES = [
    "Crackle_Pattern",       # absent / focal / bilateral_basal / diffuse
    "Crackle_Phase",         # early_inspiratory / late_inspiratory / expiratory / mixed
    "Crackle_Subtype",       # fine / coarse / velcro / uncertain
    "Wheeze_Pattern",        # absent / focal / bilateral / diffuse
    "Wheeze_Phase",          # inspiratory / expiratory / biphasic
    "Wheeze_Compound",       # 7 states — see WHEEZE_COMPOUND_STATES below
                             # NOTE: field is wheeze_compound NOT wheeze_character (removed)
    "Rhonchi_Present",       # bool
    "Diminished_Sounds",     # absent / unilateral / bilateral
    "Bilateral_Asymmetry_Score",  # none / mild / moderate / high
    "Craniocaudal_Gradient", # flat / apical / basal
    "Zone_Reliability",      # normal / reduced (reduced when dual-zone-only findings)
]

# Valid states for Wheeze_Compound — 7 states
# Derived from wheeze_morphology × wheeze_freq_band in Layer 2
WHEEZE_COMPOUND_STATES = [
    "monophonic_low",    # large airway fixed obstruction — suppresses P(bacterial)
    "monophonic_mid",    # medium airway
    "monophonic_high",   # small airway peripheral
    "polyphonic_low",    # COPD pattern
    "polyphonic_mid",    # mid-frequency polyphonic
    "polyphonic_pan",    # pan-airway severe (polyphonic mixed freq)
    "uncertain",         # marginalize — absence of subtype evidence
]

# Intermediate nodes — latent clinical constructs
INTERMEDIATE_NODES = [
    "P_Consolidation",              # probability of consolidation pattern
    "P_Obstructive_Exacerbation",   # probability of obstructive exacerbation
    "P_Severity",                   # severity of current illness
    "Anthonisen_Type",              # Type 1 / 2 / 3 (Anthonisen 1987 Ann Intern Med)
    "P_ILD_Flag",                   # ILD safety flag — suppresses overconfident bacterial dx
]

# Output nodes
OUTPUT_NODES = [
    "P_Antibiotics_Indicated",  # primary posterior [0,1]
    "Confidence_Interval",      # 95% credible interval (low/medium/high uncertainty)
    "Uncertainty_Flag",         # bool — CI spans two zones OR >2 key inputs missing
    "Disposition",              # home_safety_net / review_48h / refer
    "Contributing_Factor_Weights",  # per-node delta (computed post-inference, not a BN node)
]

ALL_NODES = ROOT_NODES + CLINICAL_NODES + ACOUSTIC_NODES + INTERMEDIATE_NODES + OUTPUT_NODES


# ---------------------------------------------------------------------------
# EDGE DEFINITIONS — causal structure of the BN
# ---------------------------------------------------------------------------

EDGES = [
    # --- Root → Clinical ---
    ("Age_Comorbidity",       "Symptom_Duration"),
    ("Age_Comorbidity",       "P_Severity"),
    ("Obstructive_Dx",        "FEV1_FVC_Known"),
    ("Obstructive_Dx",        "P_Obstructive_Exacerbation"),
    ("Biomass_Exposure",      "P_Obstructive_Exacerbation"),
    ("Prior_Antibiotic_Use",  "P_Antibiotics_Indicated"),   # modifier at inference time
    ("Season",                "Crackle_Pattern"),            # seasonal epidemiology

    # --- Root → Intermediate ---
    ("Age_Comorbidity",       "P_Consolidation"),
    ("Age_Comorbidity",       "P_ILD_Flag"),

    # --- Clinical → Intermediate ---
    ("Fever",                 "P_Consolidation"),
    ("Fever",                 "P_Obstructive_Exacerbation"),
    ("Cough_Character",       "P_Consolidation"),
    ("Symptom_Duration",      "P_Consolidation"),
    ("SpO2",                  "P_Severity"),
    ("Dyspnea_Increase",      "P_Severity"),

    # --- Clinical → Anthonisen ---
    ("Sputum_Purulence",      "Anthonisen_Type"),    # criterion 1
    ("Dyspnea_Increase",      "Anthonisen_Type"),    # criterion 2
    ("Sputum_Volume_Increase","Anthonisen_Type"),    # criterion 3

    # --- Acoustic → P_Consolidation ---
    ("Crackle_Pattern",       "P_Consolidation"),
    ("Crackle_Phase",         "P_Consolidation"),
    ("Crackle_Subtype",       "P_Consolidation"),
    ("Diminished_Sounds",     "P_Consolidation"),
    ("Bilateral_Asymmetry_Score", "P_Consolidation"),

    # --- Acoustic → P_Obstructive_Exacerbation ---
    ("Wheeze_Pattern",        "P_Obstructive_Exacerbation"),
    ("Wheeze_Phase",          "P_Obstructive_Exacerbation"),
    ("Wheeze_Compound",       "P_Obstructive_Exacerbation"),
    ("FEV1_FVC_Known",        "P_Obstructive_Exacerbation"),

    # --- Acoustic → P_ILD_Flag ---
    # Velcro state of Crackle_Subtype + late_insp phase + basal gradient → ILD
    ("Crackle_Subtype",       "P_ILD_Flag"),
    ("Crackle_Phase",         "P_ILD_Flag"),
    ("Craniocaudal_Gradient", "P_ILD_Flag"),

    # --- Acoustic → Severity ---
    ("Zone_Reliability",      "Confidence_Interval"),
    ("Bilateral_Asymmetry_Score", "P_Consolidation"),

    # --- Rhonchi ---
    ("Rhonchi_Present",       "P_Consolidation"),

    # --- Intermediate → Output ---
    ("P_Consolidation",            "P_Antibiotics_Indicated"),
    ("P_Obstructive_Exacerbation", "P_Antibiotics_Indicated"),
    ("P_Severity",                 "P_Antibiotics_Indicated"),
    ("Anthonisen_Type",            "P_Antibiotics_Indicated"),
    ("P_ILD_Flag",                 "P_Antibiotics_Indicated"),  # suppression path

    ("P_Antibiotics_Indicated",    "Confidence_Interval"),
    ("P_Antibiotics_Indicated",    "Uncertainty_Flag"),
    ("P_Antibiotics_Indicated",    "Disposition"),
    ("P_ILD_Flag",                 "Uncertainty_Flag"),         # ILD always raises uncertainty
    ("P_Severity",                 "Disposition"),
    ("SpO2",                       "Disposition"),              # escalation overlay
]


# ---------------------------------------------------------------------------
# BUILD THE DAG
# ---------------------------------------------------------------------------

def build_dag() -> DiscreteBayesianNetwork:
    """
    Constructs the Layer 3 DAG with all nodes and edges.
    CPTs are NOT set here — see cpts.py.
    Returns a DiscreteBayesianNetwork with the structure only.
    """
    model = DiscreteBayesianNetwork(EDGES)

    # Verify all expected nodes are present
    model_nodes = set(model.nodes())
    expected = set(ALL_NODES) - {"Contributing_Factor_Weights"}  # computed post-inference
    missing = expected - model_nodes
    extra   = model_nodes - expected

    if missing:
        print(f"WARNING: Nodes in spec but not in graph: {missing}")
    if extra:
        print(f"INFO: Nodes in graph not in spec list (check intended): {extra}")

    # Confirm no cycles
    assert nx.is_directed_acyclic_graph(model), "DAG has cycles — fix edges!"

    return model


# ---------------------------------------------------------------------------
# VISUALISE
# ---------------------------------------------------------------------------

def visualise_dag(model: DiscreteBayesianNetwork, save_path: str = None):
    """
    Renders the DAG with colour-coded node groups.
    Send screenshot to Arvind for sign-off before writing any CPTs.
    """
    fig, ax = plt.subplots(figsize=(20, 14))

    # Colour per node group
    color_map = {}
    for n in model.nodes():
        if n in ROOT_NODES:
            color_map[n] = "#4A90D9"       # blue
        elif n in CLINICAL_NODES:
            color_map[n] = "#7ED321"       # green
        elif n in ACOUSTIC_NODES:
            color_map[n] = "#F5A623"       # orange
        elif n in INTERMEDIATE_NODES:
            color_map[n] = "#9B59B6"       # purple
        elif n in OUTPUT_NODES:
            color_map[n] = "#E74C3C"       # red
        else:
            color_map[n] = "#95A5A6"       # grey

    node_colors = [color_map.get(n, "#95A5A6") for n in model.nodes()]

    pos = nx.drawing.nx_agraph.graphviz_layout(model, prog="dot")

    nx.draw(
        model, pos, ax=ax,
        node_color=node_colors,
        node_size=2200,
        font_size=7,
        font_weight="bold",
        font_color="white",
        arrows=True,
        arrowsize=15,
        edge_color="#555555",
        with_labels=True,
    )

    # Legend
    legend_items = [
        mpatches.Patch(color="#4A90D9", label="Root nodes"),
        mpatches.Patch(color="#7ED321", label="Clinical inputs"),
        mpatches.Patch(color="#F5A623", label="Acoustic inputs (Layer 2)"),
        mpatches.Patch(color="#9B59B6", label="Intermediate nodes"),
        mpatches.Patch(color="#E74C3C", label="Output nodes"),
    ]
    ax.legend(handles=legend_items, loc="upper left", fontsize=9)
    ax.set_title("TreBle Respire — Layer 3 DAG (v2.1.0)\nSend to Arvind for sign-off", fontsize=13)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"DAG saved to {save_path}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# MAIN — run to verify DAG and save screenshot
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model = build_dag()

    print("=" * 60)
    print("DAG BUILD COMPLETE")
    print("=" * 60)
    print(f"Total nodes : {len(model.nodes())}")
    print(f"Total edges : {len(model.edges())}")
    print()
    print("ALL NODES:")
    for n in sorted(model.nodes()):
        print(f"  {n}")
    print()
    print("ALL EDGES:")
    for e in sorted(model.edges()):
        print(f"  {e[0]:35s} → {e[1]}")
    print()

    # Confirm no cycles
    if nx.is_directed_acyclic_graph(model):
        print("✓ No cycles detected — valid DAG")
    else:
        print("✗ CYCLE DETECTED — fix edges before proceeding!")

    # Save visualisation
    visualise_dag(model, save_path="/home/claude/layer3/dag_screenshot.png")
    print("✓ DAG screenshot saved — send to Arvind for sign-off")