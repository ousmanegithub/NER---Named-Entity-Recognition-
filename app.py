"""
GARA — Reconnaissance d'entites nommees en wolof
================================================
Application de demonstration du modele NER wolof (afro-xlmr-base fine-tune
sur MasakhaNER Wolof).

Lancement :
    streamlit run app.py

Le modele entraine doit se trouver dans le dossier MODEL_DIR (voir plus bas),
ou etre accessible depuis le Hub HuggingFace.
"""

import json
import os
import re
from pathlib import Path

import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------------------------

MODEL_DIR = os.environ.get("WOLOF_NER_MODEL", "afro-xlmr-wolof-ner")

ENTITY_META = {
    "PER":  {"label": "Personne",     "color": "#C2410C", "tint": "#FDE8DC"},
    "LOC":  {"label": "Lieu",         "color": "#0E7490", "tint": "#DCF1F5"},
    "ORG":  {"label": "Organisation", "color": "#6D28D9", "tint": "#EBE3FB"},
    "DATE": {"label": "Date",         "color": "#047857", "tint": "#DBF0E7"},
}

# Resultats reels obtenus sur le jeu de test (539 phrases).
TEST_RESULTS = {
    "f1_global": 0.6254,
    "precision_globale": 0.6108,
    "rappel_global": 0.6406,
    "par_type": [
        {"type": "LOC",  "precision": 0.7689, "rappel": 0.8199, "f1": 0.7936, "support": 211},
        {"type": "PER",  "precision": 0.5693, "rappel": 0.6534, "f1": 0.6085, "support": 176},
        {"type": "ORG",  "precision": 0.3506, "rappel": 0.4909, "f1": 0.4091, "support": 55},
        {"type": "DATE", "precision": 0.3939, "rappel": 0.1857, "f1": 0.2524, "support": 70},
    ],
}

CORPUS_STATS = {
    "train": {"phrases": 1871, "tokens": 37243},
    "dev":   {"phrases": 267,  "tokens": 5294},
    "test":  {"phrases": 539,  "tokens": 11682},
    "part_o": 94.0,
    "oov": 49.0,
}

EXEMPLES = {
    "Personnes et lieu": "Maki Sall dafa dem Ndakaaru ak Usmaan Sonko.",
    "Organisation": "Njiitu réew mi jotoon na ak ñoñ ONU ci Ndakaaru.",
    "Date": "Ci atum 2019 la woon , ci weeru sulet.",
    "Phrase longue": (
        "SAFIYETU BÉEY moo doon jangale ci Universite Sheex Anta Jóob "
        "bu Ndakaaru ci atum 2020 ."
    ),
}

st.set_page_config(
    page_title="Gara - NER wolof",
    page_icon="◧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# STYLE
# --------------------------------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
        --indigo:      #1B2A4A;
        --indigo-soft: #33507F;
        --sand:        #F6F2EA;
        --ink:         #14181F;
        --muted:       #5C6577;
        --rule:        #DED6C8;
    }

    .stApp { background: var(--sand); }

    h1, h2, h3 {
        font-family: 'Fraunces', Georgia, serif !important;
        color: var(--indigo) !important;
        letter-spacing: -0.01em;
    }
    html, body, .stMarkdown, p, li, label { font-family: 'Inter', sans-serif; }

    /* En-tete */
    .gara-head {
        border-bottom: 2px solid var(--indigo);
        padding-bottom: 0.9rem;
        margin-bottom: 1.6rem;
    }
    .gara-title {
        font-family: 'Fraunces', serif;
        font-size: 2.6rem;
        font-weight: 700;
        color: var(--indigo);
        line-height: 1;
        margin: 0;
    }
    .gara-sub {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.74rem;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        color: var(--muted);
        margin-top: 0.45rem;
    }

    /* Eyebrow de section */
    .eyebrow {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: var(--indigo-soft);
        border-left: 3px solid var(--indigo-soft);
        padding-left: 0.6rem;
        margin: 1.6rem 0 0.7rem 0;
    }

    /* Zone de texte annote — element signature */
    .gloss {
        background: #FFFFFF;
        border: 1px solid var(--rule);
        border-radius: 3px;
        padding: 1.6rem 1.5rem;
        font-size: 1.12rem;
        line-height: 2.9;
        color: var(--ink);
    }
    .tok { position: relative; padding: 0.05em 0.1em; }
    .ent {
        position: relative;
        padding: 0.1em 0.28em;
        border-radius: 2px;
        font-weight: 600;
        white-space: nowrap;
    }
    .ent .tag {
        position: absolute;
        top: -1.05em;
        left: 0;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.58rem;
        letter-spacing: 0.1em;
        font-weight: 500;
        opacity: 0.95;
    }

    /* Cartes de statistiques */
    .statcard {
        background: #FFFFFF;
        border: 1px solid var(--rule);
        border-left: 3px solid var(--indigo);
        border-radius: 3px;
        padding: 0.9rem 1rem;
    }
    .statcard .k {
        font-family: 'Fraunces', serif;
        font-size: 1.85rem;
        font-weight: 700;
        color: var(--indigo);
        line-height: 1;
    }
    .statcard .v {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.66rem;
        letter-spacing: 0.11em;
        text-transform: uppercase;
        color: var(--muted);
        margin-top: 0.4rem;
    }

    /* Encadre note */
    .note {
        background: #FFFFFF;
        border: 1px solid var(--rule);
        border-left: 3px solid #B08968;
        padding: 0.9rem 1.1rem;
        border-radius: 3px;
        font-size: 0.92rem;
    }

    .legend-chip {
        display: inline-block;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem;
        letter-spacing: 0.08em;
        padding: 0.22em 0.6em;
        border-radius: 2px;
        margin-right: 0.4rem;
    }

    section[data-testid="stSidebar"] { background: #EFE9DE; }
    section[data-testid="stSidebar"] h1 { font-size: 1.15rem !important; }

    .stTabs [data-baseweb="tab-list"] { gap: 1.4rem; }
    .stTabs [data-baseweb="tab"] {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# MODELE
# --------------------------------------------------------------------------

@st.cache_resource(show_spinner="Chargement du modele…")
def charger_modele(chemin: str):
    """Charge le pipeline de NER. Mis en cache : une seule fois par session."""
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("USE_FLAX", "0")
    from transformers import (AutoTokenizer, AutoModelForTokenClassification,
                              pipeline)

    tokenizer = AutoTokenizer.from_pretrained(chemin)
    modele = AutoModelForTokenClassification.from_pretrained(chemin)
    return pipeline(
        "token-classification",
        model=modele,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
    )


def annoter(texte: str, seuil: float):
    """Renvoie la liste des entites detectees au-dessus du seuil."""
    pipe = charger_modele(MODEL_DIR)
    brut = pipe(texte)
    return [e for e in brut if float(e["score"]) >= seuil]


def rendre_gloss(texte: str, entites: list) -> str:
    """Construit le HTML du texte annote (entites surlignees + etiquette)."""
    if not entites:
        return f'<div class="gloss">{st_escape(texte)}</div>'

    entites = sorted(entites, key=lambda e: e["start"])
    morceaux, curseur = [], 0
    for ent in entites:
        debut, fin = ent["start"], ent["end"]
        groupe = ent["entity_group"]
        meta = ENTITY_META.get(groupe, {"color": "#444", "tint": "#EEE"})
        if debut > curseur:
            morceaux.append(
                f'<span class="tok">{st_escape(texte[curseur:debut])}</span>')
        morceaux.append(
            f'<span class="ent" style="background:{meta["tint"]};'
            f'box-shadow: inset 0 -2px 0 {meta["color"]};color:{meta["color"]}">'
            f'<span class="tag" style="color:{meta["color"]}">{groupe}</span>'
            f'{st_escape(texte[debut:fin])}</span>'
        )
        curseur = fin
    if curseur < len(texte):
        morceaux.append(f'<span class="tok">{st_escape(texte[curseur:])}</span>')
    return f'<div class="gloss">{"".join(morceaux)}</div>'


def st_escape(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def carte_stat(valeur, legende):
    return (f'<div class="statcard"><div class="k">{valeur}</div>'
            f'<div class="v">{legende}</div></div>')


def eyebrow(txt):
    st.markdown(f'<div class="eyebrow">{txt}</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------
# EN-TETE + NAVIGATION
# --------------------------------------------------------------------------

st.markdown(
    '<div class="gara-head">'
    '<div class="gara-title">Gara</div>'
    '<div class="gara-sub">Reconnaissance d\'entites nommees &nbsp;·&nbsp; '
    'langue wolof &nbsp;·&nbsp; afro-xlmr-base</div>'
    '</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### Navigation")
    page = st.radio(
        "Aller a",
        ["Annoter un texte", "Performances", "Documentation", "Limites du modele"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("### Reglages")
    seuil = st.slider(
        "Seuil de confiance", 0.0, 1.0, 0.50, 0.05,
        help="Les entites dont le score est inferieur a ce seuil sont ecartees. "
             "Augmentez pour plus de precision, baissez pour plus de rappel.",
    )

    st.markdown("---")
    st.markdown("### Legende")
    for code, meta in ENTITY_META.items():
        st.markdown(
            f'<span class="legend-chip" style="background:{meta["tint"]};'
            f'color:{meta["color"]}">{code}</span> {meta["label"]}',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.caption(f"Modele : `{MODEL_DIR}`")


# --------------------------------------------------------------------------
# PAGE 1 — ANNOTER
# --------------------------------------------------------------------------

if page == "Annoter un texte":

    gauche, droite = st.columns([3, 2], gap="large")

    with gauche:
        eyebrow("Texte a analyser")

        choix = st.selectbox(
            "Charger un exemple",
            ["— Saisir mon propre texte —"] + list(EXEMPLES.keys()),
        )
        defaut = "" if choix.startswith("—") else EXEMPLES[choix]

        texte = st.text_area(
            "Texte en wolof",
            value=defaut,
            height=170,
            placeholder="Collez ou saisissez un texte en wolof, puis lancez l'analyse.",
            label_visibility="collapsed",
        )

        lancer = st.button("Analyser le texte", type="primary",
                           use_container_width=True)

    with droite:
        eyebrow("Comment lire le resultat")
        st.markdown(
            '<div class="note">Chaque entite detectee est surlignee et '
            'surmontee de son type. Le tableau en dessous donne le score de '
            'confiance du modele et la position exacte dans le texte. '
            'Ajustez le seuil dans le panneau lateral pour filtrer les '
            'detections peu sures.</div>',
            unsafe_allow_html=True,
        )

    if lancer:
        if not texte.strip():
            st.warning("Saisissez un texte avant de lancer l'analyse.")
        elif not Path(MODEL_DIR).exists():
            st.error(
                f"Modele introuvable dans `{MODEL_DIR}`. Placez le dossier du "
                f"modele entraine a cet emplacement, ou definissez la variable "
                f"d'environnement `WOLOF_NER_MODEL`."
            )
        else:
            try:
                entites = annoter(texte, seuil)
            except Exception as err:
                st.error(f"Le chargement du modele a echoue : {err}")
                entites = None

            if entites is not None:
                eyebrow("Texte annote")
                st.markdown(rendre_gloss(texte, entites), unsafe_allow_html=True)

                if entites:
                    eyebrow("Entites detectees")

                    df = pd.DataFrame([{
                        "Entite":   e["word"],
                        "Type":     e["entity_group"],
                        "Libelle":  ENTITY_META.get(e["entity_group"], {}).get("label", "—"),
                        "Confiance": round(float(e["score"]), 4),
                        "Debut":    int(e["start"]),
                        "Fin":      int(e["end"]),
                    } for e in entites])

                    st.dataframe(df, use_container_width=True, hide_index=True)

                    # Compteurs par type
                    cols = st.columns(len(ENTITY_META))
                    counts = df["Type"].value_counts().to_dict()
                    for col, (code, meta) in zip(cols, ENTITY_META.items()):
                        with col:
                            st.markdown(
                                carte_stat(counts.get(code, 0), meta["label"]),
                                unsafe_allow_html=True)

                    eyebrow("Exporter")
                    c1, c2 = st.columns(2)
                    with c1:
                        st.download_button(
                            "Telecharger en CSV",
                            df.to_csv(index=False).encode("utf-8"),
                            file_name="entites_wolof.csv",
                            mime="text/csv",
                            use_container_width=True,
                        )
                    with c2:
                        st.download_button(
                            "Telecharger en JSON",
                            json.dumps(
                                {"texte": texte,
                                 "entites": df.to_dict(orient="records")},
                                ensure_ascii=False, indent=2).encode("utf-8"),
                            file_name="entites_wolof.json",
                            mime="application/json",
                            use_container_width=True,
                        )
                else:
                    st.info(
                        "Aucune entite au-dessus du seuil de "
                        f"{seuil:.2f}. Baissez le seuil dans le panneau lateral, "
                        "ou essayez un texte contenant des noms de personnes, "
                        "de lieux, d'organisations ou des dates."
                    )


# --------------------------------------------------------------------------
# PAGE 2 — PERFORMANCES
# --------------------------------------------------------------------------

elif page == "Performances":

    eyebrow("Resultats sur le jeu de test")
    st.markdown(
        "Mesures obtenues sur les **539 phrases de test**, jamais vues pendant "
        "l'entrainement. Toutes les valeurs sont calculees **au niveau de "
        "l'entite** (une entite compte juste seulement si tous ses tokens sont "
        "correctement etiquetes), avec la bibliotheque `seqeval`."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(carte_stat(f'{TEST_RESULTS["f1_global"]:.3f}', "F1 global"),
                    unsafe_allow_html=True)
    with c2:
        st.markdown(carte_stat(f'{TEST_RESULTS["precision_globale"]:.3f}',
                               "Precision"), unsafe_allow_html=True)
    with c3:
        st.markdown(carte_stat(f'{TEST_RESULTS["rappel_global"]:.3f}',
                               "Rappel"), unsafe_allow_html=True)

    eyebrow("Detail par type d'entite")
    df_perf = pd.DataFrame(TEST_RESULTS["par_type"])
    df_perf.columns = ["Type", "Precision", "Rappel", "F1", "Occurrences (test)"]
    st.dataframe(df_perf, use_container_width=True, hide_index=True)

    st.bar_chart(df_perf.set_index("Type")["F1"], height=260)

    st.markdown(
        '<div class="note"><b>Lecture.</b> Les ecarts entre types ne sont pas '
        'aleatoires : ils suivent la frequence des entites dans le corpus '
        'd\'entrainement. LOC et PER, les types les plus representes, obtiennent '
        'les meilleurs scores. ORG et DATE, rares, restent difficiles. Le cas de '
        'DATE est le plus marque : son rappel de 0,19 signifie que le modele ne '
        'retrouve qu\'une date sur cinq.</div>',
        unsafe_allow_html=True,
    )

    eyebrow("Ou vont les erreurs")
    st.markdown(
        "L'analyse de la matrice de confusion montre que **la majorite des "
        "erreurs sont des entites classees `O`** (texte ordinaire) plutot que "
        "des confusions entre types. Autrement dit, le modele **sous-detecte** "
        "au lieu de se tromper de categorie. Pour les dates, 116 tokens `I-DATE` "
        "et 47 tokens `B-DATE` ont ete predits `O`."
    )
    st.markdown(
        "C'est la consequence directe du desequilibre du corpus : "
        f"**{CORPUS_STATS['part_o']:.0f} % des tokens** portent l'etiquette `O`, "
        "ce qui pousse le modele vers la prudence."
    )


# --------------------------------------------------------------------------
# PAGE 3 — DOCUMENTATION
# --------------------------------------------------------------------------

elif page == "Documentation":

    st.markdown(
        "Cette section decrit le systeme de bout en bout : les donnees, la "
        "facon dont le texte est prepare, l'architecture du modele, "
        "l'entrainement et l'evaluation."
    )

    t1, t2, t3, t4, t5 = st.tabs(
        ["Donnees", "Preparation", "Modele", "Entrainement", "Evaluation"])

    # ---- Donnees ----
    with t1:
        eyebrow("Corpus MasakhaNER Wolof")
        st.markdown(
            "Le corpus est fourni au format **CoNLL** : un token par ligne, "
            "suivi de son etiquette, les phrases etant separees par une ligne "
            "vide."
        )
        st.code("SAFIYETU    B-PER\nBÉEY        I-PER\nCéy         O\nKoronaa     O\n!           O",
                language="text")

        c1, c2, c3 = st.columns(3)
        for col, split in zip((c1, c2, c3), ("train", "dev", "test")):
            with col:
                st.markdown(
                    carte_stat(CORPUS_STATS[split]["phrases"], f"phrases · {split}"),
                    unsafe_allow_html=True)

        eyebrow("Le schema BIO")
        st.markdown(
            "Chaque mot recoit une etiquette qui indique **s'il appartient a une "
            "entite et a quelle position** :\n\n"
            "- `B-XXX` — **B**egin : premier mot d'une entite de type XXX\n"
            "- `I-XXX` — **I**nside : mot suivant, dans la meme entite\n"
            "- `O` — **O**utside : mot hors entite\n\n"
            "Ce codage permet de distinguer deux entites voisines. Dans "
            "« *Ndakaaru Tiwaawan* », si les deux mots portaient simplement "
            "`LOC`, rien ne dirait s'il s'agit d'un lieu en deux mots ou de deux "
            "lieux distincts. Avec `B-LOC B-LOC`, la reponse est sans ambiguite."
        )

        eyebrow("Deux caracteristiques determinantes")
        st.markdown(
            f"**Desequilibre.** {CORPUS_STATS['part_o']:.0f} % des tokens portent "
            "l'etiquette `O`. Un systeme qui repondrait `O` partout obtiendrait "
            "une exactitude de 94 % tout en etant parfaitement inutile — c'est "
            "pourquoi l'evaluation se fait au niveau de l'entite, pas du token."
        )
        st.markdown(
            f"**Mots inconnus.** {CORPUS_STATS['oov']:.0f} % du vocabulaire du "
            "jeu de test n'apparait jamais dans l'entrainement. Un modele "
            "fonctionnant sur des mots entiers serait desarme ; c'est l'argument "
            "central en faveur d'un modele a sous-mots pre-entraine."
        )

    # ---- Preparation ----
    with t2:
        eyebrow("Du mot au sous-mot")
        st.markdown(
            "Le modele ne connait pas les mots entiers : son vocabulaire est "
            "compose d'environ 250 000 **sous-mots**. Un mot inconnu est "
            "decoupe en fragments connus, ce qui permet de traiter du "
            "vocabulaire jamais rencontre."
        )
        st.code('tokenizer.tokenize("SAFIYETU")\n'
                '# ["▁", "SAF", "IYE", "TU"]', language="python")

        eyebrow("Le probleme d'alignement")
        st.markdown(
            "Le corpus fournit **une etiquette par mot**, mais le modele predit "
            "**une etiquette par sous-mot**. Il faut donc redistribuer les "
            "etiquettes. La regle retenue est la convention standard :\n\n"
            "- le **premier** sous-mot d'un mot recoit l'etiquette du mot ;\n"
            "- les sous-mots suivants recoivent `-100` ;\n"
            "- les jetons speciaux (`<s>`, `</s>`, remplissage) recoivent aussi `-100`."
        )
        st.markdown(
            '<div class="note"><b>Pourquoi <code>-100</code> ?</b> C\'est la '
            'valeur que la fonction de cout de PyTorch ignore par convention. '
            'Ces positions ne participent donc ni a l\'apprentissage ni au '
            'calcul de l\'erreur : le modele n\'est juge que sur le premier '
            'sous-mot de chaque mot, soit exactement une prediction par mot.</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            "**Pourquoi ne pas etiqueter tous les sous-mots ?** Dupliquer "
            "`B-PER` sur les quatre fragments de « SAFIYETU » reviendrait a "
            "declarer quatre debuts d'entite pour une seule personne, ce qui "
            "casserait la logique BIO."
        )

        st.markdown("**Resultat de l'alignement sur un exemple reel :**")
        st.code(
            "sous-mot   word_id   etiquette\n"
            "---------------------------------\n"
            "<s>        None      -100\n"
            "▁          0         B-PER\n"
            "SAF        0         -100\n"
            "IYE        0         -100\n"
            "TU         0         -100\n"
            "▁B         1         I-PER\n"
            "É          1         -100\n"
            "EY         1         -100\n"
            "▁Cé        2         O\n"
            "y          2         -100\n"
            "</s>       None      -100",
            language="text",
        )
        st.markdown(
            "La methode `word_ids()` du tokenizer indique, pour chaque "
            "sous-mot, de quel mot d'origine il provient — c'est elle qui rend "
            "l'alignement possible."
        )

    # ---- Modele ----
    with t3:
        eyebrow("Architecture")
        st.markdown(
            "Le systeme repose sur **afro-xlmr-base**, un modele de type "
            "*transformer* de la famille XLM-RoBERTa, pre-entraine sur des "
            "langues africaines dont le wolof."
        )
        st.markdown(
            "Le traitement se fait en deux etages :\n\n"
            "1. **L'encodeur.** Chaque sous-mot est transforme en un vecteur qui "
            "encode son sens *en contexte*. Le mecanisme d'**attention** permet "
            "a chaque position de consulter toutes les autres positions de la "
            "phrase : le modele peut donc utiliser les mots voisins pour decider. "
            "C'est essentiel en NER, ou le meme mot peut etre un nom de personne "
            "ou un nom commun selon le contexte.\n\n"
            "2. **La tete de classification.** Une couche lineaire posee au-dessus "
            "de l'encodeur projette chaque vecteur vers **9 scores**, un par "
            "etiquette possible. L'etiquette retenue est celle du score le plus eleve."
        )

        eyebrow("Pourquoi le fine-tuning")
        st.markdown(
            "Le corpus compte moins de 2 000 phrases d'entrainement — trop peu "
            "pour apprendre une langue de zero. Le **fine-tuning** consiste a "
            "partir d'un modele qui *comprend deja* le texte, puis a l'ajuster "
            "sur la tache precise. L'encodeur apporte la connaissance "
            "linguistique ; l'entrainement n'a plus qu'a apprendre a reconnaitre "
            "les entites."
        )
        st.markdown(
            '<div class="note"><b>Choix du modele de base.</b> afro-xlmr-base a '
            'ete retenu plutot que xlm-roberta-base parce que son '
            'pre-entrainement couvre des langues africaines, wolof compris. '
            'Face aux 49 % de mots inconnus du jeu de test, un modele qui '
            'connait deja la langue part avec un avantage reel.</div>',
            unsafe_allow_html=True,
        )

    # ---- Entrainement ----
    with t4:
        eyebrow("Parametres")
        st.dataframe(pd.DataFrame([
            {"Parametre": "Taux d'apprentissage", "Valeur": "2e-5",
             "Raison": "Valeur faible, usuelle en fine-tuning : on ajuste sans effacer le pre-entrainement"},
            {"Parametre": "Taille de lot", "Valeur": "16",
             "Raison": "Compromis entre stabilite du gradient et memoire GPU"},
            {"Parametre": "Epoques (maximum)", "Valeur": "15",
             "Raison": "Plafond ; l'arret anticipe intervient generalement avant"},
            {"Parametre": "Decroissance des poids", "Valeur": "0.01",
             "Raison": "Regularisation, utile sur un petit corpus"},
            {"Parametre": "Patience (arret anticipe)", "Valeur": "3",
             "Raison": "Arret si le F1 de validation ne progresse plus"},
        ]), use_container_width=True, hide_index=True)

        eyebrow("Arret anticipe et selection du modele")
        st.markdown(
            "L'entrainement s'est arrete a la **8e epoque** sur 15 : le F1 de "
            "validation n'avait plus progresse depuis trois evaluations. "
            "Poursuivre n'aurait produit que du **surapprentissage** — la perte "
            "d'entrainement continuait de baisser tandis que la perte de "
            "validation remontait."
        )
        st.markdown(
            "Grace a l'option `load_best_model_at_end`, le modele conserve "
            "n'est pas celui de la derniere epoque mais celui de la **meilleure "
            "epoque mesuree sur la validation**."
        )

        eyebrow("Role des trois jeux de donnees")
        st.markdown(
            "- **Entrainement** — le modele ajuste ses poids dessus.\n"
            "- **Validation** — sert a decider *quand arreter* et *quel modele "
            "garder*. Le modele ne s'entraine pas dessus, mais ces donnees "
            "influencent les choix : le score obtenu est donc legerement optimiste.\n"
            "- **Test** — n'intervient qu'une fois, a la toute fin. C'est la "
            "seule mesure honnete de la performance reelle."
        )

    # ---- Evaluation ----
    with t5:
        eyebrow("Mesurer au niveau de l'entite")
        st.markdown(
            "Une evaluation token par token serait trompeuse. Avec 94 % de "
            "tokens `O`, un systeme muet afficherait 94 % d'exactitude. "
            "L'evaluation se fait donc **au niveau de l'entite** : une entite "
            "n'est comptee juste que si **tous** ses tokens sont correctement "
            "etiquetes, avec les bonnes frontieres et le bon type."
        )
        st.markdown(
            "Consequence concrete : les entites longues sont plus fragiles. Les "
            "dates du corpus font en moyenne **3,2 tokens** ; un seul token "
            "manque et l'entite entiere est comptee fausse. C'est l'une des "
            "raisons du faible score sur DATE."
        )

        eyebrow("Les trois mesures")
        st.markdown(
            "- **Precision** — parmi les entites annoncees, quelle proportion "
            "est correcte ? Elle chute quand le modele invente des entites.\n"
            "- **Rappel** — parmi les entites reellement presentes, quelle "
            "proportion a ete trouvee ? Il chute quand le modele en oublie.\n"
            "- **F1** — moyenne harmonique des deux. Elle ne peut etre elevee "
            "que si precision *et* rappel le sont, ce qui en fait la mesure de "
            "reference sur donnees desequilibrees."
        )
        st.latex(r"F_1 = 2 \times \frac{\text{precision} \times \text{rappel}}"
                 r"{\text{precision} + \text{rappel}}")

        st.markdown(
            '<div class="note">La bibliotheque <code>seqeval</code> reconstruit '
            'les entites a partir des etiquettes BIO avant de comparer, ce que '
            'ne fait pas une metrique de classification classique. C\'est elle '
            'qui produit les scores presentes dans la section Performances.</div>',
            unsafe_allow_html=True,
        )


# --------------------------------------------------------------------------
# PAGE 4 — LIMITES
# --------------------------------------------------------------------------

elif page == "Limites du modele":

    eyebrow("Ce que le systeme fait mal")
    st.markdown(
        "**Les dates sont largement manquees.** Avec un rappel de 0,19, quatre "
        "dates sur cinq passent inapercues. Deux causes se cumulent : leur "
        "rarete dans le corpus (130 occurrences a l'entrainement) et leur "
        "longueur moyenne de 3,2 tokens, qui multiplie les occasions d'erreur."
    )
    st.markdown(
        "**Les organisations sont instables.** Avec 55 occurrences seulement "
        "dans le jeu de test, les scores sur ORG reposent sur peu d'exemples et "
        "doivent etre interpretes avec prudence."
    )
    st.markdown(
        "**Le modele sous-detecte plutot qu'il ne se trompe.** La majorite des "
        "erreurs consistent a classer une entite en texte ordinaire. Un "
        "utilisateur doit donc s'attendre a des oublis plus qu'a des "
        "categorisations erronees."
    )
    st.markdown(
        "**Le domaine est etroit.** Le corpus provient essentiellement "
        "d'articles de presse. Sur des messages informels, des transcriptions "
        "orales ou du wolof mele au francais, les performances seront "
        "inferieures."
    )

    eyebrow("Pistes d'amelioration")
    st.dataframe(pd.DataFrame([
        {"Piste": "Ponderer la fonction de cout",
         "Effet attendu": "Donner plus de poids aux classes rares (DATE, ORG) pour contrer le desequilibre"},
        {"Piste": "Augmenter les donnees",
         "Effet attendu": "Enrichir le corpus en dates et organisations, par annotation ou generation"},
        {"Piste": "Passer a afro-xlmr-large",
         "Effet attendu": "Modele plus grand, generalement meilleur, au prix d'un entrainement plus lourd"},
        {"Piste": "Ajouter une couche CRF",
         "Effet attendu": "Modeliser les transitions entre etiquettes et garantir des sequences BIO valides"},
        {"Piste": "Assouplir l'arret anticipe",
         "Effet attendu": "Laisser plus de temps aux classes rares, qui convergent plus lentement"},
    ]), use_container_width=True, hide_index=True)

    eyebrow("Usage responsable")
    st.markdown(
        "Ce systeme est un prototype academique. Il ne doit pas etre utilise "
        "seul pour des decisions engageantes : les entites extraites demandent "
        "une verification humaine, en particulier sur les dates et les "
        "organisations."
    )

    st.markdown("---")
    st.caption(
        "Corpus MasakhaNER (Masakhane) · modele de base Davlan/afro-xlmr-base · "
        "projet academique de traitement automatique des langues."
    )
