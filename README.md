# Gara — application de démonstration NER wolof

Interface de démonstration du modèle de reconnaissance d'entités nommées
en wolof (afro-xlmr-base affiné sur MasakhaNER Wolof).

## Contenu de l'application

| Section | Rôle |
|---|---|
| **Annoter un texte** | Saisie libre ou exemples, surlignage des entités, tableau des détections, export CSV/JSON |
| **Performances** | Scores réels sur le jeu de test, détail par type, analyse des erreurs |
| **Documentation** | Explication complète du processus : données, préparation, modèle, entraînement, évaluation |
| **Limites du modèle** | Faiblesses connues, pistes d'amélioration, précautions d'usage |

## Installation

```bash
pip install -r requirements.txt
```

## Placer le modèle

L'application cherche le modèle entraîné dans un dossier nommé
`afro-xlmr-wolof-ner`, à côté de `app.py`.

Ce dossier est celui produit par le notebook à la fin de la phase 3 :

```python
trainer.save_model("afro-xlmr-wolof-ner")
tokenizer.save_pretrained("afro-xlmr-wolof-ner")
```

Si l'entraînement a été fait sur Colab, téléchargez le dossier puis
décompressez-le ici. Arborescence attendue :

```
app/
├── app.py
├── requirements.txt
└── afro-xlmr-wolof-ner/
    ├── config.json
    ├── model.safetensors
    ├── tokenizer.json
    ├── sentencepiece.bpe.model
    └── special_tokens_map.json
```

Pour utiliser un autre emplacement, définissez la variable d'environnement
`WOLOF_NER_MODEL` avec le chemin voulu.

## Lancer

```bash
streamlit run app.py
```

L'application s'ouvre sur `http://localhost:8501`.

## Notes

- Le premier chargement du modèle prend quelques secondes ; il est ensuite
  mis en cache pour toute la session (`@st.cache_resource`).
- Le curseur **seuil de confiance** filtre les détections : l'augmenter
  favorise la précision, le baisser favorise le rappel.
- Les scores affichés dans la section Performances sont ceux mesurés sur les
  539 phrases de test. Si vous réentraînez le modèle, mettez à jour le
  dictionnaire `TEST_RESULTS` en tête de `app.py`.
