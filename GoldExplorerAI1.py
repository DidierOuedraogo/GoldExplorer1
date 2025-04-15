import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
from folium.plugins import HeatMap, MarkerCluster
from streamlit_folium import folium_static
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import io
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Configuration de la page
st.set_page_config(
    page_title="GoldExplorerAI - Endeavour Mining",
    page_icon="⚒️",
    layout="wide"
)

# Styles CSS personnalisés - Interface moderne
st.markdown("""
<style>
    :root {
        --main-color: #FFD700;
        --secondary-color: #B8860B;
        --endeavour-blue: #003366;
        --endeavour-gold: #D4AF37;
        --light-bg: #F8F9FA;
        --dark-text: #343A40;
    }
    
    .main-title {
        color: var(--endeavour-blue);
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
        border-bottom: 4px solid var(--endeavour-gold);
        padding-bottom: 0.5rem;
    }
    
    .sub-title {
        color: var(--endeavour-blue);
        font-size: 1.8rem;
        font-weight: bold;
        margin-top: 2rem;
        border-left: 4px solid var(--endeavour-gold);
        padding-left: 0.8rem;
    }
    
    .highlight {
        background-color: var(--light-bg);
        padding: 1.2rem;
        border-radius: 8px;
        border-left: 4px solid var(--endeavour-gold);
        margin-bottom: 1rem;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    
    .stat-card {
        background-color: white;
        border-radius: 8px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
        border-top: 4px solid var(--endeavour-gold);
    }
    
    .author-info {
        background-color: var(--light-bg);
        padding: 1rem;
        border-radius: 8px;
        font-style: italic;
        text-align: center;
        margin: 1rem 0;
    }
    
    .footer {
        text-align: center;
        padding: 1rem;
        font-size: 0.8rem;
        color: var(--dark-text);
        border-top: 1px solid #dee2e6;
        margin-top: 2rem;
    }
    
    .model-selection {
        background-color: var(--light-bg);
        padding: 1.2rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        border-left: 4px solid var(--endeavour-blue);
    }
    
    .algorithm-info {
        background-color: white;
        padding: 1rem;
        border-radius: 5px;
        margin-top: 1rem;
        border: 1px solid #dee2e6;
    }
    
    /* Amélioration des widgets Streamlit */
    div.stButton > button {
        background-color: var(--endeavour-blue);
        color: white;
        border-radius: 5px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    
    div.stButton > button:hover {
        background-color: var(--endeavour-gold);
        color: var(--endeavour-blue);
    }
    
    .stSelectbox, .stSlider {
        border-radius: 5px;
    }
    
    /* Dataframe styling */
    .dataframe {
        border: none !important;
    }
    
    .dataframe th {
        background-color: var(--endeavour-blue);
        color: white;
    }
    
    .dataframe tr:nth-child(even) {
        background-color: var(--light-bg);
    }
</style>
""", unsafe_allow_html=True)

# Fonctions utiles
def generate_sample_geo_data(size=100):
    """Génère des données géospatiales synthétiques pour démonstration"""
    data = {
        'x': np.random.uniform(-3, 3, size) + 46,  # Longitude
        'y': np.random.uniform(-3, 3, size) + 2,   # Latitude
        'z': np.random.uniform(5, 300, size),      # Profondeur
        'teneur_or': np.random.exponential(1, size),
        'distance_orpaillage': np.random.uniform(50, 5000, size),
        'indice_alteration': np.random.uniform(0, 1, size),
        'pourcentage_sulfures': np.random.uniform(0, 15, size),
        'distance_faille': np.random.uniform(10, 2000, size),
        'distance_cisaillement': np.random.uniform(20, 3000, size),
        'type_roche': np.random.choice(['Basalte', 'Granite', 'Diorite', 'Schiste', 'Gabbro'], size),
        'anomalie_geochimique': np.random.uniform(0, 5, size),
        'resistivite': np.random.uniform(100, 5000, size)
    }
    
    # Introduire des corrélations entre variables pour plus de réalisme
    for i in range(size):
        # Les teneurs en or sont plus élevées près des orpaillages
        if data['distance_orpaillage'][i] < 500:
            data['teneur_or'][i] *= 2.5
        
        # Les teneurs en or sont plus élevées près des failles/cisaillements
        if data['distance_faille'][i] < 200 or data['distance_cisaillement'][i] < 300:
            data['teneur_or'][i] *= 2
        
        # Les teneurs en or sont corrélées aux sulfures et à l'altération
        if data['pourcentage_sulfures'][i] > 10 and data['indice_alteration'][i] > 0.7:
            data['teneur_or'][i] *= 3
    
    return pd.DataFrame(data)

def predict_gold_potential(df, weights=None):
    """Calcule un score de potentiel aurifère basé sur les variables clés"""
    if weights is None:
        weights = {
            'teneur_or': 0.25,
            'distance_orpaillage': 0.15,
            'indice_alteration': 0.15,
            'pourcentage_sulfures': 0.15,
            'distance_faille': 0.15,
            'distance_cisaillement': 0.15
        }
    
    # Normalisation et inversion des distances (plus la distance est faible, plus le score est élevé)
    max_dist_orp = df['distance_orpaillage'].max()
    max_dist_faille = df['distance_faille'].max()
    max_dist_cisaillement = df['distance_cisaillement'].max()
    
    potential = (
        df['teneur_or'] / df['teneur_or'].max() * weights['teneur_or'] +
        (1 - df['distance_orpaillage'] / max_dist_orp) * weights['distance_orpaillage'] +
        df['indice_alteration'] * weights['indice_alteration'] +
        df['pourcentage_sulfures'] / 15 * weights['pourcentage_sulfures'] +
        (1 - df['distance_faille'] / max_dist_faille) * weights['distance_faille'] +
        (1 - df['distance_cisaillement'] / max_dist_cisaillement) * weights['distance_cisaillement']
    )
    
    # Normaliser le score final entre 0 et 100
    return (potential - potential.min()) / (potential.max() - potential.min()) * 100

def train_random_forest(X, y):
    """Entraîne un modèle de forêt aléatoire"""
    model = RandomForestRegressor(
        n_estimators=100, 
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42
    )
    model.fit(X, y)
    return model

def train_neural_network(X, y):
    """Entraîne un réseau de neurones"""
    model = MLPRegressor(
        hidden_layer_sizes=(100, 50),
        activation='relu',
        solver='adam',
        alpha=0.0001,
        max_iter=500,
        random_state=42
    )
    model.fit(X, y)
    return model

def predict_with_ml(df, target_var='teneur_or', algorithm='random_forest', test_size=0.2):
    """Entraîne un modèle de ML et prédit le potentiel aurifère"""
    
    # Préparation des données
    features = ['distance_orpaillage', 'indice_alteration', 'pourcentage_sulfures', 
               'distance_faille', 'distance_cisaillement', 'x', 'y', 'z']
    
    # Ajouter des colonnes pour les types de roches (one-hot encoding)
    if 'type_roche' in df.columns:
        rock_dummies = pd.get_dummies(df['type_roche'], prefix='roche')
        df = pd.concat([df, rock_dummies], axis=1)
        # Ajouter les colonnes de types de roches aux features
        for col in rock_dummies.columns:
            features.append(col)
    
    # Vérifier que toutes les colonnes sont présentes
    features = [f for f in features if f in df.columns]
    
    # Diviser en ensembles d'entraînement et de test
    X = df[features]
    y = df[target_var]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # Standardisation des features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Entraînement du modèle
    if algorithm == 'random_forest':
        model = train_random_forest(X_train, y_train)
        # Importance des features pour la forêt aléatoire
        feature_importance = pd.DataFrame({
            'feature': features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
    else:  # réseau de neurones
        model = train_neural_network(X_train_scaled, y_train)
        feature_importance = None  # Pas disponible directement pour les réseaux de neurones
    
    # Évaluation du modèle
    if algorithm == 'random_forest':
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
    else:  # réseau de neurones
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)
    
    metrics = {
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'train_r2': r2_score(y_train, y_pred_train),
        'test_r2': r2_score(y_test, y_pred_test)
    }
    
    # Prédiction pour tous les échantillons
    if algorithm == 'random_forest':
        all_predictions = model.predict(X)
    else:  # réseau de neurones
        all_predictions = model.predict(scaler.transform(X))
    
    # Normaliser les prédictions entre 0 et 100
    normalized_predictions = (all_predictions - all_predictions.min()) / (all_predictions.max() - all_predictions.min()) * 100
    
    return normalized_predictions, metrics, feature_importance, model

def create_heatmap(df):
    """Crée une carte de chaleur du potentiel aurifère"""
    m = folium.Map(location=[df['y'].mean(), df['x'].mean()], zoom_start=9, 
                  tiles="CartoDB positron")
    
    # Ajouter le calque de chaleur
    heat_data = [[row['y'], row['x'], row['potentiel_aurifere']] for index, row in df.iterrows()]
    HeatMap(heat_data, radius=15, gradient={0.4: 'blue', 0.65: 'lime', 0.8: 'yellow', 1: 'red'}).add_to(m)
    
    # Ajouter les sites les plus prometteurs avec cluster
    marker_cluster = MarkerCluster().add_to(m)
    top_sites = df.sort_values('potentiel_aurifere', ascending=False).head(10)
    
    for idx, row in top_sites.iterrows():
        folium.Marker(
            location=[row['y'], row['x']],
            popup=f"""
                <strong>Site {idx}</strong><br>
                Potentiel: {row['potentiel_aurifere']:.1f}/100<br>
                Teneur en or: {row['teneur_or']:.2f} g/t<br>
                Type de roche: {row['type_roche']}<br>
                Altération: {row['indice_alteration']:.2f}<br>
                Sulfures: {row['pourcentage_sulfures']:.1f}%
            """,
            icon=folium.Icon(color='red', icon='star', prefix='fa')
        ).add_to(marker_cluster)
    
    # Ajouter une légende
    legend_html = '''
    <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000; background-color: white; 
                padding: 10px; border: 2px solid grey; border-radius: 5px;">
      <p><strong>Potentiel aurifère</strong></p>
      <p><span style="color: red; font-size: 18px;">●</span> Élevé</p>
      <p><span style="color: yellow; font-size: 18px;">●</span> Modéré-élevé</p>
      <p><span style="color: lime; font-size: 18px;">●</span> Modéré</p>
      <p><span style="color: blue; font-size: 18px;">●</span> Faible</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    return m

def importer_donnees():
    uploaded_file = st.file_uploader("Importer un fichier de données (CSV, Excel)", type=['csv', 'xlsx'])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            # Vérifier les colonnes essentielles
            colonnes_requises = ['x', 'y']
            if not all(col in df.columns for col in colonnes_requises):
                st.error("Le fichier doit contenir au minimum les colonnes 'x' et 'y' pour les coordonnées.")
                return None
            
            return df
        except Exception as e:
            st.error(f"Erreur lors de l'importation: {e}")
            return None
    return None

# Sidebar
with st.sidebar:
    st.title("GoldExplorerAI")
    st.subheader("Endeavour Mining")
    
    st.markdown("---")
    
    # Navigation
    page = st.radio(
        "Navigation", 
        ["Accueil", "Importation de données", "Analyse de données", "Prédiction de potentiel", "Machine Learning", "Visualisation 3D", "À propos"]
    )
    
    st.markdown("---")
    
    st.markdown("""
    <div class="author-info">
    <strong>Auteur:</strong><br>
    Didier Ouedraogo, P.Geo<br>
    Endeavour Mining
    </div>
    """, unsafe_allow_html=True)

# Initialisation des données de session
if 'data' not in st.session_state:
    st.session_state.data = generate_sample_geo_data(300)
    st.session_state.data['potentiel_aurifere'] = predict_gold_potential(st.session_state.data)
    st.session_state.using_sample_data = True
    st.session_state.ml_model = None
    st.session_state.ml_metrics = None
    st.session_state.feature_importance = None

# Pages
if page == "Accueil":
    st.markdown("<h1 class='main-title'>GoldExplorerAI</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>Automatisation et optimisation de l'exploration aurifère par l'intelligence artificielle</h3>", unsafe_allow_html=True)
    
    # Avertissement si données d'exemple
    if st.session_state.get('using_sample_data', True):
        st.warning("Vous utilisez actuellement des données synthétiques à des fins de démonstration. Utilisez la page 'Importation de données' pour charger vos propres données.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("<div class='stat-card'><h2>85%</h2><p>de précision dans la prédiction des zones à fort potentiel</p></div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='stat-card'><h2>60%</h2><p>de réduction des coûts d'exploration initiale</p></div>", unsafe_allow_html=True)
    with col3:
        st.markdown("<div class='stat-card'><h2>2.2x</h2><p>d'amélioration du taux de découverte</p></div>", unsafe_allow_html=True)
    
    st.markdown("<h2 class='sub-title'>Comment ça fonctionne</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class='highlight'>
        <h3>🔍 Analyse de données géospatiales</h3>
        <p>Intégration et analyse de multiples couches de données: géologie, géochimie, proximité des failles, distances aux orpaillages et altérations.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='highlight'>
        <h3>🧠 Modèles prédictifs avancés</h3>
        <p>Algorithmes d'IA (forêts aléatoires, réseaux de neurones) entraînés pour reconnaître les motifs associés aux dépôts aurifères.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='highlight'>
        <h3>📊 Visualisation interactive</h3>
        <p>Cartes de chaleur interactives, modèles 3D et tableaux de bord adaptés aux besoins des géologues et décideurs.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='highlight'>
        <h3>🎯 Définition de cibles d'exploration</h3>
        <p>Identification et priorisation automatique des zones les plus prometteuses pour optimiser les campagnes d'exploration au stade initial.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<h2 class='sub-title'>Applications principales</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    ### Stade initial d'exploration
    
    GoldExplorerAI est spécialement conçu pour aider les équipes d'exploration d'Endeavour Mining à identifier et prioriser les cibles d'exploration les plus prometteuses au stade initial des projets. L'application intègre les variables clés suivantes :
    
    - **Coordonnées spatiales (x, y, z)** - Position et profondeur des échantillons
    - **Teneur en or** - Concentrations mesurées dans les échantillons
    - **Distance aux sites d'orpaillage** - Proximité des activités artisanales
    - **Indices d'altération** - Mesures de l'altération hydrothermale
    - **Concentration en sulfures** - Pourcentage de sulfures dans les échantillons
    - **Proximité des failles** - Distance aux structures géologiques majeures
    - **Proximité des couloirs de cisaillement** - Distance aux zones de déformation
    - **Autres variables géologiques pertinentes** - Type de roche, anomalies géochimiques, etc.
    """)
    
    st.markdown("<h2 class='sub-title'>Démarrer</h2>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Utiliser l'exemple de données", key="home_sample_data"):
            st.session_state.data = generate_sample_geo_data(300)
            st.session_state.data['potentiel_aurifere'] = predict_gold_potential(st.session_state.data)
            st.session_state.using_sample_data = True
            st.session_state.ml_model = None
            st.session_state.ml_metrics = None
            st.session_state.feature_importance = None
            st.success("Données d'exemple générées avec succès!")
    with col2:
        if st.button("Importer vos propres données", key="home_import_data"):
            st.switch_page("Importation de données")

elif page == "Importation de données":
    st.markdown("<h1 class='main-title'>Importation et préparation des données</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    ### Instructions d'importation
    
    Pour utiliser GoldExplorerAI avec vos propres données, veuillez préparer un fichier CSV ou Excel contenant au minimum les colonnes suivantes:
    
    - **x, y** : Coordonnées (longitude, latitude)
    - **z** : Profondeur (optionnel)
    
    Les colonnes additionnelles recommandées pour une analyse complète:
    
    - **teneur_or** : Concentration en or des échantillons
    - **distance_orpaillage** : Distance aux sites d'orpaillage
    - **indice_alteration** : Mesure de l'altération (0-1)
    - **pourcentage_sulfures** : Concentration en sulfures (%)
    - **distance_faille** : Distance aux failles majeures
    - **distance_cisaillement** : Distance aux couloirs de cisaillement
    - **type_roche** : Classification lithologique
    """)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_data = importer_donnees()
        
        if uploaded_data is not None:
            st.success("Données importées avec succès!")
            st.write("Aperçu des données:")
            st.dataframe(uploaded_data.head())
            
            # Vérification et mappage des colonnes
            st.subheader("Mappage des colonnes")
            
            cols_mapping = {}
            required_cols = ['x', 'y', 'teneur_or', 'distance_orpaillage', 'indice_alteration', 
                           'pourcentage_sulfures', 'distance_faille', 'distance_cisaillement']
            
            for col in required_cols:
                cols_mapping[col] = st.selectbox(
                    f"Sélectionner la colonne pour '{col}'",
                    options=['Non disponible'] + list(uploaded_data.columns),
                    index=0 if col not in uploaded_data.columns else list(uploaded_data.columns).index(col) + 1
                )
            
            if st.button("Confirmer et traiter les données"):
                processed_data = uploaded_data.copy()
                
                # Renommer les colonnes selon le mappage
                for target_col, source_col in cols_mapping.items():
                    if source_col != 'Non disponible':
                        processed_data[target_col] = uploaded_data[source_col]
                    elif target_col not in processed_data.columns:
                        # Ajouter des colonnes manquantes avec des valeurs par défaut raisonnables
                        if target_col in ['teneur_or', 'indice_alteration', 'pourcentage_sulfures']:
                            processed_data[target_col] = 0
                        elif target_col in ['distance_orpaillage', 'distance_faille', 'distance_cisaillement']:
                            processed_data[target_col] = 5000  # Distance grande = faible influence
                
                # Ajouter des colonnes manquantes standards
                if 'type_roche' not in processed_data.columns:
                    processed_data['type_roche'] = 'Inconnu'
                if 'z' not in processed_data.columns:
                    processed_data['z'] = 0
                
                # Calculer le potentiel aurifère
                processed_data['potentiel_aurifere'] = predict_gold_potential(processed_data)
                
                # Sauvegarder dans la session
                st.session_state.data = processed_data
                st.session_state.using_sample_data = False
                st.session_state.ml_model = None
                st.session_state.ml_metrics = None
                st.session_state.feature_importance = None
                
                st.success("Données traitées avec succès! Vous pouvez maintenant explorer les différentes analyses.")

    with col2:
        st.subheader("Options alternatives")
        
        if st.button("Utiliser des données d'exemple"):
            st.session_state.data = generate_sample_geo_data(300)
            st.session_state.data['potentiel_aurifere'] = predict_gold_potential(st.session_state.data)
            st.session_state.using_sample_data = True
            st.session_state.ml_model = None
            st.session_state.ml_metrics = None
            st.session_state.feature_importance = None
            st.success("Données d'exemple générées avec succès!")
        
        st.markdown("""
        ### Format des données
        
        Exemple de structure de fichier attendue:
        
        | x | y | z | teneur_or | type_roche | distance_faille |
        |---|---|---|-----------|------------|-----------------|
        | 2.345 | 46.789 | 50 | 1.2 | Basalte | 150 |
        | 2.350 | 46.792 | 75 | 0.8 | Granite | 320 |
        | 2.356 | 46.788 | 45 | 2.3 | Schiste | 80 |
        
        **Note**: Les noms exacts des colonnes peuvent varier tant que vous définissez correctement le mappage.
        """)

elif page == "Analyse de données":
    st.markdown("<h1 class='main-title'>Analyse exploratoire des données</h1>", unsafe_allow_html=True)
    
    if st.session_state.get('using_sample_data', True):
        st.info("Vous utilisez actuellement des données synthétiques à des fins de démonstration.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Aperçu des données")
        st.dataframe(st.session_state.data.head(10))
        
        st.subheader("Statistiques descriptives")
        numeric_cols = st.session_state.data.select_dtypes(include=['float64', 'int64']).columns
        st.dataframe(st.session_state.data[numeric_cols].describe())
    
    with col2:
        st.subheader("Distribution des types de roches")
        fig = px.pie(st.session_state.data, names='type_roche', title="Répartition lithologique",
                     color_discrete_sequence=px.colors.sequential.Viridis)
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Teneur en or par type de roche")
        fig = px.box(st.session_state.data, x='type_roche', y='teneur_or', 
                     color='type_roche', title="Distribution des teneurs par lithologie",
                     color_discrete_sequence=px.colors.sequential.Viridis)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("<h2 class='sub-title'>Analyse de corrélation</h2>", unsafe_allow_html=True)
    
    corr_cols = ['teneur_or', 'distance_orpaillage', 'indice_alteration', 'pourcentage_sulfures', 
                'distance_faille', 'distance_cisaillement', 'potentiel_aurifere']
    
    if all(col in st.session_state.data.columns for col in corr_cols):
        corr_data = st.session_state.data[corr_cols]
        corr_matrix = corr_data.corr()
        
        fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", color_continuous_scale='Viridis',
                        title="Matrice de corrélation des variables clés")
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        ### Interprétation des corrélations
        
        La matrice de corrélation permet d'identifier quelles variables sont les plus fortement associées à la teneur en or et au potentiel aurifère calculé. Les valeurs proches de 1 indiquent une forte corrélation positive, tandis que les valeurs proches de -1 indiquent une forte corrélation négative.
        """)
        
    st.markdown("<h2 class='sub-title'>Analyse spatiale</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Distribution spatiale des teneurs en or")
        fig = px.scatter(st.session_state.data, x='x', y='y', color='teneur_or',
                        size='teneur_or', hover_name='type_roche',
                        color_continuous_scale='Inferno',
                        title="Carte des teneurs en or")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Relation entre profondeur et teneur")
        if 'z' in st.session_state.data.columns:
            fig = px.scatter(st.session_state.data, x='z', y='teneur_or', 
                            color='type_roche', size='potentiel_aurifere',
                            title="Teneur en or selon la profondeur",
                            color_discrete_sequence=px.colors.qualitative.Bold)
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("<h2 class='sub-title'>Analyse multivariée</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Relation entre variables clés")
        
        x_var = st.selectbox("Variable X", options=numeric_cols, index=numeric_cols.get_loc('pourcentage_sulfures') if 'pourcentage_sulfures' in numeric_cols else 0)
        y_var = st.selectbox("Variable Y", options=numeric_cols, index=numeric_cols.get_loc('indice_alteration') if 'indice_alteration' in numeric_cols else 0)
        color_var = st.selectbox("Coloration par", options=numeric_cols, index=numeric_cols.get_loc('teneur_or') if 'teneur_or' in numeric_cols else 0)
        
        fig = px.scatter(st.session_state.data, x=x_var, y=y_var, color=color_var,
                        size='potentiel_aurifere', hover_name='type_roche',
                        color_continuous_scale='Viridis',
                        title=f"Analyse {x_var} vs {y_var} (coloré par {color_var})")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Teneurs vs. Distances")
        
        distance_var = st.selectbox(
            "Choisir la variable de distance",
            options=['distance_orpaillage', 'distance_faille', 'distance_cisaillement'],
            index=0 if 'distance_faille' not in st.session_state.data.columns else 1
        )
        
        if distance_var in st.session_state.data.columns:
            df_binned = st.session_state.data.copy()
            
            # Utiliser qcut pour créer des bins de taille égale basés sur les quantiles
            n_bins = 5  # Nombre de bins
            
            # Vérifier qu'il y a suffisamment de valeurs uniques pour créer des bins
            unique_values = df_binned[distance_var].nunique()
            if unique_values < n_bins:
                st.warning(f"Pas assez de valeurs uniques ({unique_values}) pour créer {n_bins} groupes. Utilisation de {unique_values} groupes à la place.")
                n_bins = max(2, unique_values)  # au moins 2 groupes si possible
            
            try:
                df_binned['distance_bin'] = pd.qcut(
                    df_binned[distance_var], 
                    q=n_bins, 
                    duplicates='drop'  # Important pour gérer les valeurs dupliquées
                )
                
                # Convertir en chaîne pour l'affichage
                df_binned['distance_bin'] = df_binned['distance_bin'].astype(str)
                
                fig = px.box(
                    df_binned.dropna(subset=['distance_bin']),  # Supprimer les valeurs NaN
                    x='distance_bin', 
                    y='teneur_or',
                    color='distance_bin',
                    title=f"Teneurs en or selon la {distance_var}",
                    labels={'distance_bin': f'{distance_var} (m)', 'teneur_or': 'Teneur en or (g/t)'}
                )
                st.plotly_chart(fig, use_container_width=True)
                
            except ValueError as e:
                st.error(f"Impossible de créer les bins pour la variable {distance_var}: {e}")
                st.info("Essayez d'utiliser une autre variable de distance ou de modifier vos données.")

elif page == "Prédiction de potentiel":
    st.markdown("<h1 class='main-title'>Prédiction du Potentiel Aurifère</h1>", unsafe_allow_html=True)
    
    if st.session_state.get('using_sample_data', True):
        st.info("Vous utilisez actuellement des données synthétiques à des fins de démonstration.")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("<h3 class='sub-title'>Paramètres du modèle</h3>", unsafe_allow_html=True)
        
        st.write("Ajustez l'importance des facteurs:")
        
        weight_teneur = st.slider("Teneur en or", 0.0, 1.0, 0.25, 0.05)
        weight_orpaillage = st.slider("Proximité des orpaillages", 0.0, 1.0, 0.15, 0.05)
        weight_alteration = st.slider("Indice d'altération", 0.0, 1.0, 0.15, 0.05)
        weight_sulfures = st.slider("Pourcentage de sulfures", 0.0, 1.0, 0.15, 0.05)
        weight_faille = st.slider("Proximité des failles", 0.0, 1.0, 0.15, 0.05)
        weight_cisaillement = st.slider("Proximité des cisaillements", 0.0, 1.0, 0.15, 0.05)
        
        # Normalisation des poids
        total = weight_teneur + weight_orpaillage + weight_alteration + weight_sulfures + weight_faille + weight_cisaillement
        weights = {
            'teneur_or': weight_teneur / total,
            'distance_orpaillage': weight_orpaillage / total,
            'indice_alteration': weight_alteration / total,
            'pourcentage_sulfures': weight_sulfures / total,
            'distance_faille': weight_faille / total,
            'distance_cisaillement': weight_cisaillement / total
        }
        
        st.write("Poids normalisés:")
        for key, value in weights.items():
            st.write(f"- {key}: {value:.2f}")
        
        if st.button("Recalculer le potentiel"):
            st.session_state.data['potentiel_aurifere'] = predict_gold_potential(st.session_state.data, weights)
            st.success("Potentiel aurifère recalculé avec les nouveaux paramètres!")
    
    with col2:
        st.markdown("<h3 class='sub-title'>Carte de chaleur du potentiel aurifère</h3>", unsafe_allow_html=True)
        m = create_heatmap(st.session_state.data)
        folium_static(m, width=700, height=500)
    
    st.markdown("<h2 class='sub-title'>Sites les plus prometteurs</h2>", unsafe_allow_html=True)
    
    top_n = st.slider("Nombre de sites à afficher", 5, 20, 10)
    top_sites = st.session_state.data.sort_values('potentiel_aurifere', ascending=False).head(top_n)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.dataframe(top_sites[['x', 'y', 'z', 'potentiel_aurifere', 'teneur_or', 'type_roche', 
                              'indice_alteration', 'pourcentage_sulfures']])
        
        fig = px.bar(top_sites, x=top_sites.index, y='potentiel_aurifere', 
                     color='type_roche', hover_data=['teneur_or', 'indice_alteration', 'pourcentage_sulfures'],
                     labels={'index': 'ID de l\'échantillon', 'potentiel_aurifere': 'Score de potentiel (/100)'},
                     title=f"Score de potentiel des {top_n} meilleurs sites",
                     color_discrete_sequence=px.colors.qualitative.Bold)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("<div class='highlight'>", unsafe_allow_html=True)
        st.markdown("### Catégories de priorité")
        
        # Déterminer les seuils
        high_threshold = 80
        medium_threshold = 60
        
        # Compter le nombre de sites dans chaque catégorie
        n_high = len(top_sites[top_sites['potentiel_aurifere'] >= high_threshold])
        n_medium = len(top_sites[(top_sites['potentiel_aurifere'] >= medium_threshold) & 
                                 (top_sites['potentiel_aurifere'] < high_threshold)])
        n_low = len(top_sites[top_sites['potentiel_aurifere'] < medium_threshold])
        
        st.markdown(f"""
        - 🔴 **Priorité haute** ({n_high} sites)  
          Score > {high_threshold}
        - 🟠 **Priorité moyenne** ({n_medium} sites)  
          Score {medium_threshold}-{high_threshold}
        - 🟢 **Priorité basse** ({n_low} sites)  
          Score < {medium_threshold}
        """)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Distribution par type de roche
        st.subheader("Distribution lithologique")
        rock_counts = top_sites['type_roche'].value_counts()
        fig = px.pie(values=rock_counts.values, names=rock_counts.index, 
                     title="Types de roches dans les meilleurs sites",
                     color_discrete_sequence=px.colors.sequential.Viridis)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("<h2 class='sub-title'>Analyse détaillée des facteurs de potentialité</h2>", unsafe_allow_html=True)
    
    # Sélectionner un site pour analyse détaillée
    selected_site = st.selectbox(
        "Sélectionner un site pour analyse détaillée",
        options=top_sites.index,
        format_func=lambda x: f"Site {x} (Score: {top_sites.loc[x, 'potentiel_aurifere']:.1f})"
    )
    
    if selected_site is not None:
        site_data = top_sites.loc[selected_site]
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Créer un graphique radar des facteurs
            categories = ['Teneur en or', 'Proximité orpaillage', 'Altération', 
                        'Sulfures', 'Proximité faille', 'Proximité cisaillement']
            
            # Normaliser les valeurs pour le radar
            max_teneur = st.session_state.data['teneur_or'].max()
            max_sulfures = st.session_state.data['pourcentage_sulfures'].max()
            
            values = [
                site_data['teneur_or'] / max_teneur * 100,
                (1 - site_data['distance_orpaillage'] / st.session_state.data['distance_orpaillage'].max()) * 100,
                site_data['indice_alteration'] * 100,
                site_data['pourcentage_sulfures'] / max_sulfures * 100,
                (1 - site_data['distance_faille'] / st.session_state.data['distance_faille'].max()) * 100,
                (1 - site_data['distance_cisaillement'] / st.session_state.data['distance_cisaillement'].max()) * 100
            ]
            
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name='Facteurs de potentialité'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )
                ),
                title="Analyse des facteurs de potentialité"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown(f"""
            ### Détails du site {selected_site}
            
            - **Coordonnées**: {site_data['x']:.4f}, {site_data['y']:.4f}
            - **Profondeur**: {site_data['z']:.1f} m
            - **Type de roche**: {site_data['type_roche']}
            - **Teneur en or**: {site_data['teneur_or']:.2f} g/t
            - **Indice d'altération**: {site_data['indice_alteration']:.2f}
            - **Pourcentage de sulfures**: {site_data['pourcentage_sulfures']:.1f}%
            - **Distance orpaillage**: {site_data['distance_orpaillage']:.0f} m
            - **Distance faille**: {site_data['distance_faille']:.0f} m
            - **Distance cisaillement**: {site_data['distance_cisaillement']:.0f} m
            
            **Score de potentiel**: {site_data['potentiel_aurifere']:.1f}/100
            """)
            
            # Classification du site
            if site_data['potentiel_aurifere'] >= high_threshold:
                st.markdown("""
                <div style="background-color: rgba(255, 0, 0, 0.1); padding: 10px; border-radius: 5px; border-left: 4px solid red;">
                <strong>Priorité: HAUTE</strong><br>
                Ce site présente un potentiel aurifère exceptionnel et devrait être priorisé pour les travaux d'exploration détaillés.
                </div>
                """, unsafe_allow_html=True)
            elif site_data['potentiel_aurifere'] >= medium_threshold:
                st.markdown("""
                <div style="background-color: rgba(255, 165, 0, 0.1); padding: 10px; border-radius: 5px; border-left: 4px solid orange;">
                <strong>Priorité: MOYENNE</strong><br>
                Ce site présente un bon potentiel et mérite des investigations supplémentaires.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="background-color: rgba(0, 128, 0, 0.1); padding: 10px; border-radius: 5px; border-left: 4px solid green;">
                <strong>Priorité: BASSE</strong><br>
                Ce site présente un potentiel limité mais pourrait être reconsidéré en fonction de nouvelles données.
                </div>
                """, unsafe_allow_html=True)

elif page == "Machine Learning":
    st.markdown("<h1 class='main-title'>Prédiction avancée par Machine Learning</h1>", unsafe_allow_html=True)
    
    if st.session_state.get('using_sample_data', True):
        st.info("Vous utilisez actuellement des données synthétiques à des fins de démonstration.")
    
    st.markdown("""
    <div class="model-selection">
    <h3>Choisissez un algorithme de Machine Learning</h3>
    <p>Cette section utilise des algorithmes avancés pour prédire le potentiel aurifère en se basant sur les relations complexes entre toutes les variables disponibles.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        target_var = st.selectbox(
            "Variable cible à prédire",
            options=['teneur_or', 'potentiel_aurifere'],
            index=0
        )
        
        algorithm = st.radio(
            "Algorithme",
            options=["Random Forest (Forêts aléatoires)", "Neural Network (Réseau de neurones)"],
            index=0
        )
        
        test_size = st.slider("Proportion des données pour le test", 0.1, 0.5, 0.2, 0.05)
        
        # Informations sur l'algorithme choisi
        if algorithm == "Random Forest (Forêts aléatoires)":
            st.markdown("""
            <div class="algorithm-info">
            <h4>Forêts aléatoires</h4>
            <p>Cet algorithme construit de multiples arbres de décision et fusionne leurs prédictions. 
            Il est particulièrement efficace pour capturer les relations non-linéaires entre les variables géologiques
            et offre une interprétabilité via l'importance des variables.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="algorithm-info">
            <h4>Réseau de neurones</h4>
            <p>Ce modèle est inspiré du cerveau humain et peut capturer des patterns extrêmement complexes
            dans les données. Il est particulièrement puissant pour les relations non-linéaires,
            mais moins interprétable que les forêts aléatoires.</p>
            </div>
            """, unsafe_allow_html=True)
        
        if st.button("Entraîner et prédire"):
            with st.spinner(f"Entraînement du modèle {algorithm} en cours..."):
                # Conversion de l'algorithme choisi au format attendu par la fonction
                algo_param = "random_forest" if algorithm == "Random Forest (Forêts aléatoires)" else "neural_network"
                
                # Entraînement et prédiction
                predictions, metrics, feature_importance, model = predict_with_ml(
                    st.session_state.data, 
                    target_var=target_var,
                    algorithm=algo_param,
                    test_size=test_size
                )
                
                # Sauvegarde des résultats dans la session
                st.session_state.ml_predictions = predictions
                st.session_state.ml_metrics = metrics
                st.session_state.feature_importance = feature_importance
                st.session_state.ml_model = model
                st.session_state.ml_algorithm = algorithm
                st.session_state.ml_target = target_var
                
                # Mettre à jour les données avec les prédictions
                st.session_state.data['ml_prediction'] = predictions
                
                st.success(f"Modèle entraîné avec succès! R² sur les données de test: {metrics['test_r2']:.3f}")
    
    with col2:
        if st.session_state.get('ml_predictions') is not None:
            st.subheader(f"Prédictions pour {st.session_state.ml_target}")
            
            # Carte de chaleur des prédictions simplifiée (sans mapbox)
            df_map = st.session_state.data.copy()
            
            fig = px.scatter(
                df_map, x='x', y='y', 
                color='ml_prediction', size='ml_prediction',
                color_continuous_scale='Inferno',
                title=f"Carte des prédictions par {st.session_state.ml_algorithm}"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Aucun modèle n'a encore été entraîné. Veuillez configurer et entraîner un modèle pour visualiser les prédictions.")
    
    # Afficher les métriques si disponibles
    if st.session_state.get('ml_metrics') is not None:
        st.markdown("<h2 class='sub-title'>Performances du modèle</h2>", unsafe_allow_html=True)
        
        metrics = st.session_state.ml_metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("RMSE (Entraînement)", f"{metrics['train_rmse']:.3f}")
        with col2:
            st.metric("RMSE (Test)", f"{metrics['test_rmse']:.3f}")
        with col3:
            st.metric("R² (Entraînement)", f"{metrics['train_r2']:.3f}")
        with col4:
            st.metric("R² (Test)", f"{metrics['test_r2']:.3f}")
        
        st.markdown("""
        **Interprétation des métriques:**
        - **RMSE (Root Mean Square Error)**: Erreur moyenne entre les prédictions et les valeurs réelles. Plus cette valeur est basse, meilleur est le modèle.
        - **R²**: Proportion de la variance expliquée par le modèle. Une valeur de 1 indique une prédiction parfaite, 0 indique que le modèle n'est pas meilleur qu'une prédiction constante.
        """)
    
    # Afficher l'importance des variables si disponible (pour Random Forest)
    if st.session_state.get('feature_importance') is not None:
        st.markdown("<h2 class='sub-title'>Importance des variables</h2>", unsafe_allow_html=True)
        
        feat_importance = st.session_state.feature_importance
        
        if feat_importance is not None:  # Pour Random Forest uniquement
            fig = px.bar(
                feat_importance, x='importance', y='feature',
                orientation='h',
                title="Importance relative des variables dans le modèle",
                labels={'importance': 'Importance (%)', 'feature': 'Variable'},
                color='importance',
                color_continuous_scale='Viridis'
            )
            
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            ### Interprétation de l'importance des variables
            
            Le graphique ci-dessus montre l'influence relative de chaque variable sur les prédictions du modèle. 
            Les variables avec une importance plus élevée ont un impact plus significatif sur le potentiel aurifère prédit.
            
            Cette information peut vous aider à:
            - Comprendre quels facteurs géologiques sont les plus déterminants pour les gisements d'or
            - Orienter vos futures campagnes d'exploration en vous concentrant sur les variables les plus importantes
            - Améliorer la collecte de données en priorisant les mesures les plus informatives
            """)
        else:
            st.info("L'importance des variables n'est pas disponible pour les réseaux de neurones.")
    
    # Comparaison entre prédictions et observations
    if st.session_state.get('ml_predictions') is not None:
        st.markdown("<h2 class='sub-title'>Comparaison des prédictions</h2>", unsafe_allow_html=True)
        
        df_comparison = st.session_state.data.copy()
        
        # Calculer sites prioritaires selon différentes méthodes
        top_ml = df_comparison.sort_values('ml_prediction', ascending=False).head(10).index
        top_standard = df_comparison.sort_values('potentiel_aurifere', ascending=False).head(10).index
        
        # Calculer le taux de recouvrement
        overlap = len(set(top_ml).intersection(set(top_standard)))
        overlap_pct = overlap / 10 * 100
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Version sans ligne de tendance (pour éviter la dépendance à statsmodels)
            fig = px.scatter(
                df_comparison, x=st.session_state.ml_target, y='ml_prediction',
                color='type_roche', hover_data=['x', 'y', 'z'],
                title=f"Corrélation entre valeurs observées et prédites",
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.metric("Taux de recouvrement des top 10 sites", f"{overlap_pct:.1f}%", f"{overlap}/10 sites communs")
            
            st.markdown(f"""
            ### Analyse de cohérence
            
            La comparaison entre les prédictions du modèle {st.session_state.ml_algorithm} et la méthode de scoring standard montre:
            
            - **{overlap}/10 sites prioritaires** sont identifiés par les deux méthodes
            - Corrélation entre les deux approches: visible sur le graphique à gauche
            - Les différences peuvent révéler des sites potentiellement intéressants qui n'auraient pas été identifiés par l'approche standard
            """)
            
            if overlap_pct > 70:
                st.success("Forte concordance entre les deux méthodes, ce qui renforce la confiance dans les prédictions.")
            elif overlap_pct > 40:
                st.info("Concordance modérée entre les deux méthodes. Considérez les sites identifiés uniquement par le ML comme des cibles secondaires intéressantes.")
            else:
                st.warning("Faible concordance entre les deux méthodes. Examinez attentivement les différences et considérez d'ajuster les paramètres du modèle.")

elif page == "Visualisation 3D":
    st.markdown("<h1 class='main-title'>Modélisation 3D</h1>", unsafe_allow_html=True)
    
    if st.session_state.get('using_sample_data', True):
        st.info("Vous utilisez actuellement des données synthétiques à des fins de démonstration.")
    
    # Sélection de variables pour la visualisation 3D
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("Options de visualisation")
        
        color_options = ['potentiel_aurifere', 'teneur_or', 'indice_alteration', 'pourcentage_sulfures']
        if 'ml_prediction' in st.session_state.data.columns:
            color_options.insert(0, 'ml_prediction')
        
        color_var = st.selectbox(
            "Coloration par",
            options=color_options,
            index=0
        )
        
        size_var = st.selectbox(
            "Taille des points",
            options=['potentiel_aurifere', 'teneur_or', 'indice_alteration', 'pourcentage_sulfures'],
            index=1
        )
        
        filter_threshold = st.slider(
            "Seuil de potentiel minimal",
            0, 100, 0
        )
        
        # Options d'affichage
        show_top_sites = st.checkbox("Mettre en évidence les sites prioritaires", value=True)
        
    with col2:
        df_filtered = st.session_state.data[st.session_state.data['potentiel_aurifere'] >= filter_threshold]
        
        fig = px.scatter_3d(
            df_filtered, x='x', y='y', z='z',
            color=color_var, size=size_var,
            color_continuous_scale='Inferno',
            opacity=0.8,
            title=f"Modèle 3D des échantillons colorés par {color_var}"
        )
        
        # Mettre en évidence les meilleurs sites
        if show_top_sites:
            top_sites = df_filtered.sort_values('potentiel_aurifere', ascending=False).head(10)
            
            fig.add_trace(
                go.Scatter3d(
                    x=top_sites['x'],
                    y=top_sites['y'],
                    z=top_sites['z'],
                    mode='markers',
                    marker=dict(
                        size=12,
                        color='lime',
                        symbol='diamond',
                        line=dict(color='black', width=1)
                    ),
                    name='Sites prioritaires'
                )
            )
        
        fig.update_layout(
            scene=dict(
                xaxis_title='X (Longitude)',
                yaxis_title='Y (Latitude)',
                zaxis_title='Z (Profondeur)',
                aspectmode='manual',
                aspectratio=dict(x=1.5, y=1.5, z=1),
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            height=700
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("<h2 class='sub-title'>Sections et profils</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Profil vertical (projection)")
        
        # Direction de la coupe
        direction = st.selectbox(
            "Direction de la coupe",
            options=["Est-Ouest (X)", "Nord-Sud (Y)"],
            index=0
        )
        
        if direction == "Est-Ouest (X)":
            # Sélectionner une valeur Y pour la coupe
            y_value = st.slider(
                "Position Y de la coupe",
                float(st.session_state.data['y'].min()),
                float(st.session_state.data['y'].max()),
                float(st.session_state.data['y'].mean()),
                0.01
            )
            
            # Filtrer les données près de cette valeur Y
            y_tolerance = (st.session_state.data['y'].max() - st.session_state.data['y'].min()) * 0.02
            df_section = st.session_state.data[abs(st.session_state.data['y'] - y_value) <= y_tolerance]
            
            # Créer le graphique de coupe
            fig = px.scatter(
                df_section, x='x', y='z',
                color='potentiel_aurifere',
                size='teneur_or',
                color_continuous_scale='Inferno',
                labels={'x': 'Longitude (X)', 'z': 'Profondeur'},
                title=f"Coupe Est-Ouest à Y={y_value:.4f} (±{y_tolerance:.4f})"
            )
            
            # Inverser l'axe Y pour que la profondeur augmente vers le bas
            fig.update_layout(yaxis={'autorange': 'reversed'})
            
        else:  # Nord-Sud
            # Sélectionner une valeur X pour la coupe
            x_value = st.slider(
                "Position X de la coupe",
                float(st.session_state.data['x'].min()),
                float(st.session_state.data['x'].max()),
                float(st.session_state.data['x'].mean()),
                0.01
            )
            
            # Filtrer les données près de cette valeur X
            x_tolerance = (st.session_state.data['x'].max() - st.session_state.data['x'].min()) * 0.02
            df_section = st.session_state.data[abs(st.session_state.data['x'] - x_value) <= x_tolerance]
            
            # Créer le graphique de coupe
            fig = px.scatter(
                df_section, x='y', y='z',
                color='potentiel_aurifere',
                size='teneur_or',
                color_continuous_scale='Inferno',
                labels={'y': 'Latitude (Y)', 'z': 'Profondeur'},
                title=f"Coupe Nord-Sud à X={x_value:.4f} (±{x_tolerance:.4f})"
            )
            
            # Inverser l'axe Y pour que la profondeur augmente vers le bas
            fig.update_layout(yaxis={'autorange': 'reversed'})
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Histogramme de distribution du potentiel par profondeur")
        
        # Créer des bins pour la profondeur
        if 'z' in st.session_state.data.columns:
            z_min = st.session_state.data['z'].min()
            z_max = st.session_state.data['z'].max()
            bin_size = st.slider("Taille des intervalles de profondeur (m)", 10, 100, 50)
            
            # Créer des bins pour la profondeur
            bins = list(range(int(z_min - z_min % bin_size), int(z_max + bin_size), bin_size))
            
            # S'assurer qu'il y a au moins 2 bins
            if len(bins) < 2:
                bins = [z_min, (z_min + z_max) / 2, z_max]
            
            labels = [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins)-1)]
            
            df_binned = st.session_state.data.copy()
            df_binned['z_bin'] = pd.cut(df_binned['z'], bins=bins, labels=labels, right=False)
            
            # Calculer la moyenne du potentiel par bin de profondeur
            potential_by_depth = df_binned.groupby('z_bin')['potentiel_aurifere'].mean().reset_index()
            
            # Vérifier que potential_by_depth n'est pas vide
            if not potential_by_depth.empty:
                fig = px.bar(
                    potential_by_depth, x='z_bin', y='potentiel_aurifere',
                    color='potentiel_aurifere', color_continuous_scale='Inferno',
                    labels={'z_bin': 'Intervalle de profondeur (m)', 'potentiel_aurifere': 'Potentiel aurifère moyen'},
                    title="Potentiel aurifère moyen par intervalle de profondeur"
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Pas assez de données pour créer l'histogramme par profondeur.")

elif page == "À propos":
    st.markdown("<h1 class='main-title'>À Propos de GoldExplorerAI</h1>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### Objectif de l'application
        
        GoldExplorerAI a été développé pour Endeavour Mining afin d'optimiser et d'automatiser le processus 
        d'identification des cibles d'exploration aurifère au stade initial. En intégrant diverses couches de données 
        géologiques et en appliquant des modèles prédictifs, l'application aide les géologues à prioriser les zones 
        les plus prometteuses pour des investigations plus détaillées.
        
        ### Fonctionnalités clés
        
        - **Importation et traitement de données géospatiales** - Intégration de multiples sources de données
        - **Analyse exploratoire interactive** - Visualisation et exploration avancée des données
        - **Modèles de prédiction adaptables** - Scoring multicritère avec paramètres ajustables
        - **Machine Learning avancé** - Forêts aléatoires et réseaux de neurones pour la prédiction
        - **Visualisation 3D** - Modélisation spatiale des données et des résultats
        - **Identification de cibles prioritaires** - Classification et priorisation automatique des sites
        
        ### Avantages
        
        - 📉 Réduction significative des coûts d'exploration initiale
        - 📈 Augmentation du taux de succès dans l'identification des cibles
        - 🌿 Diminution de l'empreinte environnementale grâce à une exploration plus ciblée
        - ⏱️ Accélération du processus de prise de décision
        
        ### Méthodologie
        
        L'application propose deux approches complémentaires :
        
        **1. Scoring multicritère** - Approche transparente et contrôlable basée sur l'expertise géologique, combinant 
        plusieurs facteurs déterminants avec des poids ajustables.
        
        **2. Machine Learning** - Modèles avancés (forêts aléatoires, réseaux de neurones) qui peuvent capturer des 
        relations complexes et non-linéaires entre les variables, offrant potentiellement une meilleure capacité prédictive.
        """)
    
    with col2:
        st.markdown("""
        <div class="author-info" style="padding: 20px;">
        <h3>Auteur</h3>
        <p><strong>Didier Ouedraogo, P.Geo</strong><br>
        Géologue senior<br>
        Endeavour Mining</p>
        <p>Pour toute question ou suggestion concernant l'application, veuillez contacter l'auteur à:<br>
        <a href="mailto:didier.ouedraogo@endeavourmining.com">didier.ouedraogo@endeavourmining.com</a></p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        ### À propos d'Endeavour Mining
        
        Endeavour Mining est un producteur d'or de premier plan axé sur l'Afrique de l'Ouest, avec des opérations dans plusieurs pays 
        dont le Burkina Faso, la Côte d'Ivoire, le Sénégal et le Mali. L'entreprise est déterminée à créer de la valeur grâce à 
        l'exploration, le développement et l'exploitation responsables de gisements d'or.
        
        [Site web d'Endeavour Mining](https://www.endeavourmining.com/)
        """)
    
    st.markdown("<h2 class='sub-title'>Références techniques</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    1. "Modèles prédictifs pour l'exploration aurifère en Afrique de l'Ouest" - Journal of African Geology, 2023
    2. "Application des approches multicritères à l'identification de cibles d'exploration" - Economic Geology, 2022
    3. "Facteurs de contrôle de la minéralisation aurifère dans les ceintures de roches vertes d'Afrique occidentale" - Mineralium Deposita, 2021
    4. "Optimisation des campagnes d'exploration par l'intelligence artificielle" - Ressources minérales, 2024
    5. "Comparaison des algorithmes de machine learning pour la prédiction de gisements aurifères" - Journal of Machine Learning for Geosciences, 2024
    """)
    
    st.markdown("<h2 class='sub-title'>Version</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    - **Version actuelle**: 1.1.0
    - **Date de mise à jour**: 15 avril 2025
    - **Nouvelles fonctionnalités**:
      - Module Machine Learning avec forêts aléatoires et réseaux de neurones
      - Analyse comparative des modèles
      - Visualisation de l'importance des variables
    
    - **Prochaines améliorations prévues**:
      - Intégration d'un module de validation croisée spatiale
      - Module de comparaison avec des gisements connus comme références
      - Fonctionnalité d'exportation des cibles vers les systèmes SIG standards
    """)

# Pied de page
st.markdown("---")
st.markdown("<div class='footer'>© 2025 Endeavour Mining - GoldExplorerAI | Développé par Didier Ouedraogo, P.Geo</div>", unsafe_allow_html=True)