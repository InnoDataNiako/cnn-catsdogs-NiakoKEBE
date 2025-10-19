import streamlit as st
import torch
from PIL import Image
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from streamlit_utils.model_loader import load_model
from streamlit_utils.predictor import predict_image
import json
import os
from streamlit_utils.gdrive_loader import get_models

# Configuration de la page
st.set_page_config(
    page_title="🐱🐶 CNN Cats vs Dogs",
    page_icon="favicon.png",  
    layout="wide",
    initial_sidebar_state="expanded"
)


# CSS personnalisé pour rendre l'app plus jolie
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem 0;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .prediction-box {
        background: #f0f2f6;
        padding: 2rem;
        border-radius: 15px;
        border-left: 5px solid #667eea;
        margin: 1rem 0;
    }
    
    .stButton>button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 25px;
        font-weight: bold;
        transition: transform 0.2s;
    }
    
    .stButton>button:hover {
        transform: scale(1.05);
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.image("https://img.icons8.com/fluency/96/000000/dog-footprint.png", width=80)
st.sidebar.title("🐾 Navigation")

page = st.sidebar.radio(
    "Choisissez une page",
    ["🏠 Accueil", "🔮 Prédiction", "📊 Comparaison Modèles", "📈 Visualisations", "ℹ️ À propos"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
### 📌 Informations

**Projet:** Deep Learning  
**Dataset:** Cats vs Dogs  
**Meilleur modèle:** Transfer Learning + Adam  
**Accuracy:** 97.64%  
""")

# ==================== PAGE ACCUEIL ====================
if page == "🏠 Accueil":
    st.markdown('<h1 class="main-header">🐱🐶 CNN Cats vs Dogs Classifier</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    ## Bienvenue dans le Classificateur de Chats et Chiens !
    
    Cette application utilise le **Deep Learning** pour distinguer les chats des chiens avec une précision de **97.64%** !
    
    ###  Fonctionnalités
    
    - **🔮 Prédiction** : Uploadez une image et obtenez une prédiction instantanée
    - **📊 Comparaison** : Testez les 4 modèles différents sur la même image
    - **📈 Visualisations** : Explorez les performances des modèles
    - **ℹ️ À propos** : Découvrez comment ça fonctionne
    """)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>Accuracy</h3>
            <h1>97.64</h1>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>F1-Score</h3>
            <h1>97.65</h1>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>Dataset</h3>
            <h1>25K</h1>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h3>Modèles</h3>
            <h1>4</h1>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Afficher les images du dataset
    st.subheader("📸 Exemples d'Images du Dataset")
    if os.path.exists("images/sample_images.png"):
        st.image("images/sample_images.png", use_container_width=True)
    
    st.markdown("---")
    
    st.info("👈 Utilisez le menu à gauche pour naviguer dans l'application")

# ==================== PAGE PRÉDICTION ====================
elif page == "🔮 Prédiction":
    st.markdown('<h1 class="main-header">🔮 Prédiction en Temps Réel</h1>', unsafe_allow_html=True)
    
    # Télécharger les modèles (une seule fois grâce au cache)
    with st.spinner('🔄 Initialisation des modèles...'):
        models_config = get_models()
    
    st.markdown("""
    Uploadez une image de chat ou de chien et laissez notre meilleur modèle faire sa magie ! ✨
    """)
    
    # Upload d'image
    uploaded_file = st.file_uploader(
        "📤 Choisissez une image...",
        type=["jpg", "jpeg", "png"],
        help="Formats acceptés: JPG, JPEG, PNG"
    )
    
    if uploaded_file is not None:
        # Afficher l'image
        image = Image.open(uploaded_file).convert('RGB')
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("🖼️ Image Uploadée")
            st.image(image, use_container_width=True)
        
        with col2:
            st.subheader("🎯 Résultat de la Prédiction")
            
            # Vérifier si le modèle existe
            model_path = "models/transfer_adam_best.pt"
            
            if not os.path.exists(model_path):
                st.error("""
                ⚠️ **Modèle non trouvé !**
                
                Pour utiliser la prédiction, vous devez :
                1. Télécharger les modèles depuis Colab
                2. Les placer dans le dossier `models/`
                3. Renommer le fichier en `transfer_adam_best.pt`
                
                **Note:** Les modèles sont trop lourds pour GitHub (~16 MB).
                """)
            else:
                with st.spinner('🔄 Analyse en cours...'):
                    try:
                        # Charger le modèle
                        model, device = load_model(model_path, model_type='transfer')
                        
                        # Prédiction
                        prediction, confidence, probs = predict_image(model, image, device)
                        
                        # Afficher le résultat
                        st.markdown(f"""
                        <div class="prediction-box">
                            <h2 style="margin:0;">Prédiction: {prediction}</h2>
                            <h3 style="color: #667eea;">Confiance: {confidence*100:.2f}%</h3>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Graphique des probabilités
                        fig = go.Figure(data=[
                            go.Bar(
                                x=list(probs.keys()),
                                y=list(probs.values()),
                                marker_color=['#FF6B6B', '#4ECDC4'],
                                text=[f'{v*100:.2f}%' for v in probs.values()],
                                textposition='auto',
                            )
                        ])
                        
                        fig.update_layout(
                            title="Probabilités par Classe",
                            yaxis_title="Probabilité",
                            yaxis=dict(range=[0, 1]),
                            height=400
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"Erreur lors de la prédiction: {str(e)}")

# ==================== PAGE COMPARAISON ====================
elif page == "📊 Comparaison Modèles":
    st.markdown('<h1 class="main-header">📊 Comparaison des 4 Modèles</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    Comparez les prédictions des 4 modèles différents sur la même image !
    """)
    
    uploaded_file = st.file_uploader(
        "📤 Choisissez une image...",
        type=["jpg", "jpeg", "png"],
        key="comparison_uploader"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        
        st.image(image, caption="Image à analyser", use_container_width=True)
        
        st.markdown("---")
        
        # Les 4 modèles
        models_config = [
            ("models/cnn_scratch_sgd_best.pt", "CNN From Scratch + SGD", "cnn"),
            ("models/cnn_scratch_adam_best.pt", "CNN From Scratch + Adam", "cnn"),
            ("models/transfer_sgd_best.pt", "Transfer Learning + SGD", "transfer"),
            ("models/transfer_adam_best.pt", "Transfer Learning + Adam", "transfer"),
        ]
        
        cols = st.columns(4)
        
        for idx, (model_path, model_name, model_type) in enumerate(models_config):
            with cols[idx]:
                st.subheader(model_name)
                
                if not os.path.exists(model_path):
                    st.warning("Modèle non disponible")
                else:
                    try:
                        model, device = load_model(model_path, model_type=model_type)
                        prediction, confidence, probs = predict_image(model, image, device)
                        
                        st.metric("Prédiction", prediction)
                        st.metric("Confiance", f"{confidence*100:.2f}%")
                        
                        # Mini graphique
                        fig = go.Figure(data=[
                            go.Bar(
                                x=list(probs.keys()),
                                y=list(probs.values()),
                                marker_color=['#FF6B6B', '#4ECDC4']
                            )
                        ])
                        fig.update_layout(height=250, showlegend=False, margin=dict(l=0, r=0, t=0, b=0))
                        st.plotly_chart(fig, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"Erreur: {str(e)}")

# ==================== PAGE VISUALISATIONS ====================
elif page == "📈 Visualisations":
    st.markdown('<h1 class="main-header">📈 Visualisations & Analyses</h1>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["📊 Métriques", "📈 Courbes", "🔍 Matrices"])
    
    with tab1:
        st.subheader("📊 Performances des Modèles (Test Set)")
        
        # Données des résultats
        results_data = {
            'Modèle': ['CNN + SGD', 'CNN + Adam', 'Transfer + SGD', 'Transfer + Adam'],
            'Accuracy': [0.5770, 0.7819, 0.9720, 0.9764],
            'Precision': [0.9364, 0.7988, 0.9675, 0.9715],
            'Recall': [0.1649, 0.7534, 0.9768, 0.9816],
            'F1-Score': [0.2805, 0.7754, 0.9721, 0.9765]
        }
        
        df = pd.DataFrame(results_data)
        
        # Afficher le tableau
        st.dataframe(df.style.highlight_max(axis=0, subset=['Accuracy', 'Precision', 'Recall', 'F1-Score']), use_container_width=True)
        
        # Graphique à barres interactif
        fig = go.Figure()
        
        for metric in ['Accuracy', 'Precision', 'Recall', 'F1-Score']:
            fig.add_trace(go.Bar(
                name=metric,
                x=df['Modèle'],
                y=df[metric],
                text=df[metric].apply(lambda x: f'{x:.2%}'),
                textposition='auto',
            ))
        
        fig.update_layout(
            title="Comparaison des Métriques",
            barmode='group',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("📈 Courbes d'Entraînement")
        
        # Afficher les courbes depuis les images
        if os.path.exists("images/acc_curves.png"):
            st.image("images/acc_curves.png", caption="Courbes d'Accuracy", use_container_width=True)
        
        if os.path.exists("images/loss_curves.png"):
            st.image("images/loss_curves.png", caption="Courbes de Loss", use_container_width=True)
    
    with tab3:
        st.subheader("🔍 Matrices de Confusion")
        
        if os.path.exists("images/confusion_matrices.png"):
           st.image("images/confusion_matrices.png", use_container_width=True)
        
        if os.path.exists("images/final_comparison.png"):
            st.image("images/final_comparison.png", caption="Comparaison Finale", use_container_width=True)

# ==================== PAGE À PROPOS ====================
elif page == "ℹ️ À propos":
    st.markdown('<h1 class="main-header">ℹ️ À propos du Projet</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    ##  Objectif
    
    Ce projet compare deux approches de classification d'images :
    1. **CNN From Scratch** : Architecture personnalisée entraînée de zéro
    2. **Transfer Learning** : MobileNetV2 pré-entraîné adapté
    
    ##  Dataset
    
    - **Source** : Microsoft Cats vs Dogs
    - **Total** : 25,000 images (12,500 chats + 12,500 chiens)
    - **Split** : 85% train, 15% validation
    - **Test Set** : 2,499 images
    
    ##  Architectures
    
    ### CNN From Scratch
    - 4 blocs convolutionnels
    - Batch Normalization + Dropout
    - Global Average Pooling
    - **Paramètres** : 782,370
    
    ### Transfer Learning (MobileNetV2)
    - Backbone pré-entraîné sur ImageNet (gelé)
    - Classifier personnalisé
    - **Paramètres entraînables** : 657,922 (22.8%)
    
    ##  Entraînement
    
    - **Optimiseurs** : SGD et Adam
    - **Schedulers** : StepLR et CosineAnnealingLR
    - **Régularisation** : Dropout (0.5) + Batch Normalization
    - **Data Augmentation** : Flip, Rotation, ColorJitter, Affine
    - **GPU** : Tesla T4 (Google Colab)
    
    ##  Résultats
    
    **Meilleur Modèle : Transfer Learning + Adam**
    - Test Accuracy : **97.64%**
    - Test Precision : **97.15%**
    - Test Recall : **98.16%**
    - Test F1-Score : **97.65%**
    
    ##  Conclusions
    
    1. **Transfer Learning >> CNN From Scratch** (+43% d'accuracy)
    2. **Adam > SGD** (convergence plus rapide et stable)
    3. La régularisation est essentielle pour éviter l'overfitting
    
    ##  Liens
    
    - [GitHub Repository](https://github.com/InnoDataNiako/cnn-catsdogs-NiakoKEBE)
    - [Google Colab Notebook](https://colab.research.google.com/drive/154owDh7IY7tMqy3rZD0-JzfGusc7D6Qa#scrollTo=H6yxmSe1XoUj)
    
    ##  Auteur
    
    - Prenom :  **Niako KEBE** 
    - Email: **drivenindata@gmail.com**
    ---
    
    *Fait avec passion et beaucoup de perseverance*
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🐾 CNN Cats vs Dogs Classifier | Deep Learning Project 2025</p>
    <p>Powered by PyTorch & Streamlit</p>
</div>
""", unsafe_allow_html=True)