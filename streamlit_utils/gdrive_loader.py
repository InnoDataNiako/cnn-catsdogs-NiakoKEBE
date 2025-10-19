import os
import gdown
import streamlit as st

def download_model_from_gdrive(gdrive_id, output_path, model_name):
    """
    Télécharge un modèle depuis Google Drive
    
    Args:
        gdrive_id: ID du fichier Google Drive
        output_path: Chemin de sauvegarde local
        model_name: Nom du modèle pour affichage
    """
    if os.path.exists(output_path):
        return True
    
    try:
        # URL de téléchargement Google Drive
        url = f'https://drive.google.com/uc?id={gdrive_id}'
        
        # Créer le dossier si nécessaire
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Télécharger
        with st.spinner(f'⬇️ Téléchargement du modèle {model_name}... (première utilisation uniquement)'):
            gdown.download(url, output_path, quiet=False)
        
        return True
    
    except Exception as e:
        st.error(f" Erreur lors du téléchargement de {model_name}: {str(e)}")
        return False


@st.cache_resource
def get_models():
    """
    Télécharge tous les modèles (mise en cache)
    Cette fonction ne s'exécute qu'UNE SEULE FOIS
    """
    models_config = {
        'transfer_adam': {
            'gdrive_id': '1zWyBeNo4tLIzhWofpUJrzA9fXmz2t5Bo',  # À remplacer
            'path': 'models/transfer_adam_best.pt',
            'name': 'Transfer Learning + Adam'
        },
        'transfer_sgd': {
            'gdrive_id': '19bkLrmTLgJ4GxkY_TN9z7YaEEWzjFm8O',  # À remplacer
            'path': 'models/transfer_sgd_best.pt',
            'name': 'Transfer Learning + SGD'
        },
        'cnn_adam': {
            'gdrive_id': '1dH_uuixjEkIFixSRVZ_amzZThEkrIP1V',  # À remplacer
            'path': 'models/cnn_scratch_adam_best.pt',
            'name': 'CNN + Adam'
        },
        'cnn_sgd': {
            'gdrive_id': '1Hy_6fBEWYMk3-gGNo1laf0tmjmri4kzm',  # À remplacer
            'path': 'models/cnn_scratch_sgd_best.pt',
            'name': 'CNN + SGD'
        }
    }
    
    # Télécharger tous les modèles
    for key, config in models_config.items():
        download_model_from_gdrive(
            config['gdrive_id'],
            config['path'],
            config['name']
        )
    
    return models_config