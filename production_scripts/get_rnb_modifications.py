import requests
from datetime import datetime, timezone
from typing import Optional, Dict
import pandas as pd
from io import StringIO

GEO_API_URL = "https://geo.api.gouv.fr/departements"
RNB_DIFF_URL = "https://rnb-api.beta.gouv.fr/api/alpha/buildings/diff/"
MIN_ALLOWED_DATE = datetime(2024, 4, 1, tzinfo=timezone.utc)

"""
Récupère les modifications du Référentiel National des Bâtiments (RNB)
depuis une date donnée via l'API officielle.

Paramètres
----------
since : str
    Date et heure au format ISO 8601 (ex: "2024-04-02T00:00:00Z").
    Seules les dates postérieures au 1er avril 2024 sont acceptées.

insee_code : str, optionnel
    Code INSEE de la commune (5 caractères).
    Si fourni, seules les modifications concernant les bâtiments
    intersectant cette commune seront retournées.

timeout : int, optionnel
    Temps maximum d’attente de la requête HTTP en secondes (défaut: 30).

Retour
------
List[Dict]
    Liste des objets bâtiments modifiés depuis la date spécifiée.
    Chaque élément correspond à un bâtiment retourné par l’API.

Exceptions
----------
ValueError :
    - Si la date n'est pas valide
    - Si la date est antérieure au 1er avril 2024
    - Si le code INSEE n’est pas valide

requests.HTTPError :
    Si l’API retourne une erreur HTTP.

Exemple
-------
>>> mods = get_rnb_modifications(
...     since="2024-05-01T00:00:00Z",
...     insee_code="75056"
... )
>>> print(len(mods))
"""

def get_rnb_modifications(
    since: str,
    insee_code: Optional[str] = None,
    timeout: int = 30
) -> pd.DataFrame:

    # Validation de la date
    try:
        parsed_date = datetime.fromisoformat(
            since.replace("Z", "+00:00")
        )
    except ValueError:
        raise ValueError("Le paramètre 'since' doit être au format ISO 8601.")

    if parsed_date.tzinfo is None:
        raise ValueError("La date doit contenir un fuseau horaire (ex: 'Z').")

    if parsed_date < MIN_ALLOWED_DATE:
        raise ValueError(
            "La date doit être postérieure au 1er avril 2024."
        )

    # Validation code INSEE
    if insee_code is not None:
        if not (isinstance(insee_code, str) and len(insee_code) == 5 and insee_code.isdigit()):
            raise ValueError(
                "Le code INSEE doit être une chaîne de 5 caractères numériques."
            )

    # Construction des paramètres
    params = {"since": parsed_date.isoformat()}

    if insee_code:
        params["insee_code"] = insee_code

    # Appel API avec pagination
    next_url = RNB_DIFF_URL
    dfs = []  

    while next_url:
        response = requests.get(next_url, params=params if next_url == RNB_DIFF_URL else None, timeout=timeout)
        # exemple de requete : 
        # https://rnb-api.beta.gouv.fr/api/alpha/buildings/diff/?since=2026-02-04T13%3A28%3A38Z&insee_code=75056
        response.raise_for_status()
        dfs.append(decode_resp_csv(response.text))
        next_url = response.links.get("next", {}).get("url")
    
    df_final = pd.concat(dfs, ignore_index=True)    
    return df_final

def decode_resp_csv(csv_text: str):
    return pd.read_csv(StringIO(csv_text))

def get_departments_from_api():
    """
    Recupère la liste officielle des départements via l'API geo.api.gouv.fr.
    """
    response = request_geo_api(GEO_API_URL)
    return [dep["code"] for dep in response]

def get_communes_from_api(dep: str):
    """
    Recupère la liste officielle des communes (code_insee) par département via l'API geo.api.gouv.fr.
    """
    response = request_geo_api(f"{GEO_API_URL}/{dep}/communes")
    return [commune['code'] for commune in response]

def request_geo_api(url: str, timeout: int = 30) -> Dict:
    """
    Effectue une requête GET à l'API geo.api.gouv.fr et retourne la réponse JSON.
    """
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        print("HTTP error occurred")
        print("Status:", e.response.status_code)
        print("Message:", e.response.text)
    except requests.exceptions.RequestException as e:
        print("Request failed:", e)
        
def get_rnb_modifications_for_all_departments(since: str):
    """
    Récupère la liste des modifications pour tous les départements français.
    """
    departments = get_departments_from_api()
    print(f"Nombre de departements francais : {len(departments)}")
    results = []
    for dep in departments:
        print(f"Récupération des modifications pour le département {dep}...")
        df = get_rnb_modifications_by_departement(dep, since)
        print(f"Nombre de modifications pour le departement {dep} depuis {since} : {len(df)}")
        results.append([dep, df])
    return results

def get_rnb_modifications_by_departement(dep: str, since: str)-> pd.DataFrame:
    """
    Recupère la liste des modifications par departement
    """
    dfs = []
    communes = get_communes_from_api(dep)
    # pour chaque commune, recurpere les modifications
    for c_insee_code in communes:
        df = get_rnb_modifications(since=since, insee_code=c_insee_code)
        dfs.append(df)  
    df_final = pd.concat(dfs, ignore_index=True)  
    return df_final
    