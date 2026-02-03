#!/usr/bin/env python3

import requests
import json
LIMIT = 100
 

def get_dept_buildings_geojson(dept_code):
	#Probleme de cette approche la limite imposee sur le nombre de batiments 
	'''limit
	integer · min: 1 · max: 100
	Nombre maximum de bâtiments à retourner dans la page de résultats. Valeur par défaut : 20. Valeur maximale : 100.
	Default: 20'''
	# cf.Paramètres de requête décrite ici : https://rnb-fr.gitbook.io/documentation/api-et-outils/api-batiments/lister-des-batiments

	# liste des communes du département
	try:
		resp_communes = requests.get("https://geo.api.gouv.fr/departements/75/communes")
		communes = resp_communes.json()

		all_features = []
		# pour chaque commune, récupérer les bâtiments
		for commune in communes:
			insee = commune["code"]
			# par commune
			url = f"https://rnb-api.beta.gouv.fr/api/alpha/buildings/?insee_code={insee}&format=geojson&limit={LIMIT}"
			while url:
				r = requests.get(url)
				r.raise_for_status()
				data=r.json()
				all_features += data.get("features", [])
				# TODO pagination eventuelle 
				# url = data.get("links", {}).get("next")
				url = []

		# résultat final en GeoJSON
		result_geojson = {
			"type": "FeatureCollection",
			"features": all_features
		}
	except requests.exceptions.HTTPError as e:
		print("HTTP error occurred")
		print("Status:", e.response.status_code)
		print("Message:", e.response.text)

	except requests.exceptions.RequestException as e:
		print("Request failed:", e)

 
def get_bbox_buildings_geojson(bbox, limit):
	try:
		all_features = []
		url = f"https://rnb-api.beta.gouv.fr/api/alpha/buildings/?bbox={bbox}&format=geojson&limit={limit}"


		r = requests.get(url)
		r.raise_for_status()
		data=r.json()
		all_features += data.get("features", [])

		# resultat final en GeoJSON
		result_geojson = {
			"type": "FeatureCollection",
			"features": all_features}

	except requests.exceptions.HTTPError as e:
		print("HTTP error occurred")
		print("Status:", e.response.status_code)
		print("Message:", e.response.text)

	except requests.exceptions.RequestException as e:
		print("Request failed:", e)
