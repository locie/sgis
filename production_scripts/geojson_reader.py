# Documentation sur structure csv : https://rnb-fr.gitbook.io/documentation/api-et-outils/export-national-et-departemental
'''
Clé | Type |Commentaire
rnb_id | string | L'identifiant RNB (ID-RNB) du bâtiment
point | Point au format EWKT | Point situant le bâtiment en WGS84
shape | Géométrie au format EWKT | Géométrie représentant l'enveloppe du bâtiment en WGS84 
status | string | Statut physique du bâtiment
ext_ids | JSON | Clé(s) de correspondance au sein de la BD Topo et de la BDNB
addresses | JSON | Liste des adresses connues de ce bâtiment. Chaque adresse contient à minima la clé d'interopérabilité BAN
plots | JSON | Lien géométrique entre le polygone du bâtiment et les parcelles cadastrales. Cette donnée n'est absolument pas un lien "administratif". Donne la liste des parcelles que le bâtiment intersecte, avec le ratio de recouvrement. Un ratio de 50% signifie que 50% de la surface du bâtiment est située sur la parcelle en question.
'''
