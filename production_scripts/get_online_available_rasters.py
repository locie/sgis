"""
Fetch and save on disk a list of available raster 7z files.
Unused, for information purpose only.
"""


from aiofiles import open as open_
from aiohttp import TCPConnector, ClientSession, ClientTimeout
from asyncio import ensure_future, gather, run, Semaphore
from datetime import datetime
from functools import partial
import os
import pandas as pd
from pathlib import Path
import re
from time import time
from numpy.random import randint

from time import perf_counter, sleep


# code pour regénérer la liste d'URLs de téléchargement
import urllib.request
from bs4 import BeautifulSoup

URL_source = r"https://geoservices.ign.fr/bdortho"
with urllib.request.urlopen(URL_source) as f:
    content = f.read().decode("utf8")

bs = BeautifulSoup(content, features="lxml")
a_tags = bs.find_all('a')
URLs = []
for idx, link in enumerate(a_tags):
    href = link.get('href')
    if (href is not None) and (".7z" in href):
        URLs.append(href+"\n")

with open(r"online_available_rasters.txt", "w") as f:
   f.writelines(URLs)



"""
667 fichiers de résolution 20cm, en couleurs "RVB", pour la dernière date disponible
==> environ 2.7 TO
faire un ordinateur de stockage dédié à BDORTHO?

Si connection à 300 KO/s --> 2500h = 104 jours
Parallélisé sur 20 threads: 5 jours
ordre de grandeur écriture disque: 10-100 MO/s --> max 30-300 threads
plafond apparent en version asynchrone: 40 MO/s

note: aucun fichier (couleur, res, dep, date, partie) n'existe sous plus d'une projection, donc la projection est ignorée
"""


# ressources
#   exemple/inspiré de: https://github.com/caa06d9c/Examples/blob/master/Python/basic/asyncio/download_file_iterate/run.py
#   https://docs.aiohttp.org/en/stable/http_request_lifecycle.html?highlight=response.text#why-is-aiohttp-client-api-that-way
#   https://pypi.org/project/aiofile/
#   téléchargement de gros fichiers: https://github.com/aio-libs/aiohttp/issues/2249

## todo
"""
    - permettre la décompression des fichiers (union puis décompression)
    commande Linux: 7z x ORTHOHR_1-0_RVB-0M20_JP2-E080_LAMB93_D073_2019-01-01.7z.001
    attention:
        - commande différente sous Windows
        - très demandeur de CPU [15-20% du CPU et plus d'une heure pour un département de 7 parties]
            --> concurrence en multiprocessing souhaitable, mais pas obligé
                --> exploiter déjà le multithreading bas niveau 7z
                ex: https://superuser.com/questions/433945/multithreaded-support-in-7za
        - peut être nécessaire de mettre toutes les parties dans le même dossier
        - peut-être nécessaire de concaténer les parties ensemble avant (avec `cat`)
    note:
        - l'archive rassemblée et décompressée prend apriori la même place que la somme de ses parties
"""




with open(r"online_available_rasters.txt", "r") as g:
    URLs = g.readlines()


# "-E080_(?P<Proj>\w+)_" \  # projection ignorée
pattern = ".*ORTHO.*" \
          "(?P<couleur>(RVB|IRC))" \
          "-0M(?P<Res>\d\d).*" \
          "D(?P<Dep>\d\d[\dAB])_" \
          "(?P<Date1>\d{4})-(?P<Date2>\d{2,4})-(?P<Date3>\d{2,4})" \
          "\.7z\.?0{0,2}(?P<Part>(?:[1-9]{0,3})?)"

data = []
for URL in URLs:
    URL_short = URL.split("/")[-1]
    match_ = re.match(pattern, URL_short)
    if match_ is not None:
        group_values = match_.groupdict()
        if len(group_values["Date2"])==len(group_values["Date3"])==4:
            group_values["Date1"] = group_values["Date3"] # la plus récente
        else:
            assert len(group_values["Date2"])==len(group_values["Date3"])==2
        group_values["URL"] = URL
        group_values.pop("Date2")
        group_values.pop("Date3")
        data.append(group_values)
    else:
        raise ValueError

assert len(data)==len(URLs)
df = pd.DataFrame.from_dict(data)

df["Res"] = df["Res"].astype(int)
df["Part"] = df["Part"].replace("", 1)          # si pas de découpage en parties, l'unique fichier est décrit comme étant la partie 1
df["Part"] = df["Part"].astype(int)
df = df.set_index(df.columns[:-1].to_list())
df = df.sort_index()
df = df.drop_duplicates() # some duplicated links in source BDD

class Session:
    df = df
    def __init__(self,
                 dossier,
                 retelecharger=False,
                 # decompresser=False,
                 nbr_simultanes=1000,
                 interval_progression=10
                 ):
        self.dossier = Path(dossier)
        self.retelecharger = retelecharger
        # self.decompresser = decompresser # fixme
        self.URLs_parties = pd.DataFrame()
        self.__progression = {}
        self.nbr_simultanes = nbr_simultanes
        self.interval_progression = interval_progression



    def ajouter_departement(self, dep, couleur="RVB", res=-1, date=-1, partie=None):
        """
        doc help: chaine de priorité dans la détermination de l'URL:
        `res` > `date` (i.e.: on fixe res avant de fixer la date, si les deux sont non précisés)
        date:
            - nombre négatif: -1 pour le plus récent, -2 pour le précédent, etc...
              pas d'erreur si "-n" avec "n" trop grand. Prend le plus ancien
            - date explicite en chaine de caractères
        res:
            - nombre négatif: -1 pour la plus fine résolution, -2 pour la deuxième plus fine, etc...
              pas d'erreur si "-n" avec "n" trop grand. Prend la moins bonne résolution
            - résolution explicite, en entier
        """
        if not couleur in ["RVB", "IRC"]:
            raise ValueError("La couleur `color` doit être 'RVB' (rouge, vert, bleu) ou 'IRC' (infra-rouge).")
        if not (isinstance(dep, str) and len(dep)==3):
            raise TypeError("Le département `dep` doit être une chaine de caractère, ex: 073 pour la Savoie.")
        if not (isinstance(res, int) and (res<=-1 or res>=10)):
            raise TypeError("La résolution `res` doit être relative (entier négatif) ou spécifiée (entier positif, en cm).")
        if partie is not None:
            if not isinstance(partie, int):
                raise TypeError("La partie d'archive `partie` doit être un entier.")
        if not (isinstance(date, str) or (isinstance(date, int) and date <= -1)):
            raise TypeError("La date `date` doit être relative (entier négatif) ou spécifiée (chaine de caractère, ex: '2018').")
        try:
            idx = pd.IndexSlice
            df_ = df.loc[idx[couleur, :, dep, :]]
        except:
            raise IndexError(f"couleur={couleur} et département={dep} inconnus.")

        res_possibles = df_.index.get_level_values("Res").unique()
        if res <= -1:
            res = max(res, - len(res_possibles))
            res = sorted(res_possibles)[::-1][res]
        try:
            df_ = df_.loc[idx[res, :, :]]
        except:
            raise IndexError("Résolution inconnue.\n"
                             f"Les champs de résolution disponibles pour couleur={couleur}, département={dep} sont:\n"
                             f"{list(res_possibles)}.")

        dates_possibles = df_.index.get_level_values("Date1").unique()
        if isinstance(date, int):
            date = max(date, - len(dates_possibles))
            date = sorted(dates_possibles)[date]
        try:
            df_ = df_.loc[idx[date, :]]
        except:
            raise IndexError("Date inconnue.\n"
                             f"Les champs de date disponibles pour couleur={couleur}, résolution={res}, département={dep} sont:\n"
                             f"{list(dates_possibles)}.")

        parties_possibles = df.loc[idx[couleur, res, dep, date]].index.unique()
        if partie is None:
            partie = parties_possibles
        try:
            nouv_URLs = df.loc[idx[couleur, res, dep, date, partie]]
            self.URLs_parties = pd.concat([self.URLs_parties, nouv_URLs])
        except:
            raise IndexError("Partie inconnue.\n"
                             f"Les champs de partie disponibles pour couleur={couleur}, résolution={res}, département={dep}, date={date} sont:\n"
                             f"{list(parties_possibles)}.")

    def telecharger(self):
        taches = self.__rassembler_taches()
        with open(str(self.dossier/"traites.csv"), "w") as f:
            f.write(f"Couleur, Resolution (cm), Departement, Date, Partie")
        run(taches)

    async def __rassembler_taches(self):
        # self.URLs_parties = self.URLs_parties.drop_duplicates() # fixme
        taches = list()
        # timeout: le temps nécessaire pour télécharger 4 GO à une vitesse de 10 KO/s
        connector = TCPConnector(limit=self.nbr_simultanes)
        sem = Semaphore(self.nbr_simultanes)
        async with ClientSession(connector=connector,
                                 timeout=ClientTimeout(total=4e5),
                                 # auto_decompress=False      # fixme
                                 ) as session:
                                                                                 # fixme: default is limited to 100 open connections. Redundant with `Semaphore`?
            for description, URL in self.URLs_parties.itertuples():
                # note: removed `ensure_future` call
                tache = self.__telecharger(
                                            description=description,
                                            URL=URL,
                                            session=session,
                                            sem=sem
                                            )
                taches.append(tache)

            return await gather(*taches)


    async def __telecharger(self, description, URL, session, sem):
        def montrer_progression(len_, cumul, dernier_t):
            """
            Montre une progression environ toutes les `self.interval_progression` secondes
            (hyp: écriture à l'échelle infra seconde)
            """
            cumul += len_
            t = round(time())
            if not (t % self.interval_progression):
                mod = t // self.interval_progression
                if mod != dernier_t:
                    maintenant = datetime.now().strftime("%d/%m %H:%M:%S")
                    print(f"[{maintenant}][{couleur}, {dep}, {res}cm, {date}, {partie}]: {cumul / taille_totale:.1%}")
                    dernier_t = mod
            return cumul, dernier_t

        couleur, res, dep, date, partie = description
        chemin = Path(self.dossier, couleur, f"{res}cm", dep, date)
        nom_fichier = f"{URL.split(r'/')[-1][:-1]}"
        # nom_fichier = f"B{randint(10000)}_{URL.split(r'/')[-1][:-1]}" # fixme remove randint and prefix
        existant = chemin.exists() and (nom_fichier in  os.listdir(str(chemin)))
        if existant and (self.retelecharger==False):
            print(f"Le fichier {description} existe.")
        else:
            chemin.mkdir(parents=True, exist_ok=True)
            cumul = 0
            dernier_t = -1
            if True: # fixme: contournement télchargement
                async with sem:
                    async with session.get(URL) as response:
                        print(URL)
                        taille_totale = int(response.headers['content-length'])
                        print(response.status)   # returns '403'
                        # if taille_totale <1e8:
                        #     if partie==1:
                        #         print("    ", end="")
                        #     print(description, taille_totale)
                        async with open_(str(chemin/nom_fichier), 'wb') as f:
                            # l'argument de iter_chunked est une borne supérieure (en octets) des données placées en RAM
                            # Si il y a interruption avant, le contenu est écrit (degré de confiance 2/3)
                            # en pratique: les morceaux écrits dépassent rarement 16 KO
                            # téléchargement à max 400 KO/s
                            async for chunk in response.content.iter_chunked(1024 * 1e4):  # 10 MO
                                await f.write(chunk)
                                cumul, dernier_t = montrer_progression(len(chunk), cumul, dernier_t)

            with open(str(self.dossier/"traites.csv"), "a") as f:
                f.write(f"\n{couleur}, {res}, {dep}, {date}, {partie}")
                
df.to_csv("online_available_rasters.csv")


"""
deb = perf_counter()
session = Session(r"./Essais", retelecharger=True) # fixme: change args
session.ajouter_departement(f"02A", "RVB", res=-1, date=-1, partie=1)
session.telecharger()
fin = perf_counter()
with open(r"./Essais/temps_exec", "w") as f:
    f.write(f"{fin-deb: .0f}")
"""
# téléchargement de presque tous les fichiers
"""
for color in ["RVB", "IRC"]:
    for res in [-3, -2, -1]:
        for date in [-3, -2, -1]:
            for dep in range(1, 1000):
                try:
                    session.ajouter_departement(f"{dep:0>3d}", "RVB", res=res, date=date, partie=None)
                except IndexError:
                    pass
"""



## le plus petit fichier: 40 MO
# session.ajouter_departement(f"089", "RVB", res=20, date="2020", partie=9)

## le plus petit fichier sans extension de partie
# session.ajouter_departement(f"975", "RVB", res=50, date="2017", partie=1)


## problème concernant le format incorrect du fichier téléchargé lorsqu'il y a une extension de partie
# idée: décompresser les parties ensembles
"""
Essai 1: écrire en mode "wb" au lieu de "ab"
    ==> ne change rien a priori
Essai 2: régler `auto_decompress` à False (ClientSession)
    ==> ne change rien a priori
Essai 3: essayer un fichier sans extension de partie 
    ==> fonctionnel
"""


## au sujet de l'erreur 403 ("urllib.error.HTTPError: HTTP Error 403: Forbidden"):
"""
diagnostic:
- présente également sur réseau non USMB, avec adresse IP différente
- erreur présente seulement depuis Python, téléchargement manuel Firefox OK 
- idem aiohttp, requests.get, wget.download

hypothèses concernant la source du problème:
- blocage de la part de l'IGN de mes demandes
questions: pourquoi est-ce que ça ne fonctionne pas non plus en changeant de réseau (et donc d'IP)?
           pourquoi est-ce que ça fonctionne depuis Firefox?
- problème d'ouverture de port, celui utilisé par Python
question: est-ce que le port est différent de celui utilisé par Firefox? 
- connection non https désormais interdite par IGN, et Python réalise des http (à vérifier)
"""
