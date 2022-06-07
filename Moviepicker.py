import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

def converti_stringa_in_lista(x):
    lista = x.split(",")
    return lista

def ordina_lista_e_converti_in_stringa(lista):
    lista.sort()
    y = ' '.join(lista)
    return y

class consiglia_film(object):
    def __init__(self,path,file_name):
        self.path = path
        self.file_name = file_name
        
    def read_user_file(self):
        df = pd.read_csv(self.path + self.file_name,
                         header = None,
                         names = ["NomeFilm"])
        df["NomeFilm"] = df["NomeFilm"].str.lower()
        self.FilmScelti = df["NomeFilm"].tolist()
        
    def import_and_filter_file_movie(self):
        
        Film = pd.read_csv(filepath_or_buffer = self.path + "film.tsv", 
                   sep = "\t",
                   header = 0, 
                   names = ["tconst","titleType","primaryTitle","originalTitle","isAdult","startYear","endYear","runtimeMinutes","genres"], 
                   usecols=["tconst","titleType","primaryTitle","originalTitle","isAdult","startYear","runtimeMinutes","genres"],
                   dtype = {"runtimeMinutes": "str"},  
                   na_values = "\\N"
                  )
        
        Film = Film[Film["titleType"] == "movie"] 
        Film['runtimeMinutes'] = Film['runtimeMinutes'].fillna(0).astype(np.int64)
        
        Film['runtimeMinutes'] = Film['runtimeMinutes'].astype(float)
        Film = Film[Film["runtimeMinutes"] > 30] 
        
        Film["originalTitle"] = Film["originalTitle"].str.lower()
        Film["primaryTitle"] = Film["primaryTitle"].str.lower()
        
        Ratings = pd.read_csv(filepath_or_buffer = self.path + "data.tsv", 
                   sep = "\t",
                   header = 0, 
                   names = ["tconst","averageRating","numVotes"], 
                   usecols=["tconst","averageRating","numVotes"],
                   na_values = "\\N"
                  ) 
      
        Film = pd.merge(Film, 
         Ratings, 
         how = 'inner',
         left_on = "tconst", 
         right_on = "tconst")
        
        Film = Film[Film["runtimeMinutes"]<300] 
        Film = Film[(Film["runtimeMinutes"]<150) | (Film["numVotes"]>30000) ] 
        Film = Film[ (Film["numVotes"] > 30000) ] 
        
        Film = Film.drop(["titleType"],axis=1)
          
        Film['Rank'] = Film.groupby(by = ["originalTitle"])["numVotes"].transform(lambda x: x.rank(method = 'first', ascending=False))
        Film=Film[Film["Rank"]==1]
        
        self.Film = Film.copy()
        
        
    def combine_user_and_movie_file(self):
        '''create PrimoTentativo'''
        
        Film = self.Film.copy()
        MyFilm = pd.DataFrame(columns = Film.columns)

        for film in self.FilmScelti:
            MyFilm = pd.concat([MyFilm, Film[ (Film["originalTitle"]== film)  | (Film["primaryTitle"]== film)]])
            
        x = Film['genres'].str.split(pat = ',', expand=True)
        generi = list(set(x[0]).union(set(x[1])).union(set(x[2])))   
        
        Film['genres'] = Film['genres'].apply(converti_stringa_in_lista)
        MyFilm['genres'] = MyFilm['genres'].apply(converti_stringa_in_lista)
        
        best = ""
        bestvalue = 0

        for genere in generi:
            x = 0 
            for i in range(len(MyFilm)):
                if genere in MyFilm["genres"].iloc[i]:
                    x = x+ 1
            if x>bestvalue:
                best = genere
                bestvalue = x

        genres = best
        if len(MyFilm)>1:
            
            min_durata = MyFilm["runtimeMinutes"].mean() - 1.5*MyFilm["runtimeMinutes"].std()

            max_durata = MyFilm["runtimeMinutes"].mean() + 1.5*MyFilm["runtimeMinutes"].std()

            min_numVotes = MyFilm["numVotes"].mean() - 1.5*MyFilm["numVotes"].std()

            max_numVotes = MyFilm["numVotes"].mean() + 1.5*MyFilm["numVotes"].std()

            min_startYear = MyFilm["startYear"].mean() - 1.5*MyFilm["startYear"].std()

            max_startYear = MyFilm["startYear"].mean() + 1.5*MyFilm["startYear"].std()
            
        else:
            min_durata = MyFilm["runtimeMinutes"].mean() - MyFilm["runtimeMinutes"].mean()/2

            max_durata = MyFilm["runtimeMinutes"].mean() + MyFilm["runtimeMinutes"].mean()/2

            min_numVotes = MyFilm["numVotes"].mean() - MyFilm["numVotes"].mean()/2

            max_numVotes = MyFilm["numVotes"].mean() + MyFilm["numVotes"].mean()/2

            min_startYear = MyFilm["startYear"].mean() - 10

            max_startYear = MyFilm["startYear"].mean() + 10
            
                   

        mask = Film.genres.apply(lambda x: genres in x)
        FilmFinale = Film[mask]
        
        Primotentativo = FilmFinale[ (FilmFinale["runtimeMinutes"] > min_durata) & (FilmFinale["runtimeMinutes"] < max_durata) \
      & (FilmFinale["numVotes"] > min_numVotes) & (FilmFinale["numVotes"] < max_numVotes) \
      & (FilmFinale["startYear"] > min_startYear) & (FilmFinale["startYear"] < max_startYear) ]
        
        Primotentativo = Primotentativo[~Primotentativo["originalTitle"].isin(self.FilmScelti)]
        
        self.Primotentativo = Primotentativo.copy()
        self.MyFilm = MyFilm.copy()
        self.min_durata =min_durata

        self.max_durata =max_durata

        self.min_numVotes = min_numVotes

        self.max_numVotes = max_numVotes

        self.min_startYear = min_startYear

        self.max_startYear = max_startYear
        
        self.mask = mask
        
        
    
    def write_output(self,my_list):
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename_output = 'film_consigliati_'+ timestamp +'.txt'
        if len(my_list)>0:
            with open(self.path + filename_output, 'w') as f:
                for item in my_list:
                    f.write("%s\n" % item)
        else:
            with open(self.path + filename_output, 'w') as f:
                    f.write("Qualcosa è andato storto, controllare il nome dei film inseriti")

    def pre_processing_for_machine_learning(self):
        Primotentativo = self.Primotentativo.copy()
        MyFilm = self.MyFilm.copy()
        
        Primotentativo['genres'] = Primotentativo['genres'].apply(ordina_lista_e_converti_in_stringa)
        MyFilm['genres'] = MyFilm['genres'].apply(ordina_lista_e_converti_in_stringa)
        Primotentativo["Target"] = 0
        MyFilm["Target"] = 1
        
        #Creo un dataframe concatenando i film scelti (a cui associa il valore 1 nella colonna target)
        #con la rosa di film consigliati (a cui associa il valore 0 nella colonna target)
        InputMachineLearning = pd.concat([Primotentativo, MyFilm],
                  ignore_index = True)
        
        #pre-processing
        InputMachineLearning = pd.get_dummies(InputMachineLearning, 
                                      columns=['genres'])
        InputMachineLearning = InputMachineLearning.sample(frac = 1,
                               random_state = 1)
        InputMachineLearning = InputMachineLearning.reset_index(drop = True)
        
        colonne = list(InputMachineLearning.columns)
        colonne.remove("tconst")
        colonne.remove("primaryTitle")
        colonne.remove("originalTitle")
        colonne.remove("Rank")
        colonne.remove("Target")
        
        #x è l'array delle colonna features
        #y è l'array della colonna Target
        #x1 è l'array delle colonna features per i soli film scelti dall'utente
        x = InputMachineLearning[colonne].values
        y = InputMachineLearning[['Target']].values
        x1 = InputMachineLearning[InputMachineLearning["Target"]==1][colonne].values
        
        
        Normalizzatore = StandardScaler()
        Normalizzatore.fit(x)
        x = Normalizzatore.transform(x)
        x1 = Normalizzatore.transform(x1)
        
        self.x = x
        self.y = y
        self.x1 = x1
        self.InputMachineLearning=InputMachineLearning
      
        
    def knn(self):
        MyFilm = self.MyFilm.copy()
        kn = KNeighborsClassifier()
        kn.fit(self.x,self.y.ravel())
        neighbors = kn.kneighbors( X = self.x1, n_neighbors=3, return_distance=True)
        self.neighbors = neighbors
        lista_indici_film_preferiti = []
        for i in range(len(self.x1)):
            lista_indici_film_preferiti.append(neighbors[1][i][1])
            lista_indici_film_preferiti.append(neighbors[1][i][2])
        lista_indici_film_preferiti = list(set(lista_indici_film_preferiti))
        
        OutputMachineLearning = self.InputMachineLearning.reset_index().copy()
        
        OutputMachineLearning = OutputMachineLearning[(OutputMachineLearning["Target"]==0) & (OutputMachineLearning["index"].isin(lista_indici_film_preferiti)) ]
         
        self.lista_indici_film_preferiti = lista_indici_film_preferiti
        self.OutputMachineLearning = OutputMachineLearning   
            
            
    def execute_program(self):
        self.read_user_file()
        print("file di input letto")
        self.import_and_filter_file_movie()
        print("lavorazione file sorgente")
        self.combine_user_and_movie_file()
        print("combinazione file eseguita")
        
        if len(self.Primotentativo) <= 2*len(self.FilmScelti):
            self.write_output(my_list = list(self.Primotentativo["primaryTitle"]))
            print("file di output pronto")
        else:
            self.pre_processing_for_machine_learning()
            print("pre processing effettuato")
            self.knn()
            print("machine learning effettuato")
            self.write_output(my_list = list(self.OutputMachineLearning["primaryTitle"]))
            print("file di output pronto")  
    