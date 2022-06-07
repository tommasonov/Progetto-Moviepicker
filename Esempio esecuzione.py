import Moviepicker as mp

#path contiene il percorso del file system in cui sono presenti i file di input data.tsv e Film.tsv

#filename contiene il nome del file all'interno del path con la lista dei film piaciuti; il file deve contenere un titolo di film per riga

esegui = mp.consiglia_film(path="C://Users//Tommaso//Desktop//movie picker//", file_name = "elenco_film.txt")

#il programma genera in output nella cattrtella indicata nel percorso path un file di output con il timestamp d'esecuzione
esegui.execute_program()