import pandas as pd
import multiprocessing
import sistema_recomendacion

def main():
    # cargar los datos
    ratings_df = pd.read_csv('ratings.csv')  # DataFrame de calificaciones de películas por usuario
    movies_df = pd.read_csv('movies.csv')  # DataFrame de información de películas
    # mostrar información de los DataFrames cargados
    print("ratings")
    ratings_df.head()
    ratings_df.info()

    print("movies")
    movies_df.head()
    movies_df.info()

    rec1 = sistema_recomendacion.user_based_recommendation(ratings_df, movies_df)
    rec2 = sistema_recomendacion.genre_based_recommendation(ratings_df, movies_df)
    table = sistema_recomendacion.compare_recommendations(rec1, rec2)
    print(table)



if __name__ == '__main__':
    main()
