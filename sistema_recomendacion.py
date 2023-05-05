import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances
from multiprocessing import Process

# primer método: recomendación basada en usuarios
def user_based_recommendation(ratings_df,movies_df):
    # crear una matriz de usuarios por películas
    ratings_matrix = ratings_df.pivot_table(index='userId', columns='movieId', values='rating')

    # calcular la similitud entre usuarios basándose en sus calificaciones
    user_similarities = ratings_matrix.corr(method='pearson', min_periods=5)
    """
    min_periods=5 significa que se requieren al menos 5 calificaciones no nulas de ambas usuarios
    para calcular su correlación. Esto se utiliza para garantizar que la correlación sea significativa
    y no se vea influenciada por un número pequeño de calificaciones. Si una de las dos usuarios tiene
    menos de 5 calificaciones, entonces la correlación con el usuario principal no se calcula y se establece en NaN.
    """

    # seleccionar un usuario principal y encontrar los usuarios más similares a él o ella
    main_user_id = 1  # id del usuario principal
    similar_users = user_similarities.loc[main_user_id].sort_values(ascending=False)[1:10]  # usuarios más similares (sin incluir al usuario principal)

    # recomendar películas que los usuarios similares hayan calificado con alta puntuación y que el usuario principal no haya visto aún
    movies_seen_by_main_user = ratings_df[ratings_df['userId'] == main_user_id]['movieId']  # películas vistas por el usuario principal
    recommended_movies = []
    for user_id, similarity_score in similar_users.items():
        movies_seen_by_similar_user = ratings_df[ratings_df['userId'] == user_id]['movieId']  # películas vistas por el usuario similar
        movies_to_recommend = list(set(movies_seen_by_similar_user) - set(movies_seen_by_main_user))  # películas no vistas por el usuario principal
        movies_df_filtered = movies_df[movies_df['movieId'].isin(movies_to_recommend)]  # DataFrame de películas recomendadas
        recommended_movies.extend(list(movies_df_filtered['title']))  # agregar los títulos de las películas recomendadas a la lista de recomendaciones

    # concatenar las películas recomendadas en una cadena de texto
    recommended_movies_str = '\n'.join([f'{title}' for title in recommended_movies[:10]])

    # retornar la cadena de texto con las películas recomendadas
    return recommended_movies_str


# segundo método: recomendación basada en géneros
def genre_based_recommendation(ratings_df,movies_df):
    # crear una matriz de usuarios por géneros
    genre_ratings_matrix = ratings_df.merge(movies_df, on='movieId')[['userId', 'title', 'genres', 'rating']]
    genre_ratings_matrix = genre_ratings_matrix.assign(genres=genre_ratings_matrix['genres'].str.split('|'))
    genre_ratings_matrix = genre_ratings_matrix.explode('genres')
    genre_ratings_matrix = genre_ratings_matrix.pivot_table(index='userId', columns='genres', values='rating', aggfunc='mean').fillna(0)

    # normalizar las calificaciones de los géneros
    genre_ratings_matrix_norm = genre_ratings_matrix.apply(lambda x: x - x.mean(), axis=1)

    # calcular la similitud entre usuarios basándose en los géneros
    user_similarities = pd.DataFrame(1 - pairwise_distances(genre_ratings_matrix_norm, metric='cosine'), index=genre_ratings_matrix.index, columns=genre_ratings_matrix.index)
    """
    Se calcula la similitud entre los usuarios basándose en los géneros de las películas que han calificado. 
    Para hacer esto, primero se normalizan las calificaciones de las películas por género en una matriz 
    y luego se calcula la distancia coseno entre cada par de usuarios en la matriz. 
    La función pairwise_distances de scikit-learn se utiliza para calcular la distancia coseno. 
    """
    # seleccionar un usuario principal y encontrar los usuarios más similares a él o ella
    main_user_id = 1  # id del usuario principal
    similar_users = user_similarities.loc[main_user_id].sort_values(ascending=False)[1:10]  # usuarios más similares (sin incluir al usuario principal)

    # recomendar películas que los usuarios similares hayan calificado con alta puntuación y que el usuario principal no haya visto aún
    movies_seen_by_main_user = ratings_df[ratings_df['userId'] == main_user_id]['movieId']  # películas vistas por el usuario principal
    recommended_movies = []
    for user_id, similarity_score in similar_users.items():
        # películas vistas por el usuario similar
        movies_seen_by_similar_user = ratings_df[ratings_df['userId'] == user_id]['movieId']  
        # películas no vistas por el usuario principal
        movies_to_recommend = list(set(movies_seen_by_similar_user) - set(movies_seen_by_main_user))  
        # DataFrame de películas recomendadas
        movies_df_filtered = movies_df[movies_df['movieId'].isin(movies_to_recommend)]  
        # agregar los títulos de las películas recomendadas a la lista de recomendaciones
        recommended_movies.extend(list(movies_df_filtered['title']))  

    # concatenar las películas recomendadas en una cadena de texto
    recommended_movies_str = '\n'.join([f'{title}' for title in recommended_movies[:10]])

    # retornar la cadena de texto con las películas recomendadas
    return recommended_movies_str


def compare_recommendations(rec1, rec2):
    # Separar las cadenas en listas de películas
    rec1_movies = rec1.strip().split('\n')
    rec2_movies = rec2.strip().split('\n')

    # Obtener la longitud máxima de ambas listas para evitar errores de indexación
    max_len = max(len(rec1_movies), len(rec2_movies))

    # Crear la tabla concatenando las cadenas con tabuladores
    table = []

    header = ['Valores obtenidos por el metodo pearson aplicado a usuarios','Valores obtenidos por la distancia de coseno aplicado a usuarios basandose en los generos de las peliculas']

    # crear una lista de tuplas para cada par de películas recomendadas
    data = [(rec1_movies[i], rec2_movies[i]) if i < min(len(rec1_movies), len(rec2_movies)) else ('', '') for i in range(max_len)]

    # construir la tabla
    table = [f"{header[0].ljust(65)}{header[1]}"]
    for rec1, rec2 in data:
        table.append(f"{rec1.ljust(65)}{rec2}")
        
    # Unir las filas de la tabla en una cadena
    return '\n'.join(table)

        
