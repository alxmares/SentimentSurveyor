import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import concurrent.futures
import language_tool_python
from nltk.corpus import stopwords
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
from textblob import TextBlob
import scipy.stats as stats

class Analysis():
    tool = None
    stop_words = None
    Tvectorizer = None
    Cvectorizer = None
    
    def __init__(self,filepath):
        self.filepath = filepath
        self.columnas_corregidas = set()
        self.columnas_emotivas = set()
    
    def load_file(self):
        try:
            self.df = pd.read_csv(self.filepath)
            self.check_cols()
            return True
        except:
            return False
        
    def check_cols(self):
        self.tem_cols = []
        self.op_cols = [] # Opción múltiple
        self.ab_cols = [] # Opción abierta
        self.na_cols = [] # Sin respuestas
        
        temporal_posisble_names = ["Marca temporal"]
        
        for column in self.df.columns:
            
            # Temporal columns
            if column in temporal_posisble_names:
                self.tem_cols.append(column)
                continue
            
            count_series = self.df[column].value_counts()
            
            # Sin respuestas
            if len(count_series)==0:
                self.na_cols.append(column)
            # Respuestas abiertas
            elif len(count_series) < 6:
                self.op_cols.append(column)
            # Respuestas de opción múltiple
            else:
                self.ab_cols.append(column)
        #print(self.tem_cols)
        #print(self.op_cols)
        #print(self.ab_cols)
        
    def acortar_nombres(self, categoria, length=10):
        # Primero, verifica si 'categoria' es una instancia de una cadena
        if isinstance(categoria, str):
            if len(categoria) > length:  # Si es una cadena, verifica su longitud
                return categoria[:length] + "..."
            else:
                return categoria
        else:
            # Si 'categoria' no es una cadena, conviértelo a cadena o maneja el caso según necesites
            # Por ejemplo, puedes convertirlo a cadena o retornar una cadena vacía o alguna representación predeterminada
            return str(categoria)  # O manejar de otra manera
        
    # Graficar opción Múltiple
    def opmul_graph(self, column):
        count_series = self.df[column].value_counts()

        # Acortar los nombres de las etiquetas
        etiquetas_acortadas = [self.acortar_nombres(etiqueta, 45) for etiqueta in count_series.index]
        plt.figure(figsize=(6,6))
        plt.pie(count_series, labels=etiquetas_acortadas, autopct="%1.1f%%", startangle=140)
        plt.title(f"{column}")
        plt.title(f"{column}", fontweight='bold')  # Título en negritas
        plt.show()
    
    def get_correlation(self, col1, col2):
        # Crear mapeos temporales para acortar nombres sin modificar df
        mapeo_col1 = {cat: self.acortar_nombres(cat) for cat in self.df[col1].unique()}
        mapeo_col2 = {cat: self.acortar_nombres(cat) for cat in self.df[col2].unique()}

        # Aplicar el mapeo para visualización sin cambiar df
        df_temp = self.df.copy()
        df_temp[col1] = self.df[col1].map(mapeo_col1)
        df_temp[col2] = self.df[col2].map(mapeo_col2)

        # Paso 1: Crear una tabla de contingencia con el DataFrame temporal
        tabla_contingencia = pd.crosstab(df_temp[col1], df_temp[col2])

        # Paso 2: Realizar la prueba chi-cuadrado
        chi2, p, dof, expected = stats.chi2_contingency(tabla_contingencia)

        # Paso 3: Visualización - Tabla de contingencia como gráfico de barras apiladas
        tabla_contingencia.plot(kind='bar', stacked=True, figsize=(12, 6))
        plt.title('Distribución de respuestas')
        plt.xlabel(self.acortar_nombres(col1, 90))
        plt.ylabel('Frecuencia de ' + self.acortar_nombres(col2, 40))
        plt.legend(title=self.acortar_nombres(col2), bbox_to_anchor=(1.05, 1), loc='upper left')

        # Agregar los resultados de la prueba Chi-cuadrado directamente en el área del gráfico
        texto_resultados = f"Chi-cuadrado: {chi2:.2f}, p-valor: {p:.4f}"
        interpretacion = "Asociación significativa" if p < 0.05 else "Sin asociación significativa"
        plt.text(0.85, 1, texto_resultados + '\n' + interpretacion, transform=plt.gca().transAxes,
                 ha="center", va="center", fontsize=10, bbox=dict(boxstyle="round", facecolor='white', edgecolor='black'))

        plt.tight_layout()
        plt.show()
        
    # Corregir texto
    def corregir_texto(self, texto):
        if pd.isna(texto):
            return texto
        matches = self.tool.check(texto)
        texto_corregido = language_tool_python.utils.correct(texto, matches)
        return texto_corregido.lower()
    
    # Preparar texto
    def prepare_text(self, column):
        # Descargar archivos si es que aún no se ha hecho
        if Analysis.tool is None:
            print("Descargando archivos")
            Analysis.tool = language_tool_python.LanguageTool("es")
            print("Herramienta de corrección de texto lista")
        # Descargar la lista de palabras vacías
        if Analysis.stop_words is None:
            nltk.download('stopwords')
            Analysis.stop_words = set(stopwords.words('spanish'))
            print("StopWord descargadas")
        # Establecer el idioma de las palabras vacías
        if column not in self.columnas_corregidas:
            self.df[column] = self.df[column].apply(self.corregir_texto)
            self.columnas_corregidas.add(column)
            print("Texto corregido exitosamente")
    
    # Nube de palabras
    def wordcloud(self, column):
        self.prepare_text(column)
        
        # Crear vector si no se ha creado
        if Analysis.Tvectorizer is None:
            Analysis.Tvectorizer = TfidfVectorizer(stop_words=list(self.stop_words))
        if Analysis.Cvectorizer is None:
            Analysis.Cvectorizer = CountVectorizer(stop_words=list(self.stop_words))
        
        answers = self.df[column].dropna().tolist()
        tfidf_matrix = Analysis.Tvectorizer.fit_transform(answers)

        # Obtener los nombres de las características (términos) y los scores TF-IDF
        features = Analysis.Tvectorizer.get_feature_names_out()
        scores = tfidf_matrix.toarray()
        # Calcular el puntaje medio TF-IDF por término
        mean_scores = np.mean(scores, axis=0)
        tfidf_scores = dict(zip(features, mean_scores))

        # Ordenar los términos por su puntaje y obtener los top 10
        sorted_terms = sorted(tfidf_scores.items(), key=lambda item: item[1], reverse=True)[:10]
        top_words_text = "\n".join([f"{term}: {round(score, 2)}" for term, score in sorted_terms])

        # Generar la nube de palabras
        wordcloud = WordCloud(width=800, height=400, background_color='white')
        wordcloud.generate_from_frequencies(tfidf_scores)

        counts_matrix = Analysis.Cvectorizer.fit_transform(answers)
        # Sumar todas las palabras para obtener el conteo total por palabra
        sum_words = counts_matrix.sum(axis=0)
        words_freq = [(word, sum_words[0, idx]) for word, idx in Analysis.Cvectorizer.vocabulary_.items()]
        
        # Ordenar las palabras por frecuencia
        words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
        # Extraer las 10 palabras más frecuentes y sus conteos
        top_words_freq = words_freq[:10]
        # Corrección: Asegurarse de que las frecuencias son enteros
        top_words_freq = [(word, int(freq)) for word, freq in top_words_freq]
        # Preparar texto para visualización
        top_words_text = "\n".join([f"{word}: {freq}" for word, freq in top_words_freq])

        # Visualizar la nube de palabras
        plt.figure(figsize=(10, 8))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        # Agregar texto con las top 10 palabras y su conteo exacto en la parte inferior
        plt.figtext(0.5, 0.02, f"Top 10 palabras y frecuencias:\n{top_words_text}", ha="center", fontsize=10, wrap=True)
            
        plt.show()
         
    # Clasificar_sentimientos
    def clasificar_sentimiento(self, texto):
        if pd.isna(texto):
            return []
        blob = TextBlob(texto)
        if blob.sentiment.polarity > 0:
            return 'Positivo'
        elif blob.sentiment.polarity < 0:
            return 'Negativo'
        else:
            return 'Neutral'
        
    def get_emotions(self, column):
        col_name = f"Sent - {column}"
        conteo_sentimientos = pd.DataFrame()
        
        if col_name not in self.columnas_emotivas: 
            self.df[col_name] = self.df[column].apply(self.clasificar_sentimiento)
            self.columnas_emotivas.add(col_name)
            print("Emociones clasificadas")
            
        conteo = self.df[col_name].value_counts()
        conteo.index = conteo.index.map(str)
        # Ahora, eliminar específicamente el conteo para la lista vacía representada como una cadena '[]'
        conteo = conteo.drop('[]', errors='ignore')

        conteo_sentimientos = pd.concat([conteo_sentimientos, conteo], axis=1)
        
        conteo_sentimientos.plot(kind='bar', figsize=(10, 6))

        plt.title(f'Distribución de Sentimientos en {self.acortar_nombres(column, 20)}')
        plt.xlabel('Sentimiento')
        plt.ylabel('Frecuencia')
        plt.xticks(rotation=0)
        plt.legend(title=f'{self.acortar_nombres(column, 20)}')

        plt.tight_layout()
        plt.show()
    
    def get_hour(self):
        self.df[self.tem_cols[0]] = pd.to_datetime(self.df[self.tem_cols[0]], format='%d/%m/%Y %H:%M:%S', errors='coerce')
        # Extraer la hora
        self.df['Hour'] = self.df[self.tem_cols[0]].dt.hour
    
    def get_day(self):
        self.df[self.tem_cols[0]] = pd.to_datetime(self.df[self.tem_cols[0]], format='%d/%m/%Y %H:%M:%S', errors='coerce')
        # Extraer la hora
        self.df['Day'] = self.df[self.tem_cols[0]].dt.date
    
    def per_hour(self):
        if "Hour" not in self.df.columns:
            self.get_hour()
            print("Conversión de hora")
                  
        # Asume que conteo_por_hora ya ha sido calculado
        conteo_por_hora = self.df['Hour'].value_counts().sort_index()

        # Crear un gráfico de líneas
        conteo_por_hora.plot(kind='line', figsize=(12, 6), marker='o')

        plt.title('Número de Respuestas por Hora del Día')
        plt.xlabel('Hora del Día')
        plt.ylabel('Número de Respuestas')
        plt.xticks(range(0, 24), rotation=0)  # Asumiendo que las horas van de 0 a 23
        plt.grid(axis='y', linestyle='--')

        plt.show()
    
    def per_day(self):
        if "Day" not in self.df.columns:
            self.get_day()
            print("Conversión de hora")

        # Calcular el conteo de respuestas por día
        conteo_por_dia = self.df['Day'].value_counts().sort_index()

        # Crear un gráfico de líneas
        conteo_por_dia.plot(kind='line', figsize=(12, 6), marker='o', linestyle='-', color='b')

        plt.title('Número de Respuestas por Día')
        plt.xlabel('Día')
        plt.ylabel('Número de Respuestas')
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--')

        plt.tight_layout()  # Ajustar automáticamente los parámetros de la subtrama para que la subtrama encaje en el área de la figura.
        plt.show()
    
    def emotions_per_hour(self, column="¿Por qué?"):
        # Configurar los colores para cada sentimiento
        colores_sentimientos = {
            'Positivo': 'green',
            'Negativo': 'red',
            'Neutral': 'blue'
        }
        
        if "Hour" not in self.df.columns:
            self.get_hour()
            print("Conversión de hora")
            
        col_name = f"Sent - {column}"
        if col_name not in self.columnas_emotivas:
            self.df[col_name] = self.df[column].apply(self.clasificar_sentimiento)
            self.columnas_emotivas.add(col_name)
            print("Emociones clasificadas")
        
        # Filtrar df_filtrado para incluir solo filas con sentimientos reconocidos
        df_filtrado = self.df[self.df[col_name].isin(colores_sentimientos.keys())]

        # Procede con la agrupación y el conteo después de este filtrado adicional
        conteo_sentimientos = df_filtrado.groupby(['Hour', col_name]).size().unstack(fill_value=0)
        
        plt.figure(figsize=(12, 6))
        
        # Trazar cada sentimiento individualmente para poder personalizar el color
        for sentimiento in conteo_sentimientos.columns:
            plt.plot(conteo_sentimientos.index, conteo_sentimientos[sentimiento], 
                     marker='o', linestyle='-', color=colores_sentimientos.get(sentimiento, 'gray'), label=sentimiento)
        
        plt.title(f'Distribución de Sentimientos por Hora en {column[:30]}')
        plt.xlabel('Hora del Día')
        plt.ylabel('Frecuencia')
        plt.xticks(range(24), rotation=45)
        plt.legend(title='Sentimientos')
        plt.grid(axis='y', linestyle='--')
        
        plt.tight_layout()
        plt.show()
