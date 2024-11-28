import pandas as pd
import matplotlib as plt
import matplotlib.pyplot as plt
from streamlit import button
from streamlit_folium import st_folium
import streamlit as st
import plotly.graph_objects as go
import numpy as np
import mysql
import mysql.connector
import joblib
from joblib import load
import geopandas as gpd
import folium
import json
from PIL import Image
from datetime import datetime

def main():


    menu = ['Inicio', 'Valores climatológicos por comunidad', 'Comparador de valores climatológicos',
            'Mapa coroplético', 'Predicción meteorológica', 'Diagrama MYSQL: Base de datos',"Facebook Prophet", 'About']

    choice = st.sidebar.selectbox("Selecciona una opción", menu, key="menu_selectbox_unique")

    # Cambiar el color de fondo de la página y otros estilos
    st.markdown("""
        <style>
            /* Cambiar el color de fondo de toda la página */
            .reportview-container {
                background-color: #f0f8ff;  /* Color de fondo (puedes cambiarlo por cualquier color) */
            }

            /* Cambiar el color de los encabezados h1 */
            h1 {
                color: #33ffd1;  /* Color para los encabezados */
            }

            /* Cambiar el color de los subheaders h2 */
            h2 {
                color: #ff6347;  /* Color para los subheaders */
            }

            /* Cambiar el color de los párrafos */
            p {
                color: #ffce33;  /* Color para el texto en párrafos */
            }
        </style>
    """, unsafe_allow_html=True)

    @st.cache_data
    def consulta_sql(query):

            database = "AEMET"
            db = mysql.connector.connect(
                host="localhost",
                user="root",
                password= st.secrets["database_password"],
                database=database
            )

            cursor = db.cursor()
            cursor.execute(query)

            # Obtiene los datos
            data = cursor.fetchall()
            columns = [col[0] for col in cursor.description]  # Nombres de las columnas
            cursor.close()
            db.close()

            # Convierte los datos en un DataFrame de pandas
            return pd.DataFrame(data, columns=columns)

    if choice == "Inicio":
        st.markdown(
            """
            <div style="text-align: center;">
                <img src="https://facuso.es/wp-content/uploads/2023/09/6de3b76f2eeed4e2edfa5420ad9630bd.jpg" 
                     alt="Imagen oficial de la AEMET" 
                     width="250">
                <p>Imagen oficial de la AEMET</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Introducción
        st.markdown(
            "##### Bienvenido/a a la plataforma de exploración de datos de la AEMET (Agencia Estatal Meteorológica de España).")

        st.markdown(
            "##### España, con su diversidad climática y variada geografía, ofrece un rico panorama meteorológico en donde te encontrarás con temperaturas muy dispersas entre ellas en varios puntos geográfico. ")
        st.markdown("A tu izquierda encontrarás varias secciones, en donde cada apartado tendrá una breve introducción")
        st.markdown(
            "En este sitio, podrás explorar y comparar datos históricos desde 2014 para obtener una visión profunda del clima en nuestro país.")

        # Crear 3 columnas para las imágenes
        col1, col2, col3 = st.columns(3)

        # Añadir imágenes a cada columna con un tamaño mayor
        with col1:
            st.image(
                "https://i.pinimg.com/originals/73/93/14/739314e72faa8f68bc12a29dcf0ce07c.jpg",
                caption="Ordesa y Monte Perdido",
                width=225  # Ajusta el ancho según sea necesario
            )
            st.image(
                "https://fascinatingspain.com/wp-content/uploads/benasque_nieve.jpg",
                caption="Benasque",
                width=225  # Ajusta el ancho según sea necesario
            )

        with col2:
            st.image(
                "https://www.viajes.com/blog/wp-content/uploads/2021/09/sea-6580532_1920.jpg",
                caption="Galicia, tierra de Meigas",
                width=225  # Ajusta el ancho según sea necesario
            )
            st.image(
                "https://i.pinimg.com/originals/cd/14/c8/cd14c8b90c06f714899d0d17e7d7fcd4.jpg",
                caption="Mallorca, Cala Egos - Cala d'Or",
                width=225  # Ajusta el ancho según sea necesario
            )

        with col3:
            st.image(
                "https://palenciaturismo.es/system/files/Monta%C3%B1aPalentinaGaleria5.jpg",
                caption="Palencia",
                width=225  # Ajusta el ancho según sea necesario
            )
            st.image(
                "https://i.pinimg.com/originals/d8/3a/f2/d83af2c8d615f0a8393ef3eeb9325435.jpg",
                caption="Asturias",
                width=225  # Ajusta el ancho según sea necesario
            )
    if choice == "Valores climatológicos por comunidad":
        st.header("Valores Climatológicos por Comunidad")

        query_ciudades = f"""SELECT * FROM ciudades"""
        query_provincias = f"""SELECT * FROM provincias"""

        ciudades_df = consulta_sql(query_ciudades)
        provincias_df = consulta_sql(query_provincias)

        # Título de la aplicación
        st.title('Análisis Climatológico por Ciudad')

        # Selección de ciudad
        ciudad_seleccionada = st.selectbox("Selecciona una ciudad", ciudades_df['ciudad'].tolist())

        # Obtener el ID de la ciudad seleccionada
        ciudad_id = ciudades_df.loc[ciudades_df['ciudad'] == ciudad_seleccionada, 'ciudad_id'].values

        # Verificación y obtención de la provincia
        if ciudad_id.size > 0:
            ciudad_id = ciudad_id[0]
            provincia = provincias_df.loc[provincias_df['provincia_id'] == ciudad_id]

            if not provincia.empty:
                st.write("Provincia:", provincia['provincia'].values[0])

                # Mensaje introductorio sobre los datos
                st.write(
                    "Este análisis incluye datos climatológicos que reflejan las condiciones meteorológicas promedio y extremas en el tiempo."
                )
                st.write(
                    "Los datos presentados aquí pueden ayudar a entender las tendencias climáticas y las variaciones en los patrones meteorológicos a lo largo del tiempo."
                )
            else:
                st.write("No se encontró información de provincia para la ciudad seleccionada.")
        else:
            st.write("No se encontró la ciudad seleccionada.")

        queries = {
            "Promedio de Temperatura": "SELECT fecha, AVG(tmed) AS average_temperature FROM valores_climatologicos GROUP BY fecha ORDER BY fecha;",
            "Total de Precipitación": "SELECT fecha, SUM(prec) AS total_precipitation FROM valores_climatologicos GROUP BY fecha ORDER BY fecha;",
            "Temperaturas Máxima y Mínima": "SELECT fecha, MAX(tmax) AS max_temperature, MIN(tmin) AS min_temperature FROM valores_climatologicos GROUP BY fecha ORDER BY fecha;",
            "Humedad Promedio": "SELECT fecha, AVG(hrMedia) AS average_humidity FROM valores_climatologicos GROUP BY fecha ORDER BY fecha;",
            "Velocidad del Viento Promedio": "SELECT fecha, AVG(velemedia) AS average_wind_speed FROM valores_climatologicos GROUP BY fecha ORDER BY fecha;",
            "Registros de Temperatura": "SELECT tmed, tmax, tmin FROM valores_climatologicos ORDER BY fecha;",
            "Precipitación Total Mensual": "SELECT DATE_FORMAT(fecha, '%Y-%m') AS month, SUM(prec) AS total_precipitation_mes FROM valores_climatologicos GROUP BY month ORDER BY month;",
            "Rango de Temperatura": "SELECT fecha, (MAX(tmax) - MIN(tmin)) AS temperature_range FROM valores_climatologicos GROUP BY fecha ORDER BY fecha;",
            "Velocidad del Viento Promedio Mensual": "SELECT DATE_FORMAT(fecha, '%Y-%m') AS month, AVG(velemedia) AS average_wind_speed_mes FROM valores_climatologicos GROUP BY month ORDER BY month;",
            "Estadísticas de Humedad Mensual": "SELECT DATE_FORMAT(fecha, '%Y-%m') AS month, AVG(hrMedia) AS average_humidity_mes, MAX(hrMax) AS max_humidity, MIN(hrMin) AS min_humidity FROM valores_climatologicos GROUP BY month ORDER BY month;",
            }

        # Selección de consulta
        selected_query = st.selectbox("Selecciona una consulta para visualizar:", list(queries.keys()))
        data = consulta_sql(queries[selected_query])

        # Mostrar los datos
        st.subheader(selected_query)
        st.dataframe(data)

        # Gráficos de los datos
        if 'fecha' in data.columns:
            fig = go.Figure()

            # Plot average temperature
            if "average_temperature" in data.columns:
                fig.add_trace(go.Scatter(x=data['fecha'], y=data['average_temperature'], mode='lines',
                                         name='Temperatura Promedio', line=dict(color='blue')))

            # Plot total precipitation
            if "total_precipitation" in data.columns:
                fig.add_trace(go.Bar(x=data['fecha'], y=data['total_precipitation'], name='Precipitación Total',
                                     marker_color='orange', opacity=0.5))

            # Fill between max and min temperature
            if "max_temperature" in data.columns and "min_temperature" in data.columns:
                fig.add_trace(
                    go.Scatter(x=data['fecha'], y=data['max_temperature'], mode='lines', name='Temperatura Máxima',
                               line=dict(color='lightgray')))
                fig.add_trace(
                    go.Scatter(x=data['fecha'], y=data['min_temperature'], mode='lines', name='Temperatura Mínima',
                               fill='tonexty', fillcolor='rgba(211, 211, 211, 0.5)', line=dict(color='lightgray')))

            # Plot average humidity
            if "average_humidity" in data.columns:
                fig.add_trace(
                    go.Scatter(x=data['fecha'], y=data['average_humidity'], mode='lines', name='Humedad Promedio',
                               line=dict(color='green')))

            if "average_wind_speed" in data.columns:
                fig.add_trace(go.Scatter(x=data['fecha'], y=data['average_wind_speed'], mode='lines',
                                         name='Velocidad del Viento Promedio', line=dict(color='red')))

            # Additional queries
            if "average_precipitation" in data.columns:
                fig.add_trace(go.Scatter(x=data['fecha'], y=data['average_precipitation'], mode='lines',
                                         name='Precipitación Promedio', line=dict(color='purple')))

            if "tmed" in data.columns:
                fig.add_trace(go.Scatter(x=data['fecha'], y=data['tmed'], mode='lines', name='Temperatura Media (tmed)',
                                         line=dict(color='green')))

            if "tmax" in data.columns:
                fig.add_trace(
                    go.Scatter(x=data['fecha'], y=data['tmax'], mode='lines', name='Temperatura Máxima (tmax)',
                               line=dict(color='orange')))

            if "tmin" in data.columns:
                fig.add_trace(
                    go.Scatter(x=data['fecha'], y=data['tmin'], mode='lines', name='Temperatura Mínima (tmin)',
                               line=dict(color='purple')))

            if "total_precipitation_mes" in data.columns:
                fig.add_trace(
                    go.Bar(x=data['month'], y=data['total_precipitation_mes'], name='Precipitación Total Mensual',
                           marker_color='orange', opacity=0.3))

            if "average_wind_speed_mes" in data.columns:
                fig.add_trace(go.Scatter(x=data['month'], y=data['average_wind_speed_mes'], mode='lines',
                                         name='Velocidad del Viento Promedio Mensual',
                                         line=dict(color='red', dash='dash')))
            if "temperature_range" in data.columns:
                fig.add_trace(go.Scatter(x=data['fecha'], y=data['temperature_range'], mode='lines',
                                         name='Rango de Temperatura', line=dict(color='green')))

            if "month" in data.columns and "average_wind_speed_mes" in data.columns:
                fig.add_trace(go.Bar(x=data['month'], y=data['average_wind_speed_mes'],
                                     name='Velocidad del Viento Promedio Mensual',
                                     marker_color='purple', opacity=0.5))

            if "max_humidity" in data.columns and "min_humidity" in data.columns:
                fig.add_trace(go.Scatter(x=data['month'], y=data['max_humidity'], mode='lines', name='Humedad Máxima',
                                         line=dict(color='lightgreen')))
                fig.add_trace(go.Scatter(x=data['month'], y=data['min_humidity'], mode='lines', name='Humedad Mínima',
                                         fill='tonexty', fillcolor='rgba(144, 238, 144, 0.5)',
                                         line=dict(color='lightgreen')))

            # Update layout
            fig.update_layout(title='Análisis de Datos Meteorológicos',
                              xaxis_title='Fecha',
                              yaxis_title='Valores',
                              xaxis_tickangle=-45,
                              barmode='overlay')

            # Show the figure in Streamlit
            st.plotly_chart(fig)

        else:
            st.write("No se encontró la provincia para la ciudad seleccionada o la ciudad seleccionada")
            st.write("Selecciona una opción del menú para comenzar.")


    if choice == "Comparador de valores climatológicos":
        st.header("Comparativa de los valores climatologicos")
        st.write("Aqui podrás comparar una provincia y los datos de las fechas de los años:")

        def comparador(year1, year2, provincia_id):
            database = "AEMET"
            db = mysql.connector.connect(
                host="localhost",
                user="root",
                password= st.secrets["database_password"],
                database=database
            )

            query_comparador = f"""SELECT vc.Fecha, vc.Tmed, p.provincia
                                    FROM valores_climatologicos vc
                                    JOIN provincias p ON vc.Provincia_id = p.provincia_id
                                    WHERE vc.Provincia_id = '{provincia_id}' AND 
                                    (YEAR(vc.Fecha) = {year1} OR YEAR(vc.Fecha) = {year2})
                                """

            cursor = db.cursor()
            cursor.execute(query_comparador)

            data = cursor.fetchall()
            columns = [col[0] for col in cursor.description]  # Nombres de las columnas
            cursor.close()
            db.close()

            return pd.DataFrame(data, columns=columns)


        # Título de la aplicación
        st.subheader("Comparación de la temperatura por Provincia")

        # Selección de provincia y años

        query_provincias = f"""SELECT * FROM provincias"""
        provincias_df = consulta_sql(query_provincias)
        provincia = st.selectbox("Selecciona una provincia", provincias_df["provincia"].tolist())

        provincia_id = provincias_df.loc[provincias_df['provincia'] == provincia, 'provincia_id'].values[0]

        year1 = st.selectbox("Selecciona el primer año",
                             [2014, 2015, 2016, 2017,
                                 2018, 2019, 2020,
                             2021,2022, 2023, 2024])
        year2 = st.selectbox("Selecciona el segundo año",
                             [2014, 2015, 2016, 2017,
                                 2018, 2019, 2020,
                             2021,2022, 2023, 2024])

        # Cargar datos
        data = comparador(year1, year2, provincia_id)

        st.write(data)

        # Calcular estadísticas
        data['Year'] = pd.to_datetime(data['Fecha']).dt.year
        stats = data.groupby(['Year', 'Fecha'])['Tmed'].agg(['mean', 'median', 'min', 'max']).reset_index()

        st.write(stats)  # Esto te permitirá ver las estadísticas calculadas


        # Graficar
        fig, ax = plt.subplots(figsize=(16,8))
        for year in stats['Year'].unique():
            year_data = stats[stats['Year'] == year]
            ax.plot(year_data['Fecha'], year_data['mean'], label=f'Media {year}', marker='o')
            ax.plot(year_data['Fecha'], year_data['median'], label=f'Mediana {year}', marker='o')
            ax.plot(year_data['Fecha'], year_data['min'], label=f'Mínimo {year}', linestyle='--', marker='o')
            ax.plot(year_data['Fecha'], year_data['max'], label=f'Máximo {year}', linestyle='--', marker='o')

        # Personalizar gráfico
        ax.set_title(f'Comparación de Climatología en {provincia} entre {year1} y {year2}', fontsize=16)
        ax.set_xlabel('Fecha', fontsize=14)
        ax.set_ylabel('Temperatura Media (°C)', fontsize=14)
        ax.legend(fontsize=12)
        ax.grid(True)
        plt.xticks(rotation=45)
        st.pyplot(fig)

    if choice == "Mapa coroplético":

        st.title("Mapa coroplético:")
        st.subheader("Histórico de temperaturas medias en España.")
        st.info("1. Mapa filtrado por años, meses y provincias.")
        year = st.selectbox("Selecciona el año:",
                             [2014, 2015, 2016, 2017,
                                 2018, 2019, 2020,
                             2021,2022, 2023, 2024])

        month = st.selectbox("Selecciona el mes:",
                             [1,2,3,4,5,6,7,8,9,10,11,12])

        provincia_seek = st.selectbox(
            "Selecciona la provincia:",
            [
                'STA. CRUZ DE TENERIFE', 'BARCELONA', 'SEVILLA', 'CUENCA',
                'ZARAGOZA', 'ILLES BALEARS', 'VALENCIA', 'ZAMORA',
                'PALENCIA', 'CASTELLON', 'LAS PALMAS', 'MADRID',
                'CANTABRIA', 'GRANADA', 'TERUEL', 'BADAJOZ',
                'A CORUÑA', 'ASTURIAS', 'TARRAGONA', 'ALMERIA',
                'ALICANTE', 'CADIZ', 'TOLEDO', 'BURGOS',
                'GIRONA', 'MALAGA', 'JAEN', 'MURCIA',
                'LLEIDA', 'HUESCA', 'ALBACETE', 'NAVARRA',
                'CORDOBA', 'OURENSE', 'CIUDAD REAL', 'GIPUZKOA',
                'MELILLA', 'LEON', 'CACERES', 'SALAMANCA',
                'HUELVA', 'LA RIOJA', 'BIZKAIA', 'GUADALAJARA',
                'VALLADOLID', 'ARABA/ALAVA', 'PONTEVEDRA', 'SEGOVIA',
                'SORIA', 'AVILA', 'CEUTA', 'LUGO', 'BALEARES'
            ]
        )

        query = f"""
           SELECT 
        DATE_FORMAT(t1.fecha, '%Y-%m') AS mes, 
        ROUND(AVG(t1.tmed), 2) AS media_tmed_mensual,
        t1.provincia_id, 
        t2.provincia 
    FROM 
        valores_climatologicos t1 
    RIGHT JOIN 
        provincias t2 ON t1.provincia_id = t2.provincia_id
    WHERE
        YEAR(t1.fecha) = {year}
        AND MONTH(t1.fecha) = {month}
        AND t2.provincia = '{provincia_seek}'
    GROUP BY 
        mes,
        t1.provincia_id, 
        t2.provincia;"""
        # Ejecutar la consulta y cargar los datos en un DataFrame
        df = consulta_sql(query)
        st.markdown("###### Tabla de datos a usar:")

        # Carga el shapefile en un GeoDataFrame
        gdf = gpd.read_file("spain-provinces.geojson")

        map_provincia = {"STA. CRUZ DE TENERIFE": "Santa Cruz De Tenerife",
                         "BARCELONA": "Barcelona",
                         "SEVILLA": "Sevilla",
                         "CUENCA": "Cuenca",
                         "ZARAGOZA": "Zaragoza",
                         "ILLES BALEARS": "Illes Balears",
                         'VALENCIA': "València/Valencia",
                         'ZAMORA': "Zamora",
                         'PALENCIA': "Palencia",
                         'CASTELLON': "Castelló/Castellón",
                         'LAS PALMAS': "Las Palmas",
                         'MADRID': "Madrid",
                         'CANTABRIA': "Cantabria",
                         'GRANADA': "Granada",
                         'TERUEL': "Teruel",
                         'BADAJOZ': "Badajoz",
                         'A CORUÑA': "A Coruña",
                         'ASTURIAS': "Asturias",
                         'TARRAGONA': "Tarragona",
                         'ALMERIA': "Almería",
                         'ALICANTE': "Alacant/Alicante",
                         'CADIZ': "Cádiz",
                         'TOLEDO': "Toledo",
                         'BURGOS': "Burgos",
                         'GIRONA': "Girona",
                         'MALAGA': "Málaga",
                         'JAEN': "Jaén",
                         'MURCIA': "Murcia",
                         'LLEIDA': "Lleida",
                         'HUESCA': "Huesca",
                         'ALBACETE': "Albacete",
                         'NAVARRA': "Navarra",
                         'CORDOBA': "Córdoba",
                         'OURENSE': "Ourense",
                         'CIUDAD REAL': "Ciudad Real",
                         'GIPUZKOA': "Gipuzkoa/Guipúzcoa",
                         'MELILLA': "Melilla",
                         'LEON': "León",
                         'CACERES': "Cáceres",
                         'SALAMANCA': "Salamanca",
                         'HUELVA': "Huelva",
                         'LA RIOJA': "La Rioja",
                         'BIZKAIA': "Bizkaia/Vizcaya",
                         'GUADALAJARA': "Guadalajara",
                         'VALLADOLID': "Valladolid",
                         'ARABA/ALAVA': "Araba/Álava",
                         'PONTEVEDRA': "Pontevedra",
                         'SEGOVIA': "Segovia",
                         'SORIA': "Soria",
                         'AVILA': "Ávila",
                         'CEUTA': "Ceuta",
                         'LUGO': "Lugo",
                         'BALEARES': "Illes Balears"}

        df["provincia"] = df["provincia"].map(map_provincia)

        # Mostrar el DataFrame con st.dataframe
        st.dataframe(df)

        with open(file = "spain-provinces.geojson", mode = "r", encoding = "utf8") as file:
            geojson_spain = json.load(file)

        mapa_espana = folium.Map(location=[40.4168, -3.7038], zoom_start=6)
        folium.Choropleth(
            geo_data=geojson_spain,
            data=df,
            columns=["provincia", "media_tmed_mensual"],
            key_on="feature.properties.name"
        ).add_to(mapa_espana)

        # streamlit_folium
        st_folium(mapa_espana, width=725)

        st.info("2. Mapa de España filtrado por fecha.")

        # Selección de fecha
        min_date = pd.to_datetime("2014-01-01")
        max_date = pd.to_datetime("2024-09-30")

        date = st.date_input("Selecciona una fecha", value=pd.to_datetime(f"2023-01-01"), min_value=min_date,
                      max_value=max_date)

        dia = date.strftime('%Y-%m-%d')

        query1 = f"""
                SELECT 
            t1.fecha, 
            ROUND(AVG(t1.tmed), 2) AS media_tmed, -- Redondea a 2 decimales
            t1.provincia_id, 
            t2.provincia 
            FROM 
            valores_climatologicos t1 
            RIGHT JOIN 
            provincias t2 ON t1.provincia_id = t2.provincia_id
            WHERE
            t1.fecha = "{date}"
            GROUP BY 
            t1.fecha, 
            t1.provincia_id, 
            t2.provincia;
                """

        df = consulta_sql(query1)
        df = df[["fecha", "media_tmed", "provincia"]]
        df["provincia"] = df["provincia"].map(map_provincia)
        st.write(df)

        mapa_espana = folium.Map(location=[40.4168, -3.7038], zoom_start=6)
        folium.Choropleth(
            geo_data=geojson_spain,
            data=df,
            columns=["provincia", "media_tmed"],
            key_on="feature.properties.name"
        ).add_to(mapa_espana)

            # streamlit_folium
        st_folium(mapa_espana, width=725)

    # HASTA AQUÍ TODO FUNCIONA.
    if choice == "Facebook Prophet":


        # Configuración de la página
        st.title("Predicciones Climatológicas")

        # Títulos de la aplicación
        st.subheader("Modelos Predictivos de Facebook Prophet")
        st.write("Cargue modelos preentrenados para realizar predicciones sobre los datos reales.")


        # Función para cargar modelos .pkl
        def load_model(file_path):
            return load(file_path)

        # Carga de modelos
        models = {
            "Semestres": load_model("prophet_biannual.pkl"),
            "Trimestres": load_model("prophet_quarterly.pkl"),
            "Meses": load_model("prophet_monthly.pkl"),
            "Semanas": load_model("prophet_weekly.pkl"),
            "Días": load_model("prophet_daily.pkl")
        }

        # Input de datos reales (simulamos que provienen de una query)
        query1 = """
            SELECT fecha, tmed
            FROM valores_climatologicos
             """

        # Simulación de datos obtenidos (reemplaza esto por tu extracción real de datos)
        data_real = consulta_sql(query1)
        st.image("https://estaticos-cdn.prensaiberica.es/clip/c086f7c2-e053-4e0a-889e-8bbb4f55197f_16-9-discover-aspect-ratio_default_0.webp",
                caption="Temperaturas España",
                width=600  # Ajusta el ancho según sea necesario
        )

        # Conversión al formato requerido por Prophet
        data_real.rename(columns={"fecha": "ds", "tmed": "y"}, inplace=True)

        # Selección del modelo
        model_choice = st.selectbox(
            "Seleccione el modelo que desee utilizar:",
            list(models.keys())
        )

        times = {"Mañana" : 1,
                 "Semana" : 7,
                 "Quincenal" : 14,
                 "Mensual" : 30}

        times_choice = st.selectbox("Seleccione el rango de tiempo que desee predecir:",
                                    list(times.keys()))

        st.write("**Predicción de la temperatura media de mañana según los distintos modelos.**",
                 )

        # Predicción con el modelo seleccionado
        if st.button("Predecir"):

            model = models[model_choice]  # Asegúrate de que este es un modelo Prophet
            future = model.make_future_dataframe(periods=times[times_choice], freq='D')
            forecast = model.predict(future)

            st.write(f"**Temperatura para {times_choice} mediante el modelo  {model_choice}:**")
            forecast_reset = forecast[['ds', 'yhat']].tail(times[times_choice]).reset_index(drop=True)
            forecast_reset.index = range(1, len(forecast_reset) + 1)
            forecast_reset['yhat'] = forecast_reset['yhat'].round(2).astype(str) + " ºC"
            forecast_reset['ds'] = forecast_reset['ds'].dt.date
            forecast_reset = forecast_reset.rename(columns={"ds": "Fecha", "yhat": "Temperatura media"})
            st.dataframe(forecast_reset[['Fecha', 'Temperatura media']].tail(times[times_choice]))

            st.write("**Gráfico de predicciones:**")

            # Crear el gráfico
            fig = model.plot(forecast)

            # Resaltar la zona de predicción con color
            plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='gray', alpha=0.2)

            # Agregar título y etiquetas
            plt.title(f"Predicción de temperatura para {times_choice}")
            plt.xlabel("Fecha")
            plt.ylabel("Temperatura (ºC)")

            # Mostrar el gráfico en Streamlit
            st.pyplot(fig)



    if choice == "Predicción meteorológica":
        def predict_temperature(model_path, scaler_path, input_data):
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)  # Cargar el scaler entrenado
            input_data_scaled = scaler.transform(input_data)  # Usar transform en lugar de fit_transform

            # Verificar las dimensiones de input_data_scaled
            if input_data_scaled.ndim == 1:
                # Si es un array unidimensional, convertirlo a 2D
                input_data_scaled = input_data_scaled.reshape(1, -1)  # 1 muestra, n características
            elif input_data_scaled.ndim == 2:
                # Si ya es 2D, asegurarse de que tenga la forma correcta
                input_data_scaled = input_data_scaled.reshape(1, input_data_scaled.shape[0], input_data_scaled.shape[1])

            predicted_temperature = model.predict(input_data_scaled)
            return scaler.inverse_transform(predicted_temperature)

        # Aquí puedes reemplazar este valor con una estimación real o lógica para obtenerlo
        estimated_temperature_tomorrow = 27

        if st.button('Predecir datos con RNN'):
            st.title('Predicción de la temperatura de mañana')
            input_data = np.array([[estimated_temperature_tomorrow]])
            predicted_temperature = predict_temperature('modelo_RNN.pkl', 'scaler.pkl', input_data)
            st.write(f'Predicción de temperatura para el día siguiente: {predicted_temperature[0][0]} °C')

        if st.button('Predecir datos con LSTM'):
            st.title('Predicción de la temperatura de mañana')
            input_data = np.array([[estimated_temperature_tomorrow]])
            predicted_temperature = predict_temperature('modelo_LSTM.pkl', 'scaler.pkl', input_data)
            st.write(f'Predicción de temperatura para el día siguiente: {predicted_temperature[0][0]} °C')


    if choice == "Diagrama MYSQL: Base de datos":

        st.image(image            = "Esquema_AEMET.png",
                 caption          = "Esquema de la base de datos AEMET",
                 use_column_width = True)

        st.subheader("Esquema base de datos AEMET:")
        st.write("""El esquema de esta base de datos consta de 4 tablas de datos en la que la principal sería la tabla llamada valores climatológicos y de la que surgen otras tres tablas llamadas indicativo, ciudades y provincias."
                 "En la tabla principal podemos encontrar los siguientes datos:
                 
    Fecha: recoge la fecha de medición de los valores climatológicos.
                 
    Altitud: altitud de medición de estos valores.
    
    Tmed: temperatura media recogida durante el día en grados centígrados.
    
    Prec: precipitaciones acumuladas en milímetros, que equivale a un 1 litro de agua por metro cuadrado."
    
    Tmin: temperatura mínima registrada en el día.
    
    HoraTmin: registro de hora de temperatura mínima.
    
    Tmax: temperatura máxima registrada en el día.
    
    HoraTmax: registro de hora de temperatura máxima.
    
    Dir: direccional predominante del viento, expresada en grados (0°-360°) o en puntos cardinales (N, NE, E, etc.). Esto señala de dónde viene el viento, no hacia dónde va.
    
    Velemedia: se refiere a la velocidad media del viento, expresada generalmente en kilómetros por hora (km/h) o metros por segundo (m/s). Este valor representa la velocidad promedio del viento registrada en el día.
    
    Racha: se refiere a la racha máxima de viento, que es la mayor velocidad instantánea del viento registrada en un periodo determinado.
    
    Horaracha: registro de hora de Racha.
    
    HrMedia: Humedad relativa media del día.
    
    HrMax: Humedad máxima registrada en el día.
    
    HoraHrMax: Hora de registro de la humedad máxima.
    
    HrMin: Humedad mínima registrada en el día.
    
    HoraHrMin: Hora de registro de la humedad mínima.
    
    Indicativo_id: índice asignado al valor indicativo de estación meteorológica.
    
    Ciudad_id: índice asignado al valor ciudad.
    
    Provincia_id: índice asignado al valor provincia.""")

        st.subheader("""Además de esta tabla surgen existen otras tablas relacionadas con esta llamadas:
        
        indicativos: con una columna indicativo_id para identificar el número de celda y otra con el 
        indicativo de estación climatológica. 
                     
        ciudades:con una columna ciudad_id para identificar el número de celda y otra con la ciudad 
        de la medición. 
                     
        provincias: con una columna provincia_id para identificar el número de celda y otra con la 
        provincia de la medición.""")


    if choice == "About":
        st.title("Contacto: ✉️📬")
        st.subheader("Este proyecto ha sido desarrollado por los alumnos del curso de Data Science & IA:")

        # Redimensionar las imágenes a un tamaño cuadrado
        size = (300, 300)  # Define el tamaño deseado (ancho, alto)

        estela_img = Image.open("Estela.jpeg").resize(size)
        pablo_img = Image.open("Pablo Petidier.jpeg").resize(size)

        # Crear dos columnas
        col1, col2 = st.columns(2)

        # Primera columna
        with col1:
            st.image(estela_img, caption="Estela Mojena Ávila", use_column_width=False)
            st.write("📧 **Correo:** estelamojenaavila@gmail.com")
            st.write("📞 **Teléfono:** +34 622 68 33 95")
            st.markdown("💼 **LinkedIn:** [Estela Mojena Ávila](https://www.linkedin.com/in/estela-mojena-avila/)")
            st.write("💻 **GitHub:** https://github.com/Estela8")

        # Segunda columna
        with col2:
            st.image(pablo_img, caption="Pablo Petidier Smit", use_column_width=False)
            st.write("📧 **Correo:** petidiersmit@gmail.com")
            st.write("📞 **Teléfono:** +34 624 10 85 03")
            st.markdown("💼 **LinkedIn:** [Pablo Petidier Smit](https://www.linkedin.com/in/pablopetidier/)")
            st.write("💻 **GitHub:** https://github.com/ppswns1988")

if __name__ == "__main__":
    main()
