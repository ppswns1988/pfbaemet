import pandas as pd
import matplotlib as plt
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.offline as pyo
from streamlit_folium import st_folium
import streamlit as st
from sqlalchemy import create_engine
import plotly.graph_objects as go
import numpy as np

import joblib
import geopandas as gpd
import folium
import json

def main():

    engine = create_engine('mysql+pymysql://root:Estela080702.@localhost:3306/AEMET?charset=utf8')

    menu = ['Inicio','Predicción del tiempo', 'Valores climatológicos por comunidad', 'Comparador de valores climatológicos',
            'Mapa coroplético']

    choice = st.sidebar.selectbox("Selecciona una opción", menu, key="menu_selectbox_unique")

    if choice == "Inicio":
        st.image(
            image="https://facuso.es/wp-content/uploads/2023/09/6de3b76f2eeed4e2edfa5420ad9630bd.jpg",
            caption="Imagen oficial de la AEMET",
            width=350,
            use_column_width=True
        )

        # Introducción
        st.markdown(
            "### Bienvenido a la web explorativa basada en datos de la AEMET(Agencia Estatal Meteorológica), donde podrás explorar y comparar datos históricos de España desde 2014.")
        st.markdown("#### A tu izquierda encontrarás varias secciones, en donde cada apartado tendrá una breve explicación")


    if choice == "Valores climatológicos por comunidad":
        st.header("Valores Climatológicos por Comunidad")

        ciudades_df = pd.read_sql("SELECT * FROM ciudades", engine)
        provincias_df = pd.read_sql("SELECT * FROM provincias", engine)

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


        def run_query(query):
            with engine.connect() as connection:
                return pd.read_sql(query, connection)


                # Define consultas SQL predefinidas
        queries = {"Promedio de Temperatura": "SELECT fecha, AVG(tmed) AS average_temperature FROM valores_climatologicos GROUP BY fecha ORDER BY fecha;",
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
        data = run_query(queries[selected_query])

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

        def load_data(provincia_id, year1, year2):
            query = f"""
                    SELECT vc.Fecha, vc.Tmed, p.provincia
                    FROM valores_climatologicos vc
                    JOIN provincias p ON vc.Provincia_id = p.provincia_id
                    WHERE vc.Provincia_id = '{provincia_id}' AND (YEAR(vc.Fecha) = {year1} OR YEAR(vc.Fecha) = {year2})
                """
            return pd.read_sql(query, engine)

        # Título de la aplicación
        st.subheader("Comparación de la temperatura por Provincia")

        # Selección de provincia y años
        provincias_df = pd.read_sql("SELECT * FROM provincias", engine)
        provincia = st.selectbox("Selecciona una provincia", provincias_df["provincia"].tolist())

        provincia_id = provincias_df.loc[provincias_df['provincia'] == provincia, 'provincia_id'].values[0]

        year1 = st.selectbox("Selecciona el primer año", [2022, 2023, 2024])
        year2 = st.selectbox("Selecciona el segundo año", [2022, 2023, 2024])

        # Cargar datos
        data = load_data(provincia_id, year1, year2)
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


    elif choice=="Mapa coroplético":

        st.title("Mapa coroplético sobre temperaturas medias por provincia en España")

        query = f"""SELECT 
            t1.fecha, 
            AVG(t1.tmed) AS media_tmed, 
            t1.provincia_id, 
            t2.provincia 
        FROM 
            valores_climatologicos t1 
        RIGHT JOIN 
            provincias t2 ON t1.provincia_id = t2.provincia_id 
        GROUP BY 
            t1.fecha, 
            t1.provincia_id, 
            t2.provincia;"""

        # Ejecutar la consulta y cargar los datos en un DataFrame
        df = pd.read_sql(query, engine)


        # Renombrar columnas
        df.rename(columns={0: "fecha",
                           1: "temperatura",
                           2: "provincia_id",
                           3: "provincia"}, inplace=True)

        st.markdown("###### Tabla de datos a usar:")
        print(df)

        df['fecha'] = pd.to_datetime(df['fecha'])
        df['año'] = df['fecha'].dt.year
        df['mes'] = df['fecha'].dt.month

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

        df = df[["fecha", "mes", "año", "media_tmed", "provincia"]]
        df

        # Selección de fecha
        min_date = pd.to_datetime("2022-01-01")
        max_date = pd.to_datetime("2024-09-30")
        # Crear el selector de fecha
        st.date_input("Selecciona una fecha", value=pd.to_datetime("2023-01-01"), min_value=min_date,
                                      max_value=max_date)

        df['fecha'] = df['fecha'].dt.strftime('%Y-%m-%d')

        print(df.info())

        with open(file = "spain-provinces.geojson", mode = "r", encoding = "utf8") as file:
            geojson_spain = json.load(file)

        mapa_espana = folium.Map(location=[40.4168, -3.7038], zoom_start=6)
        folium.Choropleth(
            geo_data=geojson_spain,
            data=df,
            columns=["provincia", "media_tmed"],
            key_on="feature.properties.name"
        ).add_to(mapa_espana)

        # streamlit_folium
        st_folium(mapa_espana, width=725)

    if choice=="Predicción del tiempo":

        def predict_temperature(model_path, scaler_path, input_data):
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)

            input_data = input_data.reshape(1, -1)
            input_data_scaled = scaler.transform(input_data)
            input_data_scaled = input_data_scaled.reshape(1, input_data_scaled.shape[0], input_data_scaled.shape[1])

            predicted_temperature = model.predict(input_data_scaled)
            return scaler.inverse_transform(predicted_temperature)

        estimated_temperature_tomorrow = 27

        # Inicializar variables para las predicciones
        predicted_temperature_rnn = None
        predicted_temperature_lstm = None

        st.markdown('#### Predicción de la temperatura de mañana')

        # Función para predecir la temperatura
        def predict_and_display(model_file):
            input_data = np.array([[estimated_temperature_tomorrow]])
            return predict_temperature(model_file, 'scaler.pkl', input_data)

        # Botones para predecir con RNN y LSTM
        if st.button('Predecir datos con RNN'):
            predicted_temperature_rnn = predict_and_display('modelo_RNN.pkl')

        if st.button('Predecir datos con LSTM'):
            predicted_temperature_lstm = predict_and_display('modelo_LSTM.pkl')

        # Mostrar predicciones si están disponibles
        if predicted_temperature_rnn is not None:
            st.write(f'Predicción de temperatura con RNN: {predicted_temperature_rnn[0][0]} °C')

        if predicted_temperature_lstm is not None:
            st.write(f'Predicción de temperatura con LSTM: {predicted_temperature_lstm[0][0]} °C')



if __name__ == "__main__":
    main()