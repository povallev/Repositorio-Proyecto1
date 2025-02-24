import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.express as px
from scipy.stats import f_oneway, pearsonr
import pandas as pd
import numpy as np

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# Cargar el DataFrame
df = pd.read_csv("Tablero de datos\data_limpia_3.csv")

# Asegurarse de que 'price' sea numérico y eliminar filas con 'price' nulo
df['price'] = pd.to_numeric(df['price'], errors='coerce')
df = df.dropna(subset=['price'])

# Definir los nombres de las columnas para la visualización
column_names = {
    'category': 'Categoría del Inmueble', 
    'bathrooms': 'Número de Baños', 
    'bedrooms': 'Número de Habitaciones', 
    'has_photo': '¿Tiene Foto?', 
    'price': 'Precio', 
    'price_type': 'Tipo de Precio', 
    'square_feet': 'Metros Cuadrados', 
    'state': 'Estado', 
    'latitude': 'Latitud', 
    'longitude': 'Longitud', 
    'time': 'Tiempo'
}

# Filtrar las columnas para excluir 'price'
available_columns = [col for col in df.columns if col != 'price']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

app.layout = html.Div([
    html.H1("Análisis del Mercado Inmobiliario",
            style={
                'textAlign': 'center',
                'color': '#1b4d3e',  # Verde oscuro
                'fontFamily': 'Arial, sans-serif',
                'fontSize': '36px',
                'marginBottom': '30px',
                'marginTop': '20px'
            }),
    
    dcc.Dropdown(
        id='variable-selector',
        options=[{'label': column_names.get(col, col), 'value': col} 
                for col in available_columns],
        value=df.columns[0],  # Valor por defecto
        style={
            'width': '50%',
            'margin': 'auto',
            'marginBottom': '30px',
            'color': '#1b4d3e'  # Verde oscuro
        }
    ),
    
    html.Div([
        dcc.Graph(id='analysis-plot')
    ], style={
        'padding': '20px',
        'boxShadow': '0px 0px 10px rgba(27, 77, 62, 0.2)',  # Sombra en verde
        'borderRadius': '10px',
        'backgroundColor': 'white',
        'margin': '20px'
    })
], style={
    'backgroundColor': '#f8f9fa',
    'padding': '20px'
})

# Callback para actualizar el gráfico basado en la selección del usuario
@app.callback(
    Output('analysis-plot', 'figure'),
    [Input('variable-selector', 'value')]
)
def update_graph(selected_variable):
    variable_name = column_names.get(selected_variable, selected_variable)
    
    # Determinar si la variable es categórica o numérica
    if df[selected_variable].dtype == 'object':  # Variable categórica
        categories = df[selected_variable].unique()
        groups = [df[df[selected_variable] == category]['price'] for category in categories]
        f_stat, p_value = f_oneway(*groups)
        # Determinar si el p-value es significativo
        significance = "Significativo" if p_value < 0.05 else "No significativo"
        
        avg_prices = df.groupby(selected_variable)['price'].mean().reset_index()
        
        fig = px.bar(avg_prices, x=selected_variable, y='price',
                    title=f'Precio Promedio por {variable_name}<br>'
                        f'p-value: {p_value:.4f} ({significance})',
                    labels={
                        selected_variable: variable_name,
                        'price': 'Precio ($)'
                    },
                    template='plotly_white',
                    color_discrete_sequence=['#1b4d3e'])  # Verde oscuro principal
    else:  # Variable numérica
        correlation, p_value = pearsonr(df[selected_variable], df['price'])
        significance = "Significativo" if p_value < 0.05 else "No significativo"
        
        fig = px.scatter(df, x=selected_variable, y='price',
                         title=f'Relación entre {variable_name} y Precio<br>, p-value: {p_value:.4f} ({significance})',
                         labels={
                             selected_variable: variable_name,
                             'price': 'Precio ($)'
                         },
                         template='plotly_white',
                         trendline='ols',  # Línea de tendencia
                         color_discrete_sequence=['#1b4d3e'])  # Verde oscuro principal
    
    # Estilo común para ambos tipos de gráficos
    fig.update_layout(
        title={
            'font_size': 24,
            'font_family': 'Arial, sans-serif',
            'x': 0.5,
            'xanchor': 'center',
            'font_color': '#1b4d3e'  # Título en verde oscuro
        },
        plot_bgcolor='white',
        font=dict(
            family='Arial, sans-serif',
            size=14,
            color='#1b4d3e'  # Texto en verde oscuro
        ),
        hoverlabel=dict(
            bgcolor='white',
            font_size=14,
            font_family='Arial, sans-serif'
        )
    )

    if df[selected_variable].dtype == 'object':  # Solo para gráficos de barras
        fig.update_traces(
            marker_line_color='#0f2c23',  # Borde más oscuro para las barras
            marker_line_width=1.5,
            opacity=0.85
        )

    fig.update_xaxes(
        tickangle=45,
        title_font=dict(size=16),
        showgrid=False,
        title_font_color='#1b4d3e'  # Título del eje en verde oscuro
    )
    
    fig.update_yaxes(
        title_font=dict(size=16),
        gridcolor='#e8f1ee',  # Grid lines en verde muy claro
        showline=True,
        linewidth=2,
        linecolor='#1b4d3e',  # Línea del eje en verde oscuro
        title_font_color='#1b4d3e'  # Título del eje en verde oscuro
    )

    return fig

if __name__ == '__main__':
    app.run_server(debug=True)