import dash
from dash import dcc, html, Input, Output, State
import plotly.express as px
from scipy.stats import f_oneway, pearsonr
import pandas as pd
import numpy as np

# Coeficientes del modelo
intercept = 6.7940
coefficients = {
    'bathrooms': 0.1475,
    'bedrooms': 0.0678,
    'square_feet': 0.0002,
    'room_density': -46.5811,
    'state_AZ': -0.1077,
    'state_CA': 0.6688,
    'state_CO': 0.0976,
    'state_CT': 0.0627,
    'state_FL': 0.0445,
    'state_GA': -0.1134,
    'state_IA': -0.3262,
    'state_IL': 0.1108,
    'state_IN': -0.2453,
    'state_MA': 0.5855,
    'state_MD': 0.1892,
    'state_MI': -0.1177,
    'state_MN': 0.1125,
    'state_MO': -0.2788,
    'state_NC': -0.1805,
    'state_ND': -0.4019,
    'state_NE': -0.2613,
    'state_NJ': 0.4118,
    'state_NV': -0.1212,
    'state_OH': -0.2319,
    'state_OK': -0.2715,
    'state_OR': 0.1626,
    'state_Otros': -0.0721,
    'state_PA': 0.0061,
    'state_TN': -0.0478,
    'state_TX': -0.0616,
    'state_VA': 0.1113,
    'state_WA': 0.3044,
    'state_WI': -0.0286
}

# Traducción de estados
state_translation = {
    'state_AZ': 'Arizona',
    'state_CA': 'California',
    'state_CO': 'Colorado',
    'state_CT': 'Connecticut',
    'state_FL': 'Florida',
    'state_GA': 'Georgia',
    'state_IA': 'Iowa',
    'state_IL': 'Illinois',
    'state_IN': 'Indiana',
    'state_MA': 'Massachusetts',
    'state_MD': 'Maryland',
    'state_MI': 'Michigan',
    'state_MN': 'Minnesota',
    'state_MO': 'Missouri',
    'state_NC': 'North Carolina',
    'state_ND': 'North Dakota',
    'state_NE': 'Nebraska',
    'state_NJ': 'New Jersey',
    'state_NV': 'Nevada',
    'state_OH': 'Ohio',
    'state_OK': 'Oklahoma',
    'state_OR': 'Oregon',
    'state_Otros': 'Otros',
    'state_PA': 'Pennsylvania',
    'state_TN': 'Tennessee',
    'state_TX': 'Texas',
    'state_VA': 'Virginia',
    'state_WA': 'Washington',
    'state_WI': 'Wisconsin'
}

# Cargar el DataFrame
df = pd.read_csv("Repositorio-Proyecto1\Despliegue\data_limpia_3.csv")

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

# Inicializar la aplicación Dash
app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])
server = app.server

# Layout de la aplicación
app.layout = html.Div([
    dcc.Tabs(id='tabs', value='tab-analysis', children=[
        # Pestaña de Análisis
        dcc.Tab(label='Análisis', value='tab-analysis', children=[
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
                value=available_columns[0],  # Valor por defecto
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
        ]),
        # Pestaña de Predicción
        dcc.Tab(label='Predicción de Precio', value='tab-prediction', children=[
            html.Div([
                html.Div([
                    html.Img(src='https://via.placeholder.com/800x400.png?text=Imagen+de+Casa', 
                             style={'width': '100%', 'borderRadius': '10px', 'marginBottom': '20px'})
                ]),
                html.Div([
                    html.H1("Formulario de Predicción de Precio",
                            style={
                                'textAlign': 'center',
                                'color': '#1b4d3e',  # Verde oscuro
                                'fontFamily': 'Arial, sans-serif',
                                'fontSize': '36px',
                                'marginBottom': '30px',
                                'marginTop': '20px'
                            }),
                    html.Div([
                        html.Label("Número de Baños (bathrooms):", style={'fontSize': '18px', 'marginBottom': '10px'}),
                        dcc.Input(id='bathrooms', type='number', value=1, style={
                            'width': '100%',
                            'padding': '10px',
                            'borderRadius': '5px',
                            'border': '1px solid #1b4d3e',
                            'fontSize': '16px'
                        }),
                    ], style={'marginBottom': '20px'}),
                    
                    html.Div([
                        html.Label("Número de Habitaciones (bedrooms):", style={'fontSize': '18px', 'marginBottom': '10px'}),
                        dcc.Input(id='bedrooms', type='number', value=1, style={
                            'width': '100%',
                            'padding': '10px',
                            'borderRadius': '5px',
                            'border': '1px solid #1b4d3e',
                            'fontSize': '16px'
                        }),
                    ], style={'marginBottom': '20px'}),
                    
                    html.Div([
                        html.Label("Pies cuadrados (square_feet):", style={'fontSize': '18px', 'marginBottom': '10px'}),
                        dcc.Input(id='square_feet', type='number', value=1000, style={
                            'width': '100%',
                            'padding': '10px',
                            'borderRadius': '5px',
                            'border': '1px solid #1b4d3e',
                            'fontSize': '16px'
                        }),
                    ], style={'marginBottom': '20px'}),
                    
                    html.Div([
                        html.Label("Estado:", style={'fontSize': '18px', 'marginBottom': '10px'}),
                        dcc.Dropdown(
                            id='state',
                            options=[{'label': state_translation[state], 'value': state} for state in state_translation],
                            value='state_CA',
                            style={
                                'width': '100%',
                                'padding': '10px',
                                'borderRadius': '5px',
                                'border': '1px solid #1b4d3e',
                                'fontSize': '16px'
                            }
                        ),
                    ], style={'marginBottom': '20px'}),
                    
                    html.Button(
                        'Calcular Predicción',
                        id='predict-button',
                        n_clicks=0,
                        style={
                            'width': '100%',  
                            'height': '50px',  # Altura fija para evitar desbordamiento
                            'padding': '10px 20px',  
                            'borderRadius': '8px',
                            'border': '2px solid #145a32',
                            'backgroundColor': '#1b4d3e',
                            'color': 'white',
                            'fontSize': '18px',
                            'fontWeight': 'bold',  # Hace que el texto se vea más claro
                            'textAlign': 'center',
                            'display': 'flex',  # Flexbox para centrar contenido
                            'alignItems': 'center',  # Centra verticalmente el texto
                            'justifyContent': 'center',  # Centra horizontalmente el texto
                            'cursor': 'pointer',
                            'marginBottom': '20px',
                            'boxShadow': '2px 4px 8px rgba(0, 0, 0, 0.2)',
                            'transition': 'background-color 0.3s ease, transform 0.2s ease'
                        }
                    ),
                    
                    html.Div(id='prediction-output', style={
                        'padding': '20px',
                        'borderRadius': '5px',
                        'backgroundColor': '#e8f1ee',
                        'border': '1px solid #1b4d3e',
                        'fontSize': '20px',
                        'textAlign': 'center'
                    })
                ], style={
                    'maxWidth': '600px',
                    'margin': 'auto',
                    'padding': '20px',
                    'boxShadow': '0px 0px 10px rgba(27, 77, 62, 0.2)',
                    'borderRadius': '10px',
                    'backgroundColor': 'white'
                })
            ])
        ])
    ])
])

# Callback para la pestaña de Análisis
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

# Callback para la pestaña de Predicción
@app.callback(
    Output('prediction-output', 'children'),
    [Input('predict-button', 'n_clicks')],
    [State('bathrooms', 'value'),
     State('bedrooms', 'value'),
     State('square_feet', 'value'),
     State('state', 'value')]
)
def predict_price(n_clicks, bathrooms, bedrooms, square_feet, state):
    if n_clicks > 0:
        # Calcular room_density
        room_density = (bedrooms + bathrooms) / square_feet
        
        # Calcular la predicción lineal
        prediction = (
            intercept +
            coefficients['bathrooms'] * bathrooms +
            coefficients['bedrooms'] * bedrooms +
            coefficients['square_feet'] * square_feet +
            coefficients['room_density'] * room_density +
            coefficients[state]
        )
        
        # Corregir la transformación logarítmica (exponencial)
        corrected_prediction = np.exp(prediction)
        
        # Mostrar el resultado
        return f'Predicción de Precio: ${corrected_prediction:,.2f}'
    return "Ingrese los valores y haga clic en 'Calcular Predicción'."

# Ejecutar la aplicación
if __name__ == '__main__':
    app.run_server(debug=True)