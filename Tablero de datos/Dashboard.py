#Desarrollo del Dasboard
#Importar librerías 
import dash
from dash import dcc  # dash core components
from dash import html # dash html components 
from dash.dependencies import Input, Output
import plotly.express as px
from scipy.stats import f_oneway    
import pandas as pd

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

df = pd.read_csv('data_limpia_2.csv')

df['price'] = pd.to_numeric(df['price'], errors='coerce')
df = df.dropna(subset=['price'])

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

app.layout = html.Div([
    html.H1("Análisis de Significancia de Variables Categóricas"),
    dcc.Dropdown(
        id='categorical-variable',
        options=[{'label': col, 'value': col} for col in df.select_dtypes(include=['object']).columns],
        value=df.select_dtypes(include=['object']).columns[0]
    ),
    dcc.Graph(id='significance-plot')
])

# Callback para actualizar el gráfico basado en la selección del usuario
@app.callback(
    Output('significance-plot', 'figure'),
    [Input('categorical-variable', 'value')]
)
def update_graph(selected_variable):
    # Calcular la significancia usando ANOVA
    categories = df[selected_variable].unique()
    groups = [df[df[selected_variable] == category]['price'] for category in categories]
    f_stat, p_value = f_oneway(*groups)
    
    # Crear un gráfico de barras para mostrar los precios promedio por categoría
    avg_prices = df.groupby(selected_variable)['price'].mean().reset_index()
    fig = px.bar(avg_prices, x=selected_variable, y='price', 
                 title=f'Precio Promedio por {selected_variable}<br>F-stat: {f_stat:.2f}, p-value: {p_value:.4f}')
    
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
