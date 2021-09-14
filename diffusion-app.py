import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from plotly.subplots import make_subplots

import numpy as np
from math import sqrt
from scipy import special, optimize
import multidiff
from multidiff import compute_diffusion_matrix, create_diffusion_profiles

# ------------ Prepare computations --------------------

# Single diffusion 

def erf_diffusion(x, D):
    return 1/ 2 * (1 + special.erf(x / sqrt(D)))

# Multicomponent diffusion

n_comps = 2

diags = np.array([1, 5])
P = np.matrix([[1, 1], [-1, 0]])

xpoints_exp1 = np.linspace(-10, 10, 100)
exchange_vectors = np.array([[0, 1, 1],
               [1, -1, 0],
               [-1, 0, -1]])

x_points = [xpoints_exp1] * 3
concentration_profiles = create_diffusion_profiles((diags, P), x_points,
                                                    exchange_vectors,
                                                    noise_level=0.02)


fig = go.Figure()
fig2 = make_subplots(rows=1, cols=3)
for i in range(3):
    fig_tmp = px.scatter(x=xpoints_exp1, y=list(concentration_profiles[i]))
    fig2.add_traces(fig_tmp.data, cols=i + 1, rows=1)

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

app.layout = html.Div(children=[
    html.H2(children='Diffusion of a single species'),
    dcc.Slider(id='d_slider', min=0, max=20, value=10, step=1, 
               tooltip={'always_visible':True}),
    dcc.Slider(id='noise_slider', min=0, max=0.5, step=0.05, value=0, tooltip={'always_visible':True}),
    dcc.Slider(id='num_slider', min=10, max=200, step=10, value=30, tooltip={'always_visible':True}),
    html.Div(id='estimation_result', children=[]),
    html.Button('Refresh', id='redo'),
    dcc.Graph(
        id='diffusion_single_fig',
        figure=fig,
    ),
    html.H2(children='Multicomponent diffusion'),
    dcc.Slider(id='noise_slider_multi', min=0, max=0.1, step=0.01, value=0, tooltip={'always_visible':True}),
    dcc.Slider(id='num_slider_multi', min=10, max=200, step=10, value=30, tooltip={'always_visible':True}),
    html.Div(id='estimation_multi', children=[]),
    dcc.Graph(
        id='diffusion_multi_fig',
        figure=fig2,
    ),
    
])

@app.callback(
    Output(component_id='diffusion_single_fig', component_property='figure'),
    Output(component_id='estimation_result', component_property='children'),
    Input(component_id='d_slider', component_property='value'),
    Input(component_id='noise_slider', component_property='value'),
    Input(component_id='num_slider', component_property='value'),
    Input(component_id='redo', component_property='n_clicks'),
)
def plot_diffusion_single(d_val, noise, num, redo):
    x = np.linspace(-10, 10, num)
    conc = 1/ 2 * (1 + special.erf(x/sqrt(d_val)))
    conc += noise * np.random.randn(num)
    fig = px.scatter(x=x, y=conc)
    D_estimate = optimize.curve_fit(erf_diffusion, x, conc)[0][0]
    fig.add_trace(px.line(x=x, y=erf_diffusion(x, D_estimate)).data[0])
    return fig, "The estimated value of D is %.1f" %D_estimate


@app.callback(
    Output(component_id='diffusion_multi_fig', component_property='figure'),
    Output(component_id='estimation_multi', component_property='children'),
    #Input(component_id='d_slider', component_property='value'),
    Input(component_id='noise_slider_multi', component_property='value'),
    Input(component_id='num_slider_multi', component_property='value'),
    #Input(component_id='redo', component_property='n_clicks'),
)
def plot_diffusion_multi(noise, num):
    diags = np.array([1, 5])
    P = np.matrix([[1, 1], [-1, 0]])

    xpoints_exp1 = np.linspace(-10, 10, 100)
    exchange_vectors = np.array([[0, 1, 1],
        [1, -1, 0],
        [-1, 0, -1]])

    x_points = [xpoints_exp1] * 3
    concentration_profiles = create_diffusion_profiles((diags, P), x_points,
                                    exchange_vectors,
                                    noise_level=noise)
    fig2 = make_subplots(rows=1, cols=3)
    for i in range(3):
        fig_tmp = px.scatter(x=xpoints_exp1, y=list(concentration_profiles[i]))
        fig2.add_traces(fig_tmp.data, cols=i + 1, rows=1)
    diags_init = np.array([1, 1])
    P_init = np.eye(2)
    diags_res, eigvecs, _, _, _ = compute_diffusion_matrix((diags_init, P_init),
                                x_points,
                                concentration_profiles, plot=False)
    concentration_profiles_est = create_diffusion_profiles((diags_res, eigvecs[:-1]), x_points,
                                    exchange_vectors,
                                    noise_level=0)
    for i in range(3):
        fig_tmp = px.line(x=xpoints_exp1, y=list(concentration_profiles_est[i]))
        fig2.add_traces(fig_tmp.data, cols=i + 1, rows=1)
    return fig2, "bla"

if __name__ == '__main__':
    app.run_server(debug=True)
