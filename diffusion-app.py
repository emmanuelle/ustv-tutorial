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
from synthetic import create_diffusion_profiles
from diffusion_matrix import compute_diffusion_matrix
from diffusion_values import diffusion_dict, initials_dict
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
    html.P('Diffusion coefficient'),
    dcc.Slider(id='d_slider', min=0, max=20, value=10, step=1, 
               tooltip={'always_visible':True}),
    html.P('Intensity of Gaussian noise'),
    dcc.Slider(id='noise_slider', min=0, max=0.5, step=0.05, value=0, tooltip={'always_visible':True}),
    html.P('Number of experimental points'),
    dcc.Slider(id='num_slider', min=10, max=200, step=10, value=30, tooltip={'always_visible':True}),
    html.Div(id='estimation_result', children=[]),
    html.Button('Refresh', id='redo'),
    dcc.Graph(
        id='diffusion_single_fig',
        figure=fig,
    ),
    html.H2(children='Multicomponent diffusion'),
    dcc.Dropdown(id='diffusion_system',
        options=[{'label':'Na2O-CaO-SiO2, 1200°C', 'value':'NCS',},
                 {'label':'B2O3-Na2O-SiO2, 800°C', 'value':'BNS-800'},
                 {'label':'B2O3-Na2O-SiO2, 900°C', 'value':'BNS-900'},
                 {'label':'B2O3-Na2O-SiO2, 1000°C', 'value':'BNS-1000'},
                 {'label':'B2O3-Na2O-SiO2, 1100°C', 'value':'BNS-1100'},
                 ],
        value='NCS'),
    html.P('Intensity of Gaussian noise'),
    dcc.Slider(id='noise_slider_multi', min=0, max=0.1, step=0.01, value=0, tooltip={'always_visible':True}),
    html.P('Number of experimental points'),
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
    Input(component_id='diffusion_system', component_property='value'),
    #Input(component_id='redo', component_property='n_clicks'),
)
def plot_diffusion_multi(noise, num, system):
    diags = diffusion_dict[system]['eigvals']
    P = diffusion_dict[system]['eigvecs']
    comp_names = [initials_dict[s] for s in system[:3]]

    lmax = 4 * np.sqrt(np.max(diags))
    xpoints_exp1 = np.linspace(-lmax, lmax, num)
    exchange_vectors = np.array([[0, 1, 1],
        [1, -1, 0],
        [-1, 0, -1]])

    x_points = [xpoints_exp1] * 3
    concentration_profiles = create_diffusion_profiles((diags, P), x_points,
                                    exchange_vectors,
                                    noise_level=noise)
    fig2 = make_subplots(rows=1, cols=3)
    dict_data = {s:concentration_prof for (s, concentration_prof) in zip(comp_names, concentration_profiles)}
    for i in range(3):
        fig_tmp = px.scatter(x=xpoints_exp1, y=list(concentration_profiles[i]),
                labels={'wide_variable_0':comp_names[0],
                        'wide_variable_1':comp_names[1],
                        'wide_variable_2':comp_names[2]})
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
    output_msg = """
    The true eigenvalues are %.2E and %.2E.

    The true coefficients of the major eigenvector are (%.2f, %.2f).
    The true coefficients of the minor eigenvector are (%.2f, %.2f).
    
    The estimated (fitted) eigenvalues are %.2E and %.2E.

    The estimated coefficients of the major eigenvector are (%.2f, %.2f).
    The estimated coefficients of the minor eigenvector are (%.2f, %.2f).
    """ %(
          diags[0], diags[1], 
          P[0, 0], P[1, 0],
          P[0, 1], P[1, 1],
          diags_res[1], diags_res[0], 
          eigvecs[0][1], eigvecs[1][1],
          eigvecs[0][0], eigvecs[1][0]
          )
    print(fig2)
    fig2.update_traces(showlegend=False)
    fig2.update_traces(showlegend=True, selector={'yaxis':'y', 'mode':'markers'})
    for i in range(3):
        fig2.update_traces(name=comp_names[i], selector={'name':'wide_variable_%d'%i})
    return fig2, output_msg

if __name__ == '__main__':
    app.run_server(debug=True)
