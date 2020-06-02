from random import randrange, choice, uniform
import collections

import plotly
from plotly import graph_objs as go
import pandas as pd
import numpy as np


def plot_mo(problem, population=None, samples=100):
    """ Plots 2 objectives for 2-D problems.

        Final plotly figure includes: 
        - Pareto front. 
        - Objectives for function-1 and function-2 with Pareto front vectors.
        - 3D plot with objectives and Pareto vectors projections on functions surfaces

    Args:
        problem (pygmo.core.problem): Problen description
        population (pygmo.core.population): Vectors population on problem instance 
        samples (int, optional): Number of intervals on the axes. Defaults to 50.

    Returns:
        [plotly.graph_objs._figure.Figure]: Plotly figure
    """

    down, up = problem.get_bounds()
    # ZDT-4: (array([0., -5.]), array([1., 5.]))

    x = np.linspace(down[0], up[0], samples, endpoint=True)
    y = np.linspace(down[1], up[1], samples, endpoint=True)

    all_vectors = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)
    space = pd.DataFrame(all_vectors, columns=['x', 'y'])

    space[['f1', 'f2']] = space.apply(
        lambda row: pd.Series(problem.fitness(row)), axis=1)
    z_f1 = space.pivot(index='y', columns='x', values='f1').values
    z_f2 = space.pivot(index='y', columns='x', values='f2').values

    pareto_pop = pd.DataFrame([[0, 0, 0, 0]], columns=['x', 'y', 'f1', 'f2'])
    if population:
        f = pd.DataFrame(population.get_f(), columns=[
            'f1', 'f2'])  # fits on functions
        v = pd.DataFrame(population.get_x(), columns=[
            'x', 'y'])  # vectors populations
        pareto_pop = pd.concat([v, f], axis=1)
    #  ------------------------------------------------------- Plots

    # Functions values in Pareto front
    front = go.Scatter(x=pareto_pop.f1.values, y=pareto_pop.f2.values,
                       mode='markers',
                       line=dict(color='#DC3912'),
                       showlegend=False,
                       name=u'Pareto front'
                       )

    pop_vectors = go.Scatter(x=pareto_pop.x.values,
                             y=pareto_pop.y.values,
                             mode='markers',
                             name=u'Pareto vectors',
                             showlegend=False,
                             line=dict(color='#DC3912')
                             )

    f1 = go.Contour(z=z_f1,
                    x=x,
                    y=y,
                    showlegend=False,
                    colorscale='Viridis',
                    name=u'f1(x,y)')
    f2 = go.Contour(z=z_f2,
                    x=x,
                    y=y,
                    showlegend=False,
                    colorscale='Viridis',
                    name=u'f2(x,y)')

    # --- Subplot 2x3

    grid = plotly.subplots.make_subplots(
        rows=2, cols=3,
        subplot_titles=('Pareto', 'f1(x,y)', 'f2(x,y)'),
        row_heights=[0.3, 0.7],
        specs=[[{'type': 'scatter'}, {'type': 'contour'}, {'type': 'contour'}],
               [{"type": "surface", "colspan": 3}, None, None]]
    )

    population and grid.add_trace(front, row=1, col=1)
    grid['layout']['yaxis1'].update(title='f2(x,y)')
    grid['layout']['xaxis1'].update(title='f1(x,y)')

    grid.add_trace(f1, row=1, col=2)
    population and grid.add_trace(pop_vectors, row=1, col=2)

    grid.add_trace(f2, row=1, col=3)
    population and grid.add_trace(pop_vectors, row=1, col=3)

    # --- 3D
    sct = go.Scatter3d(x=pareto_pop.x.values,  # values f1() functin in pareto population
                       y=pareto_pop.y.values,
                       z=pareto_pop.f1.values,
                       name='f1()',
                       marker=dict(color='#00CC96'),
                       mode='markers',
                       showlegend=False
                       )
    sct2 = go.Scatter3d(x=pareto_pop.x.values,  # values f2() functin in pareto population
                        y=pareto_pop.y.values,
                        z=pareto_pop.f2.values,
                        name='f2()',
                        marker=dict(color='#FECB52'),
                        mode='markers',
                        showlegend=False
                        )

    srf_1 = go.Surface(z=z_f1,  # f1()
                       x=x,
                       y=y,
                       name='f1()',
                       colorscale='Viridis',
                       lighting=dict(fresnel=4.5))

    srf_2 = go.Surface(z=z_f2,  # f1()
                       x=x,
                       y=y,
                       name='f2()',
                       colorscale='Viridis',
                       lighting=dict(fresnel=4.5))

    grid.add_trace(srf_1, row=2, col=1)
    grid.add_trace(srf_2, row=2, col=1)
    population and grid.add_trace(sct, row=2, col=1)
    population and grid.add_trace(sct2, row=2, col=1)

    grid.update_layout(height=1000,
                       template='plotly_white',
                       title_text="Problem: <b>{0}</b>".format(problem.get_name()))

    return grid


def animate_plot_grid(plots):
    intro = plots[0]
    FRAME = len(plots)

    frames = [dict(name=k,
                   data=plots[k].data,
                   traces=[t for t in range(len(plots[k].data))]) for k in range(FRAME)]

    updatemenus = [dict(type='buttons',
                        buttons=[dict(label='Play',
                                      method='animate',
                                      args=[[f'{k}' for k in range(FRAME)],
                                            dict(frame=dict(duration=900, redraw=False),
                                                 transition=dict(duration=400),
                                                 easing='linear',
                                                 fromcurrent=True,
                                                 mode='immediate'
                                                 )]),
                                 dict(label='Pause',
                                      method='animate',
                                      args=[None,
                                            dict(frame=dict(duration=900, redraw=False),
                                                 transition=dict(duration=400),
                                                 easing='linear',
                                                 fromcurrent=True,
                                                 mode='immediate')])],
                        direction='left',
                        pad=dict(r=10, t=85),
                        showactive=True, x=0.1, y=0, xanchor='right', yanchor='top')
                   ]

    sliders = [{'yanchor': 'top',
                'xanchor': 'left',
                'currentvalue': {'font': {'size': 16}, 'prefix': 'Frame: ', 'visible': True, 'xanchor': 'right'},
                'transition': {'duration': 900.0, 'easing': 'linear'},
                'pad': {'b': 10, 't': 50},
                'len': 0.9, 'x': 0.1, 'y': 0,
                'steps': [{'args': [[k], {'frame': {'duration': 1300.0, 'easing': 'linear', 'redraw': True},
                                          'transition': {'duration': 800, 'easing': 'linear'}}],
                           'label': k, 'method': 'animate'} for k in range(FRAME)
                          ]}]

    intro['layout'].update(updatemenus=updatemenus,
                           sliders=sliders)
    return intro.update(frames=frames)
