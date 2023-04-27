import dash_mantine_components as dmc
from dash import dcc, html
import dash
from dash_iconify import DashIconify
from sklearn_callback import *

app = dash.Dash(__name__)
data = [["tf", "TensorFlow"], ["sk", "Sklearn"]]

app.layout = html.Div(
    dmc.MantineProvider(
        theme={"colorScheme": "dark"},
        children=[
            dmc.BackgroundImage(src='assets/cool_neural_network.jpeg',
                children=[
                    dmc.Stack(
                        children=[
                            html.Div(
                                children=
                                [
                                    dmc.Paper(
                                        children=[
                                            html.H2("Week 7 - Modeling with Neural Networks"),
                                        ],
                                        radius='md',
                                        p='md',
                                        shadow='md',
                                        style={'textAlign':'center',
                                               'marginBottom':'10px'}
                                    ),

                                    dmc.Paper(
                                        children=[
                                            html.H4("Check if the message you've received is a HAM or a SPAM"),
                                            html.H4("Enter your message and choose which model you want to use")
                                        ],
                                        radius='md',
                                        p='md',
                                        shadow='md',
                                        style={'textAlign':'center',
                                               'marginBottom':'10px'}
                                    ),

                                    dmc.Paper(
                                        children=[
                                            dmc.TextInput(
                                                style={"width": 740},
                                                placeholder="Enter your message",
                                                icon=DashIconify(icon="ic:round-message"),
                                                id="textinput"
                                            ),
                                            dmc.Group(
                                                children=[
                                                    dmc.Button("Tensorflow", id="tf-button", n_clicks=0,
                                                               variant='light', color='grape'),
                                                    dmc.Button("Sklearn", id="sk-button", n_clicks=0,
                                                               variant='light', color='grape')
                                                ],
                                                align='center',
                                                style={'marginTop':'10px'},
                                                grow=True
                                            )

                                        ],
                                        radius='md',
                                        p='md',
                                        shadow='md',
                                        style={'textAlign':'center',
                                               'marginBottom':'10px'}
                                    ),

                                    dmc.Paper(
                                        children=[

                                        ],
                                        radius='md',
                                        p='md',
                                        shadow='md',
                                        style={'textAlign': 'center',
                                               'marginBottom': '10px'},
                                        id="return-paper"
                                    )
                                ],
                                style={'height':'100vh',
                                       'width':'100vh'}
                            )
                        ],
                        align="center",
                        justify="space-around"
                    )
                ]
            )
        ]
    )
)

if __name__ == '__main__':
    app.run_server(debug=False)

