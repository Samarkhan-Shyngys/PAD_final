import pandas as pd
from dash import dcc
from dash import html
import numpy as np
from dash import dash_table, Input, Output
import plotly.express as px
from jupyter_dash import JupyterDash
import plotly.graph_objects as go
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score
import statsmodels.api as sm
import statsmodels.formula.api as smf
from collections import OrderedDict

ata = OrderedDict(
    [
        ("Engine displacement", ["0-1.4", "1.4-1.8", "1.8-3.5"]),
        ("CO g/km", [11.4, 13, 14]),
        ("СхHу g/km", [2.1, 2.6, 2.8]),
        ("NO2 g/km", [1.3, 1.5, 2.7]),
        ("SO2 g/km", [0.052, 0.076, 0.096]),
    ]
)
df = pd.DataFrame(ata)



data2 = pd.read_csv('output.csv')
data = pd.read_csv('cars3.csv')


df2_proper_columns = data.loc[:, data.columns.difference(['Unnamed: 0'])].columns.values;


X = data['year'].values.reshape(-1,1)
y = data['avg_price'].values.reshape(-1,1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


regressor = LinearRegression()
regressor.fit(X_train, y_train)

#training the algorithm
y_pred = regressor.predict(X_test)

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

X = data[[ 'volume_of_car', 'year']].values
y = data['avg_price'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
#Train the model

model = linear_model.LinearRegression()
model.fit(X_train, y_train)
print("Predict value " + str(model.predict([X_test[9]])))
print("Real value " + str(y_test[9]))
print("Accuracy --> ", model.score(X_test, y_test)*100)

df_price_without_District = data.drop(columns=['brand_of_car'])

model_price = smf.ols(formula="avg_price ~ year + volume_of_car", data=df_price_without_District).fit()

print('P Values: ', model_price.pvalues.values)
pValue = model_price.pvalues.values
coef = model_price.params.values
stdError = model_price.bse.values
print('Coef: ', model_price.params.values)
print("Std Errs", model_price.bse.values)

correlations = df_price_without_District.corr(numeric_only=True)
print(correlations)


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']


statistics = data.describe()


app = JupyterDash(__name__, external_stylesheets=external_stylesheets)
colors = {
    'background': '#111111',
    'text': '#22222'
}
# Create the histogram
histogram_dist = px.histogram(data, x='brand_of_car', width=600, height=400, title="cars by brands")
histogram_year = px.histogram(data, x='year', width=600, height=400, title="cars by year")
histogram_dist_year = px.scatter(data, "volume_of_car", "brand_of_car", "year" , title = "brands of cars with volume and year")
desc = data.describe()
statistics = data.describe()

# Create the figure
fig = go.Figure(data=[go.Table(
    header=dict(values=['Statistics'] + statistics.columns.tolist(),
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']] + [statistics[col].tolist() for col in statistics.columns],
               fill_color='lavender',
               align='left'))
])

# Update the layout
fig.update_layout(
    title='Descriptive Statistics',
    autosize=False,
    width=900,
    height=500
)


fig_corr = px.imshow(correlations)
fig_corr.update_layout(title="Variable Correlations")

app.layout = html.Div([
    html.H1(
        children='Analysis of the Impact of Old Cars on Smog Formation',
        style={
            'textAlign': 'center',
            'color': colors['text']
        }
    ),


    html.Div(children='CSV table view after cleaning data', style={
        'textAlign': 'center',
        'color': colors['text']
    }),


    html.Div([

        html.Table([
            html.Thead(
                html.Tr([html.Th(col) for col in data.columns])
            ),
            html.Tbody([
                html.Tr([
                    html.Td(data.iloc[i][col]) for col in data.columns
                ]) for i in range(min(len(data), 10))
            ])
        ])



    ], style={
        'padding': '30px 50px'
    }),
    html.H1(
        children='Data analysis',
        style={
            'textAlign': 'center',
            'color': colors['text']
        }
    ),
    html.Div(children=[

        html.Table([
            html.Tbody([
                html.Tr([
                    html.Td('Mean Absolute Error:'),
                    html.Td(metrics.mean_absolute_error(y_test, y_pred)),
                ]),
                html.Tr([
                    html.Td('Mean Squared Error:'),
                    html.Td(metrics.mean_squared_error(y_test, y_pred)),
                ]),
                html.Tr([
                    html.Td('Root Mean Squared Error:'),
                    html.Td(np.sqrt(metrics.mean_squared_error(y_test, y_pred))),
                ]),
                html.Tr([
                    html.Td('Predict value: '),
                    html.Td(str(model.predict([X_test[9]]))),
                ]),
                html.Tr([
                    html.Td('Real value: '),
                    html.Td(str(y_test[9])),
                ]),
                html.Tr([
                    html.Td('Accuracy: '),
                    html.Td(model.score(X_test, y_test)*100),
                ])
            ])
        ], style={'padding': 10, 'flex': 1}),

        html.Table([
            html.Tbody([
                html.Tr([
                    html.Td('P Values:'),
                    html.Td(str(pValue[0]) + " " + str(pValue[1]) + " " +str(pValue[2]) ),
                ]),
                html.Tr([
                    html.Td('Coefficient'),
                    html.Td(str(coef[0]) + " " +  str(coef[1]) + " " + str(coef[2]) ),
                ]),
                html.Tr([
                    html.Td('Standard Error:'),
                    html.Td(str(stdError[0]) + " " +str(stdError[1]) + " " +str(stdError[2]) ),
                ])
            ])
        ], style={'padding': 10, 'flex': 1})



    ], style={'display': 'flex', 'flex-direction': 'row'}),

    html.Br(),
    html.Div([
        dcc.Graph(figure=histogram_dist_year)
    ], style={'textAlign': 'center',  'padding': '0 20'}),


    html.Div(children=[
        html.Div([
            dcc.Graph(figure=histogram_dist)
        ], style={'padding': 10, 'flex': 1}),

        html.Div([
            dcc.Graph(figure=histogram_year)
        ], style={'padding': 10, 'flex': 1})
    ], style={'display': 'flex', 'flex-direction': 'row'}),

    html.Div([

        html.Table([
            html.Thead(
                html.Tr([html.Th(col) for col in df.columns])
            ),
            html.Tbody([
                html.Tr([
                    html.Td(df.iloc[i][col]) for col in df.columns
                ]) for i in range(min(len(df), 10))
            ])
        ])

    ], style={
        'padding': '30px 50px'
    }),








    html.P(
        children='The above statistics show that the average year is 2010, the average volume is 2.7 . The smallest volume car is 0.7. While the largest volume is 8.1. The oldest car is from 1988 and the newest is from 2022.',
        style={
            'textAlign': 'center',
            'padding': '0 30'
        }
    ),
    html.Div([
        dcc.Graph(figure=fig)
    ], style={'width': '80%', 'display': 'inline-block', 'padding': '0 20'}),

    html.Div([
        dcc.Graph(figure=fig_corr)
    ], style={'width': '80%', 'display': 'inline-block', 'padding': '0 20'}),

    html.Div([


        html.Br(),
        dcc.RadioItems(
            ['Regression', 'Classification'],
            'Regression',
            id='crossfilter-xaxis-type',
            labelStyle={'display': 'inline-block', 'marginTop': '5px'}
        ),

        dcc.Dropdown(
            ['year', 'avg_price', 'brand_of_car', 'volume_of_car'],
            'year',
            id='crossfilter-xaxis-column',
        )

    ],
        style={'width': '30%', 'display': 'inline-block'}),
    html.Div([
        dcc.Graph(
            id='crossfilter-indicator-scatter',
            hoverData={'points': [{'customdata': 'Japan'}]}
        )
    ], style={'width': '80%', 'display': 'inline-block', 'padding': '0 20'}),
])

@app.callback(
    Output('crossfilter-indicator-scatter', 'figure'),
    Input('crossfilter-xaxis-column', 'value'),
    Input('crossfilter-xaxis-type', 'value'))
def update_graph(xaxis_column_name, chart_type):
    if chart_type == 'Regression':
        fig = px.scatter(data, x=xaxis_column_name, y='avg_price', trendline="ols", trendline_color_override="red")

    else:
        fig = px.pie(data, values=xaxis_column_name, names='brand_of_car')

    return fig
if __name__ == '__main__':
    app.run_server(debug=True)