import matplotlib.pyplot as plt
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go
from darts import TimeSeries
from darts.models import ExponentialSmoothing

from seir.src.simple_seir import prepare_data

pd.plotting.register_matplotlib_converters()

# Prepare the data
df_dict, pop_dict = prepare_data()
regions = [i for i in range(1, 11)]

# Create the time series forecasts
def make_ts():
    """ Run each region through a time series exponential smoothing model
    """
    final = pd.DataFrame()
    for region in regions:
        temp_df = df_dict[region].reset_index()
        series = TimeSeries.from_dataframe(df=temp_df, time_col="Date", value_cols="Cases")
        model = ExponentialSmoothing()
        model.fit(series)
        prediction = model.predict(30)
        a = series.pd_dataframe()
        name1 = "Region {}: Cases".format(str(region))
        a.columns = [name1]
        b = prediction.pd_dataframe()
        name2 = "Region {}: Predictions".format(str(region))
        b.columns = [name2]

        if final.shape[0] == 0:
            index = [item.date() for item in a.index] + [item.date() for item in b.index]
            final["Day"] = [i for i in range(len(index))]
            final.index = index
        final[name1] = a[name1]
        final[name2] = b[name2]
    final = final.drop(["Day"], axis=1)
    return final


def make_plot():
    """ Create a basic time series line graph with a line for each region
    """
    title = "Estimated Infections by Region"
    x_data = [i for i in final.index]
    fig = go.Figure()
    colors = px.colors.qualitative.G10

    for region in regions:
        region_str = str(region)
        y_data = final["Region {}: Cases".format(region_str)]
        fig.add_trace(
            go.Scatter(
                x=x_data,
                y=y_data,
                mode="lines",
                name="Region {} Estimated Infections".format(region_str),
                connectgaps=True,
                line=dict(color=colors[region - 1]),
                legendgroup=region_str,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x_data,
                y=final["Region {}: Predictions".format(region_str)],
                mode="lines",
                name="Region {} Predicted Infections".format(str(region)),
                connectgaps=True,
                line=dict(color=colors[region - 1], dash="dot"),
                legendgroup=region_str,
                showlegend=False,
            )
        )
        # Add Balls
        i = len(x_data) - 31
        fig.add_trace(
            go.Scatter(
                x=[x_data[i]],
                y=[y_data[i]],
                mode="markers",
                showlegend=False,
                marker=dict(color=colors[region - 1], size=6),
                legendgroup=region_str,
            )
        )

    fig.update_layout(
        xaxis=dict(
            showline=True,
            showgrid=True,
            showticklabels=True,
            linecolor="rgb(204, 204, 204)",
            linewidth=2,
            ticks="outside",
            tickfont=dict(family="Arial", size=12, color="rgb(82, 82, 82)",),
        ),
        yaxis=dict(showgrid=True, zeroline=False, showline=True, showticklabels=True,),
        title=title,
        autosize=True,
        showlegend=True,
        plot_bgcolor="white",
    )

    annotations = []
    fig.update_layout(annotations=annotations)
    plotly.offline.plot(fig, filename="test.html")


final = make_ts()
make_plot()


# Graph of North Carolina
