import plotly.graph_objects as go

def draw_plot(data, name, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=data, mode='lines', name=name))

    fig.update_layout(title=title)

    fig.show()