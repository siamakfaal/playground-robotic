import numpy as np
import plotly.graph_objects as go

def main():
    x = np.linspace(0, np.pi, 100)
    y = np.sin(x)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y))
    fig.update_layout(
            title="Example",
            xaxis_title="x",
            yaxis_title="sin(x)"
        )

    fig.show()



if __name__=="__main__":
    main()