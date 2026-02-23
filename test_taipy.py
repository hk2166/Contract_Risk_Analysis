from taipy.gui import Gui
import plotly.graph_objects as go

fig = go.Figure()
fig.add_annotation(text="Test", showarrow=False)

page = """
<|{fig}|chart|figure={fig}|>
"""

gui = Gui(page=page)
if __name__ == "__main__":
    t = gui.run(run_server=False)
    print("Taipy run without errors.")
