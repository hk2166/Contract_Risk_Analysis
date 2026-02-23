from taipy.gui import Gui
page = """
<|{html_val}|text|raw=True|>
<|part|render=True|>
"""
html_val = "<h1>Hello</h1>"
gui = Gui(page=page)
if __name__ == "__main__":
    gui.run(run_server=False)
    print("Taipy HTML test run without errors.")
