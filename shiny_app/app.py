from shiny import App, render, ui
import pandas as pd

app_ui = ui.page_fluid(
    ui.h2("Hello Shiny!"),
    ui.input_slider("n", "N", 0, 100, 20),
    ui.output_text_verbatim("txt"),
    ui.include_css("shiny_app/assets/default.css"),
    ui.include_js("shiny_app/assets/default.js"),
    ui.input_file("input_data", "Choose CSV File", accept=[".csv"], multiple=False),
    ui.output_ui("contents"),
)


def server(input, output, session):
    @output
    @render.text
    def txt():
        return f"n*2 is {input.n() * 2}"
    
    @output
    @render.ui
    def contents():
        if input.input_data() is None:
            return "Please upload a csv file"
        f: list[FileInfo] = input.input_data()
        # return input.header()
        df = pd.read_csv(f[0]["datapath"], sep=";")
        # print(df)
        return str(df.columns)


app = App(app_ui, server)
