from shiny import App, render, ui, reactive
import pandas as pd

import sys
sys.path.insert(0, './python_project/src/')
from automed.automed import *

global input_data
input_data = pd.DataFrame()
app_ui = ui.page_fluid(
    ui.include_css("shiny_app/assets/default.css"),
    ui.include_js("shiny_app/assets/default.js"),
    ui.navset_tab(
        ui.nav("Upload",
            ui.div(
                ui.h2("Upload file"),
                ui.input_file("input_data", "Choose CSV File", accept=[".csv"], multiple=False)
            )
        ),
        ui.nav("Choose label",
            ui.div(
                ui.h2("Choose label to predict"),
                ui.output_ui("select_label"),
                ui.output_data_frame("show_input"),
            )
        ),
        ui.nav("Pipeline",
            ui.div(
                ui.h2("Pipeline"),
                ui.input_action_button("run_pipe", "Run Pipeline", class_="btn-success"),
                ui.output_ui('output_pipe')
            )
        ),
        ui.nav("Results",
            ui.div(
                ui.h2("Results"),
                ui.output_ui("output_result"),
                ui.h2("Output data"),
                ui.output_data_frame("show_output"),
                
            )
        )
    ),
)


def server(input, output, session):
    automed = reactive.Value(AutoMed())
    pipe_log = reactive.Value([])

    @output
    @render.data_frame
    def show_input():
        if input.input_data() is None:
            return "Please upload a csv file"
        
        return render.DataTable(automed.get().dataset.X_train)
    
    
    @output
    @render.data_frame
    def show_output():
        if input.input_data() is None:
            return "Please upload a csv file and run pipeline"
        
        return render.DataTable(automed.get().output.dataset.X_train)
    
    @output
    @render.ui
    def select_label():
        return ui.input_select(
            "label",
            "Choose a column",
            list(['Choose...']) + list(automed.get().dataset.X_train.columns)
            )
        
    
    @output
    @render.ui
    def output_result():
        return f"Prediction accuracy : {automed.get().output.metric.compute(automed.get().output)}"
        
    @output
    @render.ui
    def output_pipe():
        html = "<br/><br /><ul>"
        for pl in pipe_log.get():
            html += f"<li>✅ {pl}</li>"
        html = html + "<ul>"
        return ui.HTML(html)
    
    @reactive.Effect
    @reactive.event(input.input_data)
    def get_data():
        f: list[FileInfo] = input.input_data()
        # return input.header()
        am = AutoMed(Dataset(pd.read_csv(f[0]["datapath"], sep=",")))
        am.debug_load() # Loading pipeline
        automed.set(am)
        
    @reactive.Effect
    @reactive.event(input.label)
    def get_label():
        label = input.label()
        if label != "Choose...":
            automed.get().dataset.set_label(label) 
            fetch(automed)
            
    def fetch(reactive_var):
        tmp = reactive_var.get()
        reactive_var.set(None)
        reactive_var.set(tmp)
    
    
    @reactive.Effect
    @reactive.event(input.run_pipe)
    async def go_pipe():
        add_pipe_log("Starting Pipeline")
        print("Starting Pipeline")
        automed.get().run(callback=pipe_callback)
        add_pipe_log("End !")
    
    def pipe_callback(step):
        import time
        if step.name != 'Step':
            add_pipe_log(step.name)
        # time.sleep(1)
        
    
    def add_pipe_log(text):
        pl = pipe_log.get()
        pl.append(text)
        pipe_log.set(pl)
        fetch(pipe_log)
        
        


app = App(app_ui, server)
