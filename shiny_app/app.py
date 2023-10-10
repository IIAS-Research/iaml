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
                ui.input_file("input_data", "Choose CSV File", accept=[".csv"], multiple=False),
                ui.br(),
                ui.hr(),
                ui.br(),
                ui.input_checkbox("automedoc", "Load automedoc", False),
                ui.output_ui("automedoc_value"),
            )
        ),
        ui.nav("Choose label",
            ui.div(
                ui.h2("Choose label to predict"),
                ui.output_ui("select_label"),
                ui.row(
                ui.column(9,
                    ui.h3('Features'),
                    ui.output_data_frame("show_input")),
                ui.column(3,
                    ui.h3('Label'),
                    ui.output_data_frame("show_input_label")))
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
                ui.output_ui("select_output"),
                ui.output_ui("output_result"),
                ui.h2("Output data"),
                ui.output_data_frame("show_output"),
                ui.h2("Testing set"),
                ui.output_data_frame("show_output_test"),
                ui.h2("Stacked path"),
                ui.output_ui("show_stacked_path"),
                
            )
        )
    ),
)


def server(input, output, session):
    automed = reactive.Value(AutoMed())
    pipe_log = reactive.Value([])
    current_result = reactive.Value(None)

    @output
    @render.data_frame
    def show_input():
        if input.input_data() is None:
            return "Please upload a csv file"
        
        return render.DataTable(automed.get().dataset.X_train)
    
    @output
    @render.data_frame
    def show_input_label():
        if input.input_data() is None:
            return "Please choose a label"
        
        return render.DataTable(pd.DataFrame(automed.get().dataset.y_train))
    
    
    @output
    @render.data_frame
    def show_output():
        if current_result.get():
            return render.DataTable(current_result.get().dataset.X_train)
        else:
            return "Choose an output"
        
    
    @output
    @render.data_frame
    def show_output_test():
        if current_result.get():
            return render.DataTable(current_result.get().dataset.check_X_test)
        else:
            return "Choose an output"
        
    
    @output
    @render.ui
    def show_stacked_path():
        html = "<br/><br /><ul>"
        for stack in current_result.get().stacked:
            html += f"<li>✅ {stack}</li>"
        html = html + "<ul>"
        return ui.HTML(html)
    
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
    def select_output():
        output_dict = {-1:'Choose...'}
        for index, value in enumerate(automed.get().output):
            output_dict[index] = str(index) + " : " + str(value)
            
        return ui.input_select(
            "selected_output",
            "Choose a output",
            output_dict
            )
        
    
    @output
    @render.ui
    def output_result():
        if current_result.get():
            return f"Prediction accuracy : {current_result.get().metric.compute(current_result.get())}"
        else:
            return "Choose an output"
        
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
        
        df = pd.read_csv(f[0]["datapath"], sep=";")
        if len(df.columns) < 2:
            df = pd.read_csv(f[0]["datapath"], sep=",")
            
        am = AutoMed(Dataset(df))
        am.debug_load() # Loading pipeline
        automed.set(am)
        
    @reactive.Effect
    @reactive.event(input.label)
    def get_label():
        label = input.label()
        if label != "Choose...":
            automed.get().dataset.set_label(label) 
            fetch(automed)
            
        
    @reactive.Effect
    @reactive.event(input.selected_output)
    def selected_output():
        value = int(input.selected_output())
        if value != -1:
            current_result.set(automed.get().output[value])
            print("->", current_result.get())
            
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
        
    
    @output
    @render.ui
    def automedoc_value():
        if input.automedoc():
            from automedoc.automedoc import ActAtcCode
            return "Loaded !"
        return ""
        
    
    def pipe_callback(step):
        if step.name != 'Step':
            add_pipe_log(step.name)
        
    
    def add_pipe_log(text):
        pl = pipe_log.get()
        pl.append(text)
        pipe_log.set(pl)
        fetch(pipe_log)
        
        


app = App(app_ui, server)
