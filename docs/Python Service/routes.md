# Routes

## Create new treatment
**Path : /create/**

Parameters :
- data (binary) : Binary data to load 
- data_path (string) : Path to load the data (instead of "data" parameter)
- pipeline_name (string) : Name of the pipeline to load (ex: "classifier")

Return :
- id : Id of the treatment 
- config (json) : Configuration of all steps in the pipeline

## Update config
**Path : /configure/:id**

Parameters:
- configurations (json) : JSON containing steps configuration. 

Return :
- boolean value

Json format
```json
configurations: [
    {
        step_id: <object_id>,
        configuration: [
            {
            key: <str>,
            value: <any>
            },
            {
            key: <str>,
            value: <any>
            },
            ...
        ]
    },
    {
        step_id: <object_id>,
        configuration: [
            {
            key: <str>,
            value: <any>
            },
            {
            key: <str>,
            value: <any>
            },
            ...
        ]
    },
    ...
]
```

Only Step and Key you want to update is needed. 

## Run
**Path : /run/:id**

Parameters :
- Nothing

Return :
- Boolean value

This route will run pipeline treatment. The treatment is asynchronous so only a boolean value will be returned.
Each update in the pipeline (end of a step for example) will trigger a push information to the web server.

