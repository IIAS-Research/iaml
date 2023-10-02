# AutoMed
The main goal of AutoMed is to offer to practitioner the possibility to do DataScience by themself. To reach this goal, we are developing several bricks.

## Python Package
The first brick is a python package able the carry out almost all tabular data science project. In the future, it will also work with other king of data (like images or time series). 
With less that five lines of code, this package can generate a pipeline of data treatment and execute it. The fully automated pipeline is compose of spliting, cleaning, features engineering, learning and evaluating steps. Almost like a black box, you only have to put data in it and get a good prediction model. Almost because AutoMed is not a black box. It was developed to be self explanatory, a pipeline can explain, in natural language, all the treatment done on the data and can also generate a bibliography of used technologies.

## Web interface
The python package is great but practitioner are not used to work with python. So, we are developing a web interface with Ruby on Rails and VueJS3. This interface will offer to practitioner the ability to create AI models without coding or data science knowledge. 

## Python Service
A layer over the python package will be developed to enable communication between the package and other technologies. This service will work as and interface between our bricks. 

<- You can use the left menu for more informations