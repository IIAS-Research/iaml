# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'IAML'
copyright = '2024'
author = 'Rudy MERIEUX, Robin BOURACHOT, Hugo RUELLET'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'autoapi.extension',
    'sphinx.ext.autodoc.typehints',
]

autoclass_content = 'both'

autoapi_dirs = ['../src/iaml']
# Document each object in its defining module; package re-exports create
# duplicate entries and ambiguous links for classes such as Dataset.
autoapi_options = [
    'members',
    'undoc-members',
    'show-inheritance',
    'show-module-summary',
    'special-members',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

language = 'en'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_title = 'IAML'
html_logo = '_static/iias-logo.svg'
html_theme_options = {
    'light_css_variables': {
        'color-brand-primary': '#176b63',
        'color-brand-content': '#a95324',
        'color-foreground-primary': '#243d42',
        'color-background-secondary': '#f4f7f5',
        'color-background-border': '#dce5e1',
    },
    'dark_css_variables': {
        'color-brand-primary': '#8ddbc6',
        'color-brand-content': '#f4b183',
        'color-foreground-primary': '#e2eeea',
        'color-background-primary': '#10211e',
        'color-background-secondary': '#172824',
        'color-background-border': '#344941',
    },
}
html_static_path = ['_static']
html_css_files = ['iaml.css', 'workflow-demo.css']


def add_homepage_scripts(app, pagename, _templatename, _context, _doctree):
    """Load the locally bundled animation library only on the homepage."""
    if pagename == 'index':
        for filename in (
            'vendor/gsap/gsap.min.js',
            'vendor/gsap/ScrollTrigger.min.js',
            'workflow-demo.js',
        ):
            app.add_js_file(filename, defer='defer')


def setup(app):
    app.connect('html-page-context', add_homepage_scripts)
