# Sphinx configuration
#
# Inspired by SciPy's configuration:
# https://github.com/scipy/scipy/blob/v1.12.0/doc/source/conf.py

import math

import matplotlib
import matplotlib.pyplot as plt
import pm_remez

matplotlib.use('agg')
plt.ioff()

project = 'pm-remez'
author = pm_remez.__author__.split('<')[0].rstrip()
copyright = f'2024, {author}'
release = pm_remez.__version__

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'numpydoc',
    'matplotlib.sphinxext.plot_directive',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/devdocs', None),
    'matplotlib': ('https://matplotlib.org/stable', None),
    'scipy': ('https://docs.scipy.org/doc/scipy', None),
}

templates_path = ['_templates']
exclude_patterns = []

numpydoc_use_plots = True

plot_include_source = True
plot_formats = [('png', 96)]
plot_html_show_formats = False
plot_html_show_source_link = False

font_size = 13*72/96.0  # 13 px

plot_rcparams = {
    'font.size': font_size,
    'axes.titlesize': font_size,
    'axes.labelsize': font_size,
    'xtick.labelsize': font_size,
    'ytick.labelsize': font_size,
    'legend.fontsize': font_size,
    'figure.figsize': (7, 3.5),
    'figure.subplot.bottom': 0.2,
    'figure.subplot.left': 0.2,
    'figure.subplot.right': 0.9,
    'figure.subplot.top': 0.85,
    'figure.subplot.wspace': 0.4,
    'text.usetex': False,
}

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']
html_theme_options = {'navigation_with_keys': False}
