#!/usr/bin/env bash

# If you'd like to parallelize, do the following:
# * Create a .env file in this folder
# * Declare GITHUB_TOKENS=token1,token2,token3...

python get_tasks_pipeline.py \
    --repos 'scikit-learn/scikit-learn', 'pallets/flask' \
    --path_prs '<path to folder to save PRs to>' \
    --path_tasks '<path to folder to save tasks to>'

# python get_tasks_pipeline.py \
#     --repos 'scikit-learn/scikit-learn', 'pallets/flask', 'astropy/astropy', 'django/django', 'matplotlib/matplotlib', 'mwaskom/seaborn', 'pallets/flask', 'psf/requests', 'pydata/xarray', 'pylint-dev/pylint', 'pytest-dev/pytest', 'scikit-learn/scikit-learn', 'sphinx-doc/sphinx', 'sympy/sympy' \
#     --path_prs './collected-prs' \
#     --path_tasks './collected-tasks'