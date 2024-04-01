# Retrieve html_theme_options from docs/conf.py
from docs.conf import html_theme_options

html_theme_options["switcher"][
    "json_url"
] = "https://khulnasoft.com/docs/versions/models.json"

repo_name = "models"
