"""Build a standalone explorer and a Quarto include from the trained weights."""
from pathlib import Path
import shutil

root = Path(__file__).parent
source, output = root/"interactive", root/"results"
weights = (output/"walk_models.json").read_text().strip()
fragment = (source/"walk_demo.html").read_text().replace("__WALK_MODELS__", weights)
include = ('```{=html}\n<link rel="stylesheet" href="assets/learning_from_convergence/walk_demo.css">\n'
           + fragment + '\n<script src="assets/learning_from_convergence/walk_demo.js"></script>\n```\n')
(output/"_walk_demo.qmd").write_text(include)
standalone = ('<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
              '<meta name="viewport" content="width=device-width, initial-scale=1">'
              '<title>Learning from complete random walks</title>'
              '<link rel="stylesheet" href="walk_demo.css">'
              '<style>body{max-width:780px;margin:2rem auto;padding:0 1rem;font-family:system-ui;line-height:1.6}</style>'
              '</head><body>' + fragment + '<script src="walk_demo.js"></script></body></html>\n')
(output/"walk_demo.html").write_text(standalone)
for name in ("walk_demo.js", "walk_demo.css"):
    shutil.copy2(source/name, output/name)
print("Built the standalone random-walk explorer and Quarto include.")
