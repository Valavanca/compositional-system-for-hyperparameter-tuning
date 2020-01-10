#!/bin/bash

# -- 1. Convert content files from Latex progect to manubot content folder
for filename in stthesis_latex/content/*.tex; do
    echo "Processing ${filename##*/} "
    pandoc -s $filename -o content/${filename##*/}.md
done

# # -- 2. Copy bibliography
pandoc-citeproc --bib2json stthesis_latex/content/bibliography.bib > content/manual-references.json
# jq -r 'values | .[].id="raw:"+.[].id' content/bibliography.json > content/manual-references.json

# # -- 3. Change reference prefix to md files
for f in $(pandoc-citeproc --bib2json stthesis_latex/content/bibliography.bib | jq 'values | .[].id'); do
    f="${f%\"}"
    f="${f#\"}"
    sed -i -- 's/'${f}'/raw:'${f}'/g' content/*.md
    sed -i -- 's/'${f}'/raw:'${f}'/g' content/manual-references.json
done