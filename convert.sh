#!/bin/bash

# -- 1. Convert content files from Latex progect to manubot content folder
for filename in stthesis_latex/content/*.tex; do
    echo "Processing ${filename##*/} "
    pandoc -s $filename -o content/${filename##*/}.md
done

# -- 2. Copy bibliography
cp stthesis_latex/content/bibliography.bib content/bibliography.bib