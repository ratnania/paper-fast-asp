## Base name of main LaTeX file (change this)
BASE_NAME = main

## Full names of main LaTeX file and pdf output
MAIN_TEX = ${BASE_NAME}.tex
PDF_FILE = ${BASE_NAME}.pdf

## Available commands: make, make clean, make cleanall
all:
	latexmk -pdflatex='pdflatex -synctex=1' -pdf ${MAIN_TEX}

.PHONY:
clean:
	latexmk -c -quiet
	rm -f *.bbl *.spl

.PHONY:
cleanall: clean
	rm -f *.synctex.gz ${PDF_FILE}

