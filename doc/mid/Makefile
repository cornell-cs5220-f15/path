# Add a list of home-made PDF figures here.
PDFFIGURES=figs/*.pdf

.SUFFIXES: .fig .eps .ps .dvi .bbl .bib .tex .pdf

FIGURES=${wildcard *.fig}

all: pdf   

force: rmdone all

dvi: mid.dvi

xdvi: mid.dvi
	xdvi mid.dvi

gv: mid.ps
	gv mid.ps

ps: mid.ps 

pdf: mid.pdf

rmdone:
	rm -f done.flag

done.flag:
	touch done.flag

printerrors:
	@echo "*****************************"; \
	echo Overview of typeset warnings: ; \
        echo "****************************"; \
	grep "Underfull" mid.log; \
	grep "Overfull" mid.log; \
	echo ; \
        echo "***************************"; \
	echo Overview of latex warnings: ; \
        echo "***************************"; \
	grep "LaTeX Warning" mid.log; \
        echo "***************************"; \
	echo ; \
        echo "***************************"; \
	echo Overview of bibtex warnings: ; \
        echo "***************************"; \
        grep ^Warning-- mid.blg; \
	echo "***************************"; \
	echo ; \
        echo "*************************"; \
	echo Overview of latex errors: ; \
        echo "*************************"; \
	grep "^!" mid.log; \
        echo "*************************"; \
	echo

mid.bbl: $(BIBFILES) $(TEXFILES) Makefile done.flag
	pdflatex -interaction=batchmode mid.tex || true
	rm -f $*.pdf
	bibtex -terse $*

mid.pdf: $(TEXFILES) $(PDFFIGFILES) Makefile done.flag
	@echo Running latex once.; \
	while ( pdflatex -interaction=nonstopmode mid.tex; grep "Rerun to get cross" mid.log ) ; do \
          echo; \
          echo "*********************************"; \
          echo "*********************************"; \
          echo Reruning to complete compilation.; \
          echo "*********************************"; \
          echo "*********************************"; \
	done; \
        echo "***************************"; \
	echo DVI file generated.; \
        echo "***************************"; \
	echo ; \
	make printerrors 
			
clean:
	rm -f *.lof *.lot *.pfg *~ *.aux *.dvi *.blg *.log *.bbl *.bak mid.ps mid.pdf *.toc *.swp done.flag mid.out *.upa *.upb


