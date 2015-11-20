LATEX=pdflatex --shell-escape
BIBTEX=bibtex

PDFFILES=group003.pdf

all: clean $(PDFFILES)

$(PDFFILES): %.pdf: %.tex group003.tex
	$(LATEX) $<
	$(LATEX) $<
	$(LATEX) $<
	$(LATEX) $<

clean:
	rm -f *.aux *.log *.dvi $(PAPER).pdf *.lot *.lof *.toc *.blg *.bbl *.ent *.out
