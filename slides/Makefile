# Build the Beamer deck for the ROS 2 Camera Preprocessing Accelerator.
#
# Default target produces slides.pdf via latexmk (preferred) or a fallback
# 2-pass pdflatex. The notes-* targets toggle the speaker-notes Beamer
# option without editing slides.tex.

TEX     := slides.tex
PDF     := slides.pdf
LATEXMK := $(shell command -v latexmk 2>/dev/null)
PDFLATEX := $(shell command -v pdflatex 2>/dev/null)

.PHONY: all clean distclean notes notes-only handout

all: $(PDF)

$(PDF): $(TEX) ../.asserts/Overview.png ../.asserts/Pipeline.png
ifeq ($(strip $(LATEXMK)),)
ifeq ($(strip $(PDFLATEX)),)
	@echo "ERROR: neither latexmk nor pdflatex is on PATH."; \
	echo "Install TeX Live (texlive-latex-recommended texlive-fonts-recommended) or build on Overleaf."; \
	exit 1
else
	$(PDFLATEX) -interaction=nonstopmode $(TEX)
	$(PDFLATEX) -interaction=nonstopmode $(TEX)
endif
else
	$(LATEXMK) -pdf -interaction=nonstopmode $(TEX)
endif

# Render slides with speaker notes appended to each frame.
notes: TEX_OPT := "\PassOptionsToPackage{show notes}{beamer}"
notes:
	@echo "Tip: open slides.tex and uncomment \\setbeameroption{show notes} for a permanent toggle."
	$(MAKE) all

# Render only the notes (notes-only PDF, no slide bodies).
notes-only:
	@echo "Tip: open slides.tex and uncomment \\setbeameroption{show only notes} to render notes-only."

# Render two-up handout (slide + notes on second screen=right).
handout:
	@echo "Tip: open slides.tex and uncomment \\setbeameroption{show notes on second screen=right}."

clean:
	rm -f *.aux *.log *.nav *.out *.snm *.toc *.vrb *.fls *.fdb_latexmk *.synctex.gz

distclean: clean
	rm -f $(PDF)
