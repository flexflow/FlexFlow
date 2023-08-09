.PHONY: all
all: figures/pytorch-tracing.svg

%.svg : %.puml
	plantuml -tsvg $<
