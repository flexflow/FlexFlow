.PHONY: all
all: figures/pytorch-tracing.svg figures/local-task-tracing.svg

%.svg : %.puml
	plantuml -tsvg $<
