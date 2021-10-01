SHELL := /bin/bash

default: neural.R
	Rscript neural.R

test:
	Rscript neural.R test

