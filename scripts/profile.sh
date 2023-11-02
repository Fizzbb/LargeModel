#!/bin/bash
nsys profile -f true \
	     -o resnet \
	     --capture-range cudaProfilerApi \
	     --capture-range-end stop \
	     --export sqlite \
	     python resnet50_ddp.py
