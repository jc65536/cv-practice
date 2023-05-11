#!/bin/bash

case $1 in
    build)
        maturin develop -r
        ;;
    
    run)
        pysrc/rectify.py
        ;;
    
    profile)
        flamegraph -F 600 -o ignore/profile.svg -- ./cmd run
        rm perf.data
        ;;
esac
