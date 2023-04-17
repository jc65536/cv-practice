#!/bin/bash

case $1 in
    build)
        maturin develop -r
        ;;
    
    run)
        pysrc/main.py
        ;;
    
    profile)
        py-spy record --native -o ignore/profile.svg -- python pysrc/main.py
        ;;

