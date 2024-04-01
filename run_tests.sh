#!/bin/bash -e
docker run --rm -it -v "$(pwd)":/models khulnasoft/models:latest python3 -m pytest startai_models_tests/
