#!/usr/bin/zsh
if [ 1 -ne $# ]; then
    exit 1
fi

black ./src
conda env export | grep -v "^prefix: " > environment.yml
git add . && git commit -m $1 && git push
