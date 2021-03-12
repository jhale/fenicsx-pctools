#!/bin/bash
set -eo pipefail
tag=$1
[ $# -eq 0 ] && { echo "Usage: $0 <tag>"; exit 1; }

set -u
script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
docker build \
    -t local/fenics_pctools:${tag} \
    -f ${script_dir}/Dockerfile \
    --target ${tag} \
    ${script_dir}/..
