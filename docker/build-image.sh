#!/bin/bash
set -eo pipefail
image=$1
target=$2
tag=${3:-$target}
[ $# -le 1 ] && { echo "Usage: $0 <tag> <target> [tag (default: same as target)]"; exit 1; }

set -u
script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
docker build \
    -t local/fenics_pctools:${tag} \
    -f ${script_dir}/Dockerfile \
    --target ${target} \
    --build-arg IMAGE=${image} \
    ${script_dir}/..
