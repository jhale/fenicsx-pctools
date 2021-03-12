#!/bin/bash
set -eo pipefail
image=$1
target=$2
tag=${3:-"local/fenics_pctools:${target}"}
[ $# -le 1 ] && { echo "Usage: $0 <image> <target> [tag (default: same as target)]"; exit 1; }

set -u
script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
docker build \
    -t ${tag} \
    -f ${script_dir}/Dockerfile \
    --target ${target} \
    --build-arg IMAGE=${image} \
    ${script_dir}/..
