#!/bin/bash
set -eo pipefail
[ -z "${CONTAINER_ENGINE}" ] && CONTAINER_ENGINE=docker

${CONTAINER_ENGINE} run -ti -v $(pwd):/shared -w /shared local/fenics_pctools:latest 
