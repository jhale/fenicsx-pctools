#!/bin/bash
set -eo pipefail
[ -z "${CONTAINER_ENGINE}" ] && CONTAINER_ENGINE=docker

${CONTAINER_ENGINE} run -ti -v $(pwd):/shared -w /shared dolfinx/dolfinx:v0.9.0r1
