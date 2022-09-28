#!/bin/bash
set -eo pipefail
[ -z "${CONTAINER_ENGINE}" ] && CONTAINER_ENGINE=docker

${CONTAINER_ENGINE} run -ti -e DEB_PYTHON_INSTALL_LAYOUT='deb_system' -v $(pwd):/shared -w /shared dolfinx/dolfinx:v0.5.1-r1

