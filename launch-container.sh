#!/bin/bash
set -eo pipefail
[ -z "${CONTAINER_ENGINE}" ] && CONTAINER_ENGINE=docker

${CONTAINER_ENGINE} run --rm -ti -e DEB_PYTHON_INSTALL_LAYOUT='deb_system' -v $(pwd):/shared -w /shared dolfinx/dolfinx:v0.7.2
