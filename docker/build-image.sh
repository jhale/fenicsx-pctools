#!/bin/bash
set -eo pipefail
[ -z "${CONTAINER_ENGINE}" ] && CONTAINER_ENGINE=docker

${CONTAINER_ENGINE} build \
		    -t local/fenics_pctools:latest \
		    -f docker/Dockerfile \
		    --target dev .
