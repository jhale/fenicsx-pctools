import gc
import os

from mpi4py import MPI

import pytest


# Garbage collection
def pytest_runtest_teardown(item):
    item.teardown()
    del item

    gc.collect()
    MPI.COMM_WORLD.barrier()


# Custom command line options for pytest
def pytest_addoption(parser):
    parser.addoption("--noxdmf", action="store_true", help="do not save XDMF output")
    parser.addoption("--overwrite", action="store_true", help="overwrite existing results file")
    parser.addoption(
        "--petscconf",
        metavar="FILE",
        type=str,
        default=None,
        help="path to configuration file with PETSc options",
    )
    parser.addoption(
        "--resultsdir",
        metavar="PATH",
        action="store",
        default=None,
        help="directory where to put the results",
    )
    parser.addoption("--warmup", action="store_true", help="run main solve twice (warm up first)")


@pytest.fixture(scope="session")
def timestamp():
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y-%m-%d_UTC-%H-%M-%S")


@pytest.fixture(scope="module")
def results_dir(timestamp, request):
    foldername = request.config.getoption("resultsdir")
    if foldername is None:
        foldername = os.path.join(request.node.fspath.dirname, f"results_{timestamp}")
    os.makedirs(foldername, exist_ok=True)

    return foldername


# Other useful fixtures
@pytest.fixture(scope="session")
def comm():
    return MPI.COMM_WORLD


@pytest.fixture(scope="module")
def module_dir(request):
    """Return the directory of the current test file."""
    return request.node.fspath.dirname
