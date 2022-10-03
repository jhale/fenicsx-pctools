import pkg_resources

import fenicsx_pctools


def test_version():
    version_installed = pkg_resources.get_distribution("fenicsx_pctools").version
    assert fenicsx_pctools.__version__ == version_installed
