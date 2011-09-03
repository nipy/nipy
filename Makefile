# Automating common tasks for NIPY development

PYTHON = python

clean:
	find . -regex ".*\.pyc" -exec rm -rf "{}" \;
	find . -regex ".*\.so" -exec rm -rf "{}" \;
	find . -regex ".*\.pyd" -exec rm -rf "{}" \;
	find . -regex ".*~" -exec rm -rf "{}" \;
	find . -regex ".*#" -exec rm -rf "{}" \;
	rm -rf build
	$(MAKE) -C doc clean

dev: clean
	$(PYTHON) setup.py build_ext --inplace

test:
	cd .. && $(PYTHON) -c 'import nipy; nipy.test()'

build:
	$(PYTHON) setup.py build

install:
	$(PYTHON) setup.py install

cythonize:
	$(PYTHON) tools/nicythize

# Print out info for possible install methods
check-version-info:
	$(PYTHON) -c 'from nisext.testers import info_from_here; info_from_here("nipy")'

