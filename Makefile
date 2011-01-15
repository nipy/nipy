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
	python setup.py build_ext --inplace

test:
	cd .. && python -c 'import nipy; nipy.test()'

build:
	python setup.py build

install:
	python setup.py install

# Update nisext subtree from remote
update-nisext:
	git fetch nisext
	git merge --squash -s subtree --no-commit nisext/master

# Print out info for possible install methods
check-version-info:
	$(PYTHON) -c 'from nisext.testers import info_from_here; info_from_here("nipy")'

