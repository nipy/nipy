# Automating common tasks for NIPY development

clean:
	find . -regex ".*\.pyc" -exec rm -rf "{}" \;
	find . -regex ".*\.so" -exec rm -rf "{}" \;
	find . -regex ".*~" -exec rm -rf "{}" \;
	find . -regex ".*#" -exec rm -rf "{}" \;
	rm -rf build

dev: clean
	python setup.py build_ext --inplace
	./tools/mynipy

test:
	cd .. && python -c 'import nipy; nipy.test()'

build:
	python setup.py build

install:
	python setup.py install
