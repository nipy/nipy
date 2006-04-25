# This makefile must be included from the top-level nipy makefile
# in order for its targets to be built since it depends on variables defined
# there
docs_dest := /usr/share/doc/$(package_name)-$(version)
api_dir := $(docs_dir)/api
python_source_files := $(shell find lib/ -name "*.py")

docs: $(api_dir)

# generage api docs
$(api_dir): $(python_source_files)
	cd $(docs_dir) && ./run-epydoc

# install documentation
install-docs: docs
	mkdir -p $(docs_dest)
	#cp $(docs_dir)/release-notes $(docs_dest)/
	cp -r $(api_dir) $(docs_dest)/

# remove all generated documentation
docs-clean: FORCE
	rm -rf $(api_dir)

FORCE:
