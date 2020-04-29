.PHONY: help venv distclean install-kernel jupyter
.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

help:  ## display this help
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

venv: .venv/bin/python  ## create a version-specific virtual environment

.venv/bin/python:
	conda env create -p .venv -f environment.yml

environment.yml: .venv/bin/python
	conda env export -p .venv/ --from-history  | sed 's/name: null/name: MLforPhysicists/' | grep -v 'prefix: ' > $@

install-kernel: .venv/bin/python  ## install the kernel 'mlcourse' into userspace
	.venv/bin/python -m ipykernel install --user --name=mlphyscourse2020 --display-name="ML for Physicists"

distclean: ## remove venv
	@rm -rf .venv
	jupyter kernelspec remove -f mlphyscourse2020 || echo "No kernelspec"
