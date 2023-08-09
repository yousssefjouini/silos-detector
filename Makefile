.ONESHELL:

init-venv:
	python3 -m venv .venv
	. .venv/bin/activate
	pip install --upgrade pip
	pip install wheel

clean-venv:
	rm -rf .venv

clean-repo:
	rm -rf src/build/ src/silos.egg-info/