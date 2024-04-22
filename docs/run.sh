# /bin/bash!
# Command for producing html docs:

make clean; cd ..; sphinx-apidoc -o docs/source HELPpy; cd docs; make html
cp -r ./build/html/* ../../giordamaug.github.io/myprojects/HELP