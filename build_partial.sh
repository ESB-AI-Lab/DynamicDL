source buildvenv/bin/activate
source ~/.bash_profile
rm -rf .docs/_build
rm -rf docs
mkdir docs
cd .docs
make html
cp -R _build/html/* ../docs/
touch ../docs/.nojekyll
deactivate