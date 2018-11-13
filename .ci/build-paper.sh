LATEXDIR=.
LATEXFILE=selacc.tex
LATEXBASE=`basename $LATEXFILE .tex`

# Install tectonic using conda
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh;
bash miniconda.sh -b -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"
hash -r
conda config --set always_yes yes --set changeps1 no
conda update -q conda
conda info -a
conda create --yes -n paper
source activate paper
conda install -c conda-forge -c pkgw-forge tectonic

# Build the paper using tectonic
cd $LATEXDIR
tectonic $LATEXFILE --web-bundle "https://tectonic.newton.cx/bundles/tlextras-2018.1r0/bundle.tar" --print

# Force push the paper to GitHub
cd $TRAVIS_BUILD_DIR
git checkout --orphan $TRAVIS_BRANCH-pdf
git rm -rf .
git add -f $LATEXDIR/$LATEXBASE.pdf
git -c user.name='travis' -c user.email='travis' commit -m "building the paper"
git push -q -f https://$GITHUB_USER:$GITHUB_API_KEY@github.com/$TRAVIS_REPO_SLUG $TRAVIS_BRANCH-pdf
