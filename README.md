# CyTOF Data Visualizer

This is a quick javascript implementation of PCA and Robust Sparse K-Means (RSKM). This doesn't really have to be CyTOF data, but any high dimensional dataset. Keep in mind that everything is done in your browser so choosing a huge file may take some time to analyze.

### Usage

1. Your data should be in CSV with column names in first row. Any categorical (non-numerical) columns will be ignored.
2. Once you upload your CSV, it will perform PCA on it and RSKM.
3. Reload file to analyze again.

### Options

- Coloration based on your categorical columns or RSKM clusters
- Can toggle principal component loadings as well

### Problems

- As this was developed for some bioinformatics visualization in the lab, don't expect it to be perfect
- Haven't implemented convergence testing for RSKM
- Progress bar hasn't been implemented

### References/Attribution

1. Robust Sparse K-Means: http://arxiv.org/abs/1201.6082
2. D3.js: https://github.com/mbostock/d3/
3. Numeric.js: http://www.numericjs.com/
4. jQuery: http://jquery.com/
5. Underscore.js: http://underscorejs.org/
6. jQuery Tipsy: http://onehackoranother.com/projects/jquery/tipsy/
