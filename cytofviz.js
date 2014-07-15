
// Some globals
var xScale, yScale;
var showLoadings = false;
var loadings = []; // Dict containing Marker to PC loading map
var colorScheme;
var pointColor = d3.scale.category10();
var contPointColor;
var descriptions = {
    "PC1" : "PC1", 
    "PC2" : "PC2", 
    "PC3" : "PC3", 
    "PC4" : "PC4", 
    "PC5" : "PC5"
  }
var xAxisOptions = ["PC1", "PC2", "PC3", "PC4", "PC5"];
var yAxisOptions = ["PC1", "PC2", "PC3", "PC4", "PC5"];
var data = {};
var numeric_keys = [];
var pBar = document.getElementById('progress_bar');
var pcaData = [];

function updateProgress(value) {
  
  /* TODO: This isn't displaying correctly.

  if (value >= 100) {
    document.getElementById('progress_bar_div').style.display = 'none';
  } else {
    document.getElementById('progress_bar_div').style.display = 'block';
  }
  
  pBar.value = Math.round(value);
   //console.log("Progress: " + Math.round(value));
  pBar.getElementsByTagName('strong')[0].innerHTML = Math.round(value);
  */
}

function isNumber(n) {
  return !isNaN(parseFloat(n)) && isFinite(n);
}

function PCA(d) {
  // Call PCA function using data (d)
  // d will be preproccessed to remove non-numeric fields

  var columns = _.keys(d[0]);
  var vals = _.values(d[0]);
  var numerics = []; // holds names of numeric columns
  var non_numerics = [];
  var data_len = d.length;

  _.forEach(vals, function(v, k, l) {
      if(isNumber(v)) {
        numerics.push(columns[k]);
      } else {
        non_numerics.push(columns[k]);
      }

    });

  // Generate new dataset to pass to PCA
  // and get rid of invalid numbers
  console.log("\tConverting PCA data of length " + data_len + "...");
  for(var i = 0; i < data_len; i++) {
    var o = [];
    for(var j = 0; j < numerics.length; j++) {
      isNumber(d[i][numerics[j]]) ? o.push(d[i][numerics[j]]) : o.push(0.0);
    }
    pcaData.push(o);
    if(i % 20 == 0) {
      //console.log(o);
      updateProgress(i/data_len * 25);
    }
  }


  // Get num_cols, num_rows
  var num_cols = numerics.length;
  var num_rows = pcaData.length;
  console.log("\tPerforming PCA on " + num_rows + " samples and " + num_cols + " numeric variables...");

  // Scale/Normalize   dataset (Feature Scaling)
  console.log("\t\tNormalizing...");
  updateProgress(0);
  var bounds = getBounds(pcaData,1);
  for(var j = 0; j < num_cols; j++) {
    var span = bounds[j]['max'] - bounds[j]['min'];
    numeric.setBlock(pcaData, [0,j], [num_rows-1,j], 
      numeric.div(numeric.sub(numeric.getBlock(pcaData, [0,j], [num_rows-1,j]), bounds[j]['min']), span));
    updateProgress(j/num_cols * 25 + 25);
  }

  // Center dataset
  console.log("\t\tSubtracting means...");
  for(var j = 0; j < num_cols; j++) {
    var colMean = numeric.sum(numeric.getBlock(pcaData, [0,j], [num_rows-1,j])) / num_rows;
    numeric.setBlock(pcaData, [0,j], [num_rows-1,j], 
      numeric.sub(numeric.getBlock(pcaData, [0,j], [num_rows-1,j]), colMean));
  }

  /* Use numeric library to run PCA
   *    1. SVD pcaData matrix ==> USV
   *    2. Scores/mappings are the Us where U is left sing vec and s is corresponding sing val
   *    3. Select only first 5 PCs
   *    
   *    - A^TA is the CovMat of A (when columns are variables, rows are samples)
   *    - A = USV^T
   *    - A^TA = (USV^T)^T(USV^T) = (VSU^T)(USV^T) = VS(U^TU)SV^T = VS^2V^T
   *    - Thus V has the eigenvectors of A^TA and therefore the PCs of A
   *    - US is the principal component scores (weighted projection of data onto the principal components)
   */
  var svd = numeric.svd(pcaData);
  var pca = numeric.dot(svd.U, numeric.diag(svd.S.slice(0,5)));
  updateProgress(75);

  console.log("\tPCA done. Generating new data table...");

  // Recreate array of dicts for D3 processing
  // Restore everything and add on PCs
  var returnData = [];
  for(var i = 0; i < num_rows; i++) {
    var tmp = {};
    for(var j = 0; j < (num_cols + non_numerics.length); j++) {
      tmp[columns[j]] = d[i][columns[j]];
    }
    for(var j = 0; j < 5; j++) {
      // Add PCs
      tmp["PC"+(j+1)] = pca[i][j];
    }
    returnData.push(tmp);
    if(i % 20) updateProgress(i/num_rows * 25 + 75);
  }
  
  console.assert(_.keys(returnData[0]).length == (num_cols + non_numerics.length + 5), 
    "PCA Processing error: " + (_.keys(returnData[0]).length-5) + 
    " columns in post-pca data, " + 
    (num_cols + non_numerics.length) + " columns in pre-pca data.");

  console.log("\tData table generated. " + _.keys(returnData[0]).length + " columns. One row follows:\n\t");
  console.log(returnData[0]);
  console.log("\n\n");

  updateProgress(100);
  // Set loadings global var
  // X = USVt 
  var loadingsMatrix = numeric.dot( numeric.diag(svd.S.slice(0,5)), 
    numeric.transpose( numeric.getBlock(svd.V, [0,0], [num_cols-1, 4]) ) );
  loadingsMatrix = numeric.transpose(loadingsMatrix);
  loadingsMatrix = numeric.mul(loadingsMatrix, 0.05); // Reduce the size of loadings
  for(var i = 0; i < numerics.length; i++) {
    loadings.push({'Marker': numerics[i], 
                  'PC1': loadingsMatrix[i][0],
                  'PC2': loadingsMatrix[i][1],
                  'PC3': loadingsMatrix[i][2],
                  'PC4': loadingsMatrix[i][3],
                  'PC5': loadingsMatrix[i][4]
                });
  }
  return returnData;
}

function dist(a, b, w) {
  /*
   *  Find the weighted Euclidian distance between a and b
   *  - a and b are arrays
   *  - w is array of weights
   */

   console.assert(a.length == b.length, "Distance Calculation Error: Vectors are of different dimensionality");
   if(a.length != b.length) return -1;
   
   // Create weight vector if it wasn't given
   if(w === undefined) {
    //console.log("Distance Calculation: Creating weight vector");
    w = Array(a.length);
    for(var i = 0; i < a.length; i++) w[i] = 1.0
   }

   console.assert(a.length == w.length, 
    "Distance Calculation Error: Weight vector of different dimensionality \n\t w is " + 
    w.length + " and a is " + a.length);
   if(a.length != w.length) return -1;

   var d = 0.0;
   if(a.length == undefined) { // Just a single num
    d = w ? w * Math.pow(a - b, 2) : 0.0; // Make sure w is non-0
   } else {
    for(var i = 0; i < a.length; i++) {
      d += (w[i] * Math.pow(a[i] - b[i], 2)); 
    }
   }
   d = Math.sqrt(d);

   console.assert(d >= 0, "Distance Calculation Error: Invalid distance (" + d + ")");

   return d;
}

function RSKC(k, alpha, L1_limit) {
  /* Robust Sparse k-means
   *  (http://arxiv.org/pdf/1201.6082v1.pdf)
   *  - requires pcaData to be a numeric 2D matrix
   *  - k = number clusters
   *  - alpha = proportion of cases to be trimmed
   *  - L1 = L1 bound on weights (small means few non-zeros)
   */

  console.assert(k >= 1, "RSKC Error: k must be 1 or greater");
  console.assert(alpha >= 0 && alpha <= 1, "RSKC Error: alpha must be between 0 and 1");
  console.assert(L1_limit> 0, "RSKC Error: L1 must be greater than 0");

  var observations = pcaData.length;
  console.assert(observations >= 1, "RSKC Error: Fewer than 1 observations");

  var dimensions = pcaData[0].length;
  console.assert(dimensions >= 1, "RSKC Error: Dimensionality is less than 1");

  var bounds = getBounds(pcaData, 1);
  console.assert(bounds.length != dimensions, "RSKC Error: Getting bounds of data failed");

  if(k < 1 || !(alpha >= 0 && alpha <= 1) || 
    L1_limit <= 0 || !(observations >= 1) || 
    !(dimensions >= 1) ||
    !(bounds.length != dimensions)) return -1;

  var iter = 5;
  var totalCalculations = iter * dimensions * (observations/2);

  console.log("Beginning robust sparse k-means\n\t" + k + " clusters with " 
    + observations 
    + " observations in " 
    + dimensions 
    + " dimensions");



  console.time("Total RSKC Time");

  // Initialize
  var w = numeric.linspace(1.0/Math.sqrt(dimensions), 1.0/Math.sqrt(dimensions), dimensions);
  var centroids = [];
  var epsilon = 0.01; // Threshold for convergence

  for(var RSKMiter = 0; RSKMiter < iter; RSKMiter++) { // Overall RSKM Loop

    /****** Trimmed Weighted k-means *******/

    // Random centroids 

    var error = Number.POSITIVE_INFINITY; // Current overall error
    
    console.time("Trimmed K-Means");

    for(var i = 0; i < k; i++) {
      var centroid = Array(dimensions);
      for(var j = 0; j < dimensions; j++) {
        centroid[j] = (Math.random() * (bounds[j]['max'] - bounds[j]['min'])) + bounds[j]['min'];
      }
      centroids.push(centroid.slice());
    }

    var clusters = Array(observations); // Vector where C(i) is the cluster assignment of obs i. -1 means Trimmed
    var lastClusters = numeric.linspace(-1,-1,observations); // Stores the last list of cluster assignments for determining convergence
    var distances = Array(observations); // Vector where D(i) is obs i's distance to its centroid
      
    // Vector of weights subject to L2 <= 1 and L1 <= L1 param


    /** (a) **/
    for(var l = 0; l < iter && error > epsilon; l++) { // Main iteration loop until convergence
    
      /** (i) **/

      // Assign observations to nearest centroid by Euclidian dist
      // This is pretty expensive O(obs*k)
      for(var i = 0; i < observations; i++) {
        var minD = Number.POSITIVE_INFINITY;
        var nearestCluster = -1;
        for(var j = 0; j < k; j++) {
          var d = dist(pcaData[i], centroids[j], w);
          if(d < minD) {
            minD = d;
            nearestCluster = j;
          }
        }
        console.assert(nearestCluster >= 0, "RSKC Error: Could not find nearest cluster");
        if(!(nearestCluster >= 0)) return -1;
        clusters[i] = nearestCluster;
        distances[i] = minD;
      }

      /** (ii) **/

      // Trim the alpha*100% of the observations
      var sortedDistances = distances.slice();
      sortedDistances.sort();
      var numTrim = Math.round(alpha * observations);
      var removeObs = sortedDistances.slice(0, numTrim-1);

      for(var i = 0; i < numTrim; i++) {
        // Find and remove obs from cluster (by setting it to -1)
        clusters[distances.indexOf(removeObs[i])] = -1;
      }


      // Recalculate centroids
      var obsToProcess = numeric.linspace(0, observations-1, observations); 
      // Stores indices of observations still to process

      for(var i = 0; i < k; i++) {
        var centroid = numeric.linspace(0,0,dimensions);
        var numMembers = 0;

        for(var j = 0; j < obsToProcess.length; j++) {
          if(clusters[obsToProcess[j]] == i) {
            // Update to weighted sample mean
            numeric.addeq( centroid, numeric.mul(pcaData[obsToProcess[j]], w) );
            numMembers++;

            // Remove this obs from list to process
            obsToProcess.splice(j,1);

            j--; // Correct for shift in array
          }
        }

        //console.assert(numMembers > 0, "RSKC Note: Centroid " + i + " has no members");
        
        if(numMembers > 0) {
          numeric.diveq(centroid, numMembers);
          //console.log("\t\tNew Centroid at " + centroid);
        } else {
          //console.log("\t\tKeeping existing centroid at " + centroids[i]);
        }

        console.assert(_.every(centroid, function(d) {return !isNaN(d);}), "RSKC Error: NaN in centroid.");
        if(!(_.every(centroid, function(d) {return !isNaN(d);}))) return -1;

        if(numMembers > 0) centroids[i] = centroid.slice(); // deep copy
        // Otherwise, it stays where it is
      }

      // Compare to last cluster assignment to determine convergence
      //    using hamming dist

      error = 0;
      for(var i = 0; i < observations; i++) {
        if(clusters[i] != lastClusters[i]) error++;
      }
      error = error / observations;

      //console.log("\tTrimmed K-Means Change: " + error);

      lastClusters = clusters.slice();
    }

    
    var wtClusters = clusters.slice() // Store weighted cluster assignments

    console.timeEnd("Trimmed K-Means");


    /****** Trimmed Unweighted K-Means *******/

    console.time("Trimmed Unweighted K-Means");

    /** (c) **/

    var uwCentroids = []; // A vector where i'th entry is unweighted centroid i coordinates
    obsToProcess = numeric.linspace(0, observations-1, observations); 

    // Calculate unweighted centroid coordinates
    for(var i = 0; i < k; i++) {
      var centroid = numeric.linspace(0,0,dimensions);
      var numMembers = 0;

      for(var j = 0; j < obsToProcess.length; j++) {
        if(clusters[obsToProcess[j]] == i) {
          // Add unweighted sample 
          numeric.addeq(centroid, pcaData[obsToProcess[j]]);

          numMembers++;

          // Remove this obs from list to process
          obsToProcess.splice(j,1);

          j--; // Correct for shift in array
        }
      }

      //console.assert(numMembers > 0, "RSKC Note: Centroid " + i + " has no members");
      
      if(numMembers > 0) {
        numeric.diveq(centroid, numMembers);
        //console.log("\t\tNew Centroid at " + centroid);
      } else {
        //console.log("\t\tKeeping existing centroid at " + centroids[i]);
      }

      console.assert(_.every(centroid, function(d) {return !isNaN(d);}), "RSKC Error: NaN in centroid.");
      if(!(_.every(centroid, function(d) {return !isNaN(d);}))) return -1;

      if(numMembers > 0) uwCentroids[i] = centroid.slice(); // deep copy
      // Otherwise, it stays where it is
    }

    // Calculate unweigthed distances to unweighted centroids

    var uwDistances = []; // Unweighted dist of obs[i] to its cluster
    for(var i = 0; i < observations; i++) {
      var minD = Number.POSITIVE_INFINITY;
      var nearestCluster = -1;

      for(var j = 0; j < k; j++) {

        var d = dist(pcaData[i], centroids[j]); // UW dist

        if(d < minD) {
          minD = d;
          nearestCluster = j;
        }
      }

      console.assert(nearestCluster >= 0, "RSKC Error: Could not find nearest cluster");
      if(!(nearestCluster >= 0)) return -1;

      clusters[i] = nearestCluster;
      uwDistances[i] = minD;
    }
    
    // Trim the alpha*100% of the observations
    var sortedDistances = uwDistances.slice();
    sortedDistances.sort();
    var removeObs = sortedDistances.slice(0, numTrim-1);

    for(var i = 0; i < numTrim; i++) {
      // Find and remove obs from cluster (by setting it to -1)
      clusters[uwDistances.indexOf(removeObs[i])] = -1;
    }

    /** (d) **/

    // Make union of excluded clusters from unweighted and weighted
    var totalExcluded = 0;
    for(var i = 0; i < observations; i++) {
      if(wtClusters[i] == -1) clusters[i] = -1;
      if(clusters[i] == -1) totalExcluded++;

    }
    console.assert(observations > totalExcluded, "RSKC Error: Excluded more data points than observations");
    if(observations <= totalExcluded) return -1;

    // At the end of all this, clusters should have a -1 at i 
    // if obs i was trimmed in either weighted or unweighted k means

    console.timeEnd("Trimmed Unweighted K-Means");

    /** (e) **/

    /****** Find new set of weights without trimmed points *******/
    /*    Maximize B metric subject to ||w|| = 1, |w| = l  
     *    Can be written as... maximize (wTa) where a(j) = B metric wrt feature j
     *    Subject to ||w|| <= 1, |w| = l, w(j) >= 0
     *    This is Convex! Thus, can be solved by soft-thresholding (Witten & Tibshirani, 2010)
     **/

    console.time("Weight Vector Calculation");

    error = Number.POSITIVE_INFINITY;
    
    var distMatrix = Array();
    // Initialize per-feature distance matrices
    for(var i = 0; i < dimensions; i++) {
        distMatrix.push(numeric.identity(observations)); 
    }

    for(var l = 0; l < iter && error > epsilon; l++) {
      // Calculate B metric without the outliers

      console.log("\tWeight Vector Iteration[" + l + "]");

      for(var wCheck = 0; wCheck < dimensions; wCheck++) {
        if(isNaN(w[wCheck]) || !isFinite(w[wCheck]) || w[wCheck] < 0) {
          console.log("\tWeight Vector Invalid");
          return -1;
        }
      }

      var featB = numeric.linspace(0,0,dimensions); // B metric (mean dist overall - sum mean clust dist) per feature
      var numIncluded = 0; // Number of obs not excluded
      

      // Store distances between all points for each feature/dim

      // Distance between all points by feature/weight
      //    Horribly inefficient... O(pn^2) [ p=feat ]

      /* 
      for(var i = 0; i < observations; i++) {
        if(clusters[i] == -1) continue;

        console.log("\t\t\tTotal Distance Per Feature Progress: " + 
          (Math.round(parseFloat(i)/observations * 1000))/1000 + "%");

        numIncluded++;
        for(var j = i+1; j < observations; j++) {
          if(clusters[j] != -1) {
            for(var p = 0; p < dimensions; p++) {

              if(w[p] == 0.0) {
                var d = 0.0;
              } else {
                var d = dist(pcaData[i][p], pcaData[j][p], w[p]);
              }
              
              console.assert(d >= 0, "RSKC Error: Invalid distance returned");
              if(d < 0 || isNaN(d)) return -1;

              distMatrix[p][i][j] = d;
              featB[p] += d;
            }
          }
        }
      }
      */

      // Some optimization
      var p = 0, i = 0, j = 0, d = 0;
      var curWp = -1; // For the current weight of dimension p
      var curP = -1; // Reference to dist matrix for Current dimension
      var curObs = -1; // Reference to observation row in dist matrix we're looking at 

      for(p = 0; p < dimensions; p++) {
        curWp = w[p];
        curP = distMatrix[p];
        var curFeatB = 0.0; // Stores the overal distance for the feature

        for(i = 0; i < observations; i++) {
          if(clusters[i] == -1) continue;

          if(!p) numIncluded++; // Only increment for first dimension...not for each

          if(curWp == 0.0) continue;

          curObs = curP[i];

          for(j = i+1; j < observations; j++) {
            if(clusters[j] != -1) {
                // d = dist(pcaData[i][p], pcaData[j][p], curWp); // eliminate function call to opt
                d = Math.sqrt(Math.pow(pcaData[i][p],2) + Math.pow(pcaData[j][p],2)) * curWp;
                //console.assert(d >= 0, "RSKC Error: Invalid distance returned"); // Slows this down A LOT
                //if(d < 0 || isNaN(d)) return -1;  // Slows this down A LOT

                curObs[j] = d;
                curFeatB += d;

            }
          }

          updateProgress(parseFloat(p)/dimensions * 100);
        }

        featB[p] = curFeatB * 2; // Since by def, sum of dist is over Obs x Obs dist matrix
      }
      

      // Calculate mean distance per feature
      numeric.diveq(featB, numIncluded); 
      //console.log("\t\tOverall Feature Distances for " + numIncluded + " observations: " + featB);

      // Create lists of cluster members
      var clusterMembers = Array();
      var obsToProcess = clusters.slice();

      for(i = 0; i < k; i++) {
        clusterMembers[i] = Array();
      }

      for(j = obsToProcess.pop(); j != undefined; j = obsToProcess.pop()) {
          if(clusters[j] > -1) clusterMembers[clusters[j]].push(j);
      }

      
      // Calculate Intra-Cluster distance by feature
      for(p = 0; p < dimensions; p++) {
        var intraClusterTotal = 0.0;
        curP = distMatrix[p];

        for(i = 0; i < k; i++) {
          var intraCluster = 0.0;
          var curClust = clusterMembers[i]; 

          if(clusterMembers[i].length < 1) {
            continue; // No members of this cluster
          }

          // Get distance between all members of the cluster
          var curClusterMembers = clusterMembers[i].length;
          var currentClusterMemberRow = null;

          for(var j = 0; j < curClusterMembers; j++) {
            currentClusterMemberRow = curP[curClust[j]];
            for(var q = j+1; q < curClusterMembers; q++){
              intraCluster += currentClusterMemberRow[curClust[q]];
            }

          }

          intraClusterTotal += ((intraCluster * 2) / curClusterMembers); // Again, def has all n x n distances
          /*
          if(isNaN(intraClusterTotal)) {
                console.log("RSKC Error: clusterMembers[" + i + "] invalid; value = " + clusterMembers[i]);
                return -1;
          }
          */
          //console.log("\t\tfeature " + p + ", cluster " + i + ", " + clusterMembers[i].length + " members: " + ((intraCluster * 2) / clusterMembers[i].length));
        }
        


        if(featB[p] < 0) {
          console.log("RSKC Error: Negative overall feature distance Intra-Cluster");
          return -1;
        }

        // B(p) = (Total for feature p) - (sum of mean cluster dist for p)

        //console.log("\t\t\tIntra-Cluster Distance Per Feature Progress: " + 
        // (parseFloat(p)/dimensions * 100) + "%");
      }
      
      //console.log("\t\tBetween Cluster Distance Sum: " + featB);

      /* Soft-thresholding for w update
       *    Soft(g,T) = sign(g)*(|g| - T)+
       *    (a)+ ==> 0 if a < 0, a otherwise
       * Update Rule:
       *    w[j] = (Soft(featB[j],d))/||Soft(featB,d)|| 
       *    where d = 0 if results in |w| < L1_limit, 
       *    or d chosen so |w| = L1_limit
       */

      var nW = []; // New weight vector
      var L1 = 0.0;
      var L2 = 0.0;

      // First try d = 0
      for(var i = 0; i < dimensions; i++) {
        nW.push(featB[i]); 
        L2 += Math.pow(nW[i],2);
      }
      L2 = Math.sqrt(L2);
      
      numeric.diveq(nW, L2);

      for(var i = 0; i < dimensions; i++) {
        L1 += Math.abs(nW[i]);
      }

      if(L1 < L1_limit) { // Within L1 threshold, so we're done
        w = nW.slice();
        //console.log("\tNew w [L1 = " + L1 + "]: " + w);
        //console.log("\tCurrent Feature-based BCSM: " + Math.sqrt(numeric.dot(featB, featB)));
        //console.log("\tCurrent Feature-based BCSM: " + Math.sqrt(numeric.dot(featB, featB)));
        continue;
      }

      // Try d to get |w| = L1_limit
      nW = []; // new weight vector
      L1 = 0.0;
      L2 = 0.0;

      for(var i = 0; i < dimensions; i++) L1 += Math.abs(featB[i]);
      var delta = (L1 - L1_limit) / dimensions;

      for(var i = 0; i < dimensions; i++) {
        nW.push( (featB[i] < 0) ? (((Math.abs(featB[i]) - delta) > 0) ? (-Math.abs(featB[i]) - delta) : 0) : 
         (((Math.abs(featB[i]) - delta) > 0) ? (Math.abs(featB[i]) - delta) : 0) );
        L2 += Math.pow(nW[i],2);
      }
      L2 = Math.sqrt(L2);
    
      numeric.diveq(nW, L2); // Resulting w can have negative entries, so coerce them to 0
      
      L1 = 0.0;
      for(var i = 0; i < dimensions; i++) L1 += Math.abs(nW[i]); // Recalc L1
    
      w = nW.slice();

      //console.log("\tNew w [L1 = " + L1 + "]: " + w);
      //console.log("\tCurrent Feature-based BCSM: " + Math.sqrt(numeric.dot(featB, featB)));

    }

    console.timeEnd("Weight Vector Calculation");  
  }

  // Finally, assign observations to nearest centroid by Euclidian dist
  // This is pretty expensive O(obs*k)
  for(var i = 0; i < observations; i++) {
    var minD = Number.POSITIVE_INFINITY;
    var nearestCluster = -1;
    for(var j = 0; j < k; j++) {
      var d = dist(pcaData[i], centroids[j], w);
      if(d < minD) {
        minD = d;
        nearestCluster = j;
      }
    }
    console.assert(nearestCluster >= 0, "RSKC Error: Could not find nearest cluster");
    if(!(nearestCluster >= 0)) return -1;
    clusters[i] = nearestCluster;
    distances[i] = minD;
  }

  console.timeEnd("Total RSKC Time");
  console.log("RSKC Complete");
  //console.log('Clusters\n\n' + clusters + "\n\n" + w);

  return {'Clusters' : clusters, 'Weights' : w};
}

function parseData(d) {
  var columns = _.keys(d[0]);
  var vals = _.values(d[0]);
  var numerics = []; // holds names of numeric columns
  numeric_keys = numerics;

  _.forEach(vals, function(v, k, l) {
      if(isNumber(v)) 
        numerics.push(columns[k]);
    });

  return _.map(d, function(d) {
    var o = {};
    _.each(columns, function(k) {
      if( $.inArray(k, numerics) < 0 ) {
        o[k] = d[k];
      } else {
        o[k] = parseFloat(d[k]);
      }
    });
    return o;
  });
}

function getBounds(d, paddingFactor) {
  // Find min and maxes (for the scales)
  paddingFactor = typeof paddingFactor !== 'undefined' ? paddingFactor : 1;

  var keys = _.keys(d[0]), b = {};
  _.each(keys, function(k) {
    b[k] = {};
    _.each(d, function(d) {
      if(isNaN(d[k]))
        return;
      if(b[k].min === undefined || d[k] < b[k].min)
        b[k].min = d[k];
      if(b[k].max === undefined || d[k] > b[k].max)
        b[k].max = d[k];
    });
    b[k].max > 0 ? b[k].max *= paddingFactor : b[k].max /= paddingFactor;
    b[k].min > 0 ? b[k].min /= paddingFactor : b[k].min *= paddingFactor;
  });
  return b;
}

function getColumn(d, columnName) {
  // Return a column from a 2D array of arrays
  var newCol = [];

  d.forEach(function(r) {
    newCol.push(r[columnName]);
  });

  return newCol;
}

function loadData(datastring) {
  
  // First clear all existing data
  xScale = '';
  yScale = '';
  showLoadings = false;
  loadings = []; // Dict containing Marker to PC loading map
  colorScheme = '';
  pointColor = d3.scale.category10();
  contPointColor = '';
  data = {};
  numeric_keys = [];
  pcaData = [];

  console.log("Parsing CSV file...");
  data = d3.csv.parse(datastring);

  // Check to make sure last row isn't empty
  // This can happen when there's more newlines?
  var nullCount = 0;
  _.each(_.keys(data[data.length-1]), function(k) {
    if (data[data.length-1][k] == "") nullCount++;
  });
  if (nullCount >= _.keys(data[data.length-1]).length-1) data.pop(); // get rid of last row

  data = parseData(data);
  var keys = _.keys(data[0]); // Pre PCA and RSKC

  // Determine if data is PCA processed already
  if (data[0]['PC1'] != null) {
    console.log("Data has PCA transform applied.");
  } else {
    console.log("Performing PCA on data...");
    data = PCA(data);
  }
  
  // Perform RSKC and join assignments to data
  console.log("Performing RSKC on data...");

  var rskcResults = RSKC(4, 0.05, 2);
  var clusterNames = {0:'A', 1:'B', 2:'C', 3:'D', '-1': 'Outliers'};

  for(var i = 0; i < data.length; i++) {
    data[i]['RSKC'] = clusterNames[rskcResults['Clusters'][i]];
    //if(!(i % 50)) console.log(data[i]);
  }
  
  
  var xAxis = "PC1", yAxis = "PC2";
  var vals = _.values(data[0]);
  var numerics = []; // holds names of numeric columns
  var non_numerics = [];

  _.forEach(vals, function(v, k, l) {
      if(isNumber(v)) {
        numerics.push(keys[k]);
      } else {
        non_numerics.push(keys[k]);
      }
    });

  colorScheme = (non_numerics.length != 0) ? non_numerics[0] : numerics[0];

  var colorOptions = keys;
  colorOptions.push('RSKC');
  
  var bounds = getBounds(data, 1);

  var svg = d3.select("#chart")
    .append("svg")
    .attr("width", 1000)
    .attr("height", 640);

  svg.append('g')
    .classed('chart', true)
    .attr('transform', 'translate(80, -60)');

  d3.select('#x-axis-menu')
    .selectAll('li')
    .data(xAxisOptions)
    .enter()
    .append('li')
    .text(function(d) {return d;})
    .classed('selected', function(d) {
      return d === xAxis;
    })
    .on('click', function(d) {
      xAxis = d;
      updateChart(false, bounds, xAxis, yAxis);
      updateMenus(xAxis, yAxis);
    });

  d3.select('#y-axis-menu')
     .selectAll('li')
     .data(yAxisOptions)
     .enter()
     .append('li')
     .text(function(d) {return d;})
     .classed('selected', function(d) {
       return d === yAxis;
     })
     .on('click', function(d) {
       yAxis = d;
       updateChart(false, bounds, xAxis, yAxis);
       updateMenus(xAxis, yAxis);
     });

  d3.select('#loadings-menu')
     .selectAll('li')
     .data(["Yes", "No"])
     .enter()
     .append('li')
     .text(function(d) {return d;})
     .classed('selected', function(d) {
        if (showLoadings) {
          return d === 'Yes';
        } else {
          return d === 'No';
        }
     })
     .on('click', function(d) {
       showLoadings = d === 'Yes' ? true : false;
       updateLoadings(xAxis, yAxis);
       updateChart(false, bounds, xAxis, yAxis);
       updateMenus(xAxis, yAxis);
     });

  d3.select('#color-menu')
     .selectAll('option')
     .data(colorOptions)
     .enter()
     .append('option')
     .text(function(d) {return d;});

  d3.select('#color-menu')
    .on('change',function(d) {
       colorScheme = colorOptions[this.selectedIndex];
       updateChart(false, bounds, xAxis, yAxis);
       updateMenus(xAxis, yAxis);
     });

  d3.select('svg g.chart')
    .append('text')
    .attr({'id': 'Sample', 'x': 0, 'y': 170})
    .style({'font-size': '40px', 'font-weight': 'bold', 'fill': '#ddd'});

  d3.select('svg g.chart')
    .append('text')
    .attr({'id': 'xLabel', 'x': 400, 'y': 670, 'text-anchor': 'middle'})
    .text(descriptions[xAxis]);

  d3.select('svg g.chart')
    .append('text')
    .attr('transform', 'translate(-60, 330)rotate(-90)')
    .attr({'id': 'yLabel', 'text-anchor': 'middle'})
    .text(descriptions[yAxis]);

  // Render points
  updateScales(bounds, xAxis, yAxis);

  contPointColor = d3.scale.linear()
    .domain([bounds[numerics[0]].min, bounds[numerics[0]].max])
    .range(["darkblue", "yellow"])
    .interpolate(d3.interpolateLab);

  d3.select('svg g.chart')
    .selectAll('circle') 
    .data(data)
    .enter()
    .append('circle')
    .attr('cx', function(d) {
      return isNaN(d[xAxis]) ? d3.select(this).attr('cx') : xScale(d[xAxis]);
    })
    .attr('cy', function(d) {
      return isNaN(d[yAxis]) ? d3.select(this).attr('cy') : yScale(d[yAxis]);
    })
    .attr('fill', function(d, i) {return pointColor(i);})
    .style('cursor', 'pointer');

  // Tooltips
  $('svg circle').tipsy({ 
        gravity: 'w', 
        html: true, 
        title: function() {
          var d = this.__data__;
          var ttip = "";
          for(var i = 0; i < non_numerics.length; i++) {
            ttip = ttip + "<br> " + d[non_numerics[i]];
          }
          return ttip; 
        }
  });

  updateChart(true, bounds, xAxis, yAxis);
  updateMenus(xAxis, yAxis);

  // Render axes
  d3.select('svg g.chart')
    .append("g")
    .attr('transform', 'translate(0, 630)')
    .attr('id', 'xAxis')
    .call(makeXAxis);

  d3.select('svg g.chart')
    .append("g")
    .attr('id', 'yAxis')
    .attr('transform', 'translate(-10, 0)')
    .call(makeYAxis);
}

function updateChart(init, bounds, xAxis, yAxis) {
  updateScales(bounds, xAxis, yAxis);
  updateLoadings(xAxis, yAxis);

  d3.select('svg g.chart')
    .selectAll('circle')
    .transition()
    .duration(500)
    .ease('quad-out')
    .attr('fill', function(d) {
      if ($.inArray(colorScheme, numeric_keys) < 0) { // Categorical color scheme
        return  pointColor(d[colorScheme]);
      } else {
        contPointColor = d3.scale.linear()
        .domain([bounds[colorScheme].min, bounds[colorScheme].max])
        .range(["darkblue", "yellow"])
        .interpolate(d3.interpolateLab);
        return contPointColor(d[colorScheme]);
      }
    })
    .attr('cx', function(d) {
      return isNaN(d[xAxis]) ? d3.select(this).attr('cx') : xScale(d[xAxis]);
    })
    .attr('cy', function(d) {
      return isNaN(d[yAxis]) ? d3.select(this).attr('cy') : yScale(d[yAxis]);
    })
    .attr('r', function(d) {
      return isNaN(d[xAxis]) || isNaN(d[yAxis]) ? 0 : 5;
    });

  // Also update the axes
  d3.select('#xAxis')
    .transition()
    .call(makeXAxis);

  d3.select('#yAxis')
    .transition()
    .call(makeYAxis);

  // Update axis labels
  d3.select('#xLabel')
    .text(descriptions[xAxis]);

  d3.select('#yLabel')
    .text(descriptions[yAxis]);

  // Update legend
  d3.select('div#chart svg g.chart')
    .selectAll('.legend')
    .remove();

  if ($.inArray(colorScheme, numeric_keys) < 0) {
    // Legend only applies to categorical schemes

    var legend = d3.select('div#chart svg g.chart').selectAll(".legend")
      .data(_.uniq(getColumn(data, colorScheme)))
      .enter().append("g")
      .attr("class", "legend")
      .attr("transform", function(d, i) { return "translate(0," + (i * 20.0 + 60.0) + ")"; });

    legend.append("rect")
      .attr("x", 900 - 18)
      .attr("width", 18)
      .attr("height", 18)
      .style("fill", function(d) { 
        return pointColor(d); 
      });

    legend.append("text")
      .attr("x", 900 - 24)
      .attr("y", 9)
      .attr("dy", ".35em")
      .style("text-anchor", "end")
      .text(function(d) { return d; });
  }

  }

  function updateScales(bounds, xAxis, yAxis) {

    xScale = d3.scale.linear()
                    .domain([bounds[xAxis].min, bounds[xAxis].max])
                    .range([20, 780]);

    yScale = d3.scale.linear()
                    .domain([bounds[yAxis].min, bounds[yAxis].max])
                    .range([600, 100]);    
  }

  function makeXAxis(s) {
    s.call(d3.svg.axis()
      .scale(xScale)
      .orient("bottom"));
  }

  function makeYAxis(s) {
    s.call(d3.svg.axis()
      .scale(yScale)
      .orient("left"));
  }

  function updateMenus(xAxis, yAxis) {
    d3.select('#x-axis-menu')
      .selectAll('li')
      .classed('selected', function(d) {
        return d === xAxis;
      });
    d3.select('#y-axis-menu')
      .selectAll('li')
      .classed('selected', function(d) {
        return d === yAxis;
    });
    d3.select('#color-menu')
      .selectAll('option')
      .classed('selected', function(d) {
        return d === colorScheme;
      });
    d3.select('#loadings-menu')
     .selectAll('li')
     .classed('selected', function(d) {
        if (showLoadings) {
          return d === 'Yes';
        } else {
          return d === 'No';
        }
     });
}

function updateLoadings(xAxis, yAxis) {
    if (showLoadings) {
        // For the currently selected axes, calculate the top 3 markers
        //  using L2-norm
        var L2s = [];
        for (var i = 0; i < loadings.length; i++) {
          var L2 = Math.sqrt(Math.pow(loadings[i][xAxis], 2) + Math.pow(loadings[i][yAxis], 2));
          var x,y;

          // Check if this is going to go offscreen
          if (loadings[i][xAxis] > xScale.domain()[1] || loadings[i][xAxis] < xScale.domain()[0]) {
            x = (loadings[i][xAxis] > xScale.domain()[1]) ? xScale.domain()[1] : xScale.domain()[0];
            x *= 0.7; // Scale it
          } else {
            x = loadings[i][xAxis];
          }

          if (loadings[i][yAxis] > yScale.domain()[1] || loadings[i][yAxis] < yScale.domain()[0]) {
            y = (loadings[i][yAxis] > yScale.domain()[1]) ? yScale.domain()[1] : yScale.domain()[0];
            y *= 0.7; 
          } else {
            y = loadings[i][yAxis];
          }         
            
          L2s.push({
            'L2': L2, 
            'Marker' : loadings[i]['Marker'], 
            'x' : x, 
            'y' : y
          });
        }

        var topMarkers = _.sortBy(L2s, function(num){ return num['L2']; });
        topMarkers = _.last(topMarkers, 3);

        // Check to make sure none are too close 
        // to the others 
        // (if they are, don't display them)
        /*
        var minDist = 1000;
        var oneOut;
        for(var i = 0; i < 3; i++) {
          for(var j = 0; j < 3; j++) {
            var curDist = Math.sqrt(Math.pow(topMarkers[i]['x'] - topMarkers[j]['x'], 2) + 
              Math.pow(topMarkers[i]['y'] - topMarkers[j]['y'], 2));

            if(curDist < minDist && i != j) {
              minDist = curDist;
              oneOut = j;
            }
          }
        }
        // If they're too close...
        if(minDist < 0.3) topMarkers = topMarkers.splice(oneOut,1);
        //console.log(topMarkers);
        */

        d3.select('div#chart svg g.chart')
          .selectAll('line')
          .remove();

        d3.select('div#chart svg g.chart')
          .selectAll('line')
          .data(topMarkers)
          .enter().append("line")
          .attr("class", "loadings")
          .attr("x1", function(d) {return xScale(0);})
          .attr("y1", function(d) {return yScale(0);})
          .attr("x2", function(d) {return xScale(d['x']);})
          .attr("y2", function(d) {return yScale(d['y']);});

        d3.select('div#chart svg g.chart')
          .selectAll('text#loadings')
          .remove();

        d3.select('div#chart svg g.chart')
          .selectAll('text#loadings')
          .data(topMarkers)
          .enter().append("text")
          .attr("id", "loadings")
          .attr("x", function(d) { return xScale(d['x']);})
          .attr("y", function(d) { return yScale(d['y']);})
          .text(function(d) { return d['Marker'];});
    
    } else {
      d3.select('div#chart svg g.chart')
          .selectAll('text#loadings')
          .remove();
      d3.select('div#chart svg g.chart')
          .selectAll('line')
          .remove();
    }
}

