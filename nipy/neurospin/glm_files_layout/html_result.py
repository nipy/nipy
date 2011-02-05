# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Small utility that generates an html page to describe
the activations found in an activation SPM, similar to SPM arrays 

Author: Lise Favre, Alexis Roche, Bertrand Thirion, 2008--2010
"""

from nibabel import load

def display_results_html(zmap_file_path, mask_file_path,
                         output_html_path, threshold=0.001,
                         method='fpr', cluster_th=0, null_zmax='bonferroni',
                         null_smax=None, null_s=None, nmaxima=4,
                         cluster_pval=.05):
    """
    Parameters
    ----------
    zmap_file_path, string,
                    path of the input activation (z-transformed) image
    mask_file_path, string,
                    path of the a corresponding mask image
    output_html_path, string,
                      path where the output html should be written
    threshold, float, optional
               (p-variate) frequentist threshold of the activation image
    method, string, optional
            to be chosen as height_control in 
            nipy.neurospin.statistical_mapping
    cluster_th, scalar, optional,
             cluster size threshold
    null_zmax: optional,
               parameter for cluster level statistics (?)
    null_s: optional,
             parameter for cluster level statistics (?)
    nmaxima: optional,
             number of local maxima reported per supra-threshold cluster    
    """
    import nipy.neurospin.statistical_mapping as sm

    # Read data: z-map and mask     
    zmap = load(zmap_file_path)
    mask = load(mask_file_path)
   
    # Compute cluster statistics
    nulls={'zmax' : null_zmax, 'smax' : null_smax, 's' : null_s}
    """
    if null_smax is not None:
        print "a"
        clusters, info = sm.cluster_stats(zmap, mask, height_th=threshold,
                                          nulls=nulls)
        clusters = [c for c in clusters if c['cluster_pvalue']<cluster_pval]
    else:
        print "b"
        clusters, info = sm.cluster_stats(zmap, mask, height_th=threshold,
                                          height_control=method.lower(),
                                          cluster_th=cluster_th, nulls=nulls)
    """
    clusters, info = sm.cluster_stats(zmap, mask, height_th=threshold,
                                      nulls=nulls, cluster_th=cluster_th,)
    clusters = [c for c in clusters if c['cluster_pvalue']<cluster_pval]
        
    
    #if clusters == None or info == None:
    #    print "No results were written for %s" % zmap_file_path
    #    return
    if clusters == None:
        clusters = []
        
    
    # Make HTML page 
    output = open(output_html_path, mode = "w")
    output.write("<html><head><title> Result Sheet for %s \
    </title></head><body><center>\n" % zmap_file_path)
    output.write("<h2> Results for %s</h2>\n" % zmap_file_path)
    output.write("<table border = 1>\n")
    output.write("<tr><th colspan=4> Voxel significance </th>\
    <th colspan=3> Coordinates in MNI referential</th>\
    <th>Cluster Size</th></tr>\n")
    output.write("<tr><th>p FWE corr<br>(Bonferroni)</th>\
    <th>p FDR corr</th><th>Z</th><th>p uncorr</th>")
    output.write("<th> x (mm) </th><th> y (mm) </th><th> z (mm) </th>\
    <th>(voxels)</th></tr>\n")

    for cluster in clusters:
        maxima = cluster['maxima']
        size=cluster['size']
        for j in range(min(len(maxima), nmaxima)):
            temp = ["%f" % cluster['fwer_pvalue'][j]]
            temp.append("%f" % cluster['fdr_pvalue'][j])
            temp.append("%f" % cluster['zscore'][j])
            temp.append("%f" % cluster['pvalue'][j])
            for it in range(3):
                temp.append("%f" % maxima[j][it])
            if j == 0:
                # Main local maximum
                temp.append('%i'%size)
                output.write('<tr><th align="center">' + '</th>\
                <th align="center">'.join(temp) + '</th></tr>')
            else:
                # Secondary local maxima 
                output.write('<tr><td align="center">' + '</td>\
                <td align="center">'.join(temp) + '</td><td></td></tr>\n')

                 
    nclust = len(clusters)
    nvox = sum([clusters[k]['size'] for k in range(nclust)])
    
    output.write("</table>\n")
    output.write("Number of voxels : %i<br>\n" % nvox)
    output.write("Number of clusters : %i<br>\n" % nclust)

    if info is not None:
        output.write("Threshold Z = %f (%s control at %f)<br>\n" \
                     % (info['threshold_z'], method, threshold))
        output.write("Cluster size threshold p<0.05")
    else:
        output.write("Cluster size threshold = %i voxels"%cluster_th)

    output.write("</center></body></html>\n")
    output.close()


