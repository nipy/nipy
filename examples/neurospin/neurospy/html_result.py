import nipy.neurospin.statistical_mapping as sm
from nipy.io.imageformats import load

def ComputeResultsContents(zmap_file_path, mask_file_path,
                           output_html_path, threshold=0.001,
                           method='fpr', cluster=0, null_zmax='bonferroni',
                           null_smax=None, null_s=None, nmaxima=4):

    #print zmap_file_path, mask_file_path, output_html_path

    # Read data: z-map and mask     
    zmap = load(zmap_file_path)
    mask = load(mask_file_path)

    cluster_th = cluster
   
    # Compute cluster statistics
    #if null_smax != None:
    nulls={'zmax' : null_zmax, 'smax' : null_smax, 's' : null_s}
    clusters, info = sm.cluster_stats(zmap, mask, height_th=threshold,
                                      height_control=method.lower(),
                                      cluster_th=cluster, nulls=nulls)
    if clusters == None or info == None:
        print "No results were writen for %s" % zmap_file_path
        return
    
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
            if j == 0: ## Main local maximum
                temp.append('%i'%size)
                output.write('<tr><th align="center">' + '</th>\
                <th align="center">'.join(temp) + '</th></tr>')
                #output.write('<th>%i</th>\n'%size)
            else: ## Secondary local maxima 
                output.write('<tr><td align="center">' + '</td>\
                <td align="center">'.join(temp) + '</td><td></td></tr>\n')

                 
    nclust = len(clusters)
    nvox = sum([clusters[k]['size'] for k in range(nclust)])
    
    output.write("</table>\n")
    output.write("Number of voxels : %i<br>\n" % nvox)
    output.write("Number of clusters : %i<br>\n" % nclust)
    output.write("Threshold Z = %f (%s control at %f)<br>\n" \
                 % (info['threshold_z'], method, threshold))
    output.write("Cluster size threshold = %i voxels"%cluster_th)
    output.write("</center></body></html>\n")
    output.close()
