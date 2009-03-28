#from soma import aims
import fff2.neuro as neurospy 

def ComputeResultsContents(zmap_file_path, mask_file_path, output_html_path, threshold=0.001, method='fpr', cluster=0,
			   null_zmax='bonferroni', null_smax=None, null_s=None, nmaxima=4):

	# Read data: z-map and mask 
	zmap = neurospy.image(zmap_file_path)
	mask = neurospy.image(mask_file_path)

	# Compute cluster statistics 
	clusters, info = neurospy.cluster_stats(zmap, mask, height_th=threshold, height_control=method, cluster_th=cluster,
						null_zmax=null_zmax, null_smax=null_smax, null_s=null_s)

	# Make HTML page 
	output = open(output_html_path, mode = "w")
	output.write("<html><head><title> Result Sheet for %s </title></head><body><center>\n" % zmap_file_path)
	output.write("<h2> Results for %s</h2>\n" % zmap_file_path)
	output.write("<table border = 1>\n")
	output.write("<tr><th colspan=4> Voxel significance</th><th colspan=3> Coordinates in MNI referential</th><th>Cluster Size</th></tr>\n")
	output.write("<tr><th>p FWE corr<br>(Bonferroni)</th><th>p FDR corr</th><th>Z</th><th>p uncorr</th>")
	output.write("<th> x (mm) </th><th> y (mm) </th><th> z (mm) </th><th>(voxels)</th></tr>\n")

	for cluster in clusters:
		maxima = cluster['maxima']
		for j in range(min(len(maxima), nmaxima)):
			temp = ["%f" % cluster['fwer_pvalue'][j]]
			temp.append("%f" % cluster['fdr_pvalue'][j])
			temp.append("%f" % cluster['zscore'][j])
			temp.append("%f" % cluster['pvalue'][j])
			for it in range(3):
				temp.append("%f" % maxima[j][it])
			if j == 0: ## Main local maximum 
				output.write('<tr><th align="center">' + '</th><th align="center">'.join(temp) + '</th></tr>\n')
			else: ## Secondary local maxima 
				output.write('<tr><td align="center">' + '</td><td align="center">'.join(temp) + '</td><td></td></tr>\n')

	output.write("</table>\n")
	output.write("Number of voxels : %i<br>\n" % len(mask.array > 0))
	output.write("Threshold Z=%f (%s control at %f)<br>\n" % (info['threshold_z'], method, threshold))
	output.write("</center></body></html>\n")
	output.close()

