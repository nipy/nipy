def configuration(parent_package='',top_path=None):
	
	from numpy.distutils.misc_util import Configuration
	
	config = Configuration('spatial_models', parent_package, top_path)
		
	return config
	

if __name__ == '__main__':
	print 'This is the wrong setup.py file to run'


