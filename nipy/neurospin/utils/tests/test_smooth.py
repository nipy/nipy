import numpy as np
from nipy.neurospin.utils.smoothing import cartesian_smoothing

def test_smoothing():
    #from nipy.io.imageformats import load, save, Nifti1Image 
    
    offset = 1.0
    # generate some data
    x = offset*np.ones((30,30,30))
    ijk = np.where(np.ones((30,30,30)))
    ref = np.array([15,15,15])
    x[ref[0],ref[1],ref[2]]+=1;
    sigma = 1.87

    # take the ideal result
    dx = np.sum((np.array(ijk).T-ref)*(np.array(ijk).T-ref),1)
    target = 1/(2*np.pi*sigma**2)**(1.5) * np.exp(-dx/(2*sigma**2))
    target = target/np.sum(target)
    target += offset
    gaussian = x.copy()
    gaussian[ijk] = target
    #save(Nifti1Image(gaussian,np.eye(4)),"/tmp/target.nii")

    # compute the result with the cartesian_smoothing procedure
    data = np.reshape(x.copy(),(np.size(x),1))
    trial = x.copy()
    trial[ijk] = np.squeeze(cartesian_smoothing(np.array(ijk).T,data,sigma))
    #save(Nifti1Image(trial,np.eye(4)),"/tmp/result.nii")
    
    dt = gaussian-trial
    error = (dt*dt).sum()
    print "error:", error
    assert (error<1.e-4)

if __name__ == "__main__":
    import nose
    nose.run(argv=['', __file__])


