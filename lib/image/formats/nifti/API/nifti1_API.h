#include <string.h>
#include "nifti1.h"


#ifndef Py_NIFTI_API_H
#define Py_NIFTI_API_H
#ifdef __cplusplus
extern "C" {
#endif

  /* Headerfile for nifti1_API */

#define MALLOC(size) ((void *) malloc(size))
#define FREE(ptr) (free(ptr))

#define PyNIFTIhdr_from_NIFTIhdr_NUM 0
#define PyNIFTIhdr_from_NIFTIhdr_RETURN PyObject *
#define PyNIFTIhdr_from_NIFTIhdr_PROTO (nifti_1_header *niftihdr)

#define NIFTIhdr_from_PyNIFTIhdr_NUM 1
#define NIFTIhdr_from_PyNIFTIhdr_RETURN nifti_1_header *
#define NIFTIhdr_from_PyNIFTIhdr_PROTO (PyObject *nifti)

#define NIFTIhdrfile_from_PyNIFTI_NUM 2 
#define NIFTIhdrfile_from_PyNIFTI_RETURN char *
#define NIFTIhdrfile_from_PyNIFTI_PROTO (PyObject *nifti)

/* #define PyNIFTI_from_NIFTIhdrfile_NUM 3 */
/* #define PyNIFTI_from_NIFTIhdrfile_RETURN PyObject * */
/* #define PyNIFTI_from_NIFTIhdrfile_PROTO (nifti_1_header *niftihdr, char *niftihdrfile) */

#define PyNIFTI_API_pointers 3 

#ifdef NIFTI_API_MODULE

  static NIFTIhdr_from_PyNIFTIhdr_RETURN NIFTIhdr_from_PyNIFTIhdr NIFTIhdr_from_PyNIFTIhdr_PROTO ;

  static NIFTIhdrfile_from_PyNIFTI_RETURN NIFTIhdrfile_from_PyNIFTI NIFTIhdrfile_from_PyNIFTI_PROTO ;

  static PyNIFTIhdr_from_NIFTIhdr_RETURN PyNIFTIhdr_from_NIFTIhdr PyNIFTIhdr_from_NIFTIhdr_PROTO ;

/*   static PyNIFTI_from_NIFTIhdrfile_RETURN PyNIFTI_from_NIFTIhdrfile PyNIFTI_from_NIFTIhdrfile_PROTO ; */

#else 

  static void **PyNIFTI_API;

#define NIFTIhdr_from_PyNIFTIhdr (*(NIFTIhdr_from_PyNIFTIhdr_RETURN (*)NIFTIhdr_from_PyNIFTIhdr_PROTO) PyNIFTI_API[NIFTIhdr_from_PyNIFTIhdr_NUM])

#define NIFTIhdrfile_from_PyNIFTI (*(NIFTIhdrfile_from_PyNIFTI_RETURN (*)NIFTIhdrfile_from_PyNIFTI_PROTO) PyNIFTI_API[NIFTIhdrfile_from_PyNIFTI_NUM])

#define PyNIFTIhdr_from_NIFTIhdr (*(PyNIFTIhdr_from_NIFTIhdr_RETURN (*)PyNIFTIhdr_from_NIFTIhdr_PROTO) PyNIFTI_API[PyNIFTIhdr_from_NIFTIhdr_NUM])

/* #define PyNIFTI_from_NIFTIhdrfile (*(PyNIFTI_from_NIFTIhdrfile_RETURN (*)PyNIFTI_from_NIFTIhdrfile_PROTO) PyNIFTI_API[PyNIFTI_from_NIFTIhdrfile_NUM]) */

static int import_nifti1_API(void)
{
    PyObject *module = PyImport_ImportModule("nifti1_API");

    if (module != NULL) {
        PyObject *c_api_object = PyObject_GetAttrString(module, "_C_API");
        if (c_api_object == NULL)
            return -1;
        if (PyCObject_Check(c_api_object))
            PyNIFTI_API = (void **)PyCObject_AsVoidPtr(c_api_object);
        Py_DECREF(c_api_object);
    }
    return 0;
}

#endif

#ifdef __cplusplus
}
#endif

#endif /* !defined(Py_NIFTI_API_H) */

