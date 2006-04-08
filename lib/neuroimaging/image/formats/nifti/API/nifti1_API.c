#include <Python.h>
#define NIFTI_API_MODULE
#include "nifti1_API.h"

static PyObject *_create_niftihdr_callback(PyObject *self, PyObject *args, PyObject *keywords); 

static PyMethodDef nifti1_APIMethods[] = {
  {"_create_niftihdr_callback",  (PyCFunction) _create_niftihdr_callback, METH_VARARGS|METH_KEYWORDS, "Setup callback for creating an empty nifti header in C."},
  {NULL, NULL, 0, NULL}        /* Sentinel */
};

PyMODINIT_FUNC initnifti1_API(void) 
{
    PyObject *m;
    static void *PyNIFTI_API[PyNIFTI_API_pointers];
    PyObject *c_api_object;

    m = Py_InitModule("nifti1_API", nifti1_APIMethods);

    /* Initialize the C API pointer array */


/*     PyNIFTI_API[PyNIFTI_from_NIFTIhdrfile_NUM] = (void *) PyNIFTI_from_NIFTIhdrfile; */

    PyNIFTI_API[NIFTIhdr_from_PyNIFTIhdr_NUM] = (void *) NIFTIhdr_from_PyNIFTIhdr;

    PyNIFTI_API[NIFTIhdrfile_from_PyNIFTI_NUM] = (void *) NIFTIhdrfile_from_PyNIFTI;

/*     PyNIFTI_API[PyNIFTIhdr_from_NIFTIhdr_NUM] = (void *) PyNIFTIhdr_from_NIFTIhdr; */
    /* Create a CObject containing the API pointer array's address */

    c_api_object = PyCObject_FromVoidPtr((void *)PyNIFTI_API, NULL);

    if (c_api_object != NULL)
        PyModule_AddObject(m, "_C_API", c_api_object);
}


/* a function that creates a nifti-header struct from a python nifti header object */

static nifti_1_header * NIFTIhdr_from_PyNIFTIhdr(PyObject *nifti) {

#define INTCONVERTER(att, name) niftihdr->att = (int) PyInt_AsLong(PyObject_GetAttrString(nifti, name));
#define SHORTCONVERTER(att, name) niftihdr->att = (short) PyInt_AsLong(PyObject_GetAttrString(nifti, name));
#define CHARCONVERTER(att, name) niftihdr->att = (char) PyInt_AsLong(PyObject_GetAttrString(nifti, name));
#define FLOATCONVERTER(att, name) niftihdr->att = (float) PyFloat_AsDouble(PyObject_GetAttrString(nifti, name));
#define STRINGCONVERTER(att, name,n) tmpstr = (char *) strdup(PyString_AsString(PyObject_GetAttrString(nifti, name))); for(i=0; i<n; i++) {niftihdr->att[i] = tmpstr[i];};

  int i;
  char *tmpstr;
  nifti_1_header *niftihdr;

  niftihdr = MALLOC(sizeof(*niftihdr));

  /* Go through attributes, one at a time -- not pretty */

  INTCONVERTER(sizeof_hdr, "sizeof_hdr");
  STRINGCONVERTER(data_type, "data_type", 10);
  STRINGCONVERTER(db_name, "db_name", 18);
  INTCONVERTER(extents, "extents");
  SHORTCONVERTER(session_error, "session_error");
  CHARCONVERTER(regular, "regular");
  CHARCONVERTER(dim_info, "dim_info");

  PyObject *_dimpy;
  _dimpy = PyObject_GetAttrString(nifti, "dim");
  for (i=0; i<8; i++) {
    niftihdr->dim[i] = (short) PyInt_AsLong(PySequence_GetItem(_dimpy, i));
  }
  
  FLOATCONVERTER(intent_p1, "intent_p1");
  FLOATCONVERTER(intent_p2, "intent_p2");
  FLOATCONVERTER(intent_p3, "intent_p3");
  SHORTCONVERTER(intent_code, "intent_code");
  SHORTCONVERTER(datatype, "datatype");
  SHORTCONVERTER(bitpix, "bitpix");
  SHORTCONVERTER(slice_start, "slice_start");

  PyObject *_pixdimpy;
  _pixdimpy = PyObject_GetAttrString(nifti, "pixdim");
  for (i=0; i<8; i++) {
    niftihdr->pixdim[i] = (float) PyFloat_AsDouble(PySequence_GetItem(_pixdimpy, i));
  }

  FLOATCONVERTER(vox_offset, "vox_offset");
  FLOATCONVERTER(scl_slope, "scl_slope");
  FLOATCONVERTER(scl_inter, "scl_inter");
  SHORTCONVERTER(slice_end, "slice_end");
  CHARCONVERTER(slice_code, "slice_code");
  CHARCONVERTER(xyzt_units, "xyzt_units");
  FLOATCONVERTER(cal_max, "cal_max");
  FLOATCONVERTER(cal_min, "cal_min");
  FLOATCONVERTER(slice_duration, "slice_duration");
  FLOATCONVERTER(toffset, "toffset");
  INTCONVERTER(glmax, "glmax");
  INTCONVERTER(glmax, "glmin");
  STRINGCONVERTER(descrip, "descrip", 80);
  STRINGCONVERTER(aux_file, "aux_file", 24);
  SHORTCONVERTER(qform_code, "qform_code");
  SHORTCONVERTER(sform_code, "sform_code");
  FLOATCONVERTER(quatern_b, "quatern_b");
  FLOATCONVERTER(quatern_c, "quatern_c");
  FLOATCONVERTER(quatern_d, "quatern_d");
  FLOATCONVERTER(qoffset_x, "qoffset_x");
  FLOATCONVERTER(qoffset_y, "qoffset_y");
  FLOATCONVERTER(qoffset_z, "qoffset_z");

  PyObject *_srow_xpy;
  _srow_xpy = PyObject_GetAttrString(nifti, "srow_x");
  for (i=0; i<4; i++) {
    niftihdr->srow_x[i] = (float) PyFloat_AsDouble(PySequence_GetItem(_srow_xpy, i));
  }

  PyObject *_srow_ypy;
  _srow_ypy = PyObject_GetAttrString(nifti, "srow_y");
  for (i=0; i<4; i++) {
    niftihdr->srow_y[i] = (float) PyFloat_AsDouble(PySequence_GetItem(_srow_ypy, i));
  }

  PyObject *_srow_zpy;
  _srow_zpy = PyObject_GetAttrString(nifti, "srow_z");
  for (i=0; i<4; i++) {
    niftihdr->srow_z[i] = (float) PyFloat_AsDouble(PySequence_GetItem(_srow_zpy, i));
  }

  STRINGCONVERTER(intent_name, "intent_name", 16);
  STRINGCONVERTER(magic, "magic", 4);

  return(niftihdr);

#undef INTCONVERTER
#undef SHORTCONVERTER
#undef CHARCONVERTER
#undef FLOATCONVERTER
#undef STRINGCONVERTER


}

static char *NIFTIhdrfile_from_PyNIFTI(PyObject *nifti) {
  char *hdrfile;

  hdrfile = PyString_AsString(PyObject_GetAttrString(PyObject_GetAttrString(nifti, "hdrfile"), "name"));

  return(hdrfile);

}

/* A callback to create a NIFTIhdr Python object */

static PyObject *create_niftihdr = NULL;

static PyObject *_create_niftihdr_callback(PyObject *self, PyObject *args, PyObject *keywords) {
  PyObject *result = NULL;
  PyObject *temp;

  char *keyword_list[] = {"callback", NULL};

  if (PyArg_ParseTupleAndKeywords(args, keywords, "O", keyword_list, &temp)) {
    if (!PyCallable_Check(temp)) {
      PyErr_SetString(PyExc_TypeError, "parameter must be callable");
      return NULL;
    }
    Py_XINCREF(temp);         /* Add a reference to new callback */
    Py_XDECREF(create_niftihdr);  /* Dispose of previous callback */
    create_niftihdr = temp;       /* Remember new callback */
    /* Boilerplate to return "None" */
    Py_INCREF(Py_None);
    result = Py_None;
    }
    return result;
}


static PyObject *PyNIFTIhdr_from_NIFTIhdr(nifti_1_header *niftihdr) {

  PyObject *nifti;

  /* create an empty nifti header */
  nifti = PyEval_CallObject(create_niftihdr, Py_BuildValue("()")); 

#define INTCONVERTER(att, name) PyObject_SetAttrString(nifti, name, PyInt_FromLong((long) niftihdr->att));
#define SHORTCONVERTER(att, name) INTCONVERTER(att, name);
#define CHARCONVERTER(att, name) INTCONVERTER(att, name);
#define STRINGCONVERTER(att, name) PyObject_SetAttrString(nifti, name, PyString_FromString(niftihdr->att));
#define FLOATCONVERTER(att, name) PyObject_SetAttrString(nifti, name, PyFloat_FromDouble((double) niftihdr->att));

  /* Go through attributes, one at a time -- not pretty */

  INTCONVERTER(sizeof_hdr, "sizeof_hdr");
  STRINGCONVERTER(data_type, "data_type");
  STRINGCONVERTER(db_name, "db_name");
  INTCONVERTER(extents, "extents");
  SHORTCONVERTER(session_error, "session_error");
  CHARCONVERTER(regular, "regular");
  CHARCONVERTER(dim_info, "dim_info");

  PyObject_SetAttrString(nifti, "dim", Py_BuildValue("(f,f,f,f,f,f,f,f)", 
							  (double) niftihdr->dim[0],
							  (double) niftihdr->dim[1],
							  (double) niftihdr->dim[2],
							  (double) niftihdr->dim[3],
							  (double) niftihdr->dim[4],
							  (double) niftihdr->dim[5],
							  (double) niftihdr->dim[6],
							  (double) niftihdr->dim[7]));

  
  FLOATCONVERTER(intent_p1, "intent_p1");
  FLOATCONVERTER(intent_p2, "intent_p2");
  FLOATCONVERTER(intent_p3, "intent_p3");
  SHORTCONVERTER(intent_code, "intent_code");
  SHORTCONVERTER(datatype, "datatype");
  SHORTCONVERTER(bitpix, "bitpix");
  SHORTCONVERTER(slice_start, "slice_start");

  PyObject_SetAttrString(nifti, "pixdim", Py_BuildValue("(f,f,f,f,f,f,f,f)", 
							  (double) niftihdr->pixdim[0],
							  (double) niftihdr->pixdim[1],
							  (double) niftihdr->pixdim[2],
							  (double) niftihdr->pixdim[3],
							  (double) niftihdr->pixdim[4],
							  (double) niftihdr->pixdim[5],
							  (double) niftihdr->pixdim[6],
							  (double) niftihdr->pixdim[7]));


  FLOATCONVERTER(vox_offset, "vox_offset");
  FLOATCONVERTER(scl_slope, "scl_slope");
  FLOATCONVERTER(scl_inter, "scl_inter");
  SHORTCONVERTER(slice_end, "slice_end");
  CHARCONVERTER(slice_code, "slice_code");
  CHARCONVERTER(xyzt_units, "xyzt_units");
  FLOATCONVERTER(cal_max, "cal_max");
  FLOATCONVERTER(cal_min, "cal_min");
  FLOATCONVERTER(slice_duration, "slice_duration");
  FLOATCONVERTER(toffset, "toffset");
  INTCONVERTER(glmax, "glmax");
  INTCONVERTER(glmax, "glmin");
  STRINGCONVERTER(descrip, "descrip");
  STRINGCONVERTER(aux_file, "aux_file");
  SHORTCONVERTER(qform_code, "qform_code");
  SHORTCONVERTER(sform_code, "sform_code");
  FLOATCONVERTER(quatern_b, "quatern_b");
  FLOATCONVERTER(quatern_c, "quatern_c");
  FLOATCONVERTER(quatern_d, "quatern_d");
  FLOATCONVERTER(qoffset_x, "qoffset_x");
  FLOATCONVERTER(qoffset_y, "qoffset_y");
  FLOATCONVERTER(qoffset_z, "qoffset_z");

  PyObject_SetAttrString(nifti, "srow_x", Py_BuildValue("(f,f,f,f)", 
							(double) niftihdr->srow_x[0],
							(double) niftihdr->srow_x[1],
							(double) niftihdr->srow_x[2],
							(double) niftihdr->srow_x[3]));

  PyObject_SetAttrString(nifti, "srow_y", Py_BuildValue("(f,f,f,f)", 
							(double) niftihdr->srow_y[0],
							(double) niftihdr->srow_y[1],
							(double) niftihdr->srow_y[2],
							(double) niftihdr->srow_y[3]));

  PyObject_SetAttrString(nifti, "srow_z", Py_BuildValue("(f,f,f,f)", 
							(double) niftihdr->srow_z[0],
							(double) niftihdr->srow_z[1],
							(double) niftihdr->srow_z[2],
							(double) niftihdr->srow_z[3]));


  STRINGCONVERTER(intent_name, "intent_name");
  STRINGCONVERTER(magic, "magic");

  return((PyObject *) nifti);

#undef INTCONVERTER
#undef SHORTCONVERTER
#undef CHARCONVERTER
#undef FLOATCONVERTER
#undef STRINGCONVERTER

}
