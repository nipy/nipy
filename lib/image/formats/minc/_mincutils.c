/* ----------------------------- MNI Header -----------------------------------
@NAME       : mincutils
@INPUT      : 
@OUTPUT     : (none)
@RETURNS    : 
@DESCRIPTION: Utilities to read / write / create MINC files from Python
              largely based on mincextract.c, minc?.c.
@METHOD     : 
@GLOBALS    : 
@CALLS      : 
@CREATED    : August 2003. (J. Taylor)
@MODIFIED   : 
              Copyright 1993 Peter Neelin, McConnell Brain Imaging Centre, 
              Montreal Neurological Institute, McGill University.
              Permission to use, copy, modify, and distribute this
              software and its documentation for any purpose and without
              fee is hereby granted, provided that the above copyright
              notice appear in all copies.  The author and McGill University
              make no representations about the suitability of this
              software for any purpose.  It is provided "as is" without
              express or implied warranty.

	      Copyright 2003 Jonathan Taylor, Stanford University.
              Permission to use, copy, modify, and distribute this
              software and its documentation for any purpose and without
              fee is hereby granted, provided that the above copyright
              notice appear in all copies.  The author and Stanford University
              make no representations about the suitability of this
              software for any purpose.  It is provided "as is" without
              express or implied warranty.
---------------------------------------------------------------------------- */

#include <Python.h>

#include <numpy/arrayobject.h> 
#include <minc.h>
#include <limits.h>
#include <float.h>
#include <ctype.h>

#ifndef FREE
#define  FREE( ptr ) free( (void *) ptr )
#endif

#ifndef MALLOC
#define  MALLOC(size ) malloc( (size_t) size);
#endif

/* Constants from mincextract*/

#ifndef TRUE
#  define TRUE 1
#  define FALSE 0
#endif
#ifndef public
#  define public
#  define private static
#endif

#define VECTOR_SEPARATOR ','
#define TYPE_ASCII  0
#define TYPE_BYTE   1
#define TYPE_SHORT  2
#define TYPE_INT    3
#define TYPE_FLOAT  4
#define TYPE_DOUBLE 5
#define TYPE_FILE   6
static nc_type nc_type_list[8] = {
   NC_DOUBLE, NC_BYTE, NC_SHORT, NC_INT, NC_FLOAT, NC_DOUBLE, NC_DOUBLE
};

/*  Constants from rawtominc */

#define NORMAL_STATUS 0
#define ERROR_STATUS 1
#define MIN_DIMS 2
#define MAX_DIMS 4
#define TRANSVERSE 0
#define SAGITTAL 1
#define CORONAL 2
#define TIME_FAST 3
#define XYZ_ORIENTATION 4
#define XZY_ORIENTATION 5
#define YXZ_ORIENTATION 6
#define YZX_ORIENTATION 7
#define ZXY_ORIENTATION 8
#define ZYX_ORIENTATION 9
#define DEF_TYPE 0
#define BYTE_TYPE 1
#define SHORT_TYPE 2
#define INT_TYPE 3
#define FLOAT_TYPE 4
#define DOUBLE_TYPE 5
#define DEF_SIGN 0
#define SIGNED 1
#define UNSIGNED 2
#define TRUE 1
#define FALSE 0
#define X 0
#define Y 1
#define Z 2
#define WORLD_NDIMS 3
#define DEF_STEP DBL_MAX
#define DEF_START DBL_MAX
#define DEF_RANGE DBL_MAX
#define DEF_DIRCOS DBL_MAX
#define DEF_ORIGIN DBL_MAX

/* Macros */
#define STR_EQ(s1,s2) (strcmp(s1,s2)==0)

static PyObject *mincopen(PyObject *self, PyObject *args, PyObject *keywords);
static PyObject *mincclose(PyObject *self, PyObject *args, PyObject *keywords);

static PyObject *mincextract(PyObject *self, PyObject *args, PyObject *keywords);

static PyObject *mincwrite(PyObject *self, PyObject *args, PyObject *keywords);

static PyObject *getinfo(PyObject *self, PyObject *args, PyObject *keywords);

static PyObject *getdircos(PyObject *self, PyObject *args, PyObject *keywords);

static PyObject *getvar(PyObject *self, PyObject *args, PyObject *keywords);

static PyObject *getINVALID_DATA(PyObject *self, PyObject *args, PyObject *keywords);

static PyObject *minccreate(PyObject *self, PyObject *args, PyObject *keywords);
static PyObject *readheader(PyObject *self, PyObject *args, PyObject *keywords);
/* Module methods */

static PyMethodDef mincutilsMethods[] = {
  {"_mincextract",  (PyCFunction) mincextract, METH_VARARGS|METH_KEYWORDS, "Extract a hyperslab of a mincfile to a NumPy array."},
  {"_mincopen",  (PyCFunction) mincopen, METH_VARARGS|METH_KEYWORDS, "Open a mincfile, returning mincid."},
  {"_mincclose",  (PyCFunction) mincclose, METH_VARARGS|METH_KEYWORDS, "Close a mincfile, returning mincid."},
  {"_minccreate",  (PyCFunction) minccreate, METH_VARARGS|METH_KEYWORDS, "Create a mincfile."},
  {"_readheader",  (PyCFunction) readheader, METH_VARARGS|METH_KEYWORDS, "Read header of a mincfile."},
  {"_mincwrite",  (PyCFunction) mincwrite, METH_VARARGS|METH_KEYWORDS, "Write a hyperslab of a NumPy array to a mincfile."},
  {"_getinfo",  (PyCFunction) getinfo, METH_VARARGS|METH_KEYWORDS, "Get step, shape, dimnames information from a mincfile."},  
  {"_getdircos",  (PyCFunction) getdircos, METH_VARARGS|METH_KEYWORDS, "Get direction cosines for a dimension in a mincfile."},  
  {"_getvar",  (PyCFunction) getvar, METH_VARARGS|METH_KEYWORDS, "Get values of a variable in a MINC file -- to get image values use mincextract."},  
  {"_invalid_data",  (PyCFunction) getINVALID_DATA, METH_VARARGS|METH_KEYWORDS, "Get INVALID_DATA."},  
  {NULL, NULL, 0, NULL}        /* Sentinel */
};

/* Initialize the module mincutils */


void init_mincutils(void)
{
  PyObject *module;
  module = Py_InitModule("_mincutils", mincutilsMethods);
  import_array(); 
}

static PyObject *mincopen(PyObject *self, PyObject *args, PyObject *keywords)
{
   char *filename;
   int mincid = MI_ERROR;
   int mode = NC_NOWRITE;
   PyObject *mincid_py;

   /* Check arguments */

   char *keyword_list[] = {"filename", "mode", NULL};

   if (! PyArg_ParseTupleAndKeywords(args, keywords, "s|i", keyword_list, &filename, &mode)) {
     PyErr_SetObject(PyExc_TypeError, Py_BuildValue("OO", args, keywords));
     return NULL;
   } 

   mincid = miopen(filename, mode);
   
   mincid_py = Py_BuildValue("i", mincid);
   return(mincid_py);
}

static PyObject *mincclose(PyObject *self, PyObject *args, PyObject *keywords)
{
   int success, mincid;
   PyObject *success_py;

   /* Check arguments */

   char *keyword_list[] = {"mincid", NULL};

   if (! PyArg_ParseTupleAndKeywords(args, keywords, "i", keyword_list, &mincid)) {
     PyErr_SetObject(PyExc_TypeError, args);
     return NULL;
   } 

   success = miclose(mincid);
   
   success_py = Py_BuildValue("i", success);
   return(success_py);
}

static PyObject *mincextract(PyObject *self, PyObject *args, PyObject *keywords)
{
   char *filename;
   int mincid = MI_ERROR;
   int has_mincid = FALSE;
   int imgid, icvid, ndims, dims[MAX_VAR_DIMS];
   nc_type datatype;
   int is_signed;
   long start[MAX_VAR_DIMS], end[MAX_VAR_DIMS];
   long count[MAX_VAR_DIMS], cur[MAX_VAR_DIMS];
   long old[MAX_VAR_DIMS], diff[MAX_VAR_DIMS];
   int element_size;
   int idim;
   int nstart, ncount;
   char *data;
   double temp;
   long nelements, ntotal;
   int user_normalization;
   char *xdirection_str = "any";
   char *ydirection_str = "any";
   char *zdirection_str = "any";
   PyArrayObject *array;
   PyObject *start_seq, *count_seq, *cur_item;

   /* Variables used for argument parsing */
   int arg_odatatype = TYPE_ASCII;
   nc_type output_datatype = NC_DOUBLE;
   int output_signed = INT_MAX;
   double valid_range[2] = {DBL_MAX, DBL_MAX};
   int normalize_output = TRUE;
   double image_range[2] = {DBL_MAX, DBL_MAX};
   long hs_start[MAX_VAR_DIMS] = {LONG_MIN};
   long hs_count[MAX_VAR_DIMS] = {LONG_MIN};
   int xdirection = INT_MAX;
   int ydirection = INT_MAX;
   int zdirection = INT_MAX;
   int default_direction = INT_MAX;

   /* Check arguments */

   char *keyword_list[] = {"filename", "start", "count", "xdir", "ydir", "zdir", "mincid", NULL};

   if (!PyArg_ParseTupleAndKeywords(args, keywords, "sOO|sssi", keyword_list, &filename, &start_seq, &count_seq, &xdirection_str, &ydirection_str, &zdirection_str, &mincid)) {
     PyErr_SetObject(PyExc_TypeError, Py_BuildValue("OO", args, keywords));
     return NULL;
   } 
  

   if (mincid != MI_ERROR) 
     has_mincid = TRUE;

   nstart = PySequence_Length(start_seq);
   for (idim=0; idim < nstart; idim++) {
     cur_item = PySequence_GetItem(start_seq, idim);
     hs_start[idim] = PyInt_AsLong(cur_item);
     Py_DECREF(cur_item);
   }
   Py_DECREF(start_seq);

   ntotal = 1;
   ncount = PySequence_Length(count_seq);
   for (idim=0; idim < ncount; idim++) {
     cur_item = PySequence_GetItem(count_seq, idim);
     hs_count[idim] = PyInt_AsLong(cur_item);
     Py_DECREF(cur_item);
     ntotal *= hs_count[idim];
   }
   Py_DECREF(count_seq);

   if (xdirection_str == "+")
     xdirection = MI_ICV_POSITIVE;
   else if (xdirection_str == "-")
     xdirection = MI_ICV_NEGATIVE;
   else
     xdirection = MI_ICV_ANYDIR;

   if (ydirection_str == "+")
     ydirection = MI_ICV_POSITIVE;
   else if (ydirection_str == "-")
     ydirection = MI_ICV_NEGATIVE;
   else
     ydirection = MI_ICV_ANYDIR;

   if (zdirection_str == "+")
     zdirection = MI_ICV_POSITIVE;
   else if (zdirection_str == "-")
     zdirection = MI_ICV_NEGATIVE;
   else
     zdirection = MI_ICV_ANYDIR;

   /* Set normalization if image_range specified */

   user_normalization = TRUE;
   normalize_output = TRUE;

   /* Check direction values */
   if (default_direction == INT_MAX)
      default_direction = MI_ICV_ANYDIR;
   if (xdirection == INT_MAX)
      xdirection = default_direction;
   if (ydirection == INT_MAX)
      ydirection = default_direction;
   if (zdirection == INT_MAX)
      zdirection = default_direction;

   /* Open the file if necessary */

   if (mincid == MI_ERROR)
     mincid = miopen(filename, NC_NOWRITE);

   /* Inquire about the image variable */

   imgid = ncvarid(mincid, MIimage);
   (void) ncvarinq(mincid, imgid, NULL, NULL, &ndims, dims, NULL);
   (void) miget_datatype(mincid, imgid, &datatype, &is_signed);

   /* Check if arguments set */

   /* Check the start and count arguments */
   if (((nstart != 0) && (nstart != ndims)) || 
       ((ncount != 0) && (ncount != ndims))) {
     PyErr_SetString(PyExc_ValueError, "dimensions of start or count vectors not equal to dimensions in file");
     return NULL;
   }

   /* Get output data type */
   output_datatype = nc_type_list[arg_odatatype];
   if (arg_odatatype == TYPE_FILE) output_datatype = datatype;

   /* Get output sign */ 
   if (output_signed == INT_MAX) {
      if (arg_odatatype == TYPE_FILE)
         output_signed = is_signed;
      else 
         output_signed = (output_datatype != NC_BYTE);
   }

   /* Get output range */
   if (valid_range[0] == DBL_MAX) {
      if (arg_odatatype == TYPE_FILE) {
         (void) miget_valid_range(mincid, imgid, valid_range);
      }
      else {
         (void) miget_default_range(output_datatype, output_signed, 
                                    valid_range);
      }
   }
   if (valid_range[0] > valid_range[1]) {
      temp = valid_range[0];
      valid_range[0] = valid_range[1];
      valid_range[1] = temp;
   }

   /* Set up image conversion */
   icvid = miicv_create();
   (void) miicv_setint(icvid, MI_ICV_TYPE, output_datatype);
   (void) miicv_setstr(icvid, MI_ICV_SIGN, (output_signed ? 
                                            MI_SIGNED : MI_UNSIGNED));
   (void) miicv_setdbl(icvid, MI_ICV_VALID_MIN, valid_range[0]);
   (void) miicv_setdbl(icvid, MI_ICV_VALID_MAX, valid_range[1]);
   (void) miicv_setint(icvid, MI_ICV_DO_DIM_CONV, TRUE);
   (void) miicv_setint(icvid, MI_ICV_DO_SCALAR, FALSE);
   (void) miicv_setint(icvid, MI_ICV_XDIM_DIR, xdirection);
   (void) miicv_setint(icvid, MI_ICV_YDIM_DIR, ydirection);
   (void) miicv_setint(icvid, MI_ICV_ZDIM_DIR, zdirection);
   if ((output_datatype == NC_FLOAT) || (output_datatype == NC_DOUBLE)) {
      (void) miicv_setint(icvid, MI_ICV_DO_NORM, TRUE);
      (void) miicv_setint(icvid, MI_ICV_USER_NORM, TRUE);
   }
   else if (normalize_output) {
      (void) miicv_setint(icvid, MI_ICV_DO_NORM, TRUE);
      if (user_normalization) {
         (void) miicv_attach(icvid, mincid, imgid);
         if (image_range[0] == DBL_MAX) {
            (void) miicv_inqdbl(icvid, MI_ICV_NORM_MIN, &image_range[0]);
         }
         if (image_range[1] == DBL_MAX) {
            (void) miicv_inqdbl(icvid, MI_ICV_NORM_MAX, &image_range[1]);
         }
         (void) miicv_detach(icvid);
         (void) miicv_setint(icvid, MI_ICV_USER_NORM, TRUE);
         (void) miicv_setdbl(icvid, MI_ICV_IMAGE_MIN, image_range[0]);
         (void) miicv_setdbl(icvid, MI_ICV_IMAGE_MAX, image_range[1]);
      }
   }
   (void) miicv_attach(icvid, mincid, imgid);

   /* Set input file start, count and end vectors for reading a slice
      at a time */
   nelements = 1;

   for (idim=0; idim < ndims; idim++) {

      /* Get start */
      start[idim] = (nstart == 0) ? 0 : hs_start[idim];
      cur[idim] = start[idim];

      /* Get end */
      if (ncount!=0)
         end[idim] = start[idim]+hs_count[idim];
      else if (nstart!=0)
         end[idim] = start[idim]+1;
      else
         (void) ncdiminq(mincid, dims[idim], NULL, &end[idim]);

      /* Compare start and end */
      if (start[idim] >= end[idim]) {
	PyErr_SetString(PyExc_ValueError, "start or count out of range.");
	return NULL;
      }

      /* Get count and nelements */
      if (idim < ndims-2)
         count[idim] = 1;
      else
         count[idim] = end[idim] - start[idim];
      nelements *= count[idim];
   }
   element_size = nctypelen(output_datatype);


   /* Allocate space */

   array = (PyArrayObject *) PyArray_SimpleNew(ndims, (int *) hs_count, PyArray_DOUBLE); 

   data = PyArray_DATA((PyObject *) array);

   /* Loop over input slices */

   while (cur[0] < end[0]) {

      /* Read in the slice */
      (void) miicv_get(icvid, cur, count, (void *) data);

      for (idim=0; idim<ndims; idim++) 
	old[idim] = cur[idim];
      
      /* Increment cur counter */
      idim = ndims-1;
      cur[idim] += count[idim];
      while ( (idim>0) && (cur[idim] >= end[idim])) {
         cur[idim] = start[idim];
         idim--;
         cur[idim] += count[idim];
      }
      for (idim=0; idim<ndims; idim++) {
	diff[idim] = cur[idim] - old[idim];
	data = data + diff[idim] * array->strides[idim];
      }
   }       /* End loop over slices */

   /* Clean up */
   if (!has_mincid) 
     (void) miclose(mincid);
   (void) miicv_detach(icvid);
   (void) miicv_free(icvid);

   return((PyObject *)array);
}

static PyObject *mincwrite(PyObject *self, PyObject *args, PyObject *keywords)
{
   char *filename;
   int mincid = MI_ERROR;
   int imgid, icvid, ndims, dims[MAX_VAR_DIMS];
   nc_type datatype;
   int is_signed;
   long start[MAX_VAR_DIMS], end[MAX_VAR_DIMS];
   long count[MAX_VAR_DIMS], cur[MAX_VAR_DIMS];
   long old[MAX_VAR_DIMS], diff[MAX_VAR_DIMS];
   int element_size;
   int idim;
   int nstart, ncount;
   char *data;
/*    void *data; */
   double temp;
   long nelements;
   int user_normalization;
   int islice = 0;
   int minid, maxid;
   double in_max, in_min;
   char *xdirection_str = "any";
   char *ydirection_str = "any";
   char *zdirection_str = "any";
   PyObject *start_seq, *count_seq, *cur_item, *array_obj;
   PyObject *data_max, *data_min;
   PyArrayObject *array;

   /* Variables used for argument parsing */
   int arg_odatatype = TYPE_ASCII;
   nc_type output_datatype = NC_DOUBLE;
   int output_signed = INT_MAX;
   double valid_range[2] = {DBL_MAX, DBL_MAX};
   int normalize_output = TRUE;
   double image_range[2] = {DBL_MAX, DBL_MAX};
   char names[MAX_VAR_DIMS][NC_MAX_NAME];
   long hs_start[MAX_VAR_DIMS] = {LONG_MIN};
   long hs_count[MAX_VAR_DIMS] = {LONG_MIN};
   int xdirection = INT_MAX;
   int ydirection = INT_MAX;
   int zdirection = INT_MAX;
   int default_direction = INT_MAX;
   int set_minmax = TRUE;
   int has_vector = FALSE;
   int has_mincid = FALSE;

   /* Check arguments */

   char *keyword_list[] = {"filename", "start", "count", "data", "set_minmax", "data_max", "data_min", "xdir", "ydir", "zdir", "mincid", NULL};

   if (! PyArg_ParseTupleAndKeywords(args, keywords, "sOOO|iOOsssi", keyword_list, &filename, &start_seq, &count_seq, &array_obj, &set_minmax, &data_max, &data_min, &xdirection_str, &ydirection_str, &zdirection_str, &mincid)) {
     PyErr_SetObject(PyExc_TypeError, Py_BuildValue("OO", args, keywords));
     return NULL;
   } 
  
   if (mincid != MI_ERROR)
     has_mincid = TRUE;

   nstart = PySequence_Length(start_seq);
   for (idim=0; idim < nstart; idim++) {
     cur_item = PySequence_GetItem(start_seq, idim);
     hs_start[idim] = PyInt_AsLong(cur_item);
     Py_DECREF(cur_item);
   }
   Py_DECREF(start_seq);

   ncount = PySequence_Length(count_seq);
   for (idim=0; idim < ncount; idim++) {
     cur_item = PySequence_GetItem(count_seq, idim);
     hs_count[idim] = PyInt_AsLong(cur_item);
     Py_DECREF(cur_item);
   }
   Py_DECREF(count_seq);

   if (xdirection_str == "+")
     xdirection = MI_ICV_POSITIVE;
   else if (xdirection_str == "-")
     xdirection = MI_ICV_NEGATIVE;
   else
     xdirection = MI_ICV_ANYDIR;

   if (ydirection_str == "+")
     ydirection = MI_ICV_POSITIVE;
   else if (ydirection_str == "-")
     ydirection = MI_ICV_NEGATIVE;
   else
     ydirection = MI_ICV_ANYDIR;

   if (zdirection_str == "+")
     zdirection = MI_ICV_POSITIVE;
   else if (zdirection_str == "-")
     zdirection = MI_ICV_NEGATIVE;
   else
     zdirection = MI_ICV_ANYDIR;

   /* Set normalization if image_range specified */

   user_normalization = TRUE;
   normalize_output = TRUE;

   /* Check direction values */
   if (default_direction == INT_MAX)
      default_direction = MI_ICV_ANYDIR;
   if (xdirection == INT_MAX)
      xdirection = default_direction;
   if (ydirection == INT_MAX)
      ydirection = default_direction;
   if (zdirection == INT_MAX)
      zdirection = default_direction;

   /* Open the file */
   if (mincid == MI_ERROR) {
     mincid = miopen(filename, NC_WRITE);
   }
 
   /* Inquire about the image variable */
   imgid = ncvarid(mincid, MIimage);
   (void) ncvarinq(mincid, imgid, NULL, NULL, &ndims, dims, NULL);
   (void) miget_datatype(mincid, imgid, &datatype, &is_signed);

   /* Check if arguments set */

   /* Check the start and count arguments */
   if (((nstart != 0) && (nstart != ndims)) || 
       ((ncount != 0) && (ncount != ndims))) {
     PyErr_SetString(PyExc_ValueError, "dimensions of start or count vectors not equal to dimensions in file");
     return NULL;
   }

   /* Get output data type */
   output_datatype = nc_type_list[arg_odatatype];
   if (arg_odatatype == TYPE_FILE) output_datatype = datatype;

   /* Get output sign */ 
   if (output_signed == INT_MAX) {
      if (arg_odatatype == TYPE_FILE)
         output_signed = is_signed;
      else 
         output_signed = (output_datatype != NC_BYTE);
   }

   /* Get output range */
   if (valid_range[0] == DBL_MAX) {
      if (arg_odatatype == TYPE_FILE) {
         (void) miget_valid_range(mincid, imgid, valid_range);
      }
      else {
         (void) miget_default_range(output_datatype, output_signed, 
                                    valid_range);
      }
   }
   if (valid_range[0] > valid_range[1]) {
      temp = valid_range[0];
      valid_range[0] = valid_range[1];
      valid_range[1] = temp;
   }

   /* Set up image conversion */
   icvid = miicv_create();
   (void) miicv_setint(icvid, MI_ICV_TYPE, output_datatype);
   (void) miicv_setstr(icvid, MI_ICV_SIGN, (output_signed ? 
                                            MI_SIGNED : MI_UNSIGNED));
   (void) miicv_setdbl(icvid, MI_ICV_VALID_MIN, valid_range[0]);
   (void) miicv_setdbl(icvid, MI_ICV_VALID_MAX, valid_range[1]);
   (void) miicv_setint(icvid, MI_ICV_DO_DIM_CONV, TRUE);
   (void) miicv_setint(icvid, MI_ICV_DO_SCALAR, FALSE);
   (void) miicv_setint(icvid, MI_ICV_XDIM_DIR, xdirection);
   (void) miicv_setint(icvid, MI_ICV_YDIM_DIR, ydirection);
   (void) miicv_setint(icvid, MI_ICV_ZDIM_DIR, zdirection);
   if ((output_datatype == NC_FLOAT) || (output_datatype == NC_DOUBLE)) {
      (void) miicv_setint(icvid, MI_ICV_DO_NORM, TRUE);
      (void) miicv_setint(icvid, MI_ICV_USER_NORM, TRUE);
   }
   else if (normalize_output) {
      (void) miicv_setint(icvid, MI_ICV_DO_NORM, TRUE);
      if (user_normalization) {
         (void) miicv_attach(icvid, mincid, imgid);
         if (image_range[0] == DBL_MAX) {
            (void) miicv_inqdbl(icvid, MI_ICV_NORM_MIN, &image_range[0]);
         }
         if (image_range[1] == DBL_MAX) {
            (void) miicv_inqdbl(icvid, MI_ICV_NORM_MAX, &image_range[1]);
         }
         (void) miicv_detach(icvid);
         (void) miicv_setint(icvid, MI_ICV_USER_NORM, TRUE);
         (void) miicv_setdbl(icvid, MI_ICV_IMAGE_MIN, image_range[0]);
         (void) miicv_setdbl(icvid, MI_ICV_IMAGE_MAX, image_range[1]);
      }
   }
   (void) miicv_attach(icvid, mincid, imgid);

   maxid = ncvarid(mincid, MIimagemax);
   (void) ncvarinq(mincid, maxid, NULL, NULL, NULL, NULL, NULL);

   minid = ncvarid(mincid, MIimagemin);
   (void) ncvarinq(mincid, minid, NULL, NULL, NULL, NULL, NULL);

   /* Set input file start, count and end vectors for writing a slice
      at a time */

   nelements = 1;

   for (idim=0; idim < ndims; idim++) {
     ncdiminq(mincid, dims[idim], names[idim], NULL);
     if (STR_EQ(names[idim], MIvector_dimension))
       has_vector = 1;
   }

   for (idim=0; idim < ndims; idim++) {

      /* Get start */
      start[idim] = (nstart == 0) ? 0 : hs_start[idim];
      cur[idim] = start[idim];

      /* Get end */
      if (ncount != 0)
         end[idim] = start[idim]+hs_count[idim];
      else if (nstart!=0)
         end[idim] = start[idim]+1;
      else
         (void) ncdiminq(mincid, dims[idim], NULL, &end[idim]);

      /* Compare start and end */
      if (start[idim] >= end[idim]) {
	PyErr_SetString(PyExc_ValueError, "start or count out of range.");
	return NULL;
      }

      /* Get count and nelements */
      if (idim < ndims-2-has_vector)
         count[idim] = 1;
      else
         count[idim] = end[idim] - start[idim];
      nelements *= count[idim];
   }

   element_size = nctypelen(output_datatype);

   /* Allocate space */

   array = (PyArrayObject *) array_obj;

   data = PyArray_DATA((PyObject *) array);

   /* Loop over input slices */

   while (cur[0] < end[0]) {

     /* Get image-max for this slice */
     /* All this is done assuming we are writing entire slice at a time! */

     if (set_minmax == TRUE) {
       cur_item = PySequence_GetItem(data_max, islice);
       in_max = PyFloat_AsDouble(cur_item);
       Py_DECREF(cur_item);
       ncvarput1(mincid, maxid, cur, &in_max);
       
       cur_item = PySequence_GetItem(data_min, islice);
       in_min = PyFloat_AsDouble(cur_item);
       Py_DECREF(cur_item);
       ncvarput1(mincid, minid, cur, &in_min);
     }
     islice++;

     /* Write out the slice */
     (void) miicv_put(icvid, cur, count, (void *) data);

      for (idim=0; idim<ndims; idim++) 
	old[idim] = cur[idim];

      /* Increment cur counter */
      idim = ndims-1;
      cur[idim] += count[idim];
      while ( (idim>0) && (cur[idim] >= end[idim])) {
         cur[idim] = start[idim];
         idim--;
         cur[idim] += count[idim];
      }
      for (idim=0; idim<ndims; idim++) {
	diff[idim] = cur[idim] - old[idim];
	data += diff[idim] * array->strides[idim];
      }
   }       /* End loop over slices */

   /* Clean up */

   if (!has_mincid) 
     (void) miclose(mincid);
   (void) miicv_detach(icvid); 
   (void) miicv_free(icvid);

   Py_INCREF(Py_None);
   return(Py_None);
}

static PyObject *getdircos(PyObject *self, PyObject *args, PyObject *keywords)
{
   char *filename;
   int dircos_len;
   nc_type dircos_type;
   void *direction_cosines;
   int mincid = MI_ERROR;
   int has_mincid = FALSE;
   int dimid, icos, natts, iatt;
   char *dimname;
   PyObject *direction_cosines_py, *pyerr;
   char attname[MAX_NC_NAME];

   /* Check arguments */

   char *keyword_list[] = {"filename", "varname", "mincid", NULL};

   if (! PyArg_ParseTupleAndKeywords(args, keywords, "ss|i", keyword_list, &filename, &dimname, &mincid)) {
     PyErr_SetObject(PyExc_TypeError, Py_BuildValue("OO", args, keywords));
     return NULL;
   } 
  
   if (mincid != MI_ERROR)
     has_mincid = TRUE;

   /* Open file if necessary */

   if (mincid == MI_ERROR) 
     mincid = miopen(filename, NC_NOWRITE);

   /* Inquire about the dimension variable */

   dimid = ncvarid(mincid, dimname);
   (void) ncvarinq(mincid, dimid, dimname, NULL, NULL, NULL, &natts);

   if ((!STR_EQ(dimname, MItime)) && (!STR_EQ(dimname, MIvector_dimension))) {
     for (iatt=0; iatt<natts; iatt++) {
       (void) ncattname(mincid, dimid, iatt, attname);
       if (STR_EQ(attname, MIdirection_cosines)) {
	 if (ncattinq(mincid, dimid, MIdirection_cosines, &dircos_type, &dircos_len) != -1) {
	   direction_cosines = MALLOC(nctypelen(dircos_type) * dircos_len);
	   ncattget(mincid, dimid, MIdirection_cosines, direction_cosines);
	   direction_cosines_py = PyTuple_New(dircos_len);
	   for (icos=0; icos<dircos_len; icos++) {
	     if (PyTuple_SetItem(direction_cosines_py, icos, PyFloat_FromDouble(((double *) direction_cosines)[icos])) != 0) {
	       pyerr = PyErr_Occurred();
	       if (pyerr != NULL) {
		 PyErr_Print();
	       }
	       Py_INCREF(Py_None);
	       return(Py_None);
	     }
	   }
	 }
	 return(direction_cosines_py);
       }
     }
   }

   Py_INCREF(Py_None);
   return(Py_None);
}

static PyObject *getinfo(PyObject *self, PyObject *args, PyObject *keywords)
{
   char *filename;
   int mincid = MI_ERROR;
   int has_mincid = FALSE;
   int varid, ndims, dims[MAX_VAR_DIMS];
   int dimid;
   char **names;
   long *shape;
   double *step;
   double *start;
   int idim;
   PyObject *step_py, *shape_py, *names_py, *pyerr, *return_py, *start_py;
   char *varname = MIimage;

   /* Check arguments */

   char *keyword_list[] = {"filename", "varname", "mincid", NULL};

   if (! PyArg_ParseTupleAndKeywords(args, keywords, "s|si", keyword_list, &filename, &varname, &mincid)) {
     PyErr_SetObject(PyExc_TypeError, Py_BuildValue("OO", args, keywords));
     return NULL;
   } 
  
   if (mincid != MI_ERROR)
     has_mincid = TRUE;

   /* Open file if necessary */

   if (mincid == MI_ERROR) 
     mincid = miopen(filename, NC_NOWRITE);

   /* Inquire about the image variable */
   varid = ncvarid(mincid, varname);
   (void) ncvarinq(mincid, varid, NULL, NULL, &ndims, dims, NULL);

   step = MALLOC(sizeof(*step) * ndims);
   shape = MALLOC(sizeof(*step) * ndims);
   start = MALLOC(sizeof(*step) * ndims);
   names = MALLOC(sizeof(*names) * ndims);

   step_py = PyTuple_New(ndims);
   start_py = PyTuple_New(ndims);
   shape_py = PyTuple_New(ndims);
   names_py = PyTuple_New(ndims);

   for (idim=0; idim<ndims; idim++) {
     names[idim] = MALLOC(sizeof(*names[idim]) * NC_MAX_NAME);
     ncdiminq(mincid, dims[idim], names[idim], &(shape[idim]));
     if ((!STR_EQ(names[idim], MItime)) && (!STR_EQ(names[idim], MIvector_dimension))) {
       dimid = ncvarid(mincid, names[idim]);

       ncattget(mincid, dimid, MIstep, &(step[idim]));
       ncattget(mincid, dimid, MIstart, &(start[idim]));
     }
     else {
       step[idim] = 0.0;
       start[idim] = 0.0;
     }

     if (PyTuple_SetItem(step_py, idim, PyFloat_FromDouble(step[idim])) != 0) {
       pyerr = PyErr_Occurred();
       if (pyerr != NULL) {
	 PyErr_Print();
       }
       Py_INCREF(Py_None);
       return(Py_None);
     }

     if (PyTuple_SetItem(start_py, idim, PyFloat_FromDouble(start[idim])) != 0) {
       pyerr = PyErr_Occurred();
       if (pyerr != NULL) {
	 PyErr_Print();
       }
       Py_INCREF(Py_None);
       return(Py_None);
     }

     if (PyTuple_SetItem(shape_py, idim, PyInt_FromLong(shape[idim])) != 0) {
       pyerr = PyErr_Occurred();
       if (pyerr != NULL) {
	 PyErr_Print();
       }
       Py_INCREF(Py_None);
       return(Py_None);
     }

     if (PyTuple_SetItem(names_py, idim, PyString_FromString((const char *) names[idim])) != 0) {
       pyerr = PyErr_Occurred();
       if (pyerr != NULL) {
	 PyErr_Print();
       }
       Py_INCREF(Py_None);
       return(Py_None);
     }
   }

   /* Clean up */
   if (!has_mincid)
     (void) miclose(mincid);

   return_py = PyTuple_New(4);
   PyTuple_SetItem(return_py, 0, step_py);
   PyTuple_SetItem(return_py, 1, start_py);
   PyTuple_SetItem(return_py, 2, shape_py);
   PyTuple_SetItem(return_py, 3, names_py);

   FREE(step);
   FREE(start);
   FREE(shape);
   for (idim=0; idim<ndims; idim++) 
     FREE(names[idim]);
   FREE(names);
  
/*    Py_INCREF(return_py); */
   return(return_py);
}

static PyObject *getvar(PyObject *self, PyObject *args, PyObject *keywords)
{
   char *filename, *varname;
   int mincid = MI_ERROR;
   int varid, ndims, dims[MAX_VAR_DIMS];
   nc_type datatype;
   int idim;
   int nstart = 0;
   int ncount = 0;
   void *data;
   PyArrayObject *array;
   PyObject *start_seq, *count_seq, *cur_item;
   long *hs_start;
   long *hs_count;
   int has_mincid = FALSE;
   long ntotal;

   /* Check arguments */

   char *keyword_list[] = {"filename", "varname", "start", "count", "mincid", NULL};

   if (! PyArg_ParseTupleAndKeywords(args, keywords, "ssOO|i", keyword_list, &filename, &varname, &start_seq, &count_seq, &mincid)) {
     PyErr_SetObject(PyExc_TypeError, Py_BuildValue("OO", args, keywords));
     return NULL;
   } 

   if (mincid != MI_ERROR)
     has_mincid = TRUE;

   nstart = PySequence_Length(start_seq);
   hs_start = MALLOC(sizeof(*hs_start) * nstart);
   for (idim=0; idim < nstart; idim++) {
     cur_item = PySequence_GetItem(start_seq, idim);
     hs_start[idim] = PyInt_AsLong(cur_item);
     Py_DECREF(cur_item);
   }
   Py_DECREF(start_seq);

   ncount = PySequence_Length(count_seq);
   hs_count = MALLOC(sizeof(*hs_count) * ncount);
   for (idim=0; idim < ncount; idim++) {
     cur_item = PySequence_GetItem(count_seq, idim);
     hs_count[idim] = PyInt_AsLong(cur_item);
     Py_DECREF(cur_item);
   }
   Py_DECREF(count_seq);

   /* Open the file */
   if (mincid == MI_ERROR)
     mincid = miopen(filename, NC_NOWRITE);

   /* Inquire about the image variable */
   varid = ncvarid(mincid, varname);
   (void) ncvarinq(mincid, varid, NULL, &datatype, &ndims, dims, NULL);

   /* Check the start and count arguments */
   if (((nstart != 0) && (nstart != ndims)) || 
       ((ncount != 0) && (ncount != ndims))) {
     PyErr_SetString(PyExc_ValueError, "dimensions of start or count vectors not equal to dimensions of variable");
     return NULL;
   }

   ntotal = 1;
   ncount = PySequence_Length(count_seq);
   for (idim=0; idim < ncount; idim++) {
     ntotal *= hs_count[idim];
   }

   array = (PyArrayObject *) PyArray_FromDims(ndims, (int *) hs_count, PyArray_DOUBLE); 
   data = PyArray_DATA((PyObject *) array);

   (void) mivarget(mincid, varid,
		   hs_start, hs_count,
		   NC_DOUBLE, NULL, data);

   /* Clean up */
   if (!has_mincid) 
     (void) miclose(mincid);

   FREE(hs_start);
   FREE(hs_count);

   return((PyObject *)array);
}

static PyObject *readheader(PyObject *self, PyObject *args, PyObject *keywords)
{
   char *filename;
   int mincid = MI_ERROR;
   int has_mincid = FALSE;
   int ndims, dims[MAX_VAR_DIMS];
   int nvars, natts;
   char varname[NC_MAX_NAME], dimname[NC_MAX_NAME], attname[NC_MAX_NAME];
   nc_type datatype, atttype;
   int idim, iatt, ivar, iattval;
   long lendim;
   int lenatt;
/*    void *attval, *ptr; */
   char *attval, *ptr;
   PyObject *att_py, *varatt_py, *vardim_py, *var_dict_py, *value, *pyerr;
   PyObject *dim_dict_py, *dim_tuple, *return_py, *dim_py, *att_dict_py, *val_py;

   /* Check arguments */

   char *keyword_list[] = {"filename", "mincid", NULL};

   if (! PyArg_ParseTupleAndKeywords(args, keywords, "s|i", keyword_list, &filename, &mincid)) {
     PyErr_SetObject(PyExc_TypeError, Py_BuildValue("OO", args, keywords));
     return NULL;
   }

   if (mincid != MI_ERROR)
     has_mincid = TRUE;

   /* Open file if necessary */

   if (mincid == MI_ERROR)
     mincid = miopen(filename, NC_NOWRITE);
   (void) ncinquire(mincid, &ndims, &nvars, &natts, NULL);

   /* Get global attributes */ 

   att_dict_py = PyDict_New();

   for (iatt=0; iatt<natts; iatt++) {
     (void) ncattname(mincid, NC_GLOBAL, iatt, attname);
     (void) ncattinq(mincid, NC_GLOBAL, attname, &atttype, &lenatt);
     attval = MALLOC(nctypelen(atttype) * lenatt);
     ncattget(mincid, NC_GLOBAL, attname, attval);
     if (atttype != NC_CHAR) {
       if (lenatt > 1) {
	 att_py = PyTuple_New(lenatt);
	 for (iattval=0; iattval<lenatt; iattval++) {
	   ptr = attval + iattval * nctypelen(atttype);       
	   switch(atttype) {
	   case NC_BYTE :
	     value = PyInt_FromLong((long) *((char *) ptr));
	     break;
	   case NC_SHORT :
	     value = PyInt_FromLong((long) *((short *) ptr));
	     break;
	   case NC_INT :
	     value = PyInt_FromLong((long) *((int *) ptr));
	     break;
	   case NC_FLOAT :
	     value = PyFloat_FromDouble((double) *((float *) ptr));
	     break;
	   case NC_DOUBLE :
	     value = PyFloat_FromDouble((double) *((double *) ptr));
	     break;
	   default :
	     value = Py_BuildValue("O", Py_None);
	     break;
	   }
	   if (PyTuple_SetItem(att_py, iattval, value) != 0) {
	     pyerr = PyErr_Occurred();
	     if (pyerr != NULL) {
	       PyErr_Print();
	     }
	     Py_INCREF(Py_None);
	     return(Py_None);
	   }
	 }
       }
       else {
	 switch(atttype) {
	 case NC_BYTE :
	   att_py = PyInt_FromLong((long) *((char *) attval));
	   break;
	 case NC_SHORT :
	   att_py = PyInt_FromLong((long) *((short *) attval));
	   break;
	 case NC_INT :
	   att_py = PyInt_FromLong((long) *((int *) attval));
	   break;
	 case NC_FLOAT :
	   att_py = PyFloat_FromDouble((double) *((float *) attval));
	   break;
	 case NC_DOUBLE :
	   att_py = PyFloat_FromDouble((double) *((double *) attval));
	   break;
	 default :
	   att_py = Py_BuildValue("O", Py_None);
	   break;
	 }
       }
     }
     else {
       att_py = PyString_FromString((char *) attval);
     }
     FREE(attval);
     val_py = Py_BuildValue("(OO)", att_py, PyInt_FromLong((long) atttype));
     if (PyDict_SetItemString(att_dict_py, attname, val_py) != 0) {
       pyerr = PyErr_Occurred();
       if (pyerr != NULL) {
	 PyErr_Print();
       }
       Py_INCREF(Py_None);
       return(Py_None);
     }
   }

   /* Get dimension information */

   dim_dict_py = PyDict_New();

   for (idim=0; idim<ndims; idim++) {
     (void) ncdiminq(mincid, idim, dimname, &lendim);
     dim_tuple = PyTuple_New(2);
     PyTuple_SetItem(dim_tuple, 0, PyString_FromString(dimname));
     PyTuple_SetItem(dim_tuple, 1, PyInt_FromLong(lendim));
     dim_py = PyInt_FromLong((long) idim);
     PyDict_SetItem(dim_dict_py, dim_py, dim_tuple);
   }
   
   /* Get variable information */

   var_dict_py = PyDict_New();

   for (ivar=0; ivar<nvars; ivar++) {
     (void) ncvarinq(mincid, ivar, varname, &datatype, &ndims, dims, &natts);
     if (ndims > 0) {
       vardim_py = PyTuple_New(ndims);
       for (idim=0; idim<ndims; idim++) {
	 (void) ncdiminq(mincid, dims[idim], dimname, NULL);
	 PyTuple_SetItem(vardim_py, idim, PyString_FromString(dimname));
       }
     }
     else {
       vardim_py = Py_BuildValue("OO", Py_None, Py_None);
     }
     
     varatt_py = PyDict_New();
     for (iatt=0; iatt<natts; iatt++) {
       (void) ncattname(mincid, ivar, iatt, attname);
       (void) ncattinq(mincid, ivar, attname, &atttype, &lenatt);
       attval = MALLOC(nctypelen(atttype) * lenatt);
       ncattget(mincid, ivar, attname, attval);
       if (atttype != NC_CHAR) {
	 if (lenatt > 1) {
	   att_py = PyTuple_New(lenatt);
	   for (iattval=0; iattval<lenatt; iattval++) {
	     ptr = attval + iattval * nctypelen(atttype);       
	     switch(atttype) {
	     case NC_BYTE :
	       value = PyInt_FromLong((long) *((char *) ptr));
	       break;
	     case NC_SHORT :
	       value = PyInt_FromLong((long) *((short *) ptr));
	       break;
	     case NC_INT :
	       value = PyInt_FromLong((long) *((int *) ptr));
	       break;
	     case NC_FLOAT :
	       value = PyFloat_FromDouble((double) *((float *) ptr));
	       break;
	     case NC_DOUBLE :
	       value = PyFloat_FromDouble((double) *((double *) ptr));
	       break;
	     default :
	       break;
	     }
	     if (PyTuple_SetItem(att_py, iattval, value) != 0) {
	       pyerr = PyErr_Occurred();
	       if (pyerr != NULL) {
		 PyErr_Print();
	       }
	       Py_INCREF(Py_None);
	       return(Py_None);
	     }
	   }
	 }
	 else {
	   switch(atttype) {
	   case NC_BYTE :
	     att_py = PyInt_FromLong((long) *((char *) attval));
	     break;
	   case NC_SHORT :
	     att_py = PyInt_FromLong((long) *((short *) attval));
	     break;
	   case NC_INT :
	     att_py = PyInt_FromLong((long) *((int *) attval));
	     break;
	   case NC_FLOAT :
	     att_py = PyFloat_FromDouble((double) *((float *) attval));
	     break;
	   case NC_DOUBLE :
	     att_py = PyFloat_FromDouble((double) *((double *) attval));
	     break;
	   default :
	     break;
	   }
	 }
       }
       else {
	 att_py = PyString_FromString((char *) attval);
       }
       FREE(attval);

       val_py = Py_BuildValue("(OO)", att_py, PyInt_FromLong((long) atttype));

       if (PyDict_SetItemString(varatt_py, attname, val_py) != 0) {
	 pyerr = PyErr_Occurred();
	 if (pyerr != NULL) {
	   PyErr_Print();
	 }
	 Py_INCREF(Py_None);
	 return(Py_None);
       }
     }

     PyDict_SetItemString(varatt_py, "dimensions", vardim_py);
/*      Py_INCREF(varatt_py); */

     if (PyDict_SetItemString(var_dict_py, varname, varatt_py) != 0) {
       pyerr = PyErr_Occurred();
       if (pyerr != NULL) {
	 PyErr_Print();
       }
       Py_INCREF(Py_None);
       return(Py_None);
     }
   }
   
   return_py = PyDict_New();
   PyDict_SetItemString(return_py, "dim", dim_dict_py);
   PyDict_SetItemString(return_py, "att", att_dict_py);
   PyDict_SetItemString(return_py, "var", var_dict_py);

   if (!has_mincid)
     (void) miclose(mincid);

   return(return_py);
   
}

static PyObject *minccreate(PyObject *self, PyObject *args, PyObject *keywords)
{
   char *filename;
   int imgid, ndims;
   int datatype;
   int idim;
   int vrange_set=FALSE;
   int clobber=FALSE;
   char *history, *attname, *varname, *dimname, *signtype;
   double valid_range[2] = {0.0, -1.0};
   double *dimdircos;
   int vector_dimsize = -1;
   char *modality = NULL;
   int num_frame_times = 0;
   int num_frame_widths = 0;
   int cdfid, maxid, minid, varid;
   int dims[MAX_VAR_DIMS];
   int image_dims;
   int iatt, iframe, idir;
   long time_start, time_count, dimlen;
   double dimstep, dimstart;
   int nspecial_atts; 
   PyObject *frame_widths = Py_BuildValue("O", Py_None);
   PyObject *frame_times = Py_BuildValue("O", Py_None);
   PyObject *special_atts = Py_BuildValue("O", Py_None);
   PyObject *dimdircos_py;
   PyObject *cur_pair = Py_BuildValue("O", Py_None);
   PyObject *cur_value = Py_BuildValue("O", Py_None);
   PyObject *cur_var = Py_BuildValue("O", Py_None);
   PyObject *keys = Py_BuildValue("O", Py_None);
   PyObject *values = Py_BuildValue("O", Py_None);
   PyObject *att = Py_BuildValue("O", Py_None);
   PyObject *dimensions = Py_BuildValue("O", Py_None);
   PyObject *dim = Py_BuildValue("O", Py_None);
   PyObject *cur_obj, *cur_item;
   PyObject *cdfid_py;
   double *frame_data = NULL;

   char *keyword_list[] = {"filename", "dimensions", "datatype", "nvector", "clobber", "signtype", "modality", "history", "frame_widths", "frame_times", "special_atts", NULL};

   if (! PyArg_ParseTupleAndKeywords(args, keywords, "sO|iiisssOOO", keyword_list, &filename, &dimensions, &datatype, &vector_dimsize, &clobber, &signtype, &modality, &history, &frame_widths, &frame_times, &special_atts)) {
     PyErr_SetObject(PyExc_TypeError, Py_BuildValue("OO", args, keywords));
     return NULL;
   }

   if (PySequence_Length(frame_times) > 0) {
     num_frame_times = PySequence_Size(frame_times);
   }

   if (PySequence_Length(frame_widths) > 0) {
     num_frame_widths = PySequence_Size(frame_widths);
   }

   /* Create the file and save the time stamp */
   cdfid = micreate(filename, (clobber ? NC_CLOBBER : NC_NOCLOBBER));
   (void) miattputstr(cdfid, NC_GLOBAL, MIhistory, history);

   /* Set the number of image dimensions */
   image_dims = 2;

   ndims = PySequence_Size(dimensions);

   /* Create the dimensions */
   for (idim=0; idim<ndims; idim++) {

     dim = PySequence_GetItem(dimensions, idim);

     if (PyObject_HasAttrString(dim, "name")) {
       cur_obj = PyObject_GetAttrString(dim, "name");
       dimname = PyString_AsString(cur_obj);
       Py_DECREF(cur_obj);
     }
     else {
       PyErr_SetString(PyExc_ValueError, "dimension has no name.");
     }

     if (PyObject_HasAttrString(dim, "length")) {
       cur_obj = PyObject_GetAttrString(dim, "length");
       dimlen = PyInt_AsLong(cur_obj);
       Py_DECREF(cur_obj);
     }
     else {
       PyErr_SetString(PyExc_ValueError, "dimension has no length.");
     }

     if (PyObject_HasAttrString(dim, "step")) {
       cur_obj = PyObject_GetAttrString(dim, "step");
       dimstep = PyFloat_AsDouble(cur_obj);
       Py_DECREF(cur_obj);
     }
     else {
       dimstep = 0.0;
     }

     if (PyObject_HasAttrString(dim, "start")) {
       cur_obj = PyObject_GetAttrString(dim, "start");
       dimstart = PyFloat_AsDouble(cur_obj);
       Py_DECREF(cur_obj);
     }
     else {
       dimstart = 0.0;
     }

     if (PyObject_HasAttrString(dim, MIdirection_cosines) == TRUE) {
       dimdircos_py = PyObject_GetAttrString(dim, MIdirection_cosines);
     }
     else {
       dimdircos_py = NULL;
     }

      /* Create dimension */

     dims[idim] = ncdimdef(cdfid, dimname, dimlen);

      /* Create the variable if needed */
      if (STR_EQ(dimname, MItime)) {
	if (num_frame_times > 0) {
	  varid = micreate_std_variable(cdfid, MItime,
					NC_DOUBLE, 1, &dims[idim]);
         }
         if (num_frame_widths > 0) {
	   varid = micreate_std_variable(cdfid, MItime_width,
					 NC_DOUBLE, 1, &dims[idim]);
         }
      }
      else {
         varid = micreate_std_variable(cdfid, dimname,
                                       NC_INT, 0, NULL);
	 (void) miattputdbl(cdfid, varid, MIstep, dimstep);
	 (void) miattputdbl(cdfid, varid, MIstart, dimstart);
	 
	 if (dimdircos_py != NULL) {
	   dimdircos = MALLOC(3 * sizeof(*dimdircos));
	   cur_obj = PyObject_GetAttrString(dimdircos_py, "value");
	   for (idir=0; idir<3; idir++) {
	     cur_item = PyTuple_GetItem(cur_obj, idir);
	     dimdircos[idir] = PyFloat_AsDouble(cur_item);
	   }
	   Py_DECREF(cur_obj);
	   Py_DECREF(dimdircos_py);
	   (void) ncattput(cdfid, varid, MIdirection_cosines, NC_DOUBLE,
			   3, dimdircos);
	   FREE(dimdircos);
	 }
	 else {
	   dimdircos = MALLOC(3 * sizeof(*dimdircos));
	   for (idir=0; idir<3; idir++) {
	     dimdircos[idir] = 0.0;
	   }
	 }
      }
      Py_DECREF(dim);
   }

   /* Check for vector dimension */
   if (vector_dimsize > 0) {
      ndims++;
      image_dims++;
      dims[ndims-1] = ncdimdef(cdfid, MIvector_dimension,
                              (long) vector_dimsize);
   }

   /* Create the modality attribute */
   if (modality != NULL) {
      varid = micreate_group_variable(cdfid, MIstudy);
      (void) miattputstr(cdfid, varid, MImodality, modality);
   }

   if (PyObject_IsTrue(special_atts)) {
     nspecial_atts = PyDict_Size(special_atts);

     keys = PyDict_Keys(special_atts);
     values = PyDict_Values(special_atts);

     /* Create any special attributes */
     ncopts = 0;
     for (iatt=0; iatt < nspecial_atts; iatt++) {
       att = PySequence_GetItem(keys, iatt);
       attname = PyString_AsString(att);
       Py_DECREF(att);

       cur_pair = PySequence_GetItem(values, iatt);
       cur_var = PySequence_GetItem(cur_pair, 0);
       cur_value = PySequence_GetItem(cur_pair, 1);
       Py_DECREF(cur_pair);
       
       if (PyInt_AsLong(cur_var) == NC_GLOBAL) {
	 varid = NC_GLOBAL;
       }
       else {
	 varname = PyString_AsString(cur_var);
	 varid = ncvarid(cdfid, varname);
	 if (varid == MI_ERROR) {
	   varid = micreate_group_variable(cdfid, varname);
	 }
	 if (varid == MI_ERROR) {
	   varid = ncvardef(cdfid, varname, NC_INT, 0, NULL);
	 }
	 if (varid == MI_ERROR) {
	   continue;
	 }
       }
       
       if (PyString_AsString(cur_value) != NULL) {
	 (void) miattputstr(cdfid, varid, attname, PyString_AsString(cur_value));
       }
       else {
	 (void) miattputdbl(cdfid, varid, attname, PyFloat_AsDouble(cur_value));
       }
       Py_DECREF(cur_value);
       Py_DECREF(cur_var);
     }
     Py_DECREF(keys);
     Py_DECREF(values);

     ncopts = NC_VERBOSE | NC_FATAL;
   }

   /* Create the image */

   maxid = micreate_std_variable(cdfid, MIimagemax,
				 NC_DOUBLE, ndims-image_dims, dims);
   minid = micreate_std_variable(cdfid, MIimagemin,
				 NC_DOUBLE, ndims-image_dims, dims);

   imgid = micreate_std_variable(cdfid, MIimage, datatype, ndims, dims);
   (void) miattputstr(cdfid, imgid, MIcomplete, MI_FALSE);
   (void) miattputstr(cdfid, imgid, MIsigntype, signtype);

   if (vrange_set) 
      (void) miset_valid_range(cdfid, imgid, valid_range);

   /* End definition mode */
   (void) ncendef(cdfid);

   /* Write out the frame times and widths */
   time_start = 0;
   time_count = num_frame_times;
   if (num_frame_times > 0) {
     frame_data = MALLOC(num_frame_times * sizeof(*frame_data));
     for (iframe=0; iframe<num_frame_times; iframe++) {
       cur_value = PySequence_GetItem(frame_times, iframe);
       frame_data[iframe] = PyFloat_AsDouble(cur_value);
       Py_DECREF(cur_value);
     }

      (void) mivarput(cdfid, ncvarid(cdfid, MItime),
                      &time_start, &time_count,
                      NC_DOUBLE, NULL, (void *) frame_data);
      FREE(frame_data);
   }

   time_start = 0;
   time_count = num_frame_widths;
   if (num_frame_widths > 0) {
     frame_data = MALLOC(num_frame_widths * sizeof(*frame_data));
     for (iframe=0; iframe<num_frame_widths; iframe++) {
       cur_value = PySequence_GetItem(frame_widths, iframe);
       frame_data[iframe] = PyFloat_AsDouble(cur_value);
       Py_DECREF(cur_value);
     }

      (void) mivarput(cdfid, ncvarid(cdfid, MItime_width),
                      &time_start, &time_count,
                      NC_DOUBLE, NULL, (void *)frame_data);

      FREE(frame_data);
   }


   /* Close the file */
   
   cdfid_py = Py_BuildValue("i", cdfid);

   return(cdfid_py);

}

static PyObject *getINVALID_DATA(PyObject *self, PyObject *args, PyObject *keywords)
{
  PyObject *invalid_data;

  invalid_data = Py_BuildValue("f", -DBL_MAX);

  return(invalid_data);
}

