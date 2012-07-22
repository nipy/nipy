.. _brainvisa_repositories:

Can NIPY get something interesting from BrainVISA databases?
============================================================

I wrote this document to try to give more information to the NIPY
developers about the present and future of :term:`BrainVISA` database
system. I hope it will serve the discussion opened by Jarrod Millman
about a possible collaboration between the two projects on this
topic. Unfortunately, I do not know other projects providing similar
features (such as BIRN) so I will only focus on BrainVISA.

Yann Cointepas

2006-11-21

Introduction
------------

In BrainVISA, all the database system is home made and written in
Python. This system is based on the file system and allows to do
requests for both reading and writing (get the name of non existing
files). We will change this in the future by defining an API (such the
one introduced below) and by using at least two implementations, one
relying on a relational database system and one compatible with the
actual database system. Having one single API will make it possible,
for instance, to work on huge databases located on servers and on
smaller databases located in a laptop directory (with some
synchronization features). This system will be independent from the
BrainVISA application, it could be packaged separately. Unfortunately,
we cannot say when this work will be done (our developments are slowed
because all our lab will move in a new institute in January
2007). Here is a summary describing actual BrainVISA database system
and some thoughts of what it may become.

What is a database in BrainVISA today?
--------------------------------------

A directory is a BrainVISA database if the structure of its
sub-directories and the file names in this directory respect a set of
rules. These rules make it possible to BrainVISA to scan the whole
directory contents and to identify without ambiguity the database
elements. These elements are composed of the following information:

* *Data type:* identify the contents of a data (image, mesh,
   functional image, anatomical RM, etc). The data types are organized
   in hierarchy making it possible to decline a generic type in
   several specialized types. For example, there is a 4D Image type
   which is specialized in 3D Image. 3D Image is itself declined in
   several types of which T1 MRI and Brain mask.
* *File format:* Represent the format of files used to record a
   data. BrainVISA is able to recognize several file formats (for
   example DICOM, Analyze/SPM, GIS, etc). It is easy to add new data
   formats and to provide converters to make it possible for existing
   processes to use these new formats.
* *Files:* contains the names of the files (and/or directories) used
   to record the data.
* *Attributes:* an attribute is an association between a name and a
   value. A set of attributes is associated with each element of
   BrainVISA database. This set represents all of the characteristics
   of a data (as the image size, the name of the protocol
   corresponding to the data or the acquisition
   parameters). Attributes values are set by BrainVISA during
   directory scanning (typically protocol, group, subject, etc.).

It is possible to completely define the set of rules used to convert a
directory in a BrainVISA database. That allows the use of BrainVISA
without having to modify an existing file organization. However, the
writing of such a system of rules requires very good knowledge of
BrainVISA. This is why BrainVISA is provided with a default data
organization system that can be used easily.

A database can be used for deciding where to write data. The set of
rules is used to generate the appropriate file name according to the
data type, file format and attributes. This is a key feature that
greatly helps the users and allow automation.

It is not mandatory to use a database to process data with
BrainVISA. However, some important features are not available when you
are using data which are not in a database. For example, the BrainVISA
ability to construct a default output file name when an input data is
selected in a process relies on the database system. Moreover, some
processes use the database system to find data; for example, the brain
mask viewer tries to find the T1 MRI used to build the brain mask in
order to superimpose both images in an Anatomist window.

A few thoughts about a possible API for repositories
----------------------------------------------------

I think the most important point for data repositories is to define an
user API.  This API should be independent of data storage and of data
organization. Data organization is important because it is very
difficult to find a single organization that covers the needs of all
users in the long term. In this API, each data item should have an
unique identifier (let’s call it an URL). The rest of the API could be
divided in two parts:

#. An indexation system managing data organization. It defines
   properties attached to data items (for instance, “group” or
   “subject” can be seen as properties of an FMRI image) as well as
   possible user requests on the data. This indexation API could have
   several implementations (relational database, BIRN, BrainVISA,
   etc.).
#. A data storage system managing the link between the URL of a data
   item and its representation on a local file system. This system
   should take into account various file formats and various file
   storage systems (e.g. on a local file system, on a distant ftp
   site, as bytes blocks in a relational database).

This separation between indexation and storage is important for the
design of databases, it makes it possible, for instance, to use
distant or local data storage, or to define several indexations
(i.e. several data organizations) for the same data. However
indexation and data storage are not always independent. For example,
they are independent if we use a relational database for indexation
and URLs for storage, but they are not if file or directory names give
indexation information (like in BrainVISA databases described
above). At the user level, things can be simpler because the
separation can be hidden in one object: the repository. A repository
is composed of one indexation system and one data storage system and
manage all the links between them. The user can send requests to the
repository and receive a set of data items. Each data item contains
indexation information (via the indexation system) and gives access to
the data (via the storage system). Here is a sample of
what-user-code-could-be to illustrate what I have in mind followed by
a few comments:

::

   # Get an acces to one repository
   repository = openRepository( repositoryURL )
   # Create a request for selection of all the FMRI in the repository
   request = ‘SELECT * FROM FMRI’
   # Iterate on data items in the repository
   for item in repository.select( request ):
     print item.url
     # Item is a directory-like structure for properties access
     for property in item:
       print property, ‘=’, item[ property ]
     # Retrieve the file(s) (and directorie(s) if any) from the data storage system
     # and convert it to NIFTI format (if necessary).
     files = item.getLocalFiles( format=’NIFTI’ )
     niftiFileName = files[ 0 ]
     # Read the image and do something with it
     ...


#. I do not yet have a good idea of how to represent requests. Here, I
   chose to use SQL since it is simple to understand.
#. This code does not make any assumption on the properties that are
   associated to an FMRI image.
#. The method getLocalFiles can do nothing more than return a file
   name if the data item correspond to a local file in NIFTI
   format. But the same code can be used to acces a DICOM image
   located in a distant ftp server. In this case, getLocalFiles will
   manage the transfer of the DICOM file, then the conversion to the
   required NIFTI format and return name of temporary file(s).
#. getLocalFiles cannot always return just one file name because on
   the long term, there will be many data types (FMRI, diffusion MRI,
   EEG, MEG, etc.)  that are going to be stored in the
   repositories. These different data will use various file
   formats. Some of these formats can use a combination of files and
   directories (for instance, CTF MEG raw data are stored in a
   directory (``*.ds``), the structural sulci format of BrainVISA is
   composed of a file(``*.arg``) and a directory (``*.data``), NIFTI images
   can be in one or two files, etc. ).
#. The same kind of API can be used for writing data items in a
   repository. One could build a data item, adds properties and files
   and call something like repository.update( item ).

