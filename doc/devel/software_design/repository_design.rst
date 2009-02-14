.. _repository_design:

===================
 Repository design
===================

See also :ref:`repository_api` and :ref:`brainvisa_repositories`

For the NIPY system, there seems to be interest for the following:

* Easy distributed computing
* Easy scripting, replicating the same analysis on different data
* Flexibility - easy of inter-operation with other brain imaging systems

At a minimum, this seems to entail the following requirements for the
NIPY repository system:

* Unique identifiers of data, which can be abstracted from the most
  local or convenient data storage
* A mechanism for mapping the canonical data model(s) from NIPY to an
  arbitrary, and potentially even inconsistent repository structure
* A set of semantic primitives / metadata slots, enabling for example:
   * "all scans from this subject"
   * "the first scan from every subject in the control group"
   * "V1 localizer scans from all subjects"
   * "Extract the average timecourse for each subject from the ROI
     defined by all voxels with t > 0.005 in the V1 localizer scan for
     that subject"

These problems are not unique to the problem of brain imaging data,
and in many cases have been treated in the domains of database design,
geospatial and space telescope data, and the semantic web.
Technologies of particular interest include:

* HDF5 - the basis of MINC 2.0 (and potentially NIFTII 2), the most
  recent development in the more general CDF / HDF series (and very
  highly regarded).  There are excellent python binding available in
  `PyTables <http://www.pytables.org>`_.
* Relational database design - it would be nice to efficiently select
  data based on any arbitrary subset of attributes associated with
  that data.
* The notion of `URI <http://www.w3.org/Addressing/>`_ developed under
  the guidance of the w3c.  Briefly, a URI consists of:

   * An authority (i.e. a domain name controlled by a particular
     entity)
   * A path - a particular resource specified by that authority
   * Abstraction from storage (as opposed to a URL) - a URI does not
     necessarily include the information necessary for retrieving the
     data referred to, though it may.

* Ways of dealing with hierarchical data as developed in the XML field
  (though these strategies could be implemented potentially in other
  hierarchical data formats - even filesystems).

Note that incorporation of any of the above ideas does not require the
use of the actual technology referenced.  For example, relational
queries can be made in PyTables in many cases **more efficiently**
than in a relational database by storing everything in a single
denormalized table.  This data structure tends to be more efficient
than the equivalent normalized relational database format in the cases
where a single data field is much larger than the others (as is the
case with the data array in brain imaging data).  That said, adherance
to standards allows us to leverage existing code which may be tuned to
a degree that would be beyond the scope of this project (for example,
fast Xpath query libraries, as made available via lxml in Python).
