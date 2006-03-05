# Copyright CEA and IFR 49 (2000-2005)
#
#  This software and supporting documentation were developed by
#      CEA/DSV/SHFJ and IFR 49
#      4 place du General Leclerc
#      91401 Orsay cedex
#      France
#
# This software is governed by the CeCILL license version 2 under 
# French law and abiding by the rules of distribution of free software.
# You can  use, modify and/or redistribute the software under the 
# terms of the CeCILL license version 2 as circulated by CEA, CNRS
# and INRIA at the following URL "http://www.cecill.info". 
# 
# As a counterpart to the access to the source code and  rights to copy,
# modify and redistribute granted by the license, users are provided only
# with a limited warranty  and the software's author,  the holder of the
# economic rights,  and the successive licensors  have only  limited
# liability. 
# 
# In this respect, the user's attention is drawn to the risks associated
# with loading,  using,  modifying and/or developing or reproducing the
# software by the user in light of its specific status of free software,
# that may mean  that it is complicated to manipulate,  and  that  also
# therefore means  that it is reserved for developers  and  experienced
# professionals having in-depth computer knowledge. Users are therefore
# encouraged to load and test the software's suitability as regards their
# requirements in conditions enabling the security of their systems and/or 
# data to be ensured and,  more generally, to use and operate it in the 
# same conditions as regards security. 
# 
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license version 2 and that you accept its terms.

import struct, random, binascii

#----------------------------------------------------------------------------
class Uuid:
  '''
  An Uuid instance is a universal unique identifier. It is a 128 bits
  random value.
  '''
  
  def __init__( self, string_uuid=None ):
    if string_uuid is None:
      # Generate a new 128 bits uuid
      self.uuid = struct.pack( 'QQ', random.randrange( 2**64-1 ),
                               random.randrange( 2**64-1 ) )
    else:
      self.uuid = binascii.unhexlify( string_uuid[0:8] + string_uuid[9:13] + \
                                      string_uuid[14:18] + string_uuid[19:23] +\
                                      string_uuid[24:36] )

  def __str__( self ):
    return binascii.hexlify( self.uuid[0:4] ) + '-' + \
           binascii.hexlify( self.uuid[4:6] ) + '-' + \
           binascii.hexlify( self.uuid[6:8] ) + '-' + \
           binascii.hexlify( self.uuid[8:10] ) + '-' + \
           binascii.hexlify( self.uuid[10:16] )
           
  def __repr__( self ):
    return repr( str( self ) )
  
  def __hash__( self ):
    return hash( self.uuid )
  
  def __eq__( self, other ):
    if hasattr(other, 'uuid'):
      return self.uuid == other.uuid
    else:
      return False

#----------------------------------------------------------------------------
def getUuid( object ):
  if isinstance( object, Uuid ):
    return object
  elif isinstance( object, str ):
    return Uuid( object )
  return None
