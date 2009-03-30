"""
DataEngine - An engine handling path or collection of path.
    - file and directory search
    - file and firectory tagging
    - applying actions upon collection of files and directories
    - piping the actions
"""


#from PyQt4 import QtCore
import tarfile, bz2, os

from filesystem import path



class DataEngine:
    """
    The DataEngine Class
    """
    def __init__(self):
        
        self.catalog = {}
        self.groups = {}
        self.tags = []


    def newPath(self, itemPath):
        """
        Creates a path item.

        @param itemPath: a path
        @return: a L{PathItem} aware of this dataEngine
        """

        return PathItem(itemPath, self)

    def newCollection(self, items):
        """
        Creates a path collection.

        @param items: a list of path
        @return: a L{ItemCollection} aware of this dataEngine
        """

        return ItemCollection(items, self)

    #---------------------------------------------------------------------------------------------------------------
    # search methods

    def __find(self, itemGenerator, maxResults=0):
        results = ItemCollection([], self)
        for item in itemGenerator:
            results.append(PathItem(item, self))
            if len(results) == maxResults:
                itemGenerator.close()

        if maxResults == 1 and len(results) == 0:
            return None
        elif maxResults == 1:
            return results[0]
        else:
            return results

    def findDirectories(self, srcDir, dirPattern='*', dirDepth=0, maxResults=0):
        """
        Find all the directories matching the pattern.

        @param srcDir: directory to search from
        @param dirPattern: directory pattern to find
        @param dirDepth: max directory depth to search in (0 to infinite)
        @param maxResults: max results to return (0 to infinite)
        @return: 
            - directories matching the pattern
            - an L{ItemCollection} if maxResults is not equal to 1
            - a L{PathItem} if maxResults is equal to 1
        """

        srcDir = PathItem(srcDir, self)
        if not srcDir.isdir():
            raise Exception('DataEngine: -findDirectories- source directory %s is not a valid directory'%srcDir)

        childDirs = srcDir.walkdirs(dirPattern, dirDepth, 0, 'ignore')

        return self.__find(childDirs, maxResults)

    def findFiles(self, srcDir, filePattern='*', dirDepth=0, maxResults=0):
        """
        Find all the files matching the pattern.

        @param srcDir: directory to search from
        @param filePattern: file pattern to find
        @param dirDepth: max directory depth to search in (0 to infinite)
        @param maxResults: max results to return (0 to infinite)
        @return: 
            - files matching the pattern
            - an L{ItemCollection} if maxResults is not equal to 1
            - a L{PathItem} if maxResults is equal to 1
        """

        srcDir = PathItem(srcDir, self)
        if not srcDir.isdir():
            raise Exception('DataEngine: -findDirectories- source directory %s is not a valid directory'%srcDir)

        dirFiles = srcDir.walkfiles(filePattern, dirDepth, 0, 'ignore')

        return self.__find(dirFiles, maxResults)

    def getItem(self, *search_tags):
        """
        Finds one collected L{PathItem} that has the given tags.
        Equivalent to L{getSomeItems}(1,search_tags)

        @param search_tags: the search tags
        @return: a L{PathItem}
        """

        return self.getSomeItems(1, *search_tags)

    def getItems(self, *search_tags):
        """
        Finds all the collected L{PathItem} that has the given tags.
        Equivalent to L{getSomeItems}(0,search_tags)

        @param search_tags: the search tags
        @return: an L{ItemCollection}
        """

        return self.getSomeItems(0, *search_tags)

    def getSomeItems(self, maxResults, *search_tags):
        """
        Finds at most a specified number of L{PathItem} that has the given tags.

        @param search_tags: the search tags
        @return: an L{ItemCollection}
        """

        results = ItemCollection([], self)

        for item in self.catalog.keys():
            isResult = True
            for search_tag in search_tags:
                if not search_tag in self.catalog[item]:
                    isResult = False
                    break

            if isResult:
                results.append(PathItem(item, self))
                if len(results) == maxResults:
                    break
        
        if maxResults == 1 and len(results) == 0:
            return None
        elif maxResults == 1:
            return results[0]
        else:
            return results

    def getGroup(self, name):
        if not self.groups.has_key(name):
            raise Exception('DataEngine: -findGroupItems- group %s does not exist'%name)
        else:
            return self.getItems(0, *iter(self.groups[name]))


    #---------------------------------------------------------------------------------------------------------------
    # tag management methods

    def collect(self, item, *tags):
        """
        Tags a L{PathItem} or an L{ItemCollection}

        @param item: the L{PathItem} or the L{ItemCollection}
        @param tags: the item(s) tags
        @return: the input parameter item
        """

        if isinstance(item, list):
            collected = ItemCollection([], self)
            for el in item:
                el = PathItem(el, self)
                if not self.catalog.has_key(el):
                    self.catalog[el] = ()
                self.addItemTags(el, *tags)
                collected.append(el)

            return collected

        else:
            item = PathItem(item, self)
            if self.catalog.has_key(item):
                    self.addItemTags(item, *tags)
                    return item
            else:
                self.catalog[item] = tags
                return item

    def uncollect(self, item):
        """
        Untags a L{PathItem} or an L{ItemCollection}       

        @param item: the L{PathItem} or the L{ItemCollection}
        @return: the input parameter item
        """

        if isinstance(item, list):
            uncollected = ItemCollection([], self)
            for el in item:
                el = PathItem(el, self)
                if not self.catalog.has_key(el):
                    print 'DataEngine: item %s does not exist'%el
                    continue
                else:
                    self.catalog.pop(el)
                    uncollected.append(el)

            return uncollected
        else:
            item = PathItem(item, self)
            if self.catalog.has_key(item):
                self.catalog.pop(item)
                return item
            else:
                raise Exception('DataEngine: item %s does not exist'%item)

    def reset(self):
        """
        Resets all data structures.
        """

        self.catalog = {}
        self.groups = {}
        self.tags = []

    def addItemTags(self, item, *tags):
        return self.__addTags(self.catalog, item, *tags)

    def rmvItemTags(self, item, *tags):
        return self.__rmvTags(self.catalog, item, *tags)

    def addGroupTags(self, name, *tags):
        self.__addTags(self.groups, name, *tags)

    def rmvGroupTags(self, name, *tags):
        self.__rmvTags(self.groups, name, *tags)

    def __addTags(self, tagDict, key, *tags):
        if isinstance(key, list):
            for el in key: 
                self.__rmvTags(tagDict, el, *tags)
                tagDict[el] += tags
        else:
            self.__rmvTags(tagDict, key, *tags)
            tagDict[key] += tags

        for tag in tags:
                self.tags.append(tag)

        return key

    def __rmvTags(self, tagDict, key, *tags):
        for tag in tags:
            if tag in self.tags:
                self.tags.remove(tag)

        if isinstance(key, list):
            for el in key:
                if not tagDict.has_key(el):
                    print 'DataEngine: %s unknown in %s'%(el, tagDict)
                    continue

                remaining_tags = ()
                for tag in tagDict[el]:
                    if not tag in tags:
                        remaining_tags += (tag,)

                tagDict[el] = remaining_tags

        else:
            if not tagDict.has_key(key):
                raise Exception('DataEngine: %s unknown in %s'%(key, tagDict))

            remaining_tags = ()
            for tag in tagDict[key]:
                if not tag in tags:
                    remaining_tags += (tag,)

            tagDict[key] = remaining_tags

        return key

    #---------------------------------------------------------------------------------------------------------------
    # group management methods

    def setGroup(self, name, *tags):
        if self.groups.has_key(name):
            return False

        self.groups[name] = tags
        return True 

    def rmvGroup(self, name):
        if self.groups.has_key(name):
            return self.groups.pop(name)
        else:
            raise Exception('DataEngine: -rmvGroup- group %s does not exist'%name)

    #---------------------------------------------------------------------------------------------------------------
    # collection methods

    def delete(self, collection):
        """
        Deletes the items in the collection. If they were collected they are uncollected.

        @todo: if a directory is deleted, there is no uncollection of items within it
        @param collection: the L{ItemCollection}
        @return: None
        """

        for item in collection:
            item = path(item)
            if item.exists():
                if item.isdir():
                    item.rmtree()
                elif item.isfile():
                    item.remove()
            if self.catalog.has_key(item):
                self.uncollect(item)

    def move(self, collection, base_dest):
        """
        Moves the collection items to specified destination. Collected items are automatically updated, as well as the items within a moved directory.

        @param collection: the L{ItemCollection}
        @param base_dest: the directory path were the items are moved to
        @return: an L{ItemCollection} of the moved items
        """

        if not os.path.isdir(base_dest):
            raise Exception('DataEngine: -move- destination must be a valid directory')

        movedItems = ItemCollection([], self)

        for item in collection:
            item = path(item)
            if item.isdir():
                movedItems.append(self.__moveDir(item, base_dest))
            elif item.isfile():
                movedItems.append(self.__moveFile(item, base_dest))

        return movedItems

    def __moveFile(self, item, base_dest):
        item.move(base_dest)
        newItem = PathItem(os.path.join(base_dest, item.basename()), self)

        if self.catalog.has_key(item):
            tags = self.catalog[item]
            self.uncollect(item)
            self.collect(newItem, *iter(tags))

        return newItem

    #TODO: improve the update of catalog elements within the moved dir : look in the catalog if the element prefix matches the original dir value
    def __moveDir(self, directory, base_dest):
        dest = os.path.join(base_dest, directory.basename())

        directory.copytree(dest)            
        for item in self.findFiles(directory, '*', 0, 0):
            if self.catalog.has_key(item):
                tags = self.catalog[item]
                self.uncollect(item)
                newItem = PathItem(os.path.join(dest, item.lstrip(directory)), self)
                self.collect(newItem, *iter(tags))

        if self.catalog.has_key(directory):
            tags = self.catalog[directory]
            self.uncollect(directory)
            self.collect(dest, *iter(tags))

        directory.rmtree()

        return PathItem(dest, self)

    def copy(self, collection, dest):
        """
        Copies the collection items to specified destination

        @param collection: the L{ItemCollection}
        @param dest: the directory path were the items are copied to
        @return: an L{ItemCollection} of the copied items
        """

        if not os.path.isdir(dest):
            raise Exception('DataEngine: -copy- destination must be a valid directory')

        copies = ItemCollection([], self)

        for item in collection:
            item = path(item)
            if item.isdir():
                copies.append(self.__copyDir(item, dest))
            elif item.isfile():
                copies.append(self.__copyFile(item, dest))

        return copies

    def prefix(self, collection, prefix):
        newCollection = ItemCollection([], self)
        for item in collection:
            item = PathItem(item, self)
            newCollection.append(item.rename(prefix+item.basename()))

        return newCollection

    def suffix(self, collection, suffix):
        newCollection = ItemCollection([], self)
        for item in collection:
            item = PathItem(item, self)

            tmpItem = PathItem(item, self)
            item_ext = ''

            while tmpItem.splitext()[1] != '':
                item_ext = tmpItem.splitext()[1]+item_ext
                tmpItem = PathItem(tmpItem.splitext()[0])

            newCollection.append(item.rename(tmpItem+suffix+item_ext))

        return newCollection

    def __copyFile(self, item, dest):
        item.copy(dest)
        return PathItem(os.path.join(dest, item.basename()), self)

    def __copyDir(self, directory, base_dest):
        dest = os.path.join(base_dest, directory.basename())
        directory.copytree(dest)
        return PathItem(dest, self)

    def __updateCatalog(self, oldItem, newItem):
        if self.catalog.has_key(oldItem):
            tags = self.catalog[oldItem]
            self.uncollect(oldItem)
            self.collect(newItem, *iter(tags))

#FIXME: I can't get the object to update itself when moving or renaming it
class PathItem(path):
    def __new__(cls, itemPath, dataEngine=DataEngine()):
        self = path.__new__(cls, itemPath)
        self.dataEngine = dataEngine
        return self

    def findDirectories(self, dirPattern='*', dirDepth=0, maxResults=0):
        return self.dataEngine.findDirectories(self, dirPattern, dirDepth, maxResults)

    def findFiles(self, filePattern='*', dirDepth=0, maxResults=0):
        return self.dataEngine.findFiles(self, filePattern, dirDepth, maxResults)

    def copy(self, dest):
        return self.dataEngine.copy([self], dest)[0]

    def delete(self):
        return self.dataEngine.delete([self])[0]

    def move(self, base_dest):
        return self.dataEngine.move([self], base_dest)[0]

    def addTags(self, *tags):
        self.dataEngine.addItemsTags(self, *tags)

    def rmvTags(self, *tags):
        self.dataEngine.rmvItemsTags(self, *tags)

    def collect(self, *tags):
        return self.dataEngine.collect(self, *tags)

    def uncollect(self):
        return self.dataEngine.uncollect(self)

    def rename(self, basename):
        newName = os.path.join(self.dirname(), path(basename).basename())
        newItem = PathItem(newName, self.dataEngine)
        os.rename(self, newItem)

        if self.dataEngine.catalog.has_key(self):
            tags = self.dataEngine.catalog[self]
            self.dataEngine.uncollect(self)
            self.dataEngine.collect(newItem, *iter(tags))

        return newItem

    def prefix(self, prefix):
        return self.dataEngine.prefix([self], prefix)[0]

    def suffix(self, suffix):
        return self.dataEngine.suffix([self], suffix)[0]

#TODO: check that the itemCollection members are pathItems
class ItemCollection(list):
    def __init__(self, items=[], dataEngine=DataEngine()):
        list.__init__(self, items)
        self.dataEngine = dataEngine

    def findDirectories(self, dirPattern='*', dirDepth=0, maxResults=0):
        results = ItemCollection([], self.dataEngine)
        for item in self:
            if maxResults == 1:
                results.append(self.dataEngine.findDirectories(item, dirPattern, dirDepth, maxResults))
            else:
                results.extend(self.dataEngine.findDirectories(item, dirPattern, dirDepth, maxResults))
        return results

    def findFiles(self, filePattern='*', dirDepth=0, maxResults=0):
        results = ItemCollection([], self.dataEngine)
        for item in self:
            if maxResults == 1:
                results.append(self.dataEngine.findFiles(item, filePattern, dirDepth, maxResults))
            else:
                results.extend(self.dataEngine.findFiles(item, filePattern, dirDepth, maxResults))
        return results

    def copy(self, dest):
        return self.dataEngine.copy(self, dest)

    def delete(self):
        return self.dataEngine.delete(self)

    def move(self, base_dest):
        return self.dataEngine.move(self, base_dest)

    def collect(self, *tags):
        return self.dataEngine.collect(self, *tags)

    def uncollect(self):
        return self.dataEngine.uncollect(self)

    def addTags(self, *tags):
        return self.dataEngine.addItemsTags(self, *tags)

    def rmvTags(self, *tags):
        return self.dataEngine.rmvItemsTags(self, *tags)

    def prefix(self, prefix):
        return self.dataEngine.prefix(self, prefix)

    def suffix(self, suffix):
        return self.dataEngine.suffix(self, suffix)

#TODO:
#method gettags
#maybe add a dictionary in addition of the list of tags

if __name__ == "__main__":
    dataEngine = DataEngine()
    #results = dataEngine.findFiles('/home/ys218403/Resources/packages', '*.dcm', 0, 0)

    #dataEngine.collect(['/home/ys218403/Resources/test/pouet1', '/home/ys218403/Resources/pouet3'], 'pouet', 'a')
    #dataEngine.collect('/home/ys218403/Resources/pouet2', 'pouet', 'b')
    #dataEngine.collect('/home/ys218403/Resources/test', 'dir')
    #dataEngine.setGroup('test', 'dir')
    
    #dataEngine.findDirectories('/home/ys218403/Resources/', '*t*', 1, 0).collect('test')
    #dataEngine.setGroup('pouet', 'test')
    #pouetGroup = dataEngine.getGroup('pouet')





