.. _commit-codes:

Commit message codes
---------------------

Please prefix all commit summaries with one (or more) of the following labels.
This should help others to easily classify the commits into meaningful
categories:

  * *BF* : bug fix
  * *RF* : refactoring
  * *ENH* : new feature or extended functionality
  * *BW* : addresses backward-compatibility
  * *OPT* : optimization
  * *BK* : breaks something and/or tests fail
  * *DOC*: for all kinds of documentation related commits
  * *TEST* : for adding or changing tests
  * *STY* : PEP8 conformance, whitespace changes etc that do not affect
    function.
  # *WIP* : Work in progress; please try and avoid using this one, and rebase
    incomplete changes into functional units using e.g. ``git rebase -i``

So your commit message might look something like this::

    TEST: relax test threshold slightly

    Attempted fix for failure on windows test run when arrays are in fact
    very close (within 6 dp).

Keeping up a habit of doing this is useful because it makes it much easier to
see at a glance which changes are likely to be important when you are looking
for sources of bugs, fixes, large refactorings or new features.

Pull request codes
------------------

When you submit a pull request to github, github will ask you for a summary.  If
your code is not ready to merge, but you want to get feedback, please consider
using ``WIP - me working on image design`` or similar for the title of your pull
request. That way we will all know that it's not yet ready to merge and that
you may be interested in more fundamental comments about design.
