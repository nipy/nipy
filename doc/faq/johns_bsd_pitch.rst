.. _johns-bsd-pitch:

Why we should be using BSD
==========================
   John Hunter - 16 Dec 2004

I'll start by summarizing what many of you already know about open
source licenses.  I believe this discussion is broadly correct, though
it is not a legal document and if you want legally precise statements
you should reference the original licenses cited here.  The
`Open-Source-Initiative <http://www.opensource.org>`_ is a clearing
house for OS licenses, so you can read more there.

The two dominant license variants in the wild are GPL-style and
BSD-style.  There are countless other licenses that place specific
restrictions on code reuse, but the purpose of this document is to
discuss the differences between the GPL and BSD variants, specifically
in regards to my experience developing matplotlib_ and in my
discussions with other developers about licensing issues.

The best known and perhaps most widely used license is the
:term:`GPL`, which in addition to granting you full rights to the
source code including redistribution, carries with it an extra
obligation.  If you use GPL code in your own code, or link with it,
your product must be released under a GPL compatible license.  I.e., you
are required to give the source code to other people and give them the
right to redistribute it as well.  Many of the most famous and widely
used open source projects are released under the GPL, including linux,
gcc and emacs.

The second major class are the :term:`BSD` and BSD-style licenses
(which includes MIT and the python PSF license).  These basically
allow you to do whatever you want with the code: ignore it, include it
in your own open source project, include it in your proprietary
product, sell it, whatever.  python itself is released under a BSD
compatible license, in the sense that, quoting from the PSF
license page

  There is no GPL-like "copyleft" restriction. Distributing
  binary-only versions of Python, modified or not, is allowed. There
  is no requirement to release any of your source code. You can also
  write extension modules for Python and provide them only in binary
  form.

Famous projects released under a BSD-style license in the permissive
sense of the last paragraph are the BSD operating system, python, and
TeX.

I believe the choice of license is an important one, and I advocate a
BSD-style license.  In my experience, the most important commodity an
open source project needs to succeed is users.  Of course, doing
something useful is a prerequisite to getting users, but I also
believe users are something of a prerequisite to doing something
useful.  It is very difficult to design in a vacuum, and users drive
good software by suggesting features and finding bugs.  If you satisfy
the needs of some users, you will inadvertently end up satisfying the
needs of a large class of users.  And users become developers,
especially if they have some skills and find a feature they need
implemented, or if they have a thesis to write.  Once you have a lot
of users and a number of developers, a network effect kicks in,
exponentially increasing your users and developers.  In open source
parlance, this is sometimes called competing for mind share.

So I believe the number one (or at least number two) commodity an open
source project can possess is mind share, which means you
want as many damned users using your software as you can get.  Even
though you are giving it away for free, you have to market your
software, promote it, and support it as if you were getting paid for
it.  Now, how does this relate to licensing, you are asking?

Most software companies will not use GPL code in their own software,
even those that are highly committed to open source development, such
as enthought_, out of legitimate concern that use of the GPL will
"infect" their code base by its viral nature.  In effect, they want to
retain the right to release some proprietary code.  And in my
experience, companies make for some of the best developers, because
they have the resources to get a job done, even a boring one, if they
need it in their code.  Two of the matplotlib backends (FLTK and WX)
were contributed by private sector companies who are using matplotlib
either internally or in a commercial product -- I doubt these
companies would have been using matplotlib if the code were GPL.  In
my experience, the benefits of collaborating with the private sector
are real, whereas the fear that some private company will "steal" your
product and sell it in a proprietary application leaving you with
nothing is not.

There is a lot of GPL code in the world, and it is a constant reality
in the development of matplotlib that when we want to reuse some
algorithm, we have to go on a hunt for a non-GPL version.  Most
recently this occurred in a search for a good contouring algorithm.  I
worry that the "license wars", the effect of which are starting to be
felt on many projects, have a potential to do real harm to open source
software development.  There are two unpalatable options.  1) Go with
GPL and lose the mind-share of the private sector 2) Forgo GPL code
and retain the contribution of the private sector.  This is a very
tough decision because there is a lot of very high quality software
that is GPLd and we need to use it; they don't call the license `viral
<http://www.linuxinsider.com/story/33968.html>`_ for nothing.

The third option, which is what is motivating me to write this, is to
convince people who have released code under the GPL to re-release it
under a BSD compatible license.  Package authors retain the copyright
to their software and have discretion to re-release it under a license
of their choosing.  Many people choose the GPL when releasing a
package because it is the most famous open source license, and did not
consider issues such as those raised here when choosing a license.
When asked, these developers will often be amenable to re-releasing
their code under a more permissive license.  Fernando Perez did this
with ipython, which was released under the :term:`LGPL` and then
re-released under a BSD license to ease integration with matplotlib, 
scipy and enthought code.   The LGPL is more permissive than the GPL,
allowing you to link with it non-virally, but many companies are still
loath to use it out of legal concerns, and you cannot reuse LGPL code
in a proprietary product.

So I encourage you to release your code under a BSD compatible
license, and when you encounter an open source developer whose code
you want to use, encourage them to do the same.  Feel free to forward
this document to them.

.. include:: ../links_names.txt
