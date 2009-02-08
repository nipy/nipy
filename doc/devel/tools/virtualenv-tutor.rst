Setting up virtualenv
=====================

.. Contents::

Overview
--------

virtualenv_ is a tool that allows you to install python packages in
isolated environments. In this way you can have multiple versions of
the same package without interference.  I started using this to easily
switch between multiple versions of numpy without having to constantly
reinstall and update my symlinks.  I also did this as a way to install
software for Scipy2008_, like the Enthought Tool Suite (ETS_), in a
way that would not effect my current development environment.

This tutorial is based heavily on a blog entry from Prabhu_.  I've
extended his shell script to make switching between virtual
environments a one-command operation.  (Few others who should be
credited for encouraging me to use virtualenv_: Gael_, Jarrod_,
Fernando_)


Installing
----------

Download and install the tarball for virtualenv_::

  tar xzf virtualenv-1.1.tar.gz 
  cd virtualenv-1.1
  python setup.py install --prefix=$HOME/local

Note: I install in a local directory, your install location may differ.

Setup virtualenv
----------------

Setup a base virtualenv directory.  I create this in a local
directory, you can do this in a place of your choosing.  All virtual
environments will be installed as subdirectories in here.::

  cd ~/local
  mkdir -p virtualenv

Create a virtualenv
-------------------

Create a virtual environment.  Here I change into my virtualenv
directory and create a virtual environment for my numpy-1.1.1
install::

  cd virtualenv/
  virtualenv numpy-1.1.1

Activate a virtualenv
---------------------

Set the numpy-1.1.1 as the *active* virtual environment::

  ln -s numpy-1.1.1/bin/activate .

We *enable* the numpy-1.1.1 virtual environment by sourcing it's
activate script.  This will prepend our `PATH` with the currently
active virtual environment.::

  # note: still in the ~/local/virtualenv directory
  source activate

We can see our `PATH` with the numpy-1.1.1 virtual environment at the
beginning.  Also not the label of the virtual environment prepends our
prompt.::

  (numpy-1.1.1)cburns@~ 20:23:54 $ echo $PATH
  /Users/cburns/local/virtualenv/numpy-1.1.1/bin:
  /Library/Frameworks/Python.framework/Versions/Current/bin:
  /Users/cburns/local/bin:
  /usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin:/usr/X11/bin:/usr/local/git/bin

Install packages into a virtualenv
----------------------------------

Then we install numpy-1.1.1 into the virtual environment.  In order to install
packages in the virtual environment, you need to use the *python* or
*easy_install* from that virtualenv.::

  ~/local/virtualenv/numpy-1.1.1/bin/python setup.py install

At this point any package I install in this virtual environment will
only be used when the environment is active.

Pragmatic virtualenv
--------------------

There are a few more manual steps in the above process then I wanted,
so I extended the shell script that Prabhu_ wrote to make this a
simple one-command operation.  One still needs to manually create each
virtual environment, and install packages, but this script simplifies
activating and deactivating them.

The `venv_switch.sh` script will:

* Activate the selected virtual environment. (Or issue an error if it
  doesn't exist.)
* Launch a new bash shell using the ~/.virtualenvrc file which sources
  the virtualenv/activate script.
* The activate script modifies the PATH and prepends the bash prompt
  with the virtualenv label.

`venv_switch.sh`::

    #!/bin/sh
    # venv_switch.sh
    # switch between different virtual environments

    # verify a virtualenv is passed in
    if [ $# -ne 1 ]
    then
        echo 'Usage: venv_switch venv-label'
        exit -1
    fi

    # verify the virtualenv exists
    VENV_PATH=~/local/virtualenv/$1

    # activate env script
    ACTIVATE_ENV=~/local/virtualenv/activate

    echo $VENV_PATH
    if [ -e $VENV_PATH ]
    then
        echo 'Switching to virtualenv' $VENV_PATH
        echo "Starting new bash shell.  Simply 'exit' to return to previous shell"
    else
        echo 'Error: virtualenv' $VENV_PATH 'does not exist!'
        exit -1
    fi

    rm $ACTIVATE_ENV
    ln -s ~/local/virtualenv/$1/bin/activate $ACTIVATE_ENV

    # Launch new terminal
    bash --rcfile ~/.virtualenvrc

Now to activate our numpy-1.1.1 virtual environment, we simply do::

  venv_switch.sh numpy-1.1.1

To deactivate the virtual environment and go back to your original
environment, just exit the bash shell::

  exit

The rcfile used to source the activate script.  I first source my
.profile to setup my environment and custom prompt, then source the
virtual environment.  `.virtualenvrc`::

    # rc file to initialize bash environment for virtualenv sessions

    # first source the bash_profile
    source ~/.bash_profile

    # source the virtualenv
    source ~/local/virtualenv/activate

Installing ETS 3.0.0
--------------------

As another example, I installed ETS_ 3.0.0 for the Tutorial sessions
at Scipy2008_.  (Note the prerequisites_.)

Set up an ets-3.0.0 virtualenv::

  cburns@virtualenv 15:23:50 $ pwd
  /Users/cburns/local/virtualenv

  cburns@virtualenv 15:23:50 $ virtualenv ets-3.0.0
  New python executable in ets-3.0.0/bin/python
  Installing setuptools............done.

  cburns@virtualenv 15:24:29 $ ls
  activate	ets-3.0.0	numpy-1.1.1	numpy-1.2.0b2

Switch into my ets-3.0.0 virtualenv using the `venv_switch.sh` script::

  cburns@~ 15:29:12 $ venv_switch.sh ets-3.0.0
  /Users/cburns/local/virtualenv/ets-3.0.0
  Switching to virtualenv /Users/cburns/local/virtualenv/ets-3.0.0
  Starting new bash shell.  Simply 'exit' to return to previous shell

Install ETS_ using easy_install.  Note we need to use the easy_install
from our ets-3.0.0 virtual environment::

  (ets-3.0.0)cburns@~ 15:31:41 $ which easy_install
  /Users/cburns/local/virtualenv/ets-3.0.0/bin/easy_install

  (ets-3.0.0)cburns@~ 15:31:48 $ easy_install ETS


.. include:: ../../links_names.txt

.. _Prabhu: http://prabhuramachandran.blogspot.com/2008/03/using-virtualenv-under-linux.html
.. _Gael: http://gael-varoquaux.info/blog/
.. _Jarrod: http://jarrodmillman.blogspot.com/
.. _Fernando: http://fdoperez.blogspot.com/search/label/scipy
.. _Scipy2008: http://conference.scipy.org/
.. _prerequisites: https://svn.enthought.com/enthought/wiki/Install
