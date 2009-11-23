===================
 SSH under Windows
===================

To use ssh with bzr under windows you need to:

    * Install Putty. Do NOT add an environment variable BZR_SSH (this was needed for old bazaar versions that did not include paramiko). 

    * Generate a RSA ssh-2 key using puttygen.exe

    * You need to start pageant and add you have registered on bzr.
