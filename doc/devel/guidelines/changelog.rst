.. _changelog:

===============
 The ChangeLog
===============

**NOTE:** We have not kepted up with our ChangeLog.  This is here for
  future reference.  We will be more diligent with this when we have
  regular software releases.
 
If you are a developer with commit access, **please** fill a proper
ChangeLog entry per significant change.  The SVN commit messages may
be shorter (though a brief summary is appreciated), but a detailed
ChangeLog is critical.  It gives us a history of what has happened,
allows us to write release notes at each new release, and is often the
only way to backtrack on the rationale for a change (as the diff will
only show the change, not **why** it happened).

Please skim the existing ChangeLog for an idea of the proper level of
detail (you don't have to write a novel about a patch).

The existing ChangeLog is generated using (X)Emacs' fantastic
ChangeLog mode: all you have to do is position the cursor in the
function/method where the change was made, and hit 'C-x 4 a'.  XEmacs
automatically opens the ChangeLog file, mark a dated/named point, and
creates an entry pre-titled with the file and function name.  It
doesn't get any better than this.  If you are not using (X)Emacs,
please try to follow the same convention so we have a readable,
organized ChangeLog.

To get your name in the ChangeLog, set this in your .emacs file:

(setq user-full-name "Your Name")
(setq user-mail-address "youradddress@domain.com")

Feel free to obfuscate or omit the address, but at least leave your
name in.  For user contributions, try to give credit by name on
patches or significant ideas, but please do an @ -> -AT- replacement
in the email addresses (users have asked for this in the past).
